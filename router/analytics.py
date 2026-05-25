"""
File: router/analytics.py

Purpose:
Append-only JSONL decision log with an in-memory query API.  Every routing
decision is written as a single JSON line to routing_analytics.jsonl.  Aggregation
queries (savings, model usage, task distribution, cache/fallback rates) run over
the in-memory copy for speed; the file is the durable audit trail.

Main Classes:
  RoutingAnalytics

Config Dependencies (all in config.py):
  ANALYTICS_MAX_STARTUP_ENTRIES — max JSONL lines replayed into memory on startup

Key Methods to Modify:
  _decision_to_dict()   — add new fields here when you want to log them
  update_actual()       — patch a decision entry with post-call actuals (cost, quality, latency)
  get_savings_summary() — savings report for a user

🔧 EXTENSION POINT: to add a new analytics field:
  1. Add it to _decision_to_dict() with a sensible default
  2. Update any query methods that read it to use .get("field", default)
  3. No migration needed for existing log entries

Things NOT to change without discussion:
  - The append-only write model (the JSONL file must not be mutated, only appended)
  - The correlation_id index (_index) — it enables O(1) update_actual() lookups
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from ._io_utils import safe_resolve
from .config import ANALYTICS_MAX_STARTUP_ENTRIES, MAX_STATE_FILE_BYTES
from .schemas import RoutingDecision

log = structlog.get_logger(__name__)


class RoutingAnalytics:
    """
    Append-only JSONL decision log with in-memory query support.

    When ``log_path`` is None, analytics are in-memory only (no file is read or
    written). When an explicit path is provided, the log is written synchronously
    and replayed on startup to restore the in-memory list.
    """

    def __init__(
        self,
        log_path: str | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        self._path = safe_resolve(log_path, base_dir) if log_path else None
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}  # correlation_id → index in _entries (O(1) updates)
        if self._path is not None:
            self._load_existing()

    # ── Write path ──────────────────────────────────────────────────────────

    def log_decision(
        self,
        decision: RoutingDecision,
        user_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Append a routing decision to the log."""
        entry = self._decision_to_dict(decision)
        if user_id:
            entry["user_id"] = user_id
        entry.update(extra)
        self._append(entry)

    def log_cache_hit(
        self,
        correlation_id: str,
        cached: Any,
        user_id: str | None = None,
    ) -> None:
        """Record a cache hit (no model was called)."""
        entry: dict[str, Any] = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "cache_hit": True,
            "chosen_model": getattr(cached, "model_used", {}).model_id
            if hasattr(getattr(cached, "model_used", None), "model_id")
            else "cached",
            "estimated_cost": 0.0,
        }
        if user_id:
            entry["user_id"] = user_id
        self._append(entry)

    def update_actual(
        self,
        correlation_id: str,
        actual_cost: float | None = None,
        quality_score: float | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        fallback_used: bool = False,
        fallback_models_tried: list[str] | None = None,
        fallback_reasons: list[str] | None = None,
    ) -> None:
        """
        Patch an existing log entry with actual cost / quality data that is only
        known after the model call completes.  Uses the correlation_id index for O(1) lookup.
        """
        with self._lock:
            idx = self._index.get(correlation_id)
            if idx is None:
                return
            entry = self._entries[idx]
            if actual_cost is not None:
                entry["actual_cost"] = actual_cost
            if quality_score is not None:
                entry["quality_score"] = quality_score
            if latency_ms is not None:
                entry["latency_ms"] = latency_ms
            if input_tokens is not None:
                entry["input_tokens"] = input_tokens
            if output_tokens is not None:
                entry["output_tokens"] = output_tokens
            if fallback_used:
                entry["fallback_used"] = fallback_used
            if fallback_models_tried:
                entry["fallback_models_tried"] = fallback_models_tried
            if fallback_reasons:
                entry["fallback_reasons"] = fallback_reasons

    # ── Query API ────────────────────────────────────────────────────────────

    def get_savings_summary(self, user_id: str, period: str = "all") -> dict[str, float]:
        """Total spend, premium-equivalent cost, savings amount and percentage."""
        entries = self._filter(user_id=user_id, period=period)
        spent = sum(e.get("estimated_cost", 0.0) for e in entries)
        premium = sum(e.get("premium_equivalent_cost", 0.0) for e in entries)
        saved = max(0.0, premium - spent)
        pct = (saved / premium * 100) if premium > 0 else 0.0
        return {
            "total_spent": round(spent, 6),
            "premium_equivalent": round(premium, 6),
            "total_saved": round(saved, 6),
            "savings_percentage": round(pct, 2),
        }

    def get_model_usage_breakdown(self, user_id: str) -> dict[str, int]:
        entries = self._filter(user_id=user_id)
        counts: dict[str, int] = defaultdict(int)
        for e in entries:
            counts[e.get("chosen_model", "unknown")] += 1
        return dict(counts)

    def get_task_type_distribution(self, user_id: str) -> dict[str, int]:
        entries = self._filter(user_id=user_id)
        dist: dict[str, int] = defaultdict(int)
        for e in entries:
            dist[e.get("task_type", "unknown")] += 1
        return dict(dist)

    def get_cache_hit_rate(self, user_id: str) -> float:
        entries = self._filter(user_id=user_id)
        if not entries:
            return 0.0
        hits = sum(1 for e in entries if e.get("cache_hit"))
        return round(hits / len(entries), 4)

    def get_fallback_rate(self, user_id: str) -> float:
        entries = self._filter(user_id=user_id)
        if not entries:
            return 0.0
        fallbacks = sum(1 for e in entries if e.get("fallback_used"))
        return round(fallbacks / len(entries), 4)

    def get_avg_quality_by_model(self, user_id: str) -> dict[str, float]:
        entries = self._filter(user_id=user_id)
        sums: dict[str, list[float]] = defaultdict(list)
        for e in entries:
            q = e.get("quality_score")
            if q is not None:
                sums[e.get("chosen_model", "unknown")].append(q)
        return {m: round(sum(vs) / len(vs), 4) for m, vs in sums.items() if vs}

    def export_csv(self, user_id: str, period: str = "all") -> str:
        """Return CSV text for the filtered entries."""
        entries = self._filter(user_id=user_id, period=period)
        if not entries:
            return ""
        fieldnames = list(entries[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)
        return buf.getvalue()

    def all_entries(self) -> list[dict[str, Any]]:
        """Return a snapshot of all logged entries (for demo/testing)."""
        with self._lock:
            return list(self._entries)

    # ── Private ─────────────────────────────────────────────────────────────

    def _append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            idx = len(self._entries)
            self._entries.append(entry)
            cid = entry.get("correlation_id")
            if cid:
                self._index[cid] = idx
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
                fh.flush()
                # No os.fsync: page-cache durability is enough for telemetry,
                # and fsync-per-write was a 10-50ms bottleneck under load.
            # Restrict to owner read/write — analytics may contain user PII.
            # Best-effort: chmod can fail on Windows or unusual filesystems.
            try:
                os.chmod(self._path, 0o600)
            except OSError as chmod_exc:
                log.debug(
                    "analytics_chmod_failed",
                    path=str(self._path),
                    error=str(chmod_exc),
                )
        except OSError as exc:
            log.warning("analytics_write_failed", error=str(exc))

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        try:
            size = self._path.stat().st_size
        except OSError as exc:
            log.warning("analytics_stat_failed", error=str(exc))
            return
        if size > MAX_STATE_FILE_BYTES:
            log.error(
                "state_file_too_large",
                path=str(self._path),
                size=size,
                limit=MAX_STATE_FILE_BYTES,
            )
            return
        try:
            lines: list[str] = []
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        lines.append(line)
            # Avoid OOM on large log files: only keep the most recent entries.
            if len(lines) > ANALYTICS_MAX_STARTUP_ENTRIES:
                lines = lines[-ANALYTICS_MAX_STARTUP_ENTRIES:]
            for line in lines:
                entry = json.loads(line)
                idx = len(self._entries)
                self._entries.append(entry)
                cid = entry.get("correlation_id")
                if cid:
                    self._index[cid] = idx
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("analytics_load_failed", error=str(exc))

    def _filter(self, user_id: str | None = None, period: str = "all") -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._entries)
        if user_id:
            entries = [e for e in entries if e.get("user_id") == user_id]
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if period == "day":
            entries = [e for e in entries if _parse_ts(e.get("timestamp", "")).date() == now.date()]
        elif period == "month":
            entries = [
                e
                for e in entries
                if _parse_ts(e.get("timestamp", "")).year == now.year
                and _parse_ts(e.get("timestamp", "")).month == now.month
            ]
        return entries

    @staticmethod
    def _decision_to_dict(decision: RoutingDecision) -> dict[str, Any]:
        model_id = decision.chosen_model.model_id if decision.chosen_model else "none"
        model_tier = decision.chosen_model.tier if decision.chosen_model else "none"
        return {
            "correlation_id": decision.correlation_id,
            "timestamp": decision.timestamp.isoformat(),
            "chosen_model": model_id,
            "chosen_tier": model_tier,
            "cache_hit": decision.cache_hit,
            "context_compressed": decision.context_was_compressed,
            "estimated_cost": decision.estimated_cost,
            "premium_equivalent_cost": decision.estimated_cost + decision.estimated_savings,
            "savings": decision.estimated_savings,
            "routing_rule": decision.routing_rule_matched,
            "cost_blocked": decision.cost_blocked,
            "fallback_used": False,
            "fallback_models_tried": [],
            "fallback_reasons": [],
            "was_ab_exploration": "ab_exploration" in decision.routing_rule_matched,
            "ab_exploration_tier_used": None,
        }


def _parse_ts(ts_str: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime(1970, 1, 1)
