"""
Decision logging with correlation IDs.

Every routing decision is appended as a single JSON line to a JSONL file.
All aggregation queries work over the in-memory list so they're fast;
the file is the durable audit trail.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from .schemas import RoutingDecision

log = structlog.get_logger(__name__)

_DEFAULT_LOG_PATH = "router/routing_analytics.jsonl"


class RoutingAnalytics:
    """
    Append-only JSONL decision log with in-memory query support.

    The log is written synchronously (fast path — no async I/O) and held in
    memory for aggregation queries.  On restart the JSONL is replayed to
    restore the in-memory list.
    """

    def __init__(self, log_path: str | None = None) -> None:
        self._path   = Path(log_path or _DEFAULT_LOG_PATH)
        self._lock   = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._load_existing()

    # ── Write path ──────────────────────────────────────────────────────────

    def log_decision(self, decision: RoutingDecision, **extra: Any) -> None:
        """Append a routing decision to the log."""
        entry = self._decision_to_dict(decision)
        entry.update(extra)
        self._append(entry)

    def log_cache_hit(self, correlation_id: str, cached: Any) -> None:
        """Record a cache hit (no model was called)."""
        entry: dict[str, Any] = {
            "correlation_id": correlation_id,
            "timestamp":      datetime.utcnow().isoformat(),
            "cache_hit":      True,
            "chosen_model":   getattr(cached, "model_used", {}).model_id if hasattr(getattr(cached, "model_used", None), "model_id") else "cached",
            "estimated_cost": 0.0,
        }
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
        known after the model call completes.
        """
        with self._lock:
            for entry in reversed(self._entries):
                if entry.get("correlation_id") == correlation_id:
                    if actual_cost    is not None: entry["actual_cost"]    = actual_cost
                    if quality_score  is not None: entry["quality_score"]  = quality_score
                    if latency_ms     is not None: entry["latency_ms"]     = latency_ms
                    if input_tokens   is not None: entry["input_tokens"]   = input_tokens
                    if output_tokens  is not None: entry["output_tokens"]  = output_tokens
                    if fallback_models_tried:      entry["fallback_used"]  = fallback_used
                    if fallback_models_tried:      entry["fallback_models_tried"] = fallback_models_tried
                    if fallback_reasons:           entry["fallback_reasons"] = fallback_reasons
                    break

    # ── Query API ────────────────────────────────────────────────────────────

    def get_savings_summary(self, user_id: str, period: str = "all") -> dict[str, float]:
        """Total spend, premium-equivalent cost, savings amount and percentage."""
        entries = self._filter(user_id=user_id, period=period)
        spent    = sum(e.get("estimated_cost", 0.0) for e in entries)
        premium  = sum(e.get("premium_equivalent_cost", 0.0) for e in entries)
        saved    = max(0.0, premium - spent)
        pct      = (saved / premium * 100) if premium > 0 else 0.0
        return {
            "total_spent":        round(spent, 6),
            "premium_equivalent": round(premium, 6),
            "total_saved":        round(saved, 6),
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
            self._entries.append(entry)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            log.warning("analytics_write_failed", error=str(exc))

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._entries.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("analytics_load_failed", error=str(exc))

    def _filter(self, user_id: str | None = None, period: str = "all") -> list[dict[str, Any]]:
        with self._lock:
            entries = list(self._entries)
        if user_id:
            entries = [e for e in entries if e.get("user_id") == user_id]
        now = datetime.utcnow()
        match period:
            case "day":
                entries = [e for e in entries
                           if _parse_ts(e.get("timestamp", "")).date() == now.date()]
            case "month":
                entries = [e for e in entries
                           if (ts := _parse_ts(e.get("timestamp", ""))).year == now.year
                           and ts.month == now.month]
            case _:
                pass
        return entries

    @staticmethod
    def _decision_to_dict(decision: RoutingDecision) -> dict[str, Any]:
        model_id   = decision.chosen_model.model_id   if decision.chosen_model else "none"
        model_tier = decision.chosen_model.tier        if decision.chosen_model else "none"
        return {
            "correlation_id":          decision.correlation_id,
            "timestamp":               decision.timestamp.isoformat(),
            "chosen_model":            model_id,
            "chosen_tier":             model_tier,
            "cache_hit":               decision.cache_hit,
            "context_compressed":      decision.context_was_compressed,
            "estimated_cost":          decision.estimated_cost,
            "premium_equivalent_cost": decision.estimated_cost + decision.estimated_savings,
            "savings":                 decision.estimated_savings,
            "routing_rule":            decision.routing_rule_matched,
            "cost_blocked":            decision.cost_blocked,
            "fallback_used":           False,
            "fallback_models_tried":   [],
            "fallback_reasons":        [],
            "was_ab_exploration":      "ab_exploration" in decision.routing_rule_matched,
            "ab_exploration_tier_used": None,
        }


def _parse_ts(ts_str: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime(1970, 1, 1)
