"""
Learning loop — model quality adjustments evolve from post-response quality data.

Each (model_id, task_type) pair accumulates an exponentially-decayed running
average of observed quality scores.  Once enough samples have been collected
(ADAPTIVE_MIN_SAMPLES), the routing engine uses the adjusted score instead of
the static registry rating.

Change 3: Added per-customer adaptive weights.  When a customer_id is supplied,
the engine looks up that customer's EMA table first (min 20 samples); if not
enough data exist, it falls back to the global table.  A
get_customer_routing_profile() helper returns a usage summary.

Storage: JSON file on disk.  Writes are debounced (every 50 updates) so hot
routing paths are not stalled by I/O.  Thread-safe.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from .config import (
    ADAPTIVE_DECAY_FACTOR,
    ADAPTIVE_MIN_SAMPLES,
    ADAPTIVE_WEIGHT_FILE,
    PER_CUSTOMER_MIN_SAMPLES,
)

log = structlog.get_logger(__name__)

_WRITE_INTERVAL = 50   # flush to disk every N updates


class AdaptiveWeights:
    """
    Maintains per-(model_id, task_type) quality adjustments derived from
    empirical quality scores fed back by QualityScorer.

    The exponential decay keeps recent observations more influential than old
    ones, so the system adapts when a model is upgraded or degrades over time.

    Change 3: Per-customer weights are stored in self._customer_state keyed by
    customer_id.  They are used after PER_CUSTOMER_MIN_SAMPLES (20) observations;
    before that threshold the global weights apply.
    """

    def __init__(self, state_file: str | None = None) -> None:
        self._path   = Path(state_file or ADAPTIVE_WEIGHT_FILE) if state_file else None
        self._lock   = threading.Lock()
        # Global state: key → {adjustment, sample_count, avg_quality, last_updated}
        self._state: dict[str, dict[str, Any]] = {}
        # Per-customer state: customer_id → same structure as _state
        self._customer_state: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        # Per-customer request log for routing profile (Change 3)
        # customer_id → list of {model_id, task_type, cost, timestamp}
        self._customer_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._dirty  = 0  # updates since last flush
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def record(
        self,
        model_id: str,
        task_type: str,
        quality_score: float,
        base_quality_rating: float = 0.5,
        customer_id: str | None = None,
    ) -> None:
        """
        Update the running average for (model_id, task_type) with a new observation.
        The exponential decay means recent quality matters more than historical data.
        Persists to disk every WRITE_INTERVAL calls.

        Change 3: If customer_id is provided, also update that customer's EMA table.
        """
        key = f"{model_id}:{task_type}"
        with self._lock:
            # Update global state
            self._state[key] = self._update_ema(
                self._state.get(key, {}), quality_score, base_quality_rating
            )
            # Update per-customer state (Change 3)
            if customer_id:
                self._customer_state[customer_id][key] = self._update_ema(
                    self._customer_state[customer_id].get(key, {}),
                    quality_score,
                    base_quality_rating,
                )
            self._dirty += 1
            if self._dirty >= _WRITE_INTERVAL:
                self._flush()

    def get_adjusted_score(
        self,
        model_id: str,
        task_type: str,
        base_score: float,
        customer_id: str | None = None,
    ) -> float:
        """
        Return base_score + learned adjustment, clamped to [0, 1].
        Falls back to base_score if fewer than ADAPTIVE_MIN_SAMPLES have been
        collected — not enough data to trust the adjustment yet.

        Change 3: If customer_id is provided AND that customer has
        PER_CUSTOMER_MIN_SAMPLES or more samples, use their per-customer EMA
        instead of the global one.
        """
        key = f"{model_id}:{task_type}"
        with self._lock:
            # Check per-customer data first (Change 3)
            if customer_id:
                c_entry = self._customer_state.get(customer_id, {}).get(key)
                if c_entry and c_entry.get("sample_count", 0) >= PER_CUSTOMER_MIN_SAMPLES:
                    adjusted = base_score + c_entry["adjustment"]
                    return round(max(0.0, min(1.0, adjusted)), 4)
            # Fall back to global
            entry = self._state.get(key)
        if not entry or entry.get("sample_count", 0) < ADAPTIVE_MIN_SAMPLES:
            return base_score
        adjusted = base_score + entry["adjustment"]
        return round(max(0.0, min(1.0, adjusted)), 4)

    def record_routing_event(
        self,
        customer_id: str,
        model_id: str,
        task_type: str,
        cost: float,
    ) -> None:
        """
        Log a routing event for a customer so get_customer_routing_profile() can
        report their top task types, preferred models, and average cost.

        Change 3: Called by the routing engine after every decision with a customer_id.
        """
        with self._lock:
            self._customer_log[customer_id].append({
                "model_id":  model_id,
                "task_type": task_type,
                "cost":      cost,
                "timestamp": datetime.utcnow().isoformat(),
            })

    def get_customer_routing_profile(self, customer_id: str) -> dict[str, Any]:
        """
        Return a summary of this customer's routing history:
          - top_task_types: [(task_type, count), ...] sorted by count desc
          - top_models:     [(model_id, count), ...] sorted by count desc
          - avg_cost_per_request: float
          - total_requests: int
          - has_custom_weights: bool  (True once PER_CUSTOMER_MIN_SAMPLES reached)

        Change 3: Exposes per-customer routing intelligence for dashboards and
        debug tooling.
        """
        with self._lock:
            events = list(self._customer_log.get(customer_id, []))
            c_state = dict(self._customer_state.get(customer_id, {}))

        if not events:
            return {
                "customer_id":          customer_id,
                "total_requests":       0,
                "top_task_types":       [],
                "top_models":           [],
                "avg_cost_per_request": 0.0,
                "has_custom_weights":   False,
            }

        task_counts: dict[str, int] = defaultdict(int)
        model_counts: dict[str, int] = defaultdict(int)
        total_cost = 0.0
        for ev in events:
            task_counts[ev["task_type"]] += 1
            model_counts[ev["model_id"]] += 1
            total_cost += ev["cost"]

        # Check if any model/task pair has enough samples for custom weights
        has_custom = any(
            v.get("sample_count", 0) >= PER_CUSTOMER_MIN_SAMPLES
            for v in c_state.values()
        )

        return {
            "customer_id":          customer_id,
            "total_requests":       len(events),
            "top_task_types":       sorted(task_counts.items(), key=lambda x: -x[1]),
            "top_models":           sorted(model_counts.items(), key=lambda x: -x[1]),
            "avg_cost_per_request": round(total_cost / len(events), 6) if events else 0.0,
            "has_custom_weights":   has_custom,
        }

    def flush(self) -> None:
        """Force an immediate write to disk (e.g., on shutdown)."""
        with self._lock:
            self._flush()

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _update_ema(
        entry: dict[str, Any],
        quality_score: float,
        base_quality_rating: float,
    ) -> dict[str, Any]:
        """Compute one EMA update step; return the updated entry dict."""
        if not entry:
            entry = {
                "adjustment":   0.0,
                "sample_count": 0,
                "avg_quality":  base_quality_rating,
                "last_updated": datetime.utcnow().isoformat(),
            }
        entry["avg_quality"] = (
            ADAPTIVE_DECAY_FACTOR * entry["avg_quality"]
            + (1.0 - ADAPTIVE_DECAY_FACTOR) * quality_score
        )
        entry["adjustment"]   = entry["avg_quality"] - base_quality_rating
        entry["sample_count"] += 1
        entry["last_updated"] = datetime.utcnow().isoformat()
        return entry

    def _load(self) -> None:
        """Load existing state from disk on startup, silently skip if absent."""
        if self._path and self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    self._state = json.load(fh)
                log.info("adaptive_weights_loaded", entries=len(self._state), path=str(self._path))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("adaptive_weights_load_failed", error=str(exc))
                self._state = {}

    def _flush(self) -> None:
        """Write current state to disk.  Caller must hold self._lock."""
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
            self._dirty = 0
        except OSError as exc:
            log.warning("adaptive_weights_flush_failed", error=str(exc))
