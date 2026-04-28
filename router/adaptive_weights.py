"""
File: router/adaptive_weights.py

Purpose:
Track per-(model_id, task_type) response quality using an exponential moving
average (EMA).  After ADAPTIVE_MIN_SAMPLES observations, the routing engine uses
the learned adjustment instead of the static registry quality rating.  Supports
per-customer weights, corruption rollback, configurable decay, and on-disk
persistence with automatic format migration.

Main Classes:
  AdaptiveWeights

Config Dependencies (all in config.py):
  ADAPTIVE_MIN_SAMPLES      — minimum signals before adaptive score activates
  ADAPTIVE_DECAY_FACTOR     — EMA decay; higher = longer memory
  ADAPTIVE_WEIGHT_FILE      — path to the on-disk JSON state file
  PER_CUSTOMER_MIN_SAMPLES  — signals before per-customer EMA overrides global
  ADAPTIVE_CUSTOMER_LOG_MAX — max routing events kept per customer in memory

Key Methods (most likely to be modified):
  record()                  — feed a quality signal back after a response
  get_adjusted_score()      — called by the routing engine in Step 8
  set_decay_factor()        — override decay per (model, task) for hot retraining
  _check_for_corruption()   — rolls back to last snapshot on quality degradation

Things NOT to change without discussion:
  - The threading model (_lock guards all state mutations)
  - The snapshot format (version-controlled; bump _FORMAT_VERSION on change)
  - The rollback logic (_check_for_corruption + _maybe_snapshot)
  - The Welford online stats algorithm in _update_signal_stats()
"""

from __future__ import annotations

import copy
import json
import math
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from .config import (
    ADAPTIVE_CUSTOMER_LOG_MAX,
    ADAPTIVE_DECAY_FACTOR,
    ADAPTIVE_MIN_SAMPLES,
    ADAPTIVE_WEIGHT_FILE,
    PER_CUSTOMER_MIN_SAMPLES,
)

log = structlog.get_logger(__name__)

_WRITE_INTERVAL = 50        # flush to disk every N updates
_SNAPSHOT_INTERVAL = 1000   # take a snapshot every N signals
_MAX_SNAPSHOTS = 5
_QUALITY_FLOOR = 0.20       # never allow adaptive quality below this
_OUTLIER_STD_MULTIPLIER = 2.0
_ROLLBACK_DROP_THRESHOLD = 0.90  # roll back if avg drops below 90% of snapshot avg
_FORMAT_VERSION = 2              # bumped when the on-disk schema changes; triggers auto-migration


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
        # Per-customer request log for routing profile (capped to avoid unbounded growth)
        # customer_id → deque of {model_id, task_type, cost, timestamp}
        self._customer_log: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=ADAPTIVE_CUSTOMER_LOG_MAX)
        )
        self._dirty  = 0   # updates since last flush
        self._total_signals = 0
        # Outlier detection: key → {mean, variance, count}
        self._signal_stats: dict[str, dict[str, float]] = {}
        # Rolling snapshots of _state + _signal_stats for corruption rollback
        self._snapshots: list[dict[str, Any]] = []
        # Avg quality at last snapshot time — cached to avoid recomputing on every signal.
        # None until the first snapshot fires; _check_for_corruption is a no-op until set.
        self._last_snapshot_avg: float | None = None
        # Per-(model_id, task_type) EMA decay overrides for hot-deployment retraining.
        # Set via set_decay_factor(); persisted in memory only (re-apply after restart).
        self._decay_overrides: dict[str, float] = {}
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────
    # 🔧 EXTENSION POINT: Add new learning signals here.
    # To track a new quality dimension (e.g., factual accuracy, code correctness):
    # (1) Add a scorer in quality_scorer.py and include it in _WEIGHTS,
    # (2) Pass the composite score into record() — no changes needed here unless
    #     you want separate EMA tracks per dimension (then add a new method following
    #     the record() pattern and a parallel _state key like "model:task:dimension").

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
        Rejects outliers (>2σ from running mean), enforces a quality floor of 0.20,
        and takes periodic snapshots for corruption rollback.

        Change 3: If customer_id is provided, also update that customer's EMA table.
        """
        # Clamp before any processing — out-of-range scores corrupt the EMA and
        # the Welford variance accumulator.  Legitimate callers should never hit this.
        if not (0.0 <= quality_score <= 1.0):
            log.debug(
                "adaptive_quality_clamped",
                key=f"{model_id}:{task_type}",
                original=quality_score,
            )
            quality_score = max(0.0, min(1.0, quality_score))

        key = f"{model_id}:{task_type}"
        with self._lock:
            # Outlier guard — skip signals far from the running mean
            if not self._should_accept_signal(key, quality_score):
                log.debug("adaptive_signal_rejected_outlier", key=key, quality=quality_score)
                return

            # Update running stats for future outlier detection
            self._update_signal_stats(key, quality_score)

            # Resolve per-key decay override; fall back to global config value
            decay = self._decay_overrides.get(key, ADAPTIVE_DECAY_FACTOR)

            # Update global state
            self._state[key] = self._update_ema(
                self._state.get(key, {}), quality_score, base_quality_rating, decay
            )
            # Update per-customer state (Change 3)
            if customer_id:
                self._customer_state[customer_id][key] = self._update_ema(
                    self._customer_state[customer_id].get(key, {}),
                    quality_score,
                    base_quality_rating,
                    decay,
                )
            self._dirty += 1
            self._total_signals += 1
            if self._dirty >= _WRITE_INTERVAL:
                self._flush()
            # Periodic snapshot + corruption check
            self._maybe_snapshot()
            self._check_for_corruption()

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
                    return round(max(_QUALITY_FLOOR, min(1.0, adjusted)), 4)
            # Fall back to global
            entry = self._state.get(key)
        if not entry or entry.get("sample_count", 0) < ADAPTIVE_MIN_SAMPLES:
            return base_score
        adjusted = base_score + entry["adjustment"]
        return round(max(_QUALITY_FLOOR, min(1.0, adjusted)), 4)

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

        The log is capped at ADAPTIVE_CUSTOMER_LOG_MAX entries per customer.  When
        the cap is reached the oldest entry is dropped (deque wraps); a debug log is
        emitted so operators can tune the cap if customer histories are being truncated.

        Change 3: Called by the routing engine after every decision with a customer_id.
        """
        with self._lock:
            clog = self._customer_log[customer_id]
            if len(clog) == clog.maxlen:
                # Oldest entry is about to be silently dropped — surface it so ops can
                # raise ADAPTIVE_CUSTOMER_LOG_MAX if full history is needed.
                log.debug(
                    "adaptive_customer_log_wrapped",
                    customer_id=customer_id,
                    maxlen=clog.maxlen,
                )
            clog.append({
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

    def set_decay_factor(self, model_id: str, task_type: str, decay: float) -> None:
        """
        Override the EMA decay factor for a specific (model_id, task_type) pair.

        Smaller decay → less weight on history → faster adaptation to current quality.
        Use after a model upgrade when you want stale pre-upgrade samples to wash out
        quickly.  The override is in-memory only; re-apply after restart if needed.

        decay must be in (0.0, 1.0).  The global default is ADAPTIVE_DECAY_FACTOR.
        """
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0.0, 1.0), got {decay!r}")
        key = f"{model_id}:{task_type}"
        with self._lock:
            self._decay_overrides[key] = decay
        log.info("adaptive_decay_override_set", key=key, decay=decay)

    # ── Private ─────────────────────────────────────────────────────────────

    @staticmethod
    def _avg_quality(state: dict) -> float:
        """Mean avg_quality across all entries; returns 0.5 when state is empty."""
        vals = [v.get("avg_quality", 0.5) for v in state.values()]
        return sum(vals) / len(vals) if vals else 0.5

    @staticmethod
    def _update_ema(
        entry: dict[str, Any],
        quality_score: float,
        base_quality_rating: float,
        decay_factor: float = ADAPTIVE_DECAY_FACTOR,
    ) -> dict[str, Any]:
        """Compute one EMA update step; return the updated entry dict.

        decay_factor overrides the global ADAPTIVE_DECAY_FACTOR for this key.
        Smaller values weight recent observations more heavily (faster adaptation).
        """
        if not entry:
            entry = {
                "adjustment":   0.0,
                "sample_count": 0,
                "avg_quality":  base_quality_rating,
                "last_updated": datetime.utcnow().isoformat(),
            }
        entry["avg_quality"] = (
            decay_factor * entry["avg_quality"]
            + (1.0 - decay_factor) * quality_score
        )
        # Quality floor — prevents permanent blacklisting from bad data
        entry["avg_quality"] = max(_QUALITY_FLOOR, entry["avg_quality"])
        entry["adjustment"]   = entry["avg_quality"] - base_quality_rating
        entry["sample_count"] += 1
        entry["last_updated"] = datetime.utcnow().isoformat()
        return entry

    def _should_accept_signal(self, key: str, quality: float) -> bool:
        """Reject signals that are >2σ from the running mean (caller holds lock)."""
        stats = self._signal_stats.get(key)
        if stats is None or stats["count"] < 10:
            return True
        std = math.sqrt(stats["variance"])
        if std < 0.01:
            return True
        return abs(quality - stats["mean"]) <= _OUTLIER_STD_MULTIPLIER * std

    def _update_signal_stats(self, key: str, quality: float) -> None:
        """Welford online algorithm for running mean and variance (caller holds lock)."""
        if key not in self._signal_stats:
            self._signal_stats[key] = {"mean": quality, "variance": 0.0, "count": 1}
            return
        s = self._signal_stats[key]
        s["count"] += 1
        delta = quality - s["mean"]
        s["mean"] += delta / s["count"]
        delta2 = quality - s["mean"]
        s["variance"] = (s["variance"] * (s["count"] - 2) + delta * delta2) / (s["count"] - 1) if s["count"] > 1 else 0.0

    def _maybe_snapshot(self) -> None:
        """Take a snapshot of global weights every _SNAPSHOT_INTERVAL signals (caller holds lock)."""
        if self._total_signals > 0 and self._total_signals % _SNAPSHOT_INTERVAL == 0:
            snap_avg = self._avg_quality(self._state)
            self._snapshots.append({
                "state":        copy.deepcopy(self._state),
                "signal_stats": copy.deepcopy(self._signal_stats),
                # Pre-computed avg so _check_for_corruption never recomputes it from
                # the (potentially large) snapshot state dict on every incoming signal.
                "avg_quality":  snap_avg,
            })
            self._snapshots = self._snapshots[-_MAX_SNAPSHOTS:]
            # Cache the baseline for continuous between-snapshot drift detection.
            self._last_snapshot_avg = snap_avg
            log.info(
                "adaptive_weights_snapshot_taken",
                snapshot_count=len(self._snapshots),
                avg_quality=snap_avg,
            )

    def _check_for_corruption(self) -> None:
        """Roll back to last snapshot if quality has drifted >10% below the snapshot baseline.

        Runs on every signal (not just at snapshot boundaries) to catch gradual degradation
        that wouldn't cross the threshold in a single step.  Uses the pre-computed
        _last_snapshot_avg so this hot path stays O(1) instead of O(keys).
        """
        if not self._snapshots or not self._state or self._last_snapshot_avg is None:
            return

        current_avg = self._avg_quality(self._state)
        if current_avg < self._last_snapshot_avg * _ROLLBACK_DROP_THRESHOLD:
            snapshot = self._snapshots[-1]
            self._state        = copy.deepcopy(snapshot["state"])
            self._signal_stats = copy.deepcopy(snapshot["signal_stats"])
            # Reset baseline to the restored snapshot so the next signals are compared
            # against the clean state, not the pre-rollback degraded average.
            self._last_snapshot_avg = snapshot["avg_quality"]
            log.warning(
                "adaptive_weights_rolled_back",
                current_avg=current_avg,
                snapshot_avg=self._last_snapshot_avg,
            )

    def _load(self) -> None:
        """Load existing state from disk on startup, silently skip if absent.

        Handles three formats:
          v0 — flat dict (pre-customer-weights era): migrated to v2 on next flush.
          v1 — {"global": ..., "customers": ...} without version key: migrated to v2.
          v2 — {"version": 2, "global": ..., "customers": ...}: current format.

        Any format older than _FORMAT_VERSION triggers an immediate flush so the
        file is upgraded before the next crash window.
        """
        if not (self._path and self._path.exists()):
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "global" in data:
                file_version = data.get("version", 1)
                self._state = data["global"]
                for cid, cstate in data.get("customers", {}).items():
                    self._customer_state[cid] = cstate
                if file_version < _FORMAT_VERSION:
                    # Schedule an immediate re-flush so the file is upgraded now.
                    self._dirty = _WRITE_INTERVAL
                    log.info(
                        "adaptive_weights_migrated",
                        from_version=file_version,
                        to_version=_FORMAT_VERSION,
                    )
            else:
                # v0 flat-dict format — treat as global state only and migrate.
                self._state = data
                self._dirty = _WRITE_INTERVAL
                log.info(
                    "adaptive_weights_migrated",
                    from_version=0,
                    to_version=_FORMAT_VERSION,
                )
            log.info(
                "adaptive_weights_loaded",
                entries=len(self._state),
                customers=len(self._customer_state),
                path=str(self._path),
            )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("adaptive_weights_load_failed", error=str(exc))
            self._state = {}

    def _flush(self) -> None:
        """Write global and per-customer weights to disk.  Caller must hold self._lock."""
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version":   _FORMAT_VERSION,
                "global":    self._state,
                "customers": {k: dict(v) for k, v in self._customer_state.items()},
            }
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            self._dirty = 0
        except OSError as exc:
            log.warning("adaptive_weights_flush_failed", error=str(exc))
