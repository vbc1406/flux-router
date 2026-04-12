"""
Learning loop — model quality adjustments evolve from post-response quality data.

Each (model_id, task_type) pair accumulates an exponentially-decayed running
average of observed quality scores.  Once enough samples have been collected
(ADAPTIVE_MIN_SAMPLES), the routing engine uses the adjusted score instead of
the static registry rating.

Storage: JSON file on disk.  Writes are debounced (every 50 updates) so hot
routing paths are not stalled by I/O.  Thread-safe.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from .config import (
    ADAPTIVE_DECAY_FACTOR,
    ADAPTIVE_MIN_SAMPLES,
    ADAPTIVE_WEIGHT_FILE,
)

log = structlog.get_logger(__name__)

_WRITE_INTERVAL = 50   # flush to disk every N updates


class AdaptiveWeights:
    """
    Maintains per-(model_id, task_type) quality adjustments derived from
    empirical quality scores fed back by QualityScorer.

    The exponential decay keeps recent observations more influential than old
    ones, so the system adapts when a model is upgraded or degrades over time.
    """

    def __init__(self, state_file: str | None = None) -> None:
        self._path   = Path(state_file or ADAPTIVE_WEIGHT_FILE)
        self._lock   = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}
        self._dirty  = 0  # updates since last flush
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def record(
        self,
        model_id: str,
        task_type: str,
        quality_score: float,
        base_quality_rating: float = 0.5,
    ) -> None:
        """
        Update the running average for (model_id, task_type) with a new observation.
        The exponential decay means recent quality matters more than historical data.
        Persists to disk every WRITE_INTERVAL calls.
        """
        key = f"{model_id}:{task_type}"
        with self._lock:
            entry = self._state.get(key, {
                "adjustment":   0.0,
                "sample_count": 0,
                "avg_quality":  base_quality_rating,
                "last_updated": datetime.utcnow().isoformat(),
            })
            # Exponential moving average
            entry["avg_quality"] = (
                ADAPTIVE_DECAY_FACTOR * entry["avg_quality"]
                + (1.0 - ADAPTIVE_DECAY_FACTOR) * quality_score
            )
            entry["adjustment"]   = entry["avg_quality"] - base_quality_rating
            entry["sample_count"] += 1
            entry["last_updated"] = datetime.utcnow().isoformat()
            self._state[key] = entry
            self._dirty += 1
            if self._dirty >= _WRITE_INTERVAL:
                self._flush()

    def get_adjusted_score(
        self,
        model_id: str,
        task_type: str,
        base_score: float,
    ) -> float:
        """
        Return base_score + learned adjustment, clamped to [0, 1].
        Falls back to base_score if fewer than ADAPTIVE_MIN_SAMPLES have been
        collected — not enough data to trust the adjustment yet.
        """
        key = f"{model_id}:{task_type}"
        with self._lock:
            entry = self._state.get(key)
        if not entry or entry.get("sample_count", 0) < ADAPTIVE_MIN_SAMPLES:
            return base_score
        adjusted = base_score + entry["adjustment"]
        return round(max(0.0, min(1.0, adjusted)), 4)

    def flush(self) -> None:
        """Force an immediate write to disk (e.g., on shutdown)."""
        with self._lock:
            self._flush()

    # ── Private ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load existing state from disk on startup, silently skip if absent."""
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as fh:
                    self._state = json.load(fh)
                log.info("adaptive_weights_loaded", entries=len(self._state), path=str(self._path))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("adaptive_weights_load_failed", error=str(exc))
                self._state = {}

    def _flush(self) -> None:
        """Write current state to disk.  Caller must hold self._lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
            self._dirty = 0
        except OSError as exc:
            log.warning("adaptive_weights_flush_failed", error=str(exc))
