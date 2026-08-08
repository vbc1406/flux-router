"""
File: router/tests/test_model_registry.py

Purpose:
Tests for ModelRegistry — JSON file loading, hardcoded fallback registry,
model filtering by tier/capability/provider, and load-tracking helpers.

How to run:
  pytest -v router/tests/test_model_registry.py

How to add a test:
  1. Use ModelRegistry() for the default (JSON-backed) registry.
  2. Use _load_registry_from_json(path) to test custom JSON shapes.
  3. Assert on registry.all_available_models() or registry.models_for_tier(tier).

Test classes:
  TestRegistryLoadsFromJson — JSON file loading, field parsing, hardcoded fallback
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from router.model_registry import (
    ModelRegistry,
    _build_hardcoded_registry,
    _check_staleness,
    _load_registry_from_json,
)

_HARDCODED_COUNT = len(_build_hardcoded_registry())


class TestRegistryLoadsFromJson:
    def test_registry_loads_from_json(self):
        # Default registry (loads from models.json in package dir).
        # JSON is the source of truth and is allowed to grow beyond the
        # hardcoded fallback as new model SKUs ship.
        reg = ModelRegistry()
        models = reg.all_available_models()
        assert len(models) >= _HARDCODED_COUNT
        model_ids = {m.model_id for m in models}
        assert "gpt-4o" in model_ids
        assert "claude-opus-4-7" in model_ids

    def test_registry_falls_back_to_hardcoded(self, tmp_path, monkeypatch):
        # Point _load_registry_from_json at a non-existent path
        import router.model_registry as mr

        original = mr._load_registry_from_json

        def patched():
            return None  # simulate missing / broken JSON

        monkeypatch.setattr(mr, "_load_registry_from_json", patched)
        reg = ModelRegistry()
        models = reg.all_available_models()
        assert len(models) == _HARDCODED_COUNT

    def test_registry_all_models_have_general(self):
        reg = ModelRegistry()
        for m in reg.all_available_models():
            assert "general" in m.quality_ratings, f"{m.model_id} missing 'general' quality rating"

    def test_registry_model_count(self):
        # JSON is the source of truth; hardcoded is the fallback floor.
        # JSON must contain at least every model in the hardcoded registry.
        from_json = _load_registry_from_json()
        from_code = _build_hardcoded_registry()
        assert from_json is not None, "models.json not found"
        missing = set(from_code) - set(from_json)
        assert not missing, f"models.json missing hardcoded fallback models: {missing}"

    def test_registry_includes_latest_multi_provider_models(self):
        # Coverage for the mid-2026 catalog refresh: GPT-5.6 family, Gemini
        # 3.6 Flash, Mistral Small 4/Large 3, and Groq's current lineup.
        reg = ModelRegistry()
        expected_tiers = {
            "gpt-5.6-luna": "cheap",
            "gpt-5.6-terra": "mid",
            "gpt-5.6-sol": "premium",
            "gemini-3.6-flash": "mid",
            "mistral-small-4": "cheap",
            "mistral-large-3": "mid",
            "gpt-oss-120b": "mid",
            "gpt-oss-20b": "free",
            "qwen-3.6-27b": "cheap",
        }
        for model_id, tier in expected_tiers.items():
            m = reg.get_model(model_id)
            assert m is not None, f"{model_id} missing from models.json"
            assert m.tier == tier
            assert m.is_available

    def test_deprecated_llama_4_scout_is_excluded_from_routing(self):
        # Groq deprecated llama-4-scout on 2026-06-17 — it must no longer be
        # a routable candidate, but the entry stays in the catalog (with
        # is_available=false) rather than being silently deleted.
        reg = ModelRegistry()
        assert not any(m.model_id == "llama-4-scout" for m in reg.all_available_models())
        stale_entry = reg.get_model("llama-4-scout")
        assert stale_entry is not None
        assert stale_entry.is_available is False

    def test_registry_includes_current_gen_claude_models(self):
        # Bugfix coverage: models.json used to be missing the current Claude
        # 5-family SKUs. Assert they're present with sane tiers, not just
        # that SOME models exist.
        reg = ModelRegistry()
        expected_tiers = {
            "claude-fable-5": "premium",
            "claude-sonnet-5": "mid",
            "claude-opus-5": "premium",
        }
        for model_id, tier in expected_tiers.items():
            m = reg.get_model(model_id)
            assert m is not None, f"{model_id} missing from models.json"
            assert m.tier == tier
            assert m.is_available


class TestModelsJsonStaleness:
    """Bugfix coverage: models.json's last_updated field used to be
    write-only — loaded and never checked. _check_staleness() now logs a
    warning once the catalog is older than MODELS_JSON_STALE_AFTER_DAYS."""

    def test_recent_date_does_not_warn(self, monkeypatch):
        import router.model_registry as mr

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(mr.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        today = dt.date.today().isoformat()
        _check_staleness(today)
        assert events == []

    def test_date_just_past_threshold_warns(self, monkeypatch):
        import router.model_registry as mr
        from router.config import MODELS_JSON_STALE_AFTER_DAYS

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(mr.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        old_date = (
            dt.date.today() - dt.timedelta(days=MODELS_JSON_STALE_AFTER_DAYS + 1)
        ).isoformat()
        _check_staleness(old_date)
        assert any(ev == "model_registry_json_stale" for ev, _ in events)
        kw = next(kw for ev, kw in events if ev == "model_registry_json_stale")
        assert kw["age_days"] == MODELS_JSON_STALE_AFTER_DAYS + 1

    def test_date_just_under_threshold_does_not_warn(self, monkeypatch):
        import router.model_registry as mr
        from router.config import MODELS_JSON_STALE_AFTER_DAYS

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(mr.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        recent_date = (
            dt.date.today() - dt.timedelta(days=MODELS_JSON_STALE_AFTER_DAYS)
        ).isoformat()
        _check_staleness(recent_date)
        assert events == []

    def test_missing_last_updated_warns(self, monkeypatch):
        import router.model_registry as mr

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(mr.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        _check_staleness(None)
        assert any(ev == "model_registry_json_missing_last_updated" for ev, _ in events)

    def test_malformed_date_warns_but_does_not_raise(self, monkeypatch):
        import router.model_registry as mr

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(mr.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        _check_staleness("not-a-date")
        assert any(ev == "model_registry_json_invalid_last_updated" for ev, _ in events)

    def test_real_models_json_last_updated_parses_as_a_valid_date(self):
        """Guards against the field silently rotting into an unparseable
        format, independent of whether the actual value happens to be stale
        (that's a real-clock concern for the warning, not the test suite)."""
        import json

        import router.model_registry as mr

        raw = json.loads((Path(mr.__file__).parent / "models.json").read_text())
        dt.date.fromisoformat(raw["last_updated"])


class TestLoadTrackingReset:
    """The sliding-window RPM counter is process-global and wall-clock based,
    which made ModelRegistry a silent carrier of state between unrelated
    callers: a free-tier model capped at 15 RPM drops out of the candidate
    list at RATE_LIMIT_SAFETY_MARGIN (0.85) — i.e. the 13th request in any 60
    second window — and nothing put it back. reset_load_tracking() exists so a
    caller can return the registry to a known-idle state.
    """

    def _registry(self):
        from router.model_registry import ModelRegistry

        return ModelRegistry()

    def test_reset_clears_the_window_and_the_reported_load(self):
        reg = self._registry()
        for _ in range(5):
            reg.update_load("gpt-oss-20b")
        assert reg._models["gpt-oss-20b"].current_load_rpm == 5

        reg.reset_load_tracking()
        assert reg._models["gpt-oss-20b"].current_load_rpm == 0
        assert not reg._rpm_window

    def test_reset_makes_a_rate_limited_model_eligible_again(self):
        reg = self._registry()
        model_id = "gpt-oss-20b"
        cap = reg._models[model_id].rate_limit_rpm
        for _ in range(cap):
            reg.update_load(model_id)
        assert reg.is_near_rate_limit(model_id) is True

        reg.reset_load_tracking()
        assert reg.is_near_rate_limit(model_id) is False

    def test_threshold_is_the_documented_safety_margin(self):
        """Pins the boundary the conftest fixture exists to defend against:
        one request below the margin is fine, one at it is not."""
        from router.config import RATE_LIMIT_SAFETY_MARGIN

        reg = self._registry()
        model_id = "gpt-oss-20b"
        cap = reg._models[model_id].rate_limit_rpm
        threshold = cap * RATE_LIMIT_SAFETY_MARGIN

        for _ in range(int(threshold)):
            reg.update_load(model_id)
        assert reg.is_near_rate_limit(model_id) is False

        reg.update_load(model_id)
        assert reg.is_near_rate_limit(model_id) is True
