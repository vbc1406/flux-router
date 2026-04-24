"""Tests for model registry JSON loading (Fix 4)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from router.model_registry import ModelRegistry, _build_hardcoded_registry, _load_registry_from_json
from router.schemas import ModelOption

_HARDCODED_COUNT = len(_build_hardcoded_registry())


class TestRegistryLoadsFromJson:
    def test_registry_loads_from_json(self):
        # Default registry (loads from models.json in package dir)
        reg = ModelRegistry()
        models = reg.all_available_models()
        assert len(models) == _HARDCODED_COUNT
        model_ids = {m.model_id for m in models}
        assert "gpt-4o" in model_ids
        assert "claude-opus-4-20250514" in model_ids

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
            assert "general" in m.quality_ratings, (
                f"{m.model_id} missing 'general' quality rating"
            )

    def test_registry_model_count(self):
        # JSON and hardcoded registry must agree on model count
        from_json = _load_registry_from_json()
        from_code = _build_hardcoded_registry()
        assert from_json is not None, "models.json not found"
        assert len(from_json) == len(from_code), (
            f"JSON has {len(from_json)} models, hardcoded has {len(from_code)}"
        )
