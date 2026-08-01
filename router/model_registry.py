"""
Model catalog with verified 2025 pricing, capabilities, and live load tracking.

Pricing sources: Anthropic, OpenAI, Google, Groq, and Mistral public pricing pages
as of mid-2025.  Per-1K-token costs are derived from per-million-token list prices.

Provider sensitivity restrictions:
  restricted   → only anthropic, openai (strict data-residency / compliance)
  confidential → anthropic, openai, google
  internal     → all providers
  public       → all providers
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path

import structlog

from .config import MODELS_JSON_STALE_AFTER_DAYS, RATE_LIMIT_SAFETY_MARGIN
from .schemas import ModelOption

log = structlog.get_logger(__name__)

# Quality rating tables per task type (0.0 – 1.0).
# Values reflect documented benchmark results and community evals.
_FREE_FLASH_LITE_QUALITY: dict[str, float] = {
    "simple_qa": 0.75,
    "conversation": 0.80,
    "translation": 0.75,
    "classification": 0.70,
    "extraction": 0.65,
    "summarization": 0.65,
    "code_generation": 0.50,
    "code_review": 0.45,
    "creative_writing": 0.55,
    "analysis": 0.55,
    "reasoning": 0.45,
    "function_calling": 0.50,
    "vision": 0.60,
    "long_document": 0.60,
    "unknown": 0.60,
    "general": 0.60,
}
_LLAMA_70B_QUALITY: dict[str, float] = {
    "simple_qa": 0.80,
    "conversation": 0.82,
    "translation": 0.78,
    "classification": 0.82,
    "extraction": 0.78,
    "summarization": 0.78,
    "code_generation": 0.75,
    "code_review": 0.72,
    "creative_writing": 0.72,
    "analysis": 0.75,
    "reasoning": 0.72,
    "function_calling": 0.70,
    "vision": 0.00,
    "long_document": 0.70,
    "unknown": 0.72,
    "general": 0.72,
}
_MISTRAL_SMALL_QUALITY: dict[str, float] = {
    "simple_qa": 0.75,
    "conversation": 0.78,
    "translation": 0.82,
    "classification": 0.78,
    "extraction": 0.75,
    "summarization": 0.73,
    "code_generation": 0.70,
    "code_review": 0.68,
    "creative_writing": 0.68,
    "analysis": 0.70,
    "reasoning": 0.65,
    "function_calling": 0.65,
    "vision": 0.00,
    "long_document": 0.65,
    "unknown": 0.68,
    "general": 0.68,
}
_FREE_FLASH_QUALITY: dict[str, float] = {
    "simple_qa": 0.80,
    "conversation": 0.82,
    "translation": 0.80,
    "classification": 0.80,
    "extraction": 0.78,
    "summarization": 0.78,
    "code_generation": 0.72,
    "code_review": 0.68,
    "creative_writing": 0.70,
    "analysis": 0.72,
    "reasoning": 0.68,
    "function_calling": 0.70,
    "vision": 0.80,
    "long_document": 0.78,
    "unknown": 0.72,
    "general": 0.72,
}
_HAIKU_QUALITY: dict[str, float] = {
    "simple_qa": 0.82,
    "conversation": 0.85,
    "translation": 0.83,
    "classification": 0.85,
    "extraction": 0.83,
    "summarization": 0.80,
    "code_generation": 0.75,
    "code_review": 0.73,
    "creative_writing": 0.72,
    "analysis": 0.75,
    "reasoning": 0.70,
    "function_calling": 0.78,
    "vision": 0.75,
    "long_document": 0.75,
    "unknown": 0.76,
    "general": 0.76,
}
_GPT4O_MINI_QUALITY: dict[str, float] = {
    "simple_qa": 0.83,
    "conversation": 0.82,
    "translation": 0.82,
    "classification": 0.84,
    "extraction": 0.82,
    "summarization": 0.80,
    "code_generation": 0.78,
    "code_review": 0.76,
    "creative_writing": 0.74,
    "analysis": 0.76,
    "reasoning": 0.72,
    "function_calling": 0.80,
    "vision": 0.78,
    "long_document": 0.74,
    "unknown": 0.76,
    "general": 0.76,
}
_FLASH_PAID_QUALITY: dict[str, float] = {
    "simple_qa": 0.82,
    "conversation": 0.82,
    "translation": 0.83,
    "classification": 0.82,
    "extraction": 0.80,
    "summarization": 0.80,
    "code_generation": 0.76,
    "code_review": 0.74,
    "creative_writing": 0.74,
    "analysis": 0.76,
    "reasoning": 0.72,
    "function_calling": 0.76,
    "vision": 0.85,
    "long_document": 0.83,
    "unknown": 0.78,
    "general": 0.78,
}
_SONNET_QUALITY: dict[str, float] = {
    "simple_qa": 0.90,
    "conversation": 0.88,
    "translation": 0.88,
    "classification": 0.90,
    "extraction": 0.90,
    "summarization": 0.88,
    "code_generation": 0.92,
    "code_review": 0.90,
    "creative_writing": 0.87,
    "analysis": 0.90,
    "reasoning": 0.88,
    "function_calling": 0.90,
    "vision": 0.85,
    "long_document": 0.88,
    "unknown": 0.88,
    "general": 0.88,
}
_GPT4O_QUALITY: dict[str, float] = {
    "simple_qa": 0.88,
    "conversation": 0.87,
    "translation": 0.87,
    "classification": 0.88,
    "extraction": 0.88,
    "summarization": 0.87,
    "code_generation": 0.88,
    "code_review": 0.87,
    "creative_writing": 0.85,
    "analysis": 0.88,
    "reasoning": 0.87,
    "function_calling": 0.92,
    "vision": 0.90,
    "long_document": 0.86,
    "unknown": 0.87,
    "general": 0.87,
}
_GEMINI25_PRO_QUALITY: dict[str, float] = {
    "simple_qa": 0.87,
    "conversation": 0.85,
    "translation": 0.87,
    "classification": 0.87,
    "extraction": 0.87,
    "summarization": 0.88,
    "code_generation": 0.87,
    "code_review": 0.86,
    "creative_writing": 0.83,
    "analysis": 0.88,
    "reasoning": 0.90,
    "function_calling": 0.86,
    "vision": 0.88,
    "long_document": 0.92,
    "unknown": 0.87,
    "general": 0.87,
}
_OPUS_QUALITY: dict[str, float] = {
    "simple_qa": 0.93,
    "conversation": 0.93,
    "translation": 0.92,
    "classification": 0.93,
    "extraction": 0.93,
    "summarization": 0.93,
    "code_generation": 0.95,
    "code_review": 0.95,
    "creative_writing": 0.92,
    "analysis": 0.95,
    "reasoning": 0.93,
    "function_calling": 0.92,
    "vision": 0.90,
    "long_document": 0.93,
    "unknown": 0.92,
    "general": 0.92,
}
_GPT45_PREVIEW_QUALITY: dict[str, float] = {
    "simple_qa": 0.90,
    "conversation": 0.92,
    "translation": 0.90,
    "classification": 0.90,
    "extraction": 0.90,
    "summarization": 0.90,
    "code_generation": 0.91,
    "code_review": 0.91,
    "creative_writing": 0.93,
    "analysis": 0.92,
    "reasoning": 0.90,
    "function_calling": 0.93,
    "vision": 0.91,
    "long_document": 0.88,
    "unknown": 0.90,
    "general": 0.90,
}
_GEMINI25_PRO_THINKING_QUALITY: dict[str, float] = {
    "simple_qa": 0.90,
    "conversation": 0.87,
    "translation": 0.88,
    "classification": 0.90,
    "extraction": 0.90,
    "summarization": 0.90,
    "code_generation": 0.91,
    "code_review": 0.91,
    "creative_writing": 0.85,
    "analysis": 0.93,
    "reasoning": 0.95,
    "function_calling": 0.88,
    "vision": 0.89,
    "long_document": 0.94,
    "unknown": 0.90,
    "general": 0.90,
}
_O3_QUALITY: dict[str, float] = {
    "simple_qa": 0.88,
    "conversation": 0.80,
    "translation": 0.85,
    "classification": 0.88,
    "extraction": 0.87,
    "summarization": 0.87,
    "code_generation": 0.93,
    "code_review": 0.94,
    "creative_writing": 0.75,
    "analysis": 0.95,
    "reasoning": 0.98,
    "function_calling": 0.90,
    "vision": 0.90,
    "long_document": 0.90,
    "unknown": 0.88,
    "general": 0.88,
}
_O4_MINI_QUALITY: dict[str, float] = {
    "simple_qa": 0.85,
    "conversation": 0.78,
    "translation": 0.83,
    "classification": 0.86,
    "extraction": 0.85,
    "summarization": 0.85,
    "code_generation": 0.90,
    "code_review": 0.90,
    "creative_writing": 0.73,
    "analysis": 0.90,
    "reasoning": 0.94,
    "function_calling": 0.88,
    "vision": 0.85,
    "long_document": 0.87,
    "unknown": 0.85,
    "general": 0.85,
}
_OPUS_EXTENDED_QUALITY: dict[str, float] = {
    "simple_qa": 0.92,
    "conversation": 0.90,
    "translation": 0.92,
    "classification": 0.94,
    "extraction": 0.94,
    "summarization": 0.94,
    "code_generation": 0.97,
    "code_review": 0.97,
    "creative_writing": 0.93,
    "analysis": 0.97,
    "reasoning": 0.99,
    "function_calling": 0.93,
    "vision": 0.91,
    "long_document": 0.95,
    "unknown": 0.93,
    "general": 0.93,
}

# Sensitivity levels allowed per provider category.
_ALL_LEVELS = ["public", "internal", "confidential", "restricted"]
_NO_RESTRICTED = ["public", "internal", "confidential"]
_PUBLIC_INTERNAL = ["public", "internal"]


def _check_staleness(last_updated: str | None) -> None:
    """
    Bugfix: models.json's "last_updated" field used to be write-only — loaded
    and immediately discarded. Log a loud startup warning once the catalog is
    more than MODELS_JSON_STALE_AFTER_DAYS old, so a forgotten pricing/model
    refresh doesn't just silently drift out of date. Never raises: a missing
    or malformed field is itself worth a warning, not a load failure.
    """
    if not last_updated:
        log.warning("model_registry_json_missing_last_updated")
        return
    try:
        parsed = date.fromisoformat(last_updated)
    except (TypeError, ValueError):
        log.warning("model_registry_json_invalid_last_updated", last_updated=last_updated)
        return
    age_days = (datetime.now().date() - parsed).days
    if age_days > MODELS_JSON_STALE_AFTER_DAYS:
        log.warning(
            "model_registry_json_stale",
            last_updated=last_updated,
            age_days=age_days,
            stale_after_days=MODELS_JSON_STALE_AFTER_DAYS,
        )


def _load_registry_from_json() -> dict[str, ModelOption] | None:
    """Load model registry from models.json if present; return None on any failure."""
    config_path = Path(__file__).parent / "models.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        _check_staleness(data.get("last_updated"))
        models = []
        for m in data["models"]:
            models.append(ModelOption(**m))
        log.info("model_registry_loaded_from_json", count=len(models), path=str(config_path))
        return {m.model_id: m for m in models}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # OSError: file unreadable. JSONDecodeError: malformed JSON.
        # KeyError: missing "models" key. TypeError/ValueError: ModelOption(**m)
        # failure (pydantic ValidationError subclasses ValueError).
        log.warning("model_registry_json_load_failed", error=str(exc))
        return None
    except Exception as exc:
        # Unexpected — capture traceback so unknown failure modes don't vanish.
        log.exception("model_registry_json_load_unexpected", error=str(exc))
        return None


def _build_hardcoded_registry() -> dict[str, ModelOption]:
    """Instantiate the canonical model catalog with verified 2025 pricing (hardcoded fallback)."""
    models: list[ModelOption] = [
        # ── FREE TIER ────────────────────────────────────────────────────────
        ModelOption(
            provider="google",
            model_id="gemini-2.0-flash-lite",
            display_name="Gemini 2.0 Flash Lite (free)",
            tier="free",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_context_window=1_048_576,
            max_output_tokens=8_192,
            capabilities=["vision"],
            supports_streaming=True,
            quality_ratings=_FREE_FLASH_LITE_QUALITY,
            avg_latency_ms=400,
            rate_limit_rpm=15,  # Free tier: 15 RPM
            allowed_sensitivity_levels=_PUBLIC_INTERNAL,
        ),
        ModelOption(
            provider="google",
            model_id="gemini-2.0-flash-free",
            display_name="Gemini 2.0 Flash (free)",
            tier="free",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_context_window=1_048_576,
            max_output_tokens=8_192,
            capabilities=["vision"],
            supports_streaming=True,
            quality_ratings=_FREE_FLASH_QUALITY,
            avg_latency_ms=450,
            rate_limit_rpm=15,
            allowed_sensitivity_levels=_PUBLIC_INTERNAL,
        ),
        ModelOption(
            provider="groq",
            model_id="llama-3.3-70b",
            display_name="Llama 3.3 70B (Groq free)",
            tier="free",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_context_window=128_000,
            max_output_tokens=8_192,
            capabilities=[],
            supports_streaming=True,
            quality_ratings=_LLAMA_70B_QUALITY,
            avg_latency_ms=250,  # Groq is extremely fast
            rate_limit_rpm=30,
            allowed_sensitivity_levels=_PUBLIC_INTERNAL,
        ),
        ModelOption(
            provider="mistral",
            model_id="mistral-small-latest",
            display_name="Mistral Small (free)",
            tier="free",
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_context_window=128_000,
            max_output_tokens=32_768,
            capabilities=[],
            supports_streaming=True,
            quality_ratings=_MISTRAL_SMALL_QUALITY,
            avg_latency_ms=500,
            rate_limit_rpm=20,
            allowed_sensitivity_levels=_PUBLIC_INTERNAL,
        ),
        # ── CHEAP TIER ───────────────────────────────────────────────────────
        # claude-haiku-4-5: $0.80/M input, $4.00/M output (Anthropic 2025)
        ModelOption(
            provider="anthropic",
            model_id="claude-haiku-4-5-20251001",
            display_name="Claude Haiku 4.5",
            tier="cheap",
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.0040,
            max_context_window=200_000,
            max_output_tokens=8_192,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_HAIKU_QUALITY,
            avg_latency_ms=700,
            rate_limit_rpm=1000,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # gpt-4o-mini: $0.15/M input, $0.60/M output (OpenAI 2025)
        ModelOption(
            provider="openai",
            model_id="gpt-4o-mini",
            display_name="GPT-4o Mini",
            tier="cheap",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.00060,
            max_context_window=128_000,
            max_output_tokens=16_384,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_GPT4O_MINI_QUALITY,
            avg_latency_ms=800,
            rate_limit_rpm=500,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # gemini-2.0-flash paid: $0.10/M input, $0.40/M output (Google 2025)
        ModelOption(
            provider="google",
            model_id="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash (paid)",
            tier="cheap",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            max_context_window=1_048_576,
            max_output_tokens=8_192,
            capabilities=["vision"],
            supports_streaming=True,
            quality_ratings=_FLASH_PAID_QUALITY,
            avg_latency_ms=500,
            rate_limit_rpm=2000,
            allowed_sensitivity_levels=_NO_RESTRICTED,
        ),
        # ── MID TIER ─────────────────────────────────────────────────────────
        # claude-sonnet-4: $3.00/M input, $15.00/M output (Anthropic 2025)
        ModelOption(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            tier="mid",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            max_context_window=200_000,
            max_output_tokens=8_192,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_SONNET_QUALITY,
            avg_latency_ms=2000,
            rate_limit_rpm=500,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # gpt-4o: $2.50/M input, $10.00/M output (OpenAI 2025)
        ModelOption(
            provider="openai",
            model_id="gpt-4o",
            display_name="GPT-4o",
            tier="mid",
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.010,
            max_context_window=128_000,
            max_output_tokens=16_384,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_GPT4O_QUALITY,
            avg_latency_ms=2500,
            rate_limit_rpm=500,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # gemini-2.5-pro: $1.25/M input (≤200K), $10.00/M output (Google 2025)
        ModelOption(
            provider="google",
            model_id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            tier="mid",
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.010,
            max_context_window=1_048_576,
            max_output_tokens=8_192,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_GEMINI25_PRO_QUALITY,
            avg_latency_ms=3000,
            rate_limit_rpm=360,
            allowed_sensitivity_levels=_NO_RESTRICTED,
        ),
        # ── PREMIUM TIER ─────────────────────────────────────────────────────
        # claude-opus-4: $15.00/M input, $75.00/M output (Anthropic 2025)
        ModelOption(
            provider="anthropic",
            model_id="claude-opus-4-20250514",
            display_name="Claude Opus 4",
            tier="premium",
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            max_context_window=200_000,
            max_output_tokens=8_192,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_OPUS_QUALITY,
            avg_latency_ms=4000,
            rate_limit_rpm=200,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # gpt-4.5-preview: $75/M input, $150/M output (OpenAI 2025)
        ModelOption(
            provider="openai",
            model_id="gpt-4.5-preview",
            display_name="GPT-4.5 Preview",
            tier="premium",
            cost_per_1k_input=0.075,
            cost_per_1k_output=0.150,
            max_context_window=128_000,
            max_output_tokens=16_384,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_GPT45_PREVIEW_QUALITY,
            avg_latency_ms=5000,
            rate_limit_rpm=100,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # gemini-2.5-pro with thinking: same model, different config tier
        ModelOption(
            provider="google",
            model_id="gemini-2.5-pro-thinking",
            display_name="Gemini 2.5 Pro (thinking)",
            tier="premium",
            cost_per_1k_input=0.00350,
            cost_per_1k_output=0.015,
            max_context_window=1_048_576,
            max_output_tokens=16_384,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_GEMINI25_PRO_THINKING_QUALITY,
            avg_latency_ms=6000,
            rate_limit_rpm=100,
            allowed_sensitivity_levels=_NO_RESTRICTED,
        ),
        # ── PREMIUM TIER (continued — formerly "ultra") ───────────────────────
        # Change 8: ultra tier merged into premium. These models were previously
        # tagged "ultra"; they are now premium alongside Opus 4, GPT-4.5, and
        # Gemini 2.5 Pro (thinking).
        # o3: $10/M input, $40/M output (OpenAI 2025)
        ModelOption(
            provider="openai",
            model_id="o3",
            display_name="o3",
            tier="premium",
            cost_per_1k_input=0.010,
            cost_per_1k_output=0.040,
            max_context_window=200_000,
            max_output_tokens=100_000,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_O3_QUALITY,
            avg_latency_ms=15_000,
            rate_limit_rpm=50,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # o4-mini: $1.10/M input, $4.40/M output (OpenAI 2025)
        ModelOption(
            provider="openai",
            model_id="o4-mini",
            display_name="o4-mini",
            tier="premium",
            cost_per_1k_input=0.0011,
            cost_per_1k_output=0.0044,
            max_context_window=200_000,
            max_output_tokens=100_000,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_O4_MINI_QUALITY,
            avg_latency_ms=8_000,
            rate_limit_rpm=100,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
        # claude-opus-4 with extended thinking
        ModelOption(
            provider="anthropic",
            model_id="claude-opus-4-20250514-extended",
            display_name="Claude Opus 4 (extended thinking)",
            tier="premium",
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            max_context_window=200_000,
            max_output_tokens=16_000,
            capabilities=["vision", "function_calling"],
            supports_streaming=True,
            quality_ratings=_OPUS_EXTENDED_QUALITY,
            avg_latency_ms=20_000,
            rate_limit_rpm=50,
            allowed_sensitivity_levels=_ALL_LEVELS,
        ),
    ]
    return {m.model_id: m for m in models}


def _build_default_registry() -> dict[str, ModelOption]:
    """Load from JSON first; fall back to hardcoded catalog."""
    from_json = _load_registry_from_json()
    if from_json is not None:
        return from_json
    log.info("model_registry_using_hardcoded_fallback")
    return _build_hardcoded_registry()


class ModelRegistry:
    """
    Thread-safe model catalog with live RPM tracking.

    The registry stores canonical ModelOption objects.  The routing engine
    MUST call all_available_models() which returns .model_copy() instances —
    never mutate registry objects directly.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelOption] = _build_default_registry()
        self._lock = threading.Lock()
        # Sliding-window RPM tracking: model_id → deque of monotonic timestamps
        self._rpm_window: dict[str, deque[float]] = defaultdict(deque)

    # ── Public API ──────────────────────────────────────────────────────────

    def all_available_models(self) -> list[ModelOption]:
        """Return mutable copies of every available model (safe to mutate)."""
        with self._lock:
            return [m.model_copy() for m in self._models.values() if m.is_available]

    def get_model(self, model_id: str) -> ModelOption | None:
        """Fetch a single model by ID, or None if not found."""
        with self._lock:
            m = self._models.get(model_id)
            return m.model_copy() if m else None

    def most_expensive_model(self) -> ModelOption:
        """Return the model with the highest combined cost (used for worst-case estimates)."""
        with self._lock:
            available = [m for m in self._models.values() if m.is_available]
            return max(
                available, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output
            ).model_copy()

    def get_max_context_window(self) -> int:
        """
        Return the largest context window across all available models.
        Used by the context compressor before a specific model has been chosen.
        """
        with self._lock:
            available = [m for m in self._models.values() if m.is_available]
            return max(m.max_context_window for m in available) if available else 4096

    def update_load(self, model_id: str) -> None:
        """Increment the RPM counter for a model (called after routing decision)."""
        now = time.monotonic()
        with self._lock:
            window = self._rpm_window[model_id]
            while window and now - window[0] >= 60.0:
                window.popleft()
            window.append(now)
            if model_id in self._models:
                self._models[model_id].current_load_rpm = len(window)

    def is_near_rate_limit(self, model_id: str) -> bool:
        """
        Return True when current load is within RATE_LIMIT_SAFETY_MARGIN of the
        model's stated RPM cap, so we can pre-empt the actual 429.
        """
        with self._lock:
            model = self._models.get(model_id)
            if not model:
                return False
            return model.current_load_rpm >= model.rate_limit_rpm * RATE_LIMIT_SAFETY_MARGIN

    def register_model(self, model: ModelOption) -> None:
        """Add or replace a model entry (for custom / enterprise models)."""
        with self._lock:
            self._models[model.model_id] = model
        log.info("model_registered", model_id=model.model_id, tier=model.tier)

    def refresh(self) -> None:
        """
        Stub: in production, this would re-fetch live availability and pricing
        from each provider's status / pricing API.
        """
        log.debug("model_registry_refresh_stub_called")
