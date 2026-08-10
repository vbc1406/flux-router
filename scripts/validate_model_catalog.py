"""Validate the routable model catalog without making billable provider calls.

The default mode is deterministic and CI-safe. ``--online`` additionally
checks that each enabled model's documented identifier appears on a current
official provider page. This establishes catalog existence, not account-level
availability; only the opt-in provider smoke tests can prove an API key may
actually dispatch the model.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "router" / "models.json"

OFFICIAL_SOURCES = {
    "openai": "https://platform.openai.com/docs/models",
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models/overview",
    "google": "https://ai.google.dev/gemini-api/docs/models",
    "groq": "https://console.groq.com/docs/models",
    "mistral": "https://docs.mistral.ai/models/overview",
}

OFFICIAL_MODEL_SOURCES = {
    "gpt-5-mini": "https://platform.openai.com/docs/models/gpt-5-mini",
    "gpt-5.5": "https://platform.openai.com/docs/models/gpt-5.5",
    "o3": "https://platform.openai.com/docs/models/o3",
    "o4-mini": "https://platform.openai.com/docs/models/o4-mini",
    "mistral-small-latest": "https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03",
    "mistral-small-4": "https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03",
    "mistral-large-3": "https://docs.mistral.ai/models/model-cards/mistral-large-3-25-12",
}

# Mistral exposes friendly Flux catalog names separately from versioned API IDs.
DOCUMENTED_PROVIDER_IDS = {
    "mistral-medium-3": "mistral-medium-2505",
    "mistral-small-4": "mistral-small-2603",
    "mistral-large-3": "mistral-large-2512",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen-3.6-27b": "qwen/qwen3.6-27b",
}


def load_models() -> list[dict]:
    return json.loads(CATALOG.read_text())["models"]


def validate_local(models: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for model in models:
        model_id = model.get("model_id", "")
        if not model_id or model_id in seen:
            errors.append(f"missing or duplicate model_id: {model_id!r}")
        seen.add(model_id)
        if model.get("provider") not in OFFICIAL_SOURCES:
            errors.append(f"{model_id}: unsupported provider {model.get('provider')!r}")
        if model.get("max_context_window", 0) <= 0 or model.get("max_output_tokens", 0) <= 0:
            errors.append(f"{model_id}: context/output limits must be positive")
        for field in ("cost_per_1k_input", "cost_per_1k_output"):
            if not isinstance(model.get(field), (int, float)) or model[field] < 0:
                errors.append(f"{model_id}: {field} must be a non-negative number")
        expected = DOCUMENTED_PROVIDER_IDS.get(model_id)
        if expected and model.get("provider_model_id") != expected:
            errors.append(
                f"{model_id}: provider_model_id must be {expected!r}, "
                f"got {model.get('provider_model_id')!r}"
            )
    return errors


def validate_online(models: list[dict]) -> list[str]:
    pages: dict[str, str] = {}
    errors: list[str] = []
    headers = {"User-Agent": "FluxCatalogValidator/1.0"}
    for provider, url in OFFICIAL_SOURCES.items():
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                pages[provider] = response.read().decode("utf-8", errors="replace").lower()
        except Exception as exc:  # noqa: BLE001 - aggregate every provider failure
            errors.append(f"{provider}: could not read official source: {exc}")

    for model in models:
        if model.get("is_available") is False:
            continue
        provider = model["provider"]
        page = pages.get(provider)
        if page is None:
            continue
        documented_id = model.get("provider_model_id") or model["model_id"]
        if documented_id.lower() in page:
            continue
        model_url = OFFICIAL_MODEL_SOURCES.get(model["model_id"])
        if model_url:
            try:
                request = urllib.request.Request(model_url, headers=headers)
                with urllib.request.urlopen(request, timeout=30) as response:
                    detail = response.read().decode("utf-8", errors="replace").lower()
                if documented_id.lower() in detail:
                    continue
            except Exception as exc:  # noqa: BLE001 - report aggregate result
                errors.append(f"{provider}/{model['model_id']}: {model_url}: {exc}")
                continue
        errors.append(
            f"{provider}/{model['model_id']}: {documented_id!r} not found on "
            f"{model_url or OFFICIAL_SOURCES[provider]}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true", help="check official provider pages")
    args = parser.parse_args()
    models = load_models()
    errors = validate_local(models)
    if args.online:
        errors.extend(validate_online(models))
    if errors:
        print("Catalog validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    available = sum(model.get("is_available") is not False for model in models)
    mode = "local + official-source" if args.online else "local"
    print(f"Catalog validation passed ({mode}): {available}/{len(models)} models available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
