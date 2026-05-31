"""
completions.py — get a model's answer to one sample.

Two modes:
  mock (default) — deterministic *simulated* answer. Correctness is sampled from
                   the model's quality_ratings for the task type (seeded per
                   model+sample, so runs are reproducible). Objective samples
                   encode the outcome in the answer text so the REAL graders run
                   unchanged; open-ended samples carry sim_quality for the mock
                   judge. No network, no spend.
  live           — real provider call (wired in Phase 5).

Cost is computed the same way the router estimates it
(routing_engine._estimate_cost): tokens/1000 × per-1k price, using the shared
estimate_tokens() heuristic so mock and live accounting match.
"""

from __future__ import annotations

import random
import time

from ..classifier import estimate_tokens
from ..schemas import ModelOption, RoutingRequest
from .schemas import Completion, EvalSample


class LiveCompletionError(Exception):
    """A live completion could not be produced (missing key or provider failure).

    The runner catches this and skips the (sample, strategy) pair rather than
    crashing the whole run or scoring an infra gap as zero quality.
    """

# Fallback quality by tier when a model has no rating for the task type.
_TIER_DEFAULT_QUALITY = {"free": 0.55, "cheap": 0.70, "mid": 0.82, "premium": 0.92}

_OBJECTIVE_GRADERS = {"gsm8k", "mmlu", "humaneval"}


def _sim_quality(model: ModelOption, task_type: str) -> float:
    """The model's expected quality on this task type, in [0, 1]."""
    q = model.quality_ratings.get(task_type)
    if q is None:
        q = model.quality_ratings.get("general")
    if q is None:
        q = _TIER_DEFAULT_QUALITY.get(model.tier, 0.7)
    return max(0.0, min(1.0, float(q)))


def _rng(model: ModelOption, sample: EvalSample, seed: str) -> random.Random:
    """Deterministic per (model, sample) RNG — string seed is stable across runs."""
    return random.Random(f"{seed}|{model.model_id}|{sample.id}")


def estimate_cost(model: ModelOption, input_tokens: int, output_tokens: int) -> float:
    """Mirror routing_engine._estimate_cost so eval cost matches router cost."""
    cost = (input_tokens / 1000.0) * model.cost_per_1k_input + (
        output_tokens / 1000.0
    ) * model.cost_per_1k_output
    return round(cost, 6)


# ── Simulated answer text per objective grader ──────────────────────────────


def _wrong_number(reference: str, rng: random.Random) -> str:
    try:
        val = float(reference)
        delta = rng.choice([-5, -3, -2, -1, 1, 2, 3, 7])
        wrong = int(val) + delta if val.is_integer() else round(val + delta, 2)
        return str(wrong)
    except (ValueError, AttributeError):
        return "0"


def _wrong_letter(reference: str, n_choices: int, rng: random.Random) -> str:
    letters = [chr(ord("A") + i) for i in range(max(2, n_choices))]
    alts = [x for x in letters if x != reference]
    return rng.choice(alts) if alts else reference


def _simulated_text(sample: EvalSample, correct: bool, rng: random.Random) -> str:
    ref = sample.reference or ""
    if sample.grader == "gsm8k":
        ans = ref if correct else _wrong_number(ref, rng)
        return f"Working through the problem step by step, the result is {ans}.\n#### {ans}"
    if sample.grader == "mmlu":
        n = len(sample.metadata.get("choices", [])) or 4
        ans = ref if correct else _wrong_letter(ref, n, rng)
        return ans
    if sample.grader == "humaneval":
        header = sample.metadata.get("prompt_header", "")
        if correct and sample.reference:
            body = sample.reference  # canonical solution body
        else:
            body = "    raise NotImplementedError\n"
        return header + body
    # open-ended — generic plausible prose; the mock judge uses sim_quality.
    return (
        f"Here is a thoughtful response to the request:\n\n"
        f"{sample.prompt[:80]}...\n\n"
        "This addresses the key points clearly and with relevant detail."
    )


async def get_completion(
    model: ModelOption,
    sample: EvalSample,
    *,
    mode: str = "mock",
    api_keys: dict | None = None,
    max_tokens: int = 1024,
    seed: str = "flux-eval",
) -> Completion:
    """Produce ``model``'s Completion for ``sample`` under the given mode."""
    if mode == "mock":
        return _mock_completion(model, sample, seed)
    if mode == "live":
        return await _live_completion(model, sample, api_keys or {}, max_tokens)
    raise ValueError(f"Unknown completion mode '{mode}' (use 'mock' or 'live')")


def _mock_completion(model: ModelOption, sample: EvalSample, seed: str) -> Completion:
    q = _sim_quality(model, sample.task_type)
    rng = _rng(model, sample, seed)

    sim_quality: float | None = None
    if sample.grader in _OBJECTIVE_GRADERS:
        correct = rng.random() < q
        text = _simulated_text(sample, correct, rng)
    else:
        text = _simulated_text(sample, True, rng)
        # Jitter the open-ended quality a little so the report isn't perfectly flat.
        sim_quality = max(0.0, min(1.0, q + rng.uniform(-0.05, 0.05)))

    input_tokens = estimate_tokens(sample.prompt)
    output_tokens = estimate_tokens(text)
    return Completion(
        text=text,
        model_id=model.model_id,
        provider=model.provider,
        tier=model.tier,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=estimate_cost(model, input_tokens, output_tokens),
        latency_ms=model.avg_latency_ms,
        simulated=True,
        sim_quality=sim_quality,
    )


async def _live_completion(
    model: ModelOption, sample: EvalSample, api_keys: dict, max_tokens: int
) -> Completion:
    """Call the real provider for ``model`` with the sample's prompt.

    The model is forced — we call provider_caller directly rather than routing —
    so each strategy/baseline hits exactly the model it selected. temperature=0
    keeps answers as reproducible as the provider allows.
    """
    from ..provider_caller import ProviderCallError, call_provider

    provider = model.provider.lower()
    key = api_keys.get(provider)
    if not key:
        raise LiveCompletionError(
            f"no API key for provider '{provider}' — set {provider.upper()}_API_KEY"
        )

    request = RoutingRequest(
        raw_prompt=sample.prompt,
        user_id="flux-eval",
        max_tokens_requested=max_tokens,
        temperature=0.0,
    )
    t0 = time.perf_counter()
    try:
        text = await call_provider(model, request, key)
    except ProviderCallError as exc:
        raise LiveCompletionError(f"provider call failed for {model.model_id}: {exc}") from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)

    input_tokens = estimate_tokens(sample.prompt)
    output_tokens = estimate_tokens(text)
    return Completion(
        text=text,
        model_id=model.model_id,
        provider=model.provider,
        tier=model.tier,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=estimate_cost(model, input_tokens, output_tokens),
        latency_ms=latency_ms,
        simulated=False,
        sim_quality=None,
    )
