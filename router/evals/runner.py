"""
runner.py — orchestrate one eval: every sample × every strategy.

For each (sample, strategy) it resolves the model the strategy would use,
gets a completion (simulated or live, cached in live mode), grades it, and
records a GradedResult. Items the grader can't score (e.g. HumanEval without
exec permission, or a judge with no key) are skipped and counted.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from ..config import DOMAIN_TIER_FLOORS, TIER_ORDER
from ..flux import Flux, make_flux
from ..schemas import ModelOption, RoutingRequest
from .cache import DiskCache, make_key
from .completions import LiveCompletionError, get_completion
from .datasets import DATASETS, load_datasets
from .graders import grade, make_judge
from .graders.objective import grade_budget_ladder
from .schemas import Completion, EvalSample, GradedResult
from .strategies import pick_model

log = structlog.get_logger(__name__)


def _safety_escalated(sample: EvalSample, model: ModelOption) -> bool | None:
    """For a high-stakes legal/medical sample, did the chosen model's tier
    meet config.DOMAIN_TIER_FLOORS[domain]? None when the sample isn't a
    high-stakes domain sample."""
    domain = sample.metadata.get("domain", "")
    stakes = sample.metadata.get("stakes", "")
    if domain not in DOMAIN_TIER_FLOORS or stakes != "high":
        return None
    floor_tier = DOMAIN_TIER_FLOORS[domain]
    try:
        return TIER_ORDER.index(model.tier) >= TIER_ORDER.index(floor_tier)
    except ValueError:
        return False


@dataclass
class RunConfig:
    datasets: list[str] = field(default_factory=lambda: list(DATASETS))
    strategies: list[str] = field(default_factory=lambda: ["flux", "premium", "cheapest"])
    n: int = 50
    mode: str = "mock"  # "mock" | "live"
    source: str = "fixture"  # "fixture" | "hub"
    judge_model: str = "claude-opus-4-7"
    allow_code_exec: bool = False
    seed: str = "flux-eval"
    cache_dir: str | None = ".eval_cache"


@dataclass
class RunOutput:
    config: RunConfig
    results: list[GradedResult]
    n_samples: int
    n_skipped: int
    # Deterministic, non-per-sample system checks (fallback recovery, etc.) —
    # see _check_fallback_recovery(). Not graded per-sample because a
    # fallback only fires on an actual failure, which mock-mode completions
    # never produce.
    system_checks: dict[str, Any] = field(default_factory=dict)


async def _check_fallback_recovery() -> bool:
    """System check: does FallbackExecutor actually recover onto the next
    model in the chain when the primary model's call fails? Exercises the
    real router.fallback_chain.FallbackExecutor end to end with a fake
    api_caller (no network) rather than inferring this from routing config.
    """
    from ..analytics import RoutingAnalytics
    from ..fallback_chain import FallbackExecutor
    from ..schemas import RoutingDecision

    flux = make_flux()
    registry = flux._engine._registry
    models = registry.all_available_models()
    if len(models) < 2:
        return False
    primary, fallback = models[0], models[1]
    decision = RoutingDecision(chosen_model=primary, fallback_chain=[fallback])
    request = RoutingRequest(raw_prompt="fallback recovery system check", user_id="flux-eval")

    async def flaky_caller(_req: RoutingRequest, model: ModelOption) -> str:
        if model.model_id == primary.model_id:
            raise ConnectionError("simulated primary-model failure")
        return "recovered response from the fallback model"

    executor = FallbackExecutor(RoutingAnalytics(log_path=None))
    _response, used_model, events = await executor.execute_with_fallback(
        request, decision, flaky_caller
    )
    return used_model.model_id == fallback.model_id and len(events) >= 1


def _question_type(sample: EvalSample) -> str:
    """Human-readable type label: dataset + the finest topic tag we already have."""
    topic = sample.metadata.get("subject") or sample.metadata.get("category")
    base = f"{sample.dataset}/{topic}" if topic else sample.dataset
    return f"{base} ({sample.task_type})"


def _provider_keys(flux: Flux) -> dict[str, str]:
    """Extract plain-text provider keys from a Flux instance for live calls."""
    return {prov: sec.get_secret_value() for prov, sec in flux._provider_keys.items()}


async def _completion(
    cache: DiskCache,
    model: ModelOption,
    sample: EvalSample,
    mode: str,
    api_keys: dict[str, Any],
    seed: str,
) -> Completion:
    """get_completion with a live-mode disk cache wrapped around it."""
    if mode != "live" or not cache.enabled:
        return await get_completion(model, sample, mode=mode, api_keys=api_keys, seed=seed)

    key = make_key("completion", model.model_id, sample.id, sample.prompt)
    hit = cache.get(key)
    if hit is not None:
        return Completion(**hit)
    comp = await get_completion(model, sample, mode=mode, api_keys=api_keys, seed=seed)
    cache.set(key, asdict(comp))
    return comp


async def run_eval(config: RunConfig) -> RunOutput:
    samples = load_datasets(config.datasets, n=config.n, source=config.source)

    flux = make_flux()
    engine = flux._engine
    registry = engine._registry

    cache = DiskCache(
        f"{config.cache_dir}/cache.json" if (config.cache_dir and config.mode == "live") else None
    )

    api_keys: dict[str, str] = {}
    judge = None
    if config.mode == "live":
        api_keys = _provider_keys(flux)
        if any(s.grader == "llm_judge" for s in samples):
            judge = make_judge(registry, config.judge_model, api_keys, cache=cache)

    results: list[GradedResult] = []
    skipped = 0

    for i, sample in enumerate(samples, start=1):
        question_type = _question_type(sample)

        # budget_degradation/budget_stop: a deterministic check of the real
        # RunBudget ladder, not a model-quality question — their step_type
        # isn't even a valid RoutingRequest.step_type (there's no routing
        # decision to make here), so this must run before a RoutingRequest is
        # ever built for the sample. There is no "premium vs cheapest"
        # comparison for a system behavior check, so it runs once (not per
        # strategy) and the identical outcome is recorded against every
        # configured strategy so per-strategy aggregation in report.py
        # doesn't need a special case.
        if sample.grader == "budget_ladder":
            quality, correct = grade_budget_ladder(sample)
            if quality is None:
                skipped += 1
                continue
            for strategy in config.strategies:
                results.append(
                    GradedResult(
                        sample_id=sample.id,
                        dataset=sample.dataset,
                        task_type=sample.task_type,
                        strategy=strategy,
                        model_id="n/a (deterministic system check)",
                        tier="",
                        cost=0.0,
                        quality=quality,
                        correct=correct,
                        simulated=False,
                        prompt=sample.prompt,
                        question_type=question_type,
                        complexity=0.0,
                        quality_rating=0.0,
                        step_type=sample.metadata.get("step_type", ""),
                    )
                )
            continue

        request = RoutingRequest(
            raw_prompt=sample.prompt,
            user_id="flux-eval",
            required_capabilities=sample.metadata.get("required_capabilities", []),
            # Item 6: agentic-dataset samples carry an explicit step_type
            # (plan/tool_select/...) so routing for those cases is
            # step-type-floor-aware (config.STEP_TYPE_FLOORS), same as a real
            # agent loop passing step_type= — not just task-type-classified
            # like every other dataset here.
            step_type=sample.metadata.get("step_type"),
            # Disable A/B exploration so the flux strategy is deterministic and
            # the published tradeoff numbers reproduce across runs.
            exploration_rate=0.0,
        )
        # Classify once per sample (strategy-independent) for the per-question view:
        # the complexity score + a human-readable question-type label.
        complexity = engine._classifier.analyze(request).complexity_score

        for strategy in config.strategies:
            t0 = time.perf_counter()
            model = await pick_model(strategy, request, engine, registry)
            routing_latency_ms = (time.perf_counter() - t0) * 1000 if strategy == "flux" else None
            try:
                comp = await _completion(
                    cache, model, sample, config.mode, api_keys, config.seed
                )
            except LiveCompletionError as exc:
                skipped += 1
                log.warning(
                    "eval_completion_skipped",
                    strategy=strategy,
                    sample=sample.id,
                    model=model.model_id,
                    error=str(exc),
                )
                continue
            quality, correct = await grade(
                sample, comp, allow_code_exec=config.allow_code_exec, judge=judge
            )
            if quality is None:
                skipped += 1
                continue
            results.append(
                GradedResult(
                    sample_id=sample.id,
                    dataset=sample.dataset,
                    task_type=sample.task_type,
                    strategy=strategy,
                    model_id=comp.model_id,
                    tier=comp.tier,
                    cost=comp.cost,
                    quality=quality,
                    correct=correct,
                    simulated=comp.simulated,
                    prompt=sample.prompt,
                    question_type=question_type,
                    complexity=complexity,
                    quality_rating=model.quality_ratings.get(sample.task_type, 0.0),
                    step_type=sample.metadata.get("step_type", ""),
                    provider=comp.provider,
                    input_tokens=comp.input_tokens,
                    output_tokens=comp.output_tokens,
                    latency_ms=comp.latency_ms,
                    routing_latency_ms=routing_latency_ms,
                    domain=sample.metadata.get("domain", ""),
                    safety_escalated=(
                        _safety_escalated(sample, model) if strategy == "flux" else None
                    ),
                    tool_call_valid=(correct if sample.grader == "agentic_tool_select" else None),
                    structured_output_valid=(
                        correct if sample.grader == "json_schema" else None
                    ),
                )
            )
        if i % 25 == 0:
            log.info("eval_progress", done=i, total=len(samples))

    system_checks = {"fallback_recovery_ok": await _check_fallback_recovery()}

    return RunOutput(
        config=config,
        results=results,
        n_samples=len(samples),
        n_skipped=skipped,
        system_checks=system_checks,
    )
