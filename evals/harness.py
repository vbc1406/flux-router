"""
harness.py — run the golden set through the router once per routing_priority.

For each (task, routing_priority) pair:
  1. Build a RoutingRequest (step_type + tools carried over from the task) and
     call the real RoutingEngine.route() — this is the actual thing being
     measured, not a stand-in.
  2. Produce an answer from the chosen model (mock: simulated; live: a real
     provider call via router.provider_caller) and grade it — exact-match
     against `expected`, or Claude-as-judge 1-5 against `rubric`.
  3. Record a TaskResult: which model got picked, what the router estimated
     it would cost, and how it scored.

Mock mode (the default) makes no network calls and needs no API keys — it's
what "dry-run" means throughout this harness. Live mode makes real provider
calls (and a real judge call for rubric tasks) and needs the relevant
*_API_KEY env vars set.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import structlog

from router.flux import make_flux
from router.schemas import ModelOption, RoutingRequest

from .golden_set import GOLDEN_SET, GoldenTask
from .judge import DEFAULT_JUDGE_MODEL, live_score, mock_score, resolve_judge_model

log = structlog.get_logger(__name__)

ROUTING_PRIORITIES: list[str] = ["cascade", "cost-optimized", "quality_max"]

_TIER_DEFAULT_QUALITY = {"free": 0.55, "cheap": 0.70, "mid": 0.82, "premium": 0.92}


@dataclass
class TaskResult:
    routing_priority: str
    task_id: str
    step_type: str
    task_type: str
    grader: str
    model_id: str
    provider: str
    tier: str
    cost_usd: float
    quality_score: float  # normalized to a 1-5 scale for every task, exact or rubric
    passed: bool | None  # True/False for exact-match tasks, None for rubric tasks


def _task_quality(model: ModelOption, task: GoldenTask) -> float:
    q = model.quality_ratings.get(task.task_type)
    if q is None:
        q = model.quality_ratings.get("general")
    if q is None:
        q = _TIER_DEFAULT_QUALITY.get(model.tier, 0.7)
    return max(0.0, min(1.0, float(q)))


def _normalize_exact(text: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", text.strip().lower())


def _mock_answer(model: ModelOption, task: GoldenTask, routing_priority: str) -> tuple[str, bool]:
    """Simulated answer text + whether it was correct, for exact-match tasks.

    Correctness is drawn from the chosen model's task-type quality rating —
    same convention as router/evals/completions.py's mock mode — seeded per
    (priority, task, model) so runs are reproducible.
    """
    rng = random.Random(f"{routing_priority}|{task.id}|{model.model_id}|answer")
    q = _task_quality(model, task)
    correct = rng.random() < q
    if task.grader == "exact":
        if correct:
            return task.expected or "", True
        # A plausible-but-wrong distractor: for numeric answers, offset the
        # number; for text answers, swap in a different plausible string.
        try:
            val = float(task.expected)
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            wrong = val + delta
            wrong_str = str(int(wrong)) if wrong.is_integer() else str(round(wrong, 2))
        except (TypeError, ValueError):
            wrong_str = f"not-{task.expected}"
        return wrong_str, False
    # Rubric task: text content doesn't matter in mock mode (mock_score()
    # derives the score directly from quality_ratings), just return a stub.
    return f"[simulated answer for {task.id}]", correct


async def _live_answer(model: ModelOption, task: GoldenTask, api_key: str) -> str:
    from router.provider_caller import ProviderCallError, call_provider

    prompt = task.prompt
    if task.tools:
        tool_names = ", ".join(t["function"]["name"] for t in task.tools)
        prompt = (
            f"{prompt}\n\nAvailable tools: {tool_names}. "
            "Respond with ONLY the name of the single tool you would call."
        )
    request = RoutingRequest(
        raw_prompt=prompt,
        user_id="flux-eval-priority",
        tools=task.tools,
        temperature=0.0,
    )
    try:
        result = await call_provider(model, request, api_key)
    except ProviderCallError as exc:
        log.warning("live_answer_failed", model=model.model_id, task=task.id, error=str(exc))
        return ""
    return result.text


def _provider_keys(flux) -> dict[str, str]:
    return {prov: sec.get_secret_value() for prov, sec in flux._provider_keys.items()}


async def run_golden_set(
    *,
    mode: str = "mock",
    routing_priorities: list[str] | None = None,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    tasks: list[GoldenTask] | None = None,
) -> list[TaskResult]:
    """Run every task in `tasks` (default: the full GOLDEN_SET) through every
    routing_priority in `routing_priorities` (default: ROUTING_PRIORITIES).
    Returns one TaskResult per (task, routing_priority) pair.
    """
    if mode not in ("mock", "live"):
        raise ValueError(f"Unknown mode '{mode}' (use 'mock' or 'live')")

    routing_priorities = routing_priorities or ROUTING_PRIORITIES
    tasks = tasks if tasks is not None else GOLDEN_SET

    flux = make_flux()
    engine = flux._engine
    registry = engine._registry

    api_keys: dict[str, str] = {}
    judge_model = None
    if mode == "live":
        api_keys = _provider_keys(flux)
        judge_model = resolve_judge_model(registry, judge_model_id)

    results: list[TaskResult] = []

    for priority in routing_priorities:
        for task in tasks:
            request = RoutingRequest(
                raw_prompt=task.prompt,
                user_id="flux-eval-priority",
                routing_priority=priority,
                step_type=task.step_type,
                tools=task.tools,
                exploration_rate=0.0,
            )
            decision = await engine.route(request)
            if decision.chosen_model is None:
                log.warning(
                    "no_model_chosen", priority=priority, task=task.id, reason=decision.reasoning
                )
                continue
            model = decision.chosen_model
            cost = decision.estimated_cost

            if task.grader == "exact":
                if mode == "mock":
                    answer, correct = _mock_answer(model, task, priority)
                else:
                    api_key = api_keys.get(model.provider.lower(), "")
                    answer = await _live_answer(model, task, api_key) if api_key else ""
                    correct = _normalize_exact(answer) == _normalize_exact(task.expected or "")
                score = 5.0 if correct else 1.0
                passed: bool | None = correct
            else:
                passed = None
                if mode == "mock":
                    score = mock_score(model, task, priority)
                else:
                    api_key = api_keys.get(model.provider.lower(), "")
                    answer = await _live_answer(model, task, api_key) if api_key else ""
                    if judge_model is None:
                        score = 1.0
                    else:
                        judge_key = api_keys.get(judge_model.provider.lower(), "")
                        score = (
                            await live_score(judge_model, judge_key, task, answer)
                            if judge_key
                            else 1.0
                        )

            results.append(
                TaskResult(
                    routing_priority=priority,
                    task_id=task.id,
                    step_type=task.step_type,
                    task_type=task.task_type,
                    grader=task.grader,
                    model_id=model.model_id,
                    provider=model.provider,
                    tier=model.tier,
                    cost_usd=cost,
                    quality_score=round(score, 3),
                    passed=passed,
                )
            )

    return results
