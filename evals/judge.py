"""
judge.py — score rubric-graded answers 1-5 with Claude as the judge.

Live mode makes a real call to the configured judge model via the router's
own provider_caller (so judge calls are accounted for the same way any other
router-mediated call would be). Mock mode (the default, and what the dry-run
in the PR description uses) never calls a network — it derives a deterministic
1-5 score from the chosen model's quality_ratings for the task_type, jittered
by a seed of (routing_priority, task_id, model_id) so repeated runs are
reproducible but different tasks/models don't all land on the same score.
"""

from __future__ import annotations

import random
import re

import structlog

from router.model_registry import ModelRegistry
from router.schemas import ModelOption, RoutingRequest

from .golden_set import GoldenTask

log = structlog.get_logger(__name__)

DEFAULT_JUDGE_MODEL = "claude-opus-5"

_RUBRIC_PROMPT = (
    "You are an impartial judge scoring an AI assistant's answer to a task.\n\n"
    "[Task]\n{prompt}\n\n"
    "[What a strong answer looks like]\n{rubric}\n\n"
    "[Assistant's answer]\n{answer}\n\n"
    "Score the answer from 1 to 5, where 1 = fails the rubric badly and "
    "5 = fully satisfies it. Respond with ONLY a single integer 1-5 — no "
    "words, no punctuation."
)

_SCORE_RE = re.compile(r"\b([1-5])\b")

# Fallback quality by tier when a model has no rating for the task type —
# mirrors router/evals/completions.py's _TIER_DEFAULT_QUALITY so mock scores
# stay consistent with the rest of the codebase's simulated-quality convention.
_TIER_DEFAULT_QUALITY = {"free": 0.55, "cheap": 0.70, "mid": 0.82, "premium": 0.92}


def _base_quality(model: ModelOption, task_type: str) -> float:
    q = model.quality_ratings.get(task_type)
    if q is None:
        q = model.quality_ratings.get("general")
    if q is None:
        q = _TIER_DEFAULT_QUALITY.get(model.tier, 0.7)
    return max(0.0, min(1.0, float(q)))


def mock_score(model: ModelOption, task: GoldenTask, routing_priority: str) -> float:
    """Deterministic simulated 1-5 rubric score, seeded per (priority, task, model)."""
    rng = random.Random(f"{routing_priority}|{task.id}|{model.model_id}")
    q = _base_quality(model, task.task_type)
    jittered = max(0.0, min(1.0, q + rng.uniform(-0.07, 0.07)))
    return round(1.0 + jittered * 4.0, 2)  # map [0,1] -> [1,5]


def _parse_score(raw: str) -> float:
    m = _SCORE_RE.search(raw.strip())
    if not m:
        return 1.0
    return float(m.group(1))


async def live_score(
    judge_model: ModelOption,
    judge_api_key: str,
    task: GoldenTask,
    answer_text: str,
) -> float:
    from router.provider_caller import ProviderCallError, call_provider

    prompt = _RUBRIC_PROMPT.format(prompt=task.prompt, rubric=task.rubric, answer=answer_text)
    request = RoutingRequest(raw_prompt=prompt, user_id="flux-eval-priority-judge", temperature=0.0)
    try:
        raw = await call_provider(judge_model, request, judge_api_key)
    except ProviderCallError as exc:
        log.warning("judge_call_failed", model=judge_model.model_id, task=task.id, error=str(exc))
        return 1.0
    return _parse_score(raw)


def resolve_judge_model(registry: ModelRegistry, judge_model_id: str) -> ModelOption:
    model = registry.get_model(judge_model_id)
    if model is None:
        raise ValueError(
            f"Judge model '{judge_model_id}' not found in the registry (router/models.json)."
        )
    return model
