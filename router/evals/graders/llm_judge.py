"""
llm_judge.py — grade open-ended answers with an LLM (Claude Opus by default).

Used for MT-Bench-style prompts that have no objective answer. The judge rates
the answer 1–10 against a short rubric; the score is normalized to [0, 1].

In mock mode there is no judge — the completion carries ``sim_quality`` and we
report that directly, so offline/CI runs still produce a quality number without
any network call or spend.
"""

from __future__ import annotations

import re

import structlog

from ...model_registry import ModelRegistry
from ...schemas import ModelOption, RoutingRequest
from ..cache import DiskCache
from ..schemas import Completion, EvalSample

log = structlog.get_logger(__name__)

_RUBRIC = (
    "You are an impartial judge evaluating the quality of an AI assistant's answer "
    "to a user's request. Consider helpfulness, relevance, accuracy, depth, and "
    "clarity. Rate the answer on a scale from 1 to 10, where 10 is excellent.\n\n"
    "[User request]\n{prompt}\n\n[Assistant answer]\n{answer}\n\n"
    "Respond with ONLY a single integer from 1 to 10 — no words, no punctuation."
)

_SCORE_RE = re.compile(r"\b(10|[1-9])\b")


class Judge:
    """Calls a provider to score an answer. Constructed only for live runs."""

    def __init__(
        self, model: ModelOption, api_key: str, cache: DiskCache | None = None
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._cache = cache  # optional DiskCache; verdicts keyed by answer text

    async def score(self, sample: "EvalSample", completion: "Completion") -> float:
        from ...provider_caller import ProviderCallError, call_provider
        from ..cache import make_key

        cache_key = None
        if self._cache is not None and self._cache.enabled:
            cache_key = make_key("judge", self._model.model_id, sample.id, completion.text)
            hit = self._cache.get(cache_key)
            if hit is not None:
                return float(hit["score"])

        prompt = _RUBRIC.format(prompt=sample.prompt, answer=completion.text)
        request = RoutingRequest(raw_prompt=prompt, user_id="flux-eval-judge", temperature=0.0)
        try:
            # Bugfix: call_provider() returns a ProviderResult, not a bare
            # str — same class of bug as completions.py::_live_completion.
            # This would have raised AttributeError (ProviderResult has no
            # .strip()) the moment a live judge call actually ran; caught by
            # mypy --strict, not by any test (no test exercised a live judge
            # call — see TestJudge in test_evals.py, all mock-mode).
            result = await call_provider(self._model, request, self._api_key)
        except ProviderCallError as exc:
            log.warning("judge_call_failed", model=self._model.model_id, error=str(exc))
            return 0.0
        score = _parse_score(result.text)
        if cache_key is not None and self._cache is not None:
            self._cache.set(cache_key, {"score": score})
        return score


def _parse_score(raw: str) -> float:
    """Map the judge's reply (an integer 1–10) to a [0, 1] quality score."""
    m = _SCORE_RE.search(raw.strip())
    if not m:
        return 0.0
    return max(0.0, min(1.0, int(m.group(1)) / 10.0))


def make_judge(
    registry: ModelRegistry,
    model_id: str,
    api_keys: dict[str, str],
    cache: DiskCache | None = None,
) -> Judge:
    """Build a Judge for ``model_id`` using the matching provider key."""
    model = registry.get_model(model_id)
    if model is None:
        raise ValueError(
            f"Judge model '{model_id}' not found in the registry. "
            "Pass a --judge-model that exists in models.json."
        )
    key = api_keys.get(model.provider.lower(), "")
    if not key:
        raise ValueError(
            f"No API key for judge provider '{model.provider}'. "
            f"Set the appropriate *_API_KEY env var."
        )
    return Judge(model, key, cache=cache)


async def grade_llm_judge(
    sample: "EvalSample",
    completion: "Completion",
    *,
    judge: Judge | None = None,
) -> tuple[float | None, bool | None]:
    # Mock mode: trust the simulated quality the completion carries.
    if completion.sim_quality is not None:
        return (round(completion.sim_quality, 4), None)
    if judge is None:
        # Live run but no judge configured — leave ungraded.
        return (None, None)
    return (await judge.score(sample, completion), None)
