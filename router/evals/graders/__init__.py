"""
graders — score a completion against its sample.

``grade()`` dispatches on ``sample.grader`` to the objective scorers or the LLM
judge and returns ``(quality, correct)``:
  quality — 0.0–1.0, or None if the item could not be graded (skipped).
  correct — bool for objective graders, None for the judge.

HumanEval code execution runs only when the code is trusted: the completion is
simulated (mock mode) OR the user passed --allow-code-exec.
"""

from __future__ import annotations

from ..schemas import Completion, EvalSample
from . import objective
from .llm_judge import Judge, grade_llm_judge, make_judge

__all__ = ["Judge", "grade", "make_judge"]


async def grade(
    sample: EvalSample,
    completion: Completion,
    *,
    allow_code_exec: bool = False,
    judge: Judge | None = None,
) -> tuple[float | None, bool | None]:
    g = sample.grader
    if g == "gsm8k":
        return objective.grade_gsm8k(sample, completion)
    if g == "mmlu":
        return objective.grade_mmlu(sample, completion)
    if g == "humaneval":
        return objective.grade_humaneval(
            sample, completion, allow_exec=allow_code_exec or completion.simulated
        )
    if g == "llm_judge":
        return await grade_llm_judge(sample, completion, judge=judge)
    raise ValueError(f"Unknown grader '{g}'")
