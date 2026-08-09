"""
objective.py — graders for samples with a verifiable ground truth.

  gsm8k     → extract the final number, compare numerically to the reference.
  mmlu      → extract the chosen letter (A–D), compare to the reference.
  humaneval → execute the completion against the unit tests (pass@1).

Each returns ``(quality, correct)`` where quality is 1.0/0.0 (or None if the
item could not be graded) and correct is the bool outcome (or None).
"""

from __future__ import annotations

import re

from ..schemas import Completion, EvalSample

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_LETTER_RE = re.compile(r"\b([A-Da-d])\b")

def _last_number(text: str) -> str | None:
    """Return the final numeric token, preferring text after the last '####'."""
    tail = text.split("####")[-1] if "####" in text else text
    nums = _NUM_RE.findall(tail)
    if not nums:
        nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def _nums_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def grade_gsm8k(sample: EvalSample, completion: Completion) -> tuple[float | None, bool | None]:
    pred = _last_number(completion.text)
    correct = _nums_equal(pred, sample.reference)
    return (1.0 if correct else 0.0, correct)


def grade_mmlu(sample: EvalSample, completion: Completion) -> tuple[float | None, bool | None]:
    m = _LETTER_RE.search(completion.text.strip())
    pred = m.group(1).upper() if m else None
    correct = pred is not None and pred == (sample.reference or "").upper()
    return (1.0 if correct else 0.0, correct)


def _extract_code(text: str) -> str:
    """Pull Python source out of a completion, unwrapping a markdown fence if present."""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return fence.group(1) if fence else text


def grade_humaneval(
    sample: EvalSample,
    completion: Completion,
    allow_exec: bool,
) -> tuple[float | None, bool | None]:
    """Leave HumanEval ungraded; host execution is intentionally unavailable."""
    if allow_exec:
        raise RuntimeError(
            "HumanEval code execution is disabled: Flux has no security sandbox for "
            "model-generated code. Run HumanEval in a purpose-built isolated runner."
        )
    return (None, None)


_TOOL_NAME_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def grade_tool_select(
    sample: EvalSample, completion: Completion
) -> tuple[float | None, bool | None]:
    """Item 6: the one agent-step-type grader with a verifiable right answer.

    Correct iff the completion names the expected tool and no other tool
    from the offered set — a live model's text-only answer ("I'll call
    get_weather") and a real tool_calls response (whose JSON naturally
    contains the function name as a bare token) both parse the same way,
    since this only ever runs against Completion.text.
    """
    expected = sample.metadata.get("expected_tool")
    if not expected:
        return (None, None)
    offered = {
        t.get("function", {}).get("name")
        for t in sample.metadata.get("tools", [])
        if t.get("function", {}).get("name")
    }
    mentioned = set(_TOOL_NAME_RE.findall(completion.text)) & offered
    correct = mentioned == {expected}
    return (1.0 if correct else 0.0, correct)
