"""
datasets.py — load and normalize public benchmarks into EvalSample lists.

Two sources, one normalizer:
  source="fixture" (default)  → read the tiny JSON fixtures bundled in fixtures/.
                                No network, no extra deps — used by tests, CI,
                                and simulated (mock) runs.
  source="hub"                → download the full datasets via the optional
                                HuggingFace `datasets` library (--live only).

Both paths feed the same per-dataset _build_* functions, so the prompt
construction and answer normalization are identical whether an item came from a
fixture or the real corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import EvalSample

_FIXTURE_DIR = Path(__file__).parent / "fixtures"

DATASETS = ("gsm8k", "mmlu", "humaneval", "mtbench", "agentic", "wrapper_tasks")

# wrapper_tasks.json "category" -> router task_type. legal_benign/medical_benign
# stay on their transformation task_type (extraction/summarization) so they do
# NOT trip the domain tier floor; legal_highstakes/medical_highstakes carry
# prompt text that trips classifier._LEGAL_SUBSTANTIVE_RE /
# _MEDICAL_SUBSTANTIVE_RE for real, so the *actual* classifier (not this map)
# assigns task_type "legal"/"medical" at routing time — this map only sets the
# EvalSample.task_type used for reporting/mock-quality lookup, matching what
# the real classifier is expected to produce.
_WRAPPER_TASK_TYPE = {
    "summarization": "summarization",
    "extraction": "extraction",
    "translation": "translation",
    "basic_coding": "code_generation",
    "distributed_systems_coding": "code_generation",
    "math_proof": "reasoning",
    "long_document": "summarization",
    "tool_calling": "function_calling",
    "legal_benign": "extraction",
    "legal_highstakes": "legal",
    "medical_benign": "extraction",
    "medical_highstakes": "medical",
}

# Filler paragraph repeated to build a genuinely long prompt for the
# long_document category, so required_capabilities=["long_document"] actually
# exercises the router's context-window filtering (routing_engine._capability_
# satisfied), not just a short prompt tagged with the capability.
_LONG_DOC_FILLER = (
    "Section {n}: This section restates the service's failover invariants — "
    "every region must be able to serve reads within 200ms of a primary "
    "outage, writes fail closed rather than risking a split-brain, and the "
    "on-call runbook must be re-verified quarterly against the current "
    "topology. No action is required for this section beyond acknowledging "
    "the invariant holds.\n\n"
)

# Answer-format instructions appended to objective prompts so extraction is
# reliable across providers. Applied identically to fixture and hub items.
_GSM8K_SUFFIX = (
    "\n\nSolve the problem step by step. On the final line, write the answer as "
    "'#### <number>' with no units or extra text."
)
_MMLU_SUFFIX = "\n\nRespond with only the single letter (A, B, C, or D) of the correct answer."
_HUMANEVAL_SUFFIX = (
    "\n\nComplete the function above. Return ONLY the full function definition as "
    "Python code in a single code block — no explanation."
)

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# MT-Bench category → router task_type (drives mock quality + classifier behavior).
_MTBENCH_TASK_TYPE = {
    "writing": "creative_writing",
    "roleplay": "creative_writing",
    "humanities": "analysis",
    "stem": "analysis",
    "reasoning": "analysis",
    "extraction": "extraction",
    "math": "reasoning",
    "coding": "code_generation",
}


# ── Per-dataset builders (shared by fixture + hub) ──────────────────────────────


def _extract_gsm8k_answer(answer_field: str) -> str:
    """Pull the canonical final number out of a GSM8K answer string."""
    tail = answer_field.split("####")[-1] if "####" in answer_field else answer_field
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return nums[-1].replace(",", "") if nums else tail.strip()


def _build_gsm8k(rec: dict[str, Any], idx: int) -> EvalSample:
    return EvalSample(
        id=f"gsm8k-{idx}",
        dataset="gsm8k",
        task_type="reasoning",
        grader="gsm8k",
        prompt=rec["question"].strip() + _GSM8K_SUFFIX,
        reference=_extract_gsm8k_answer(rec["answer"]),
    )


def _build_mmlu(rec: dict[str, Any], idx: int) -> EvalSample:
    choices = list(rec["choices"])
    answer = rec["answer"]
    letter = answer if isinstance(answer, str) else _LETTERS[int(answer)]
    lettered = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return EvalSample(
        id=f"mmlu-{idx}",
        dataset="mmlu",
        task_type="simple_qa",
        grader="mmlu",
        prompt=f"{rec['question'].strip()}\n\n{lettered}{_MMLU_SUFFIX}",
        reference=letter.strip().upper(),
        metadata={"choices": choices, "subject": rec.get("subject", "")},
    )


def _build_humaneval(rec: dict[str, Any], idx: int) -> EvalSample:
    header = rec["prompt"]
    return EvalSample(
        id=rec.get("task_id", f"humaneval-{idx}"),
        dataset="humaneval",
        task_type="code_generation",
        grader="humaneval",
        prompt=header + _HUMANEVAL_SUFFIX,
        reference=rec.get("canonical_solution"),
        metadata={
            "prompt_header": header,
            "test": rec["test"],
            "entry_point": rec["entry_point"],
        },
    )


def _build_mtbench(rec: dict[str, Any], idx: int) -> EvalSample:
    turns = rec["prompt"]
    first = turns[0] if isinstance(turns, list) else turns
    category = rec.get("category", "writing")
    return EvalSample(
        id=f"mtbench-{rec.get('prompt_id', idx)}",
        dataset="mtbench",
        task_type=_MTBENCH_TASK_TYPE.get(category, "analysis"),
        grader="llm_judge",
        prompt=first.strip(),
        reference=None,
        metadata={"category": category},
    )


def _build_agentic(rec: dict[str, Any], idx: int) -> EvalSample:
    """Item 6: agent-step-type eval cases — planning, tool selection,
    tool-result interpretation, reflection, final answer. Fixture-only (no
    hub source — this is a Flux-original set, not a public benchmark).

    The sample's task_type (a router task_type, e.g. "reasoning") drives mock
    quality simulation and model selection as usual; step_type (an agent-step
    label, e.g. "plan") is carried in metadata and threaded into the
    RoutingRequest by runner.py so routing itself is step-type-aware for
    these cases too (see config.STEP_TYPE_FLOORS).
    """
    step_type = rec["step_type"]
    if step_type == "tool_select":
        grader = "agentic_tool_select"
    elif step_type in ("budget_degradation", "budget_stop"):
        # Deterministic system check against the real RunBudget ladder —
        # no model call, no LLM judge. See graders.objective.grade_budget_ladder.
        grader = "budget_ladder"
    else:
        grader = "llm_judge"
    metadata: dict[str, Any] = {"step_type": step_type}
    if "tools" in rec:
        metadata["tools"] = rec["tools"]
    if "expected_tool" in rec:
        metadata["expected_tool"] = rec["expected_tool"]
    if grader == "budget_ladder":
        metadata["max_cost_usd"] = rec["max_cost_usd"]
        metadata["pre_spent_cost_usd"] = rec["pre_spent_cost_usd"]
        metadata["expected_result"] = rec["expected_result"]
    return EvalSample(
        id=f"agentic-{step_type}-{idx}",
        dataset="agentic",
        task_type=rec.get("task_type", "unknown"),
        grader=grader,
        prompt=rec["prompt"].strip(),
        reference=rec.get("expected_tool"),
        metadata=metadata,
    )


def _build_wrapper_task(rec: dict[str, Any], idx: int) -> EvalSample:
    """Ordinary wrapper-request categories not covered by the
    public benchmarks above — summarization, extraction, translation, basic
    vs. complex-distributed-systems coding, math proofs, long-document,
    wrapper-level tool calling, and benign vs. high-stakes legal/medical
    transformations. Fixture-only (Flux-original set, like "agentic")."""
    category = rec["category"]
    task_type = rec.get("task_type", _WRAPPER_TASK_TYPE.get(category, "analysis"))
    grader = "llm_judge"
    metadata: dict[str, Any] = {"category": category}
    if "domain" in rec:
        metadata["domain"] = rec["domain"]
    if "stakes" in rec:
        metadata["stakes"] = rec["stakes"]
    if "required_capabilities" in rec:
        metadata["required_capabilities"] = rec["required_capabilities"]

    prompt = rec["prompt"]
    reference = None
    if category == "extraction" and "required_keys" in rec:
        grader = "json_schema"
        metadata["required_keys"] = rec["required_keys"]
    elif category == "tool_calling" and "expected_tool" in rec:
        grader = "agentic_tool_select"
        metadata["tools"] = rec["tools"]
        metadata["expected_tool"] = rec["expected_tool"]
        reference = rec["expected_tool"]
    elif category == "long_document":
        filler = "".join(_LONG_DOC_FILLER.format(n=n) for n in range(1, 400))
        prompt = prompt.replace("{REPEATED_SECTION}", filler)

    return EvalSample(
        id=rec.get("id", f"wrapper-{category}-{idx}"),
        dataset="wrapper_tasks",
        task_type=task_type,
        grader=grader,
        prompt=prompt.strip(),
        reference=reference,
        metadata=metadata,
    )


_BUILDERS = {
    "gsm8k": _build_gsm8k,
    "mmlu": _build_mmlu,
    "humaneval": _build_humaneval,
    "mtbench": _build_mtbench,
    "agentic": _build_agentic,
    "wrapper_tasks": _build_wrapper_task,
}


# ── Loaders ─────────────────────────────────────────────────────────────────


def _load_fixture_records(name: str) -> list[dict[str, Any]]:
    path = _FIXTURE_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No fixture for dataset '{name}' at {path}")
    with path.open(encoding="utf-8") as fh:
        records: list[dict[str, Any]] = json.load(fh)
        return records


def _load_hub_records(name: str, n: int) -> list[dict[str, Any]]:
    """Download up to n raw records from the HuggingFace hub.

    Imported lazily so the core package never hard-depends on `datasets`.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:  # pragma: no cover - exercised only with --source hub
        raise RuntimeError(
            "The 'datasets' package is required for --source hub. "
            "Install it with:  pip install -e '.[evals]'"
        ) from exc

    # Pinned to each dataset's current `main` commit rather than a mutable
    # branch name, so a tampered or force-pushed upstream repo can't silently
    # swap benchmark content under us (CWE-494).
    if name == "gsm8k":
        ds = hf_load(
            "gsm8k", "main", split="test", revision="740312add88f781978c0658806c59bc2815b9866"
        )
    elif name == "mmlu":
        ds = hf_load(
            "cais/mmlu", "all", split="test", revision="c30699e8356da336a370243923dbaf21066bb9fe"
        )
    elif name == "humaneval":
        ds = hf_load(
            "openai_humaneval", split="test", revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
        )
    elif name == "mtbench":
        ds = hf_load(
            "HuggingFaceH4/mt_bench_prompts",
            split="train",
            revision="e3a795c5e9a82ee40611c416b8a7786c73198991",
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    return [dict(ds[i]) for i in range(min(n, len(ds)))]


def load_dataset(name: str, n: int = 50, source: str = "fixture") -> list[EvalSample]:
    """Return up to ``n`` normalized EvalSamples for ``name`` from ``source``."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {', '.join(DATASETS)}")

    if source == "fixture":
        records = _load_fixture_records(name)
    elif source == "hub":
        records = _load_hub_records(name, n)
    else:
        raise ValueError(f"Unknown source '{source}' (use 'fixture' or 'hub')")

    build = _BUILDERS[name]
    return [build(rec, i) for i, rec in enumerate(records[:n])]


def load_datasets(
    names: list[str], n: int = 50, source: str = "fixture"
) -> list[EvalSample]:
    """Load and concatenate several datasets."""
    samples: list[EvalSample] = []
    for name in names:
        samples.extend(load_dataset(name, n=n, source=source))
    return samples
