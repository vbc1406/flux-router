"""
schemas.py — dataclasses passed between the eval modules.

These are plain dataclasses (not pydantic) because they never cross a trust
boundary — they only move between eval modules in-process. The router's own
contracts (RoutingRequest, ModelOption, …) stay pydantic; we reuse those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Grader kinds. Each EvalSample names exactly one.
#   gsm8k             — extract final number, exact match against reference
#   mmlu              — parse chosen letter A–D, match against reference
#   humaneval         — execute completion against unit tests (pass@1)
#   llm_judge         — Claude grades the answer 1–10 against a rubric (open-ended)
#   agentic_tool_select — objective: does the completion name the one correct
#                         tool (sample.metadata["expected_tool"])? See the
#                         "agentic" dataset — this is the one agent-step type
#                         (tool_select) with a verifiable right answer; the
#                         other four (plan/tool_result_summarize/reflect/
#                         final_answer) are inherently open-ended and use
#                         llm_judge like mtbench. Also reused by the
#                         wrapper-level "tool_calling" category.
#   json_schema        — objective: does the completion parse as JSON and
#                         contain every key in sample.metadata["required_keys"]?
#                         Used by the "extraction" wrapper category.
#   budget_ladder       — deterministic, no model call: exercises the real
#                         router.run_budget.RunBudget ladder directly and
#                         checks it returns/raises what
#                         sample.metadata["expected_result"] says. Used by the
#                         budget_degradation/budget_stop agent steps.
GRADERS = (
    "gsm8k",
    "mmlu",
    "humaneval",
    "llm_judge",
    "agentic_tool_select",
    "json_schema",
    "budget_ladder",
)

# Strategy names understood by strategies.py.
#   flux/premium/cheapest/mid  — the routing engine and the synthetic baselines.
#   default_openai/anthropic/google — the flagship a company would "normally" call
#   per provider when it isn't routing (the ChatGPT / Claude / Gemini defaults).
STRATEGIES = (
    "flux",
    "premium",
    "cheapest",
    "mid",
    "default_openai",
    "default_anthropic",
    "default_google",
)


@dataclass
class EvalSample:
    """One labeled benchmark item, normalized across datasets."""

    id: str
    dataset: str  # "gsm8k" | "mmlu" | "humaneval" | "mtbench"
    task_type: str  # router task type: reasoning / simple_qa / code_generation / ...
    grader: str  # one of GRADERS
    prompt: str  # the full prompt sent to the model
    reference: str | None = None  # gold answer (number / letter / canonical), if any
    # grader-specific extras, e.g. HumanEval {"test": ..., "entry_point": ...}
    # or MMLU {"choices": [...]}.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    """A model's answer to one sample, plus the cost/latency accounting."""

    text: str
    model_id: str
    provider: str
    tier: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int
    simulated: bool = False
    # Mock mode only: the simulated quality the LLM judge should report for
    # open-ended samples (objective samples encode correctness in .text instead).
    # None in live mode — the real judge scores the real text.
    sim_quality: float | None = None


@dataclass
class GradedResult:
    """A graded (sample, strategy) pair — one row of the eval."""

    sample_id: str
    dataset: str
    task_type: str
    strategy: str
    model_id: str
    tier: str
    cost: float
    quality: float  # 0.0–1.0
    # True/False for objective graders; None for the LLM judge (graded, not pass/fail).
    correct: bool | None
    simulated: bool = False
    # ── per-question drill-down fields (for the --per-question report) ──────
    # The question text, a human-readable type label (dataset + subject/category),
    # the classifier's complexity score (0–1), and the chosen model's *rated*
    # benchmark quality for this task type (registry quality_ratings[task_type]).
    prompt: str = ""
    question_type: str = ""
    complexity: float = 0.0
    quality_rating: float = 0.0
    # Item 3: resolved agent step_type ("plan"/"tool_select"/.../"unknown")
    # for samples that carry one (currently only the "agentic" dataset via
    # sample.metadata["step_type"]) — "" for every non-agentic sample, not
    # "unknown", so report.py can tell "no step concept applies here" apart
    # from "step_type resolved to unknown". Used for the "quality by agent
    # step type" eval rollup — see report.py::_print_by_step_type().
    step_type: str = ""

    # ── Task 3 (live quality evaluation) per-sample recording fields ────────
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # Provider-call latency (Completion.latency_ms) — end-to-end for the
    # completion itself. None when the sample used no completion (budget_ladder).
    latency_ms: int | None = None
    # Wall-clock time of the routing decision alone (pick_model()), separate
    # from the provider call above. Only meaningful for strategy="flux" — the
    # other strategies are a lookup, not a routing decision.
    routing_latency_ms: float | None = None
    # "legal" / "medical" for samples tagged with a domain in wrapper_tasks.json
    # metadata; "" otherwise.
    domain: str = ""
    # True/False when `domain` is set and stakes="high" and strategy="flux":
    # did the chosen model's tier meet config.DOMAIN_TIER_FLOORS[domain]? None
    # when not applicable (benign samples, non-flux strategies, no domain).
    safety_escalated: bool | None = None
    # Mirrors `correct` but only populated for the grader it names, so
    # report.py can compute a rate without re-checking sample.grader.
    tool_call_valid: bool | None = None
    structured_output_valid: bool | None = None


@dataclass
class StrategyReport:
    """Aggregate metrics for one strategy across all graded samples."""

    strategy: str
    n: int
    total_cost: float
    mean_quality: float
    # dataset -> {"n": int, "cost": float, "quality": float}
    per_dataset: dict[str, dict[str, Any]] = field(default_factory=dict)
    # vs the premium baseline (filled in by report.py); None for premium itself.
    cost_savings_pct: float | None = None
    quality_retention_pct: float | None = None
    quality_drop: float | None = None
