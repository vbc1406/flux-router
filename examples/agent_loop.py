"""
agent_loop.py — a budgeted ReAct-style agent loop using Flux's run-scoped
budget enforcement and step-type routing.

The loop alternates "plan" and "final_answer" steps against a small set of
subtasks, all under one run_id opened via flux.start_run(). Each call passes
step_type= explicitly rather than relying on inference from prompt text
alone — the same vocabulary the HTTP proxy's X-Flux-Step-Type header accepts
(see README's "Step types" section): plan, tool_select, tool_result_summarize,
reflect, extract, format, final_answer, unknown. STEP_TYPE_FLOORS
(router/config.py) gives "plan" and "final_answer" a quality floor so a
budget-constrained run can't degrade a planning step down to a model too
weak to plan with, even while cost-optimized routing is active for the rest
of the run.

As the run's cumulative cost approaches the budget, Flux automatically
forces cost-optimized routing for the remaining steps (the degradation
ladder); if the run still exceeds its budget, flux.complete() raises
RunBudgetExceeded with a structured summary (steps taken, total cost,
per-step model/cost breakdown) so the loop can report partial progress
instead of just dying.

This is the answer to "what happens today when an agent run costs 10x what
you expected?" — with plain per-request routing, nothing; with a run budget,
the run degrades, warns, then stops itself.

Run (needs at least one provider key set, e.g. OPENAI_API_KEY):
    python examples/agent_loop.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from router import RunBudgetExceeded
from router.errors import FluxAPIError
from router.flux import make_flux

# Toy task: the agent works through a short list of subtasks, one per loop
# iteration, each broken into a "plan" step (decide the approach) followed
# by a "final_answer" step (produce the deliverable for that subtask).
_SUBTASKS = [
    "Summarize the key risks of the current plan.",
    "List three follow-up questions for the user.",
    "Draft a one-line status update.",
    "Identify the next blocking dependency.",
    "Propose a fallback if the primary approach fails.",
]

# (phase, step_type, prompt template) — step_type is passed straight to
# flux.complete() as request.step_type, which wins over inference (see
# classifier.py::_infer_step_type) and gets STEP_TYPE_FLOORS applied.
_PHASES = [
    ("plan", "plan", "Plan how you will address this subtask: {subtask}"),
    ("final_answer", "final_answer", "Now give your final answer for: {subtask}"),
]


async def main() -> None:
    flux = make_flux()

    # A deliberately small budget so the ladder is visible in a short demo run
    # ($0.05 total, capped at 20 steps). Tune both up for a real workload.
    with flux.start_run(max_cost_usd=0.05, max_steps=20) as run_id:
        print(f"Starting run {run_id} (max_cost_usd=0.05, max_steps=20)\n")

        step = 0
        for subtask in _SUBTASKS:
            for phase, step_type, template in _PHASES:
                step += 1
                prompt = template.format(subtask=subtask)
                try:
                    resp = await flux.complete(
                        prompt,
                        run_id=run_id,
                        user_id="agent-loop-demo",
                        step_type=step_type,
                    )
                except RunBudgetExceeded as exc:
                    summary = exc.summary
                    print(f"\n--- Run budget exceeded after {summary['steps_taken']} step(s) ---")
                    print(f"Total spent: ${summary['total_cost_usd']:.4f}")
                    print(f"Reason: {summary['exceeded_reason']}")
                    print("Per-step breakdown:")
                    for i, s in enumerate(summary["step_breakdown"], start=1):
                        print(f"  {i:2d}. {s['model_id']:<30} ${s['cost_usd']:.4f}")
                    print("\nReturning partial progress instead of crashing.")
                    return
                except FluxAPIError as exc:
                    print(f"[error] step {step} failed: {exc}")
                    continue

                d = resp.decision
                marker = f" ({d.budget_state})" if d.budget_state != "ok" else ""
                print(f"step {step:2d} [{phase:5s}] -> {resp.model.model_id}{marker}")
                if d.budget_warning:
                    print(f"           ! {d.budget_warning}")

        print("\nAll subtasks completed within budget.")


if __name__ == "__main__":
    asyncio.run(main())
