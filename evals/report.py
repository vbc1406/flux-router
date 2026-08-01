"""
report.py — turn a list[TaskResult] into the comparison table, the
"what am I actually losing" regression list, and CSV output.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass

from .harness import TaskResult

# A task counts as a regression when the losing priority's quality_score is
# at least this many points (on the 1-5 scale) below quality_max's score on
# the SAME task. 1.0 point ~= one full rubric grade band — a difference big
# enough to matter, not scoring noise.
REGRESSION_THRESHOLD = 1.0


@dataclass
class PrioritySummary:
    routing_priority: str
    n_tasks: int
    total_cost_usd: float
    avg_quality_score: float  # mean of the 1-5 scale across all tasks
    cost_per_quality_point: float  # total_cost_usd / sum(quality_score)


@dataclass
class Regression:
    task_id: str
    step_type: str
    routing_priority: str  # the cheaper priority being compared
    priority_score: float
    quality_max_score: float
    delta: float  # quality_max_score - priority_score, always > 0 for a flagged row


def build_summary(results: list[TaskResult]) -> list[PrioritySummary]:
    by_priority: dict[str, list[TaskResult]] = {}
    for r in results:
        by_priority.setdefault(r.routing_priority, []).append(r)

    summaries: list[PrioritySummary] = []
    for priority, rows in by_priority.items():
        total_cost = sum(r.cost_usd for r in rows)
        total_quality = sum(r.quality_score for r in rows)
        n = len(rows)
        summaries.append(
            PrioritySummary(
                routing_priority=priority,
                n_tasks=n,
                total_cost_usd=round(total_cost, 6),
                avg_quality_score=round(total_quality / n, 3) if n else 0.0,
                cost_per_quality_point=round(total_cost / total_quality, 6)
                if total_quality > 0
                else 0.0,
            )
        )
    return summaries


def find_regressions(
    results: list[TaskResult], threshold: float = REGRESSION_THRESHOLD
) -> list[Regression]:
    """Flag tasks where cascade or cost-optimized scored meaningfully lower
    than quality_max on the SAME task — the "what am I actually losing"
    signal, not just an aggregate quality gap."""
    by_task: dict[str, dict[str, TaskResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, {})[r.routing_priority] = r

    regressions: list[Regression] = []
    for task_id, by_priority in by_task.items():
        qm = by_priority.get("quality_max")
        if qm is None:
            continue
        for priority, r in by_priority.items():
            if priority == "quality_max":
                continue
            delta = qm.quality_score - r.quality_score
            if delta >= threshold:
                regressions.append(
                    Regression(
                        task_id=task_id,
                        step_type=r.step_type,
                        routing_priority=priority,
                        priority_score=r.quality_score,
                        quality_max_score=qm.quality_score,
                        delta=round(delta, 3),
                    )
                )
    regressions.sort(key=lambda x: x.delta, reverse=True)
    return regressions


def format_summary_table(summaries: list[PrioritySummary]) -> str:
    header = f"{'routing mode':<16} {'total cost':>12} {'avg quality':>12} {'cost / quality pt':>18}"
    lines = [header, "-" * len(header)]
    for s in sorted(summaries, key=lambda x: x.routing_priority):
        lines.append(
            f"{s.routing_priority:<16} "
            f"${s.total_cost_usd:>10.4f} "
            f"{s.avg_quality_score:>12.2f} "
            f"${s.cost_per_quality_point:>16.4f}"
        )
    return "\n".join(lines)


def format_regressions(regressions: list[Regression]) -> str:
    if not regressions:
        return "No regressions >= threshold — cascade/cost-optimized matched quality_max everywhere."
    header = f"{'task':<10} {'step_type':<22} {'mode':<14} {'mode score':>10} {'quality_max':>12} {'delta':>7}"
    lines = [header, "-" * len(header)]
    for r in regressions:
        lines.append(
            f"{r.task_id:<10} {r.step_type:<22} {r.routing_priority:<14} "
            f"{r.priority_score:>10.2f} {r.quality_max_score:>12.2f} {r.delta:>7.2f}"
        )
    return "\n".join(lines)


def write_results_csv(results: list[TaskResult], path: str) -> None:
    fieldnames = [
        "routing_priority",
        "task_id",
        "step_type",
        "task_type",
        "grader",
        "model_id",
        "provider",
        "tier",
        "cost_usd",
        "quality_score",
        "passed",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def write_summary_csv(summaries: list[PrioritySummary], path: str) -> None:
    fieldnames = ["routing_priority", "n_tasks", "total_cost_usd", "avg_quality_score", "cost_per_quality_point"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
