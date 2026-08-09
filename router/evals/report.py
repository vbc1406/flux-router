"""
report.py — turn GradedResults into the cost-vs-quality tradeoff report.

Produces three things:
  - a rich console report (overview + per-dataset breakdown + headline),
  - a machine-readable JSON snapshot under eval_results/ (tracked over time —
    this is the North Star artifact),
  - an optional markdown table (to_markdown) for pasting into docs.

The cost comparison assumes every strategy was graded on the same set of
samples (true in mock mode and in live mode, where skips depend on the sample,
not the strategy), so per-strategy totals line up apples-to-apples.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .runner import RunOutput
from .schemas import GradedResult, StrategyReport

console = Console()


def aggregate(results: list[GradedResult]) -> dict[str, StrategyReport]:
    """Group graded results by strategy and compute totals + vs-premium deltas."""
    by_strat: dict[str, list[GradedResult]] = defaultdict(list)
    for r in results:
        by_strat[r.strategy].append(r)

    reports: dict[str, StrategyReport] = {}
    for strat, rows in by_strat.items():
        ds_groups: dict[str, list[GradedResult]] = defaultdict(list)
        for r in rows:
            ds_groups[r.dataset].append(r)
        per_dataset = {
            ds: {
                "n": len(g),
                "cost": round(sum(r.cost for r in g), 6),
                "quality": round(mean(r.quality for r in g), 4),
            }
            for ds, g in ds_groups.items()
        }
        reports[strat] = StrategyReport(
            strategy=strat,
            n=len(rows),
            total_cost=round(sum(r.cost for r in rows), 6),
            mean_quality=round(mean(r.quality for r in rows), 4),
            per_dataset=per_dataset,
        )

    premium = reports.get("premium")
    if premium:
        for strat, rep in reports.items():
            if strat == "premium":
                continue
            rep.cost_savings_pct = (
                round((premium.total_cost - rep.total_cost) / premium.total_cost * 100, 1)
                if premium.total_cost > 0
                else 0.0
            )
            rep.quality_retention_pct = (
                round(rep.mean_quality / premium.mean_quality * 100, 1)
                if premium.mean_quality > 0
                else 0.0
            )
            rep.quality_drop = round(premium.mean_quality - rep.mean_quality, 4)
    return reports


# ── Console rendering ────────────────────────────────────────────────────────


def _print_overview(reports: dict[str, StrategyReport]) -> None:
    console.rule("[bold cyan]Strategy Overview (vs always-premium baseline)[/bold cyan]")
    tbl = Table(box=box.ROUNDED)
    tbl.add_column("Strategy", style="bold")
    tbl.add_column("N", justify="right")
    tbl.add_column("Total Cost", justify="right")
    tbl.add_column("Mean Quality", justify="right")
    tbl.add_column("Cost Savings", justify="right")
    tbl.add_column("Quality Retained", justify="right")
    tbl.add_column("Quality Drop", justify="right")

    # Stable, readable ordering.
    order = ["premium", "flux", "mid", "cheapest"]
    for strat in order + [s for s in reports if s not in order]:
        rep = reports.get(strat)
        if rep is None:
            continue
        savings = "—" if rep.cost_savings_pct is None else f"{rep.cost_savings_pct:.1f}%"
        retain = (
            "—" if rep.quality_retention_pct is None else f"{rep.quality_retention_pct:.1f}%"
        )
        drop = "—" if rep.quality_drop is None else f"{rep.quality_drop:+.3f}"
        style = "bold green" if strat == "flux" else ""
        tbl.add_row(
            f"[{style}]{strat}[/]" if style else strat,
            str(rep.n),
            f"${rep.total_cost:.5f}",
            f"{rep.mean_quality:.3f}",
            savings,
            retain,
            drop,
        )
    console.print(tbl)


def _print_per_dataset(reports: dict[str, StrategyReport]) -> None:
    console.rule("[bold cyan]Mean Quality by Dataset × Strategy[/bold cyan]")
    datasets = sorted({ds for rep in reports.values() for ds in rep.per_dataset})
    strategies = [s for s in ["premium", "flux", "mid", "cheapest"] if s in reports]
    strategies += [s for s in reports if s not in strategies]

    tbl = Table(box=box.SIMPLE)
    tbl.add_column("Dataset", style="bold")
    for s in strategies:
        tbl.add_column(s, justify="right")
    for ds in datasets:
        row = [ds]
        for s in strategies:
            cell = reports[s].per_dataset.get(ds)
            row.append(f"{cell['quality']:.3f}" if cell else "—")
        tbl.add_row(*row)
    console.print(tbl)


def _print_headline(reports: dict[str, StrategyReport], simulated: bool) -> None:
    flux = reports.get("flux")
    premium = reports.get("premium")
    if not flux or not premium:
        return
    msg = (
        f"[bold]Flux saves {flux.cost_savings_pct:.1f}% of cost vs always-premium[/bold] "
        f"while retaining [bold]{flux.quality_retention_pct:.1f}%[/bold] of its quality "
        f"(Δ {flux.quality_drop:+.3f} mean quality)."
    )
    if simulated:
        banner = "[yellow]SIMULATED — not real model quality (run --live for real numbers)[/yellow]"
        msg = f"{banner}\n\n{msg}"
    console.print(Panel(msg, border_style="green" if not simulated else "yellow", expand=False))


def print_report(out: RunOutput, reports: dict[str, StrategyReport]) -> None:
    simulated = any(r.simulated for r in out.results)
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Flux Router — Cost vs Quality Eval[/bold cyan]\n"
            f"[dim]mode={out.config.mode} · source={out.config.source} · "
            f"datasets={','.join(out.config.datasets)} · n={out.config.n}/dataset · "
            f"{out.n_samples} samples · {out.n_skipped} skipped[/dim]",
            border_style="cyan",
        )
    )
    console.print()
    _print_overview(reports)
    console.print()
    _print_per_dataset(reports)
    console.print()
    _print_headline(reports, simulated)
    console.print()


# ── Per-question drill-down ──────────────────────────────────────────────────

# Display order: flux first (the thing under test), then the provider defaults a
# company would normally call, then the synthetic baselines.
_STRATEGY_ORDER = [
    "flux",
    "default_openai",
    "default_anthropic",
    "default_google",
    "premium",
    "mid",
    "cheapest",
]


def _ordered_strategies(results: list[GradedResult]) -> list[str]:
    present = {r.strategy for r in results}
    ordered = [s for s in _STRATEGY_ORDER if s in present]
    return ordered + sorted(present - set(ordered))


def _complexity_band(c: float) -> str:
    """Mirror the classifier's clamps: low ≤ 0.40, high ≥ 0.70, else medium."""
    return "low" if c <= 0.40 else "high" if c >= 0.70 else "med"


def _by_sample(results: list[GradedResult]) -> tuple[list[str], dict[str, dict[str, GradedResult]]]:
    """Pivot results into sample_id → {strategy → GradedResult}, preserving order."""
    order: list[str] = []
    rows: dict[str, dict[str, GradedResult]] = defaultdict(dict)
    for r in results:
        if r.sample_id not in rows:
            order.append(r.sample_id)
        rows[r.sample_id][r.strategy] = r
    return order, rows


def per_question_payload(out: RunOutput) -> dict[str, Any]:
    """Machine-readable per-question rows + a by-question-type rollup.

    Bugfix (Item 6): this used to report each strategy's model's static
    catalog quality_rating here — a real number, but never a graded answer,
    even when the run itself was --live (which DOES invoke real routed
    models and grade real output; see runner.py::run_eval). That let a
    --per-question --live run silently mix a genuinely measured aggregate
    table with a per-question drill-down that was still catalog-rating-only,
    with no signal in the data to tell them apart. Now uses GradedResult.quality
    (the actual graded value in both modes — simulated-but-graded in mock,
    measured in live) and stamps an explicit "mode" field so a consumer of
    the JSON can't accidentally treat one as the other.
    """
    strategies = _ordered_strategies(out.results)
    order, rows = _by_sample(out.results)
    mode = "measured" if out.config.mode == "live" else "simulated"

    questions = []
    for sid in order:
        per_strat = rows[sid]
        any_r = next(iter(per_strat.values()))
        questions.append(
            {
                "sample_id": sid,
                "question": any_r.prompt,
                "question_type": any_r.question_type,
                "complexity": round(any_r.complexity, 3),
                "strategies": {
                    s: {
                        "model_id": per_strat[s].model_id,
                        "quality": per_strat[s].quality,
                        "cost": per_strat[s].cost,
                    }
                    for s in strategies
                    if s in per_strat
                },
            }
        )

    by_type_groups: dict[str, dict[str, list[GradedResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in out.results:
        by_type_groups[r.question_type][r.strategy].append(r)
    by_type = {
        qtype: {
            s: {
                "n": len(rs),
                "quality": round(mean(x.quality for x in rs), 4),
                "cost": round(sum(x.cost for x in rs), 6),
            }
            for s, rs in by_strat.items()
        }
        for qtype, by_strat in by_type_groups.items()
    }
    return {
        "mode": mode,
        "strategies": strategies,
        "questions": questions,
        "by_question_type": by_type,
    }


def _flux_vs_baselines(out: RunOutput) -> dict[str, dict[str, Any]]:
    """Per-strategy mean quality + total cost, with flux's deltas vs each.

    Bugfix (Item 6): same class of bug as per_question_payload — this used
    to average each strategy's static catalog quality_rating rather than the
    actual graded GradedResult.quality, so the per-question headline panel
    ("flux: mean rated quality ...") never reflected --live grading either.
    """
    strategies = _ordered_strategies(out.results)
    by_strat: dict[str, list[GradedResult]] = defaultdict(list)
    for r in out.results:
        by_strat[r.strategy].append(r)
    summary = {
        s: {
            "mean_quality": round(mean(x.quality for x in rs), 4),
            "total_cost": round(sum(x.cost for x in rs), 6),
        }
        for s, rs in by_strat.items()
        if s in strategies
    }
    flux = summary.get("flux")
    if flux:
        for s, row in summary.items():
            if s == "flux":
                continue
            row["flux_cost_savings_pct"] = (
                round((row["total_cost"] - flux["total_cost"]) / row["total_cost"] * 100, 1)
                if row["total_cost"] > 0
                else 0.0
            )
            row["flux_quality_retention_pct"] = (
                round(flux["mean_quality"] / row["mean_quality"] * 100, 1)
                if row["mean_quality"] > 0
                else 0.0
            )
    return summary


def _short_model(model_id: str) -> str:
    """Trim long date suffixes so model ids fit table cells."""
    return model_id.replace("-20250514", "").replace("-20251001", "")


def print_per_question(out: RunOutput) -> None:
    strategies = _ordered_strategies(out.results)
    order, rows = _by_sample(out.results)
    mode_label = (
        "[yellow]SIMULATED — quality is a graded mock completion, not a live "
        "answer (run --live for real numbers)[/yellow]"
        if out.config.mode != "live"
        else "[green]MEASURED — live-graded real completions[/green]"
    )

    console.rule("[bold cyan]Per-Question: flux vs provider defaults[/bold cyan]")
    console.print(mode_label)
    tbl = Table(box=box.SIMPLE, show_lines=True)
    tbl.add_column("Question", style="bold", max_width=42, overflow="fold")
    tbl.add_column("Type")
    tbl.add_column("Cplx", justify="center")
    for s in strategies:
        tbl.add_column(s, justify="right")

    for sid in order:
        per_strat = rows[sid]
        any_r = next(iter(per_strat.values()))
        q = any_r.prompt.replace("\n", " ").strip()
        q = (q[:80] + "…") if len(q) > 80 else q
        cells = [q, any_r.question_type, _complexity_band(any_r.complexity)]
        for s in strategies:
            r = per_strat.get(s)
            if r is None:
                cells.append("—")
                continue
            colour = "green" if s == "flux" else ""
            body = f"{_short_model(r.model_id)}\nq={r.quality:.2f} ${r.cost:.5f}"
            cells.append(f"[{colour}]{body}[/]" if colour else body)
        tbl.add_row(*cells)
    console.print(tbl)

    # Mean quality + total cost per question type.
    console.print()
    console.rule("[bold cyan]Mean quality / total cost by question type[/bold cyan]")
    payload = per_question_payload(out)
    rollup = Table(box=box.SIMPLE)
    rollup.add_column("Question Type", style="bold")
    for s in strategies:
        rollup.add_column(s, justify="right")
    for qtype in sorted(payload["by_question_type"]):
        cells = [qtype]
        for s in strategies:
            cell = payload["by_question_type"][qtype].get(s)
            cells.append(f"q={cell['quality']:.2f} ${cell['cost']:.5f}" if cell else "—")
        rollup.add_row(*cells)
    console.print(rollup)

    # Headline: flux vs each baseline overall.
    summary = _flux_vs_baselines(out)
    flux = summary.get("flux")
    if flux:
        lines = [
            f"[bold]flux[/bold]: mean quality {flux['mean_quality']:.3f} "
            f"at total cost ${flux['total_cost']:.5f}",
        ]
        for s in strategies:
            if s == "flux":
                continue
            row = summary[s]
            lines.append(
                f"vs [bold]{s}[/bold] ({_short_model(_PROVIDER_DEFAULT_IDS.get(s, s))}): "
                f"flux saves [bold]{row.get('flux_cost_savings_pct', 0.0):.1f}%[/bold] cost, "
                f"retains [bold]{row.get('flux_quality_retention_pct', 0.0):.1f}%[/bold] quality"
            )
        console.print()
        console.print(Panel("\n".join(lines), border_style="green", expand=False))
    console.print()


def per_question_to_markdown(out: RunOutput) -> str:
    payload = per_question_payload(out)
    strategies = payload["strategies"]
    mode_note = (
        "> **MEASURED** — live-graded real completions.\n"
        if payload["mode"] == "measured"
        else "> ⚠️ **SIMULATED** — quality is a graded mock completion, not a "
        "live answer. Run `--live` for real numbers.\n"
    )
    lines = ["## Per-question: flux vs provider defaults\n", mode_note]
    header = "| Question | Type | Cplx | " + " | ".join(strategies) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (3 + len(strategies)))
    order, rows = _by_sample(out.results)
    for sid in order:
        per_strat = rows[sid]
        any_r = next(iter(per_strat.values()))
        q = any_r.prompt.replace("\n", " ").strip()
        q = (q[:60] + "…") if len(q) > 60 else q
        cells = [q, any_r.question_type, _complexity_band(any_r.complexity)]
        for s in strategies:
            r = per_strat.get(s)
            cells.append(
                f"{_short_model(r.model_id)} q={r.quality:.2f} ${r.cost:.5f}" if r else "—"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# Reverse map for the headline (strategy → pinned model id), imported lazily to
# avoid a hard import cycle with strategies.py at module load.
try:  # pragma: no cover - trivial import wiring
    from .strategies import _PROVIDER_DEFAULTS as _PROVIDER_DEFAULT_IDS
except Exception:  # pragma: no cover
    _PROVIDER_DEFAULT_IDS = {}


# ── Persistence ──────────────────────────────────────────────────────────────


def _snapshot(
    out: RunOutput,
    reports: dict[str, StrategyReport],
    per_question: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": out.config.mode,
        "source": out.config.source,
        "simulated": any(r.simulated for r in out.results),
        "config": asdict(out.config),
        "n_samples": out.n_samples,
        "n_skipped": out.n_skipped,
        "strategies": {s: asdict(rep) for s, rep in reports.items()},
    }
    if per_question is not None:
        snap["per_question"] = per_question
    return snap


def write_json(
    out: RunOutput,
    reports: dict[str, StrategyReport],
    out_dir: str,
    per_question: dict[str, Any] | None = None,
) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(out_dir) / f"{stamp}.json"
    path.write_text(
        json.dumps(_snapshot(out, reports, per_question), indent=2), encoding="utf-8"
    )
    return path


def to_markdown(out: RunOutput, reports: dict[str, StrategyReport]) -> str:
    simulated = any(r.simulated for r in out.results)
    lines: list[str] = []
    if simulated:
        lines.append("> ⚠️ **SIMULATED** results (mock mode) — run `--live` for real numbers.\n")
    lines.append(
        f"_mode={out.config.mode}, source={out.config.source}, "
        f"{out.n_samples} samples, {out.n_skipped} skipped_\n"
    )
    lines.append(
        "| Strategy | N | Total Cost | Mean Quality | Cost Savings | "
        "Quality Retained | Quality Drop |"
    )
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    order = [s for s in ["premium", "flux", "mid", "cheapest"] if s in reports]
    order += [s for s in reports if s not in order]
    for s in order:
        rep = reports[s]
        savings = "—" if rep.cost_savings_pct is None else f"{rep.cost_savings_pct:.1f}%"
        retain = "—" if rep.quality_retention_pct is None else f"{rep.quality_retention_pct:.1f}%"
        drop = "—" if rep.quality_drop is None else f"{rep.quality_drop:+.3f}"
        lines.append(
            f"| {s} | {rep.n} | ${rep.total_cost:.5f} | {rep.mean_quality:.3f} | "
            f"{savings} | {retain} | {drop} |"
        )
    return "\n".join(lines) + "\n"
