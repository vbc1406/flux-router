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


# ── Persistence ──────────────────────────────────────────────────────────────


def _snapshot(out: RunOutput, reports: dict[str, StrategyReport]) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": out.config.mode,
        "source": out.config.source,
        "simulated": any(r.simulated for r in out.results),
        "config": asdict(out.config),
        "n_samples": out.n_samples,
        "n_skipped": out.n_skipped,
        "strategies": {s: asdict(rep) for s, rep in reports.items()},
    }


def write_json(out: RunOutput, reports: dict[str, StrategyReport], out_dir: str) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(out_dir) / f"{stamp}.json"
    path.write_text(json.dumps(_snapshot(out, reports), indent=2), encoding="utf-8")
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
