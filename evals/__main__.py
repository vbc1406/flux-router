"""
CLI entrypoint for the routing_priority cost-vs-quality harness.

Usage:
  python -m evals                      # dry-run, mock mode, all 25 tasks
  python -m evals --mode live          # real provider + judge calls
  python -m evals --out results.csv    # custom output path

Mock mode (default) makes no network calls: it runs the real RoutingEngine
to see which model each routing_priority picks and what it would cost, then
simulates answer quality from that model's quality_ratings — no API keys
needed. Live mode makes real provider calls and a real Claude judge call for
rubric tasks; it needs the relevant *_API_KEY env vars set.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone

import structlog

from .harness import ROUTING_PRIORITIES, run_golden_set
from .judge import DEFAULT_JUDGE_MODEL
from .report import (
    build_summary,
    find_regressions,
    format_regressions,
    format_summary_table,
    write_results_csv,
    write_summary_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cost-vs-quality eval for routing_priority modes.")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument(
        "--priorities",
        nargs="+",
        default=ROUTING_PRIORITIES,
        help=f"routing_priority values to compare (default: {ROUTING_PRIORITIES})",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--out", default=None, help="results CSV path (default: evals/results/<timestamp>.csv)")
    parser.add_argument(
        "--summary-out", default=None, help="summary CSV path (default: evals/results/<timestamp>_summary.csv)"
    )
    return parser.parse_args()


def main() -> None:
    # The router logs a DEBUG line per classification — fine for library use,
    # too noisy for this CLI's stdout. Raise the floor to INFO for a clean run.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    args = _parse_args()
    results = asyncio.run(
        run_golden_set(mode=args.mode, routing_priorities=args.priorities, judge_model_id=args.judge_model)
    )

    if not results:
        print("No results produced — check router/models.json and golden_set.py constraints.")
        return

    summaries = build_summary(results)
    regressions = find_regressions(results)

    print(f"\nRan {len(results)} (task x routing_priority) pairs in '{args.mode}' mode.\n")
    print("Comparison:")
    print(format_summary_table(summaries))
    print("\nRegressions (routing_priority scored meaningfully lower than quality_max on the same task):")
    print(format_regressions(regressions))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or f"evals/results/{stamp}.csv"
    summary_out_path = args.summary_out or f"evals/results/{stamp}_summary.csv"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_out_path) or ".", exist_ok=True)
    write_results_csv(results, out_path)
    write_summary_csv(summaries, summary_out_path)
    print(f"\nWrote per-task results to {out_path}")
    print(f"Wrote summary to {summary_out_path}")


if __name__ == "__main__":
    main()
