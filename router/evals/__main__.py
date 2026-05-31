"""
CLI entry point — python -m router.evals

Examples:
    python -m router.evals                         # mock, offline, CI-safe
    python -m router.evals --datasets gsm8k,mmlu --n 100
    python -m router.evals --live --n 30 --allow-code-exec
    python -m router.evals --live --judge-model claude-opus-4-20250514 --md EVALS_RESULT.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .datasets import DATASETS
from .report import aggregate, print_report, to_markdown, write_json
from .runner import RunConfig, run_eval
from .schemas import STRATEGIES


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m router.evals", description=__doc__)
    p.add_argument("--datasets", type=_csv, default=list(DATASETS),
                   help=f"comma-separated subset of {','.join(DATASETS)}")
    p.add_argument("--strategies", type=_csv, default=["flux", "premium", "cheapest"],
                   help=f"comma-separated subset of {','.join(STRATEGIES)}")
    p.add_argument("--n", type=int, default=50, help="max samples per dataset")
    p.add_argument("--live", action="store_true",
                   help="make real provider calls (needs *_API_KEY env vars); implies --source hub")
    p.add_argument("--source", choices=["fixture", "hub"], default=None,
                   help="dataset source (default: fixture, or hub when --live)")
    p.add_argument("--judge-model", default="claude-opus-4-20250514",
                   help="model id used as the LLM judge for open-ended samples (live only)")
    p.add_argument("--allow-code-exec", action="store_true",
                   help="permit executing live model-generated code for HumanEval (off by default)")
    p.add_argument("--seed", default="flux-eval", help="seed for deterministic mock completions")
    p.add_argument("--cache-dir", default=".eval_cache", help="disk cache dir (live mode)")
    p.add_argument("--out", default="eval_results", help="dir to write the JSON snapshot")
    p.add_argument("--md", default=None, help="optional path to write a markdown results table")
    return p.parse_args(argv)


def _validate(args: argparse.Namespace) -> None:
    bad_ds = [d for d in args.datasets if d not in DATASETS]
    if bad_ds:
        sys.exit(f"Unknown dataset(s): {', '.join(bad_ds)}. Known: {', '.join(DATASETS)}")
    bad_st = [s for s in args.strategies if s not in STRATEGIES]
    if bad_st:
        sys.exit(f"Unknown strategy(ies): {', '.join(bad_st)}. Known: {', '.join(STRATEGIES)}")


async def _run(args: argparse.Namespace) -> int:
    mode = "live" if args.live else "mock"
    source = args.source or ("hub" if args.live else "fixture")
    config = RunConfig(
        datasets=args.datasets,
        strategies=args.strategies,
        n=args.n,
        mode=mode,
        source=source,
        judge_model=args.judge_model,
        allow_code_exec=args.allow_code_exec,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    out = await run_eval(config)
    if not out.results:
        sys.exit("No samples were graded — nothing to report.")
    reports = aggregate(out.results)
    print_report(out, reports)
    json_path = write_json(out, reports, args.out)
    print(f"Wrote snapshot: {json_path}")
    if args.md:
        Path(args.md).write_text(to_markdown(out, reports), encoding="utf-8")
        print(f"Wrote markdown: {args.md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _validate(args)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
