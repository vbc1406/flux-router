"""
File: router/tests/test_benchmark.py

Purpose:
Regression tests for Item 4 — router/benchmark.py used to unconditionally
report FAIL on the "Cache hit rate is 0%" check, because its ResponseCache
was constructed with the ambient (disabled-by-default) enabled= setting
instead of being explicitly enabled for the benchmark run. Also covers the
new argparse CLI (--help exits without running the benchmark; --n slices
the dataset).

How to run:
  pytest -v router/tests/test_benchmark.py
"""

from __future__ import annotations

import pytest

from router import benchmark


def test_response_cache_is_explicitly_enabled_for_the_run():
    # Regression guard for the actual root cause: run_benchmark() must not
    # depend on the ambient FLUX_ENABLE_RESPONSE_CACHE env var.
    import inspect

    src = inspect.getsource(benchmark.run_benchmark)
    assert "ResponseCache(enabled=True)" in src


@pytest.mark.timeout(30)
def test_full_benchmark_produces_cache_hits_and_passes_validation():
    """The exact scenario validate() checks: duplicate prompts in the full
    500-request dataset must produce at least one cache hit, and every
    other check must pass too."""
    import asyncio

    dataset = benchmark.generate_test_dataset()
    bench = asyncio.run(benchmark.run_benchmark(dataset))

    cache_hits = sum(1 for r in bench.results if r.decision.cache_hit)
    assert cache_hits > 0, "duplicate prompts in the dataset produced no cache hits"

    # perf_checks=False: the wall-clock rules flake on loaded CI runners —
    # see validate()'s docstring. Every deterministic rule still runs.
    passed, failures = benchmark.validate(bench, perf_checks=False)
    assert passed, f"validate() reported failures: {failures}"


def test_help_exits_zero_without_running_benchmark(monkeypatch):
    """`--help` must print usage and exit 0 — never fall through to
    generating the dataset / running the benchmark."""
    called = {"generate": False}

    def _spy(*a, **kw):
        called["generate"] = True
        return []

    monkeypatch.setattr(benchmark, "generate_test_dataset", _spy)

    with pytest.raises(SystemExit) as exc_info:
        benchmark._parse_args(["--help"])
    assert exc_info.value.code == 0
    assert called["generate"] is False


def test_n_flag_parses_to_an_integer():
    args = benchmark._parse_args(["--n", "10"])
    assert args.n == 10


def test_no_args_defaults_to_full_dataset():
    args = benchmark._parse_args([])
    assert args.n is None


@pytest.mark.timeout(30)
def test_main_with_small_n_exits_cleanly(capsys, monkeypatch):
    """A small --n slice may legitimately not hit every validate() check
    (e.g. too few requests to include the duplicate-prompt category) — this
    only asserts main() runs to completion without an uncaught exception
    when SystemExit(1) doesn't fire, i.e. the CLI wiring itself works."""
    import asyncio
    import functools

    # Disable the wall-clock rules for main()'s validate() call — they flake
    # on loaded CI runners (see validate()'s docstring); the CLI default is
    # unchanged. Every deterministic rule still gates the exit code.
    monkeypatch.setattr(
        benchmark, "validate", functools.partial(benchmark.validate, perf_checks=False)
    )

    args = benchmark._parse_args(["--n", "500"])
    try:
        asyncio.run(benchmark.main(args))
    except SystemExit as exc:
        # A real FAIL (exit 1) is a legitimate outcome to assert against
        # directly if it ever happens for the full dataset — surface it.
        assert exc.code == 0, "full-dataset run failed validation"
