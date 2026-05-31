"""
Tests for the cost-vs-quality eval harness (router/evals).

Everything here runs fully offline: fixtures only, simulated completions, and
the real graders (HumanEval executes the bundled canonical/broken code in a
subprocess — no network, no provider keys, no spend).
"""

from __future__ import annotations

import asyncio

from router.evals.completions import get_completion
from router.evals.datasets import DATASETS, load_dataset, load_datasets
from router.evals.graders import grade
from router.evals.graders.llm_judge import _parse_score
from router.evals.report import aggregate
from router.evals.runner import RunConfig, run_eval
from router.evals.schemas import Completion, EvalSample, GradedResult
from router.model_registry import ModelRegistry


def _run(coro):
    return asyncio.run(coro)


def _completion(text: str, *, simulated: bool = True, sim_quality=None) -> Completion:
    return Completion(
        text=text,
        model_id="m",
        provider="p",
        tier="mid",
        input_tokens=1,
        output_tokens=1,
        cost=0.0,
        latency_ms=1,
        simulated=simulated,
        sim_quality=sim_quality,
    )


# ── Dataset normalization ──────────────────────────────────────────────────


class TestDatasets:
    def test_all_fixtures_load(self):
        for name in DATASETS:
            samples = load_dataset(name, n=50)
            assert samples, f"{name} fixture produced no samples"
            assert all(isinstance(s, EvalSample) for s in samples)

    def test_gsm8k_reference_is_final_number(self):
        s = load_dataset("gsm8k", n=1)[0]
        assert s.dataset == "gsm8k"
        assert s.grader == "gsm8k"
        assert s.reference == "72"  # Natalia clips problem

    def test_mmlu_builds_letters_and_reference(self):
        s = load_dataset("mmlu", n=1)[0]
        assert s.reference in {"A", "B", "C", "D"}
        assert "A." in s.prompt and "B." in s.prompt
        assert s.metadata["choices"]

    def test_humaneval_carries_test_and_entry_point(self):
        s = load_dataset("humaneval", n=1)[0]
        assert s.metadata["entry_point"]
        assert "def check(candidate)" in s.metadata["test"]

    def test_mtbench_is_judge_graded(self):
        s = load_dataset("mtbench", n=1)[0]
        assert s.grader == "llm_judge"
        assert s.reference is None

    def test_load_datasets_concatenates(self):
        samples = load_datasets(list(DATASETS), n=50)
        assert len(samples) == sum(len(load_dataset(n, n=50)) for n in DATASETS)


# ── Objective graders ──────────────────────────────────────────────────────


class TestObjectiveGraders:
    def test_gsm8k_correct_and_wrong(self):
        s = load_dataset("gsm8k", n=1)[0]  # reference "72"
        q_ok, ok = _run(grade(s, _completion("the answer is 72\n#### 72")))
        q_no, no = _run(grade(s, _completion("clearly the result is 71\n#### 71")))
        assert (q_ok, ok) == (1.0, True)
        assert (q_no, no) == (0.0, False)

    def test_mmlu_letter_match(self):
        s = load_dataset("mmlu", n=1)[0]
        ref = s.reference
        wrong = "A" if ref != "A" else "B"
        assert _run(grade(s, _completion(ref))) == (1.0, True)
        assert _run(grade(s, _completion(wrong))) == (0.0, False)

    def test_humaneval_pass_and_fail_via_subprocess(self):
        s = load_dataset("humaneval", n=1)[0]  # add(a, b)
        header = s.metadata["prompt_header"]
        good = _completion(header + s.reference)  # canonical body
        bad = _completion(header + "    return 0\n")
        # simulated=True ⇒ exec is permitted without --allow-code-exec.
        assert _run(grade(s, good)) == (1.0, True)
        assert _run(grade(s, bad)) == (0.0, False)

    def test_humaneval_skipped_when_exec_not_allowed(self):
        s = load_dataset("humaneval", n=1)[0]
        live_like = _completion(s.metadata["prompt_header"] + s.reference, simulated=False)
        q, correct = _run(grade(s, live_like, allow_code_exec=False))
        assert q is None and correct is None


# ── LLM judge (mock path) ──────────────────────────────────────────────────


class TestJudge:
    def test_mock_judge_uses_sim_quality(self):
        s = load_dataset("mtbench", n=1)[0]
        q, correct = _run(grade(s, _completion("a fine answer", sim_quality=0.83)))
        assert q == 0.83 and correct is None

    def test_judge_skipped_without_judge_or_simquality(self):
        s = load_dataset("mtbench", n=1)[0]
        q, correct = _run(grade(s, _completion("x", simulated=False, sim_quality=None)))
        assert q is None and correct is None

    def test_parse_score_normalizes(self):
        assert _parse_score("8") == 0.8
        assert _parse_score("The rating is 10/10") == 1.0
        assert _parse_score("no number here") == 0.0


# ── Mock completion determinism ────────────────────────────────────────────


class TestMockCompletions:
    def test_same_seed_is_deterministic(self):
        s = load_dataset("gsm8k", n=1)[0]
        model = ModelRegistry().most_expensive_model()
        c1 = _run(get_completion(model, s, mode="mock", seed="x"))
        c2 = _run(get_completion(model, s, mode="mock", seed="x"))
        assert c1.text == c2.text and c1.cost == c2.cost

    def test_cost_scales_with_price(self):
        s = load_dataset("gsm8k", n=1)[0]
        reg = ModelRegistry()
        premium = reg.most_expensive_model()
        cheapest = min(
            reg.all_available_models(),
            key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output,
        )
        cp = _run(get_completion(premium, s, mode="mock"))
        cc = _run(get_completion(cheapest, s, mode="mock"))
        assert cp.cost >= cc.cost


# ── Report math ────────────────────────────────────────────────────────────


def _gr(strategy, dataset, quality, cost) -> GradedResult:
    return GradedResult(
        sample_id="s",
        dataset=dataset,
        task_type="t",
        strategy=strategy,
        model_id="m",
        tier="mid",
        cost=cost,
        quality=quality,
        correct=None,
        simulated=True,
    )


class TestReportMath:
    def test_savings_and_retention(self):
        results = [
            _gr("premium", "gsm8k", 1.0, 1.0),
            _gr("flux", "gsm8k", 0.8, 0.2),
        ]
        reports = aggregate(results)
        flux = reports["flux"]
        assert flux.cost_savings_pct == 80.0  # (1.0 - 0.2) / 1.0
        assert flux.quality_retention_pct == 80.0  # 0.8 / 1.0
        assert flux.quality_drop == 0.2
        assert reports["premium"].cost_savings_pct is None

    def test_per_dataset_breakdown(self):
        results = [
            _gr("flux", "gsm8k", 1.0, 0.1),
            _gr("flux", "mmlu", 0.0, 0.1),
        ]
        rep = aggregate(results)["flux"]
        assert rep.mean_quality == 0.5
        assert rep.per_dataset["gsm8k"]["quality"] == 1.0
        assert rep.per_dataset["mmlu"]["quality"] == 0.0


# ── End-to-end (mock, offline) ─────────────────────────────────────────────


class TestEndToEnd:
    def test_mock_run_produces_report(self):
        config = RunConfig(
            datasets=["gsm8k", "mmlu", "humaneval", "mtbench"],
            strategies=["flux", "premium", "cheapest"],
            n=50,
            mode="mock",
            source="fixture",
            cache_dir=None,
        )
        out = _run(run_eval(config))
        assert out.results
        assert out.n_skipped == 0  # nothing should skip in mock mode
        reports = aggregate(out.results)
        # Premium is the cost ceiling; cheapest is the floor.
        assert reports["premium"].total_cost >= reports["flux"].total_cost
        assert reports["premium"].total_cost >= reports["cheapest"].total_cost

    def test_run_is_reproducible(self):
        config = RunConfig(
            datasets=["gsm8k"], strategies=["flux", "premium"], n=50,
            mode="mock", source="fixture", cache_dir=None,
        )
        a = aggregate(_run(run_eval(config)).results)["flux"]
        b = aggregate(_run(run_eval(config)).results)["flux"]
        assert a.mean_quality == b.mean_quality and a.total_cost == b.total_cost
