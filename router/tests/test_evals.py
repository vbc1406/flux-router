"""
Tests for the cost-vs-quality eval harness (router/evals).

Everything here runs fully offline: fixtures only and simulated completions.
HumanEval remains ungraded because the process provides no code sandbox.
"""

from __future__ import annotations

import asyncio

import pytest

from router.evals import __main__ as eval_cli
from router.evals.completions import get_completion
from router.evals.datasets import DATASETS, load_dataset, load_datasets
from router.evals.graders import grade
from router.evals.graders.llm_judge import _parse_score
from router.evals.report import aggregate, per_question_payload
from router.evals.runner import RunConfig, run_eval
from router.evals.schemas import Completion, EvalSample, GradedResult
from router.evals.strategies import _PROVIDER_DEFAULTS, pick_model
from router.flux import make_flux
from router.model_registry import ModelRegistry
from router.schemas import RoutingRequest


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

    def test_agentic_covers_all_seven_step_types(self):
        samples = load_dataset("agentic", n=50)
        step_types = {s.metadata["step_type"] for s in samples}
        assert step_types == {
            "plan",
            "tool_select",
            "tool_result_summarize",
            "reflect",
            "final_answer",
            "budget_degradation",
            "budget_stop",
        }

    def test_agentic_tool_select_is_objectively_graded(self):
        samples = load_dataset("agentic", n=50)
        tool_select = next(s for s in samples if s.metadata["step_type"] == "tool_select")
        assert tool_select.grader == "agentic_tool_select"
        assert tool_select.reference == tool_select.metadata["expected_tool"]

    def test_agentic_budget_steps_are_deterministically_graded(self):
        samples = load_dataset("agentic", n=50)
        for step in ("budget_degradation", "budget_stop"):
            s = next(sm for sm in samples if sm.metadata["step_type"] == step)
            assert s.grader == "budget_ladder"
            assert "max_cost_usd" in s.metadata
            assert "expected_result" in s.metadata

    def test_agentic_other_step_types_are_judge_graded(self):
        samples = load_dataset("agentic", n=50)
        for s in samples:
            if s.metadata["step_type"] not in ("tool_select", "budget_degradation", "budget_stop"):
                assert s.grader == "llm_judge"

    def test_wrapper_tasks_cover_required_categories(self):
        samples = load_dataset("wrapper_tasks", n=50)
        categories = {s.metadata["category"] for s in samples}
        assert categories == {
            "summarization",
            "extraction",
            "translation",
            "basic_coding",
            "distributed_systems_coding",
            "math_proof",
            "long_document",
            "tool_calling",
            "legal_benign",
            "legal_highstakes",
            "medical_benign",
            "medical_highstakes",
        }

    def test_wrapper_tasks_extraction_uses_json_schema_grader(self):
        samples = load_dataset("wrapper_tasks", n=50)
        s = next(sm for sm in samples if sm.metadata["category"] == "extraction")
        assert s.grader == "json_schema"
        assert s.metadata["required_keys"]

    def test_wrapper_tasks_high_stakes_trips_classifier_domain_regex(self):
        """legal_highstakes/medical_highstakes prompts must actually match the
        classifier's substantive regexes — otherwise the domain tier floor
        never engages and the "high-stakes" label would be a no-op."""
        from router.classifier import _LEGAL_SUBSTANTIVE_RE, _MEDICAL_SUBSTANTIVE_RE

        samples = load_dataset("wrapper_tasks", n=50)
        for s in samples:
            if s.metadata.get("domain") == "legal" and s.metadata.get("stakes") == "high":
                assert _LEGAL_SUBSTANTIVE_RE.search(s.prompt), s.prompt
            if s.metadata.get("domain") == "medical" and s.metadata.get("stakes") == "high":
                assert _MEDICAL_SUBSTANTIVE_RE.search(s.prompt), s.prompt

    def test_wrapper_tasks_benign_legal_medical_do_not_trip_domain_regex(self):
        from router.classifier import _LEGAL_SUBSTANTIVE_RE, _MEDICAL_SUBSTANTIVE_RE

        samples = load_dataset("wrapper_tasks", n=50)
        for s in samples:
            if s.metadata.get("domain") == "legal" and s.metadata.get("stakes") == "benign":
                assert not _LEGAL_SUBSTANTIVE_RE.search(s.prompt), s.prompt
            if s.metadata.get("domain") == "medical" and s.metadata.get("stakes") == "benign":
                assert not _MEDICAL_SUBSTANTIVE_RE.search(s.prompt), s.prompt


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

    def test_humaneval_simulated_completion_is_not_executed(self):
        s = load_dataset("humaneval", n=1)[0]  # add(a, b)
        header = s.metadata["prompt_header"]
        good = _completion(header + s.reference)  # canonical body
        assert _run(grade(s, good)) == (None, None)

    def test_humaneval_explicit_exec_request_fails_clearly(self):
        s = load_dataset("humaneval", n=1)[0]
        completion = _completion(s.metadata["prompt_header"] + s.reference, simulated=False)
        with pytest.raises(RuntimeError, match="no security sandbox"):
            _run(grade(s, completion, allow_code_exec=True))

    def test_agentic_tool_select_correct(self):
        samples = load_dataset("agentic", n=50)
        s = next(x for x in samples if x.metadata["step_type"] == "tool_select")
        q, ok = _run(grade(s, _completion(s.metadata["expected_tool"])))
        assert (q, ok) == (1.0, True)

    def test_agentic_tool_select_wrong_tool(self):
        samples = load_dataset("agentic", n=50)
        s = next(x for x in samples if x.metadata["step_type"] == "tool_select")
        wrong = next(
            t["function"]["name"]
            for t in s.metadata["tools"]
            if t["function"]["name"] != s.metadata["expected_tool"]
        )
        q, ok = _run(grade(s, _completion(wrong)))
        assert (q, ok) == (0.0, False)

    def test_agentic_tool_select_ignores_prose_around_the_name(self):
        """A real model rarely answers with the bare identifier alone —
        this must still match through natural-language wrapping."""
        samples = load_dataset("agentic", n=50)
        s = next(x for x in samples if x.metadata["step_type"] == "tool_select")
        text = f"I would call the `{s.metadata['expected_tool']}` function."
        q, ok = _run(grade(s, _completion(text)))
        assert (q, ok) == (1.0, True)

    def test_removed_code_exec_cli_flag_fails_clearly(self):
        args = eval_cli._parse_args(["--allow-code-exec"])
        eval_cli._resolve_strategies(args)
        with pytest.raises(SystemExit, match="no security sandbox"):
            eval_cli._validate(args)

    def test_humaneval_skipped_when_exec_not_allowed(self):
        s = load_dataset("humaneval", n=1)[0]
        live_like = _completion(s.metadata["prompt_header"] + s.reference, simulated=False)
        q, correct = _run(grade(s, live_like, allow_code_exec=False))
        assert q is None and correct is None

    def test_json_schema_grader_accepts_valid_json_with_required_keys(self):
        s = load_dataset("wrapper_tasks", n=50)
        s = next(x for x in s if x.metadata["category"] == "extraction")
        import json as _json

        body = _json.dumps(dict.fromkeys(s.metadata["required_keys"], "x"))
        assert _run(grade(s, _completion(body))) == (1.0, True)

    def test_json_schema_grader_rejects_missing_key(self):
        s = load_dataset("wrapper_tasks", n=50)
        s = next(x for x in s if x.metadata["category"] == "extraction")
        import json as _json

        body = _json.dumps({s.metadata["required_keys"][0]: "x"})
        assert _run(grade(s, _completion(body))) == (0.0, False)

    def test_json_schema_grader_rejects_non_json(self):
        s = load_dataset("wrapper_tasks", n=50)
        s = next(x for x in s if x.metadata["category"] == "extraction")
        assert _run(grade(s, _completion("not json at all"))) == (0.0, False)

    def test_json_schema_grader_unwraps_markdown_fence(self):
        s = load_dataset("wrapper_tasks", n=50)
        s = next(x for x in s if x.metadata["category"] == "extraction")
        import json as _json

        body = _json.dumps(dict.fromkeys(s.metadata["required_keys"], "x"))
        fenced = f"Here you go:\n```json\n{body}\n```"
        assert _run(grade(s, _completion(fenced))) == (1.0, True)

    def test_budget_ladder_degradation_case(self):
        samples = load_dataset("agentic", n=50)
        s = next(x for x in samples if x.metadata["step_type"] == "budget_degradation")
        q, correct = _run(grade(s, _completion("")))
        assert (q, correct) == (1.0, True)

    def test_budget_ladder_stop_case(self):
        samples = load_dataset("agentic", n=50)
        s = next(x for x in samples if x.metadata["step_type"] == "budget_stop")
        q, correct = _run(grade(s, _completion("")))
        assert (q, correct) == (1.0, True)

    def test_budget_ladder_detects_wrong_expected_result(self):
        """If the ladder's real behavior ever regressed, this grader must
        fail loudly rather than always reporting success."""
        from router.evals.schemas import EvalSample

        bad = EvalSample(
            id="budget-bad",
            dataset="agentic",
            task_type="budget",
            grader="budget_ladder",
            prompt="",
            metadata={
                "max_cost_usd": 1.0,
                "pre_spent_cost_usd": 0.75,
                "expected_result": "exceeded",  # wrong: 0.75/1.0 is "degraded", not "exceeded"
            },
        )
        q, correct = _run(grade(bad, _completion("")))
        assert (q, correct) == (0.0, False)


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

    def test_judge_score_extracts_text_from_provider_result(self, monkeypatch):
        """Regression test: Judge.score() used to pass the whole
        ProviderResult call_provider() returns to _parse_score() instead of
        .text — _parse_score calls .strip() on it, which ProviderResult
        doesn't have. A live judge call would have raised AttributeError.
        No test exercised it before (needs a real key) — mocked here."""
        from router.evals.graders.llm_judge import Judge
        from router.provider_caller import ProviderResult

        async def fake_call_provider(model, request, api_key):
            return ProviderResult(
                text="7",
                input_tokens=10,
                output_tokens=1,
                usage_source="provider",
            )

        monkeypatch.setattr("router.provider_caller.call_provider", fake_call_provider)
        sample = load_dataset("mtbench", n=1)[0]
        model = ModelRegistry().most_expensive_model()
        judge = Judge(model, api_key="fake-key")
        score = _run(judge.score(sample, _completion("some answer")))
        assert score == 0.7

    def test_judge_call_failure_returns_none_not_zero(self, monkeypatch):
        """Regression test: a failed judge *call* (bad request params, auth,
        network — anything raising ProviderCallError) used to be scored as
        0.0, indistinguishable from a genuine low-quality answer. A live run
        with the (real, callable) claude-opus-4-7 judge model silently
        produced a uniform 0.00 across every judge-graded sample because
        Judge.score() hardcoded temperature=0.0, which that model's live API
        rejects with a 400 (see the fix comment in llm_judge.py's
        Judge.score) — masking the real error as a fake low quality score.
        Per this module's own contract (graders/__init__.py docstring),
        "could not be graded" is None, not 0.0 — Judge.score() and
        grade_llm_judge() must surface call failures as None so callers skip
        them instead of averaging in a fake zero."""
        from router.evals.graders.llm_judge import Judge, grade_llm_judge
        from router.provider_caller import ProviderCallError

        async def fake_call_provider(model, request, api_key):
            raise ProviderCallError("HTTP 400 from anthropic", http_status=400)

        monkeypatch.setattr("router.provider_caller.call_provider", fake_call_provider)
        sample = load_dataset("mtbench", n=1)[0]
        model = ModelRegistry().most_expensive_model()
        judge = Judge(model, api_key="fake-key")

        score = _run(judge.score(sample, _completion("some answer")))
        assert score is None

        quality, correct = _run(
            grade_llm_judge(sample, _completion("some answer", simulated=False), judge=judge)
        )
        assert quality is None and correct is None

    def test_judge_call_failure_not_cached_as_a_false_zero(self, monkeypatch, tmp_path):
        """A cached failure must replay as None on a later run too — not get
        frozen in as a permanent fake 0.0 score."""
        from router.evals.cache import DiskCache
        from router.evals.graders.llm_judge import Judge
        from router.provider_caller import ProviderCallError

        calls = {"n": 0}

        async def failing_call_provider(model, request, api_key):
            calls["n"] += 1
            raise ProviderCallError("HTTP 400 from anthropic", http_status=400)

        monkeypatch.setattr("router.provider_caller.call_provider", failing_call_provider)
        sample = load_dataset("mtbench", n=1)[0]
        model = ModelRegistry().most_expensive_model()
        cache = DiskCache(str(tmp_path / "cache"))
        judge = Judge(model, api_key="fake-key", cache=cache)
        completion = _completion("some answer")

        first = _run(judge.score(sample, completion))
        second = _run(judge.score(sample, completion))
        assert first is None and second is None
        assert calls["n"] == 1  # second call served from cache, not re-dispatched

    def test_judge_does_not_set_temperature(self, monkeypatch):
        """Regression test for the actual root cause of the uniform-0.00 bug:
        Judge.score() used to hardcode temperature=0.0 on its RoutingRequest.
        Several real Anthropic models (any with extended thinking on by
        default, e.g. claude-opus-4-7/claude-opus-5/claude-sonnet-5) reject
        any explicit non-default temperature live with a 400
        ("`temperature` is deprecated for this model"), while omitting the
        field works everywhere. Assert the judge never sets it, so this
        can't silently regress for any --judge-model choice."""
        from router.evals.graders.llm_judge import Judge
        from router.provider_caller import ProviderResult

        captured = {}

        async def fake_call_provider(model, request, api_key):
            captured["temperature"] = request.temperature
            return ProviderResult(
                text="9", input_tokens=10, output_tokens=1, usage_source="provider"
            )

        monkeypatch.setattr("router.provider_caller.call_provider", fake_call_provider)
        sample = load_dataset("mtbench", n=1)[0]
        model = ModelRegistry().most_expensive_model()
        judge = Judge(model, api_key="fake-key")
        _run(judge.score(sample, _completion("some answer")))
        assert captured["temperature"] is None


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
        # HumanEval is deliberately skipped: this process is not a code sandbox.
        assert out.n_skipped == len(load_dataset("humaneval", n=50)) * 3
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

    def test_full_default_dataset_run_covers_new_categories(self):
        """Wiring check: agentic budget steps and wrapper_tasks run
        end-to-end through the real runner (not just the grader unit tests
        above), against every default strategy."""
        config = RunConfig(
            datasets=list(DATASETS),
            strategies=["flux", "premium", "cheapest"],
            n=50,
            mode="mock",
            source="fixture",
            cache_dir=None,
        )
        out = _run(run_eval(config))
        assert out.results

        budget_rows = [
            r for r in out.results if r.step_type in ("budget_degradation", "budget_stop")
        ]
        assert budget_rows
        assert all(r.correct is True for r in budget_rows)
        assert all(r.cost == 0.0 for r in budget_rows)

        json_rows = [r for r in out.results if r.structured_output_valid is not None]
        assert json_rows

        tool_rows = [r for r in out.results if r.tool_call_valid is not None]
        assert tool_rows

        high_stakes_rows = [r for r in out.results if r.safety_escalated is not None]
        assert high_stakes_rows
        # The domain tier floor is enforced regardless of routing_priority, so
        # flux must always escalate these — a real regression signal.
        assert all(r.safety_escalated for r in high_stakes_rows if r.strategy == "flux")

        assert "fallback_recovery_ok" in out.system_checks
        assert out.system_checks["fallback_recovery_ok"] is True


# ── Provider-default baselines + per-question drill-down ─────────────────────

_PQ_STRATEGIES = ["flux", "default_openai", "default_anthropic", "default_google"]


def _pq_run() -> "RunConfig":
    return _run(
        run_eval(
            RunConfig(
                datasets=["gsm8k", "mmlu"],
                strategies=_PQ_STRATEGIES,
                n=3,
                mode="mock",
                source="fixture",
                cache_dir=None,
            )
        )
    )


class TestProviderDefaults:
    def test_default_strategies_pin_expected_models(self):
        flux = make_flux()
        engine = flux._engine
        registry = engine._registry
        req = RoutingRequest(raw_prompt="hello there", user_id="t", exploration_rate=0.0)
        for strat, model_id in _PROVIDER_DEFAULTS.items():
            model = _run(pick_model(strat, req, engine, registry))
            assert model.model_id == model_id, f"{strat} should pin {model_id}"


class TestPerQuestion:
    def test_quality_rating_matches_registry_table(self):
        out = _pq_run()
        reg = ModelRegistry()
        assert out.results
        for r in out.results:
            model = reg.get_model(r.model_id)
            assert model is not None
            assert r.quality_rating == model.quality_ratings.get(r.task_type, 0.0)

    def test_payload_has_one_row_per_sample_with_all_strategies(self):
        out = _pq_run()
        payload = per_question_payload(out)
        n_samples = len({r.sample_id for r in out.results})
        assert len(payload["questions"]) == n_samples
        for q in payload["questions"]:
            assert set(q["strategies"]) == set(_PQ_STRATEGIES)
            assert q["question"] and q["question_type"]

    def test_rollup_quality_is_mean_of_graded_quality(self):
        """Item 6 bugfix: the per-question rollup used to report the mean of
        each model's static catalog quality_rating here, even in --live mode
        (where run_eval() genuinely grades real completions). It now reports
        the mean of the actual graded GradedResult.quality field — the same
        field the aggregate report already uses — so mock vs live is a
        difference in how quality was produced, not which field is read."""
        out = _pq_run()
        payload = per_question_payload(out)
        assert payload["mode"] == "simulated"  # _pq_run() runs in mock mode
        # Recompute one (question_type, strategy) cell straight from the rows.
        qtype = next(iter(payload["by_question_type"]))
        strat = _PQ_STRATEGIES[0]
        graded = [
            r.quality for r in out.results if r.question_type == qtype and r.strategy == strat
        ]
        expected = round(sum(graded) / len(graded), 4)
        assert payload["by_question_type"][qtype][strat]["quality"] == expected

    def test_per_question_payload_is_reproducible(self):
        a = per_question_payload(_pq_run())
        b = per_question_payload(_pq_run())
        assert a == b

    def test_mode_label_reflects_run_config(self):
        """RunOutput.config.mode drives the "mode" field, not a guess from
        the results themselves — mock -> "simulated", live -> "measured"."""
        out = _pq_run()
        assert out.config.mode == "mock"
        payload = per_question_payload(out)
        assert payload["mode"] == "simulated"

        out.config.mode = "live"  # simulate what a --live run's config would carry
        live_payload = per_question_payload(out)
        assert live_payload["mode"] == "measured"


# ── Live-completion plumbing (Item 6 bugfix, no network) ──────────────────────


class TestLiveCompletionBugfix:
    def test_live_completion_stores_text_not_provider_result(self, monkeypatch):
        """Regression test: _live_completion used to assign the whole
        ProviderResult returned by call_provider() directly to `text`
        instead of `.text` — every downstream consumer expecting a string
        would have broken the moment this path ran with real keys. No test
        exercised it before (mode="live" needs real keys), so this mocks
        call_provider instead of hitting a real API."""
        from router.evals.completions import _live_completion
        from router.provider_caller import ProviderResult

        async def fake_call_provider(model, request, api_key):
            return ProviderResult(
                text="a real answer",
                input_tokens=5,
                output_tokens=3,
                usage_source="provider",
            )

        # _live_completion does `from ..provider_caller import call_provider`
        # locally at call time, so the patch target is the source module.
        monkeypatch.setattr("router.provider_caller.call_provider", fake_call_provider)
        sample = load_dataset("gsm8k", n=1)[0]
        model = ModelRegistry().most_expensive_model()
        completion = _run(
            _live_completion(model, sample, {model.provider: "fake-key"}, max_tokens=64)
        )
        assert completion.text == "a real answer"
        assert isinstance(completion.text, str)
