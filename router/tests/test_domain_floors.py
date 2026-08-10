"""
File: router/tests/test_domain_floors.py

Purpose:
Tests for Item 1 — high-stakes legal/medical domain detection
(classifier.py::_MEDICAL_SUBSTANTIVE_RE / _LEGAL_SUBSTANTIVE_RE) and the
DOMAIN_TIER_FLOORS minimum-tier floor enforced in routing_engine.py via
_composed_min_tier() / _passes_hard_constraints().

How to run:
  pytest -v router/tests/test_domain_floors.py
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import RoutingEngine
from router.schemas import ModelOption, RoutingRequest


def _engine(models: list[ModelOption] | None = None) -> RoutingEngine:
    registry = ModelRegistry()
    if models is not None:
        registry._models = {m.model_id: m for m in models}
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, **kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {
        "user_id": "u_domain_floor",
        "plan": "business_plan",
        "exploration_rate": 0.0,
    }
    defaults.update(kw)
    return RoutingRequest(raw_prompt=prompt, correlation_id=str(uuid.uuid4()), **defaults)


def rr(coro):
    return asyncio.run(coro)


def _classify(prompt: str, **kw: Any):
    classifier = RequestClassifier(ResponseCache(enabled=False))
    return classifier.analyze(_req(prompt, **kw))


class TestMedicalDetection:
    def test_diagnosis_request_classified_medical(self):
        analysis = _classify("I've had a fever for a week, can you diagnose me?")
        assert analysis.task_type == "medical"

    def test_urgent_symptoms_classified_medical(self):
        analysis = _classify(
            "I have chest pain and shortness of breath, am I having a heart attack?"
        )
        assert analysis.task_type == "medical"

    def test_medication_question_classified_medical(self):
        analysis = _classify("What medication should I take for a migraine that won't go away?")
        assert analysis.task_type == "medical"

    def test_drug_interaction_classified_medical(self):
        analysis = _classify("Are there drug interactions between ibuprofen and my blood thinner?")
        assert analysis.task_type == "medical"

    def test_benign_medical_summarization_stays_cheap(self):
        report = "Patient presents with mild cough. " * 15
        analysis = _classify(f"Summarize this medical report in plain English: {report}")
        assert analysis.task_type != "medical"
        assert analysis.task_type == "summarization"

    def test_incidental_doctor_mention_not_medical(self):
        analysis = _classify("Write a short story about a doctor who loves gardening.")
        assert analysis.task_type != "medical"


class TestLegalDetection:
    def test_liability_question_classified_legal(self):
        analysis = _classify("If this feature breaks, am I legally liable for the damages?")
        assert analysis.task_type == "legal"

    def test_contract_enforceability_classified_legal(self):
        analysis = _classify("Is this non-compete clause enforceable in California?")
        assert analysis.task_type == "legal"

    def test_regulatory_compliance_classified_legal(self):
        analysis = _classify("How do we comply with GDPR when storing this customer data?")
        assert analysis.task_type == "legal"

    def test_breach_of_contract_classified_legal(self):
        analysis = _classify("My vendor missed the deadline — is this a breach of contract?")
        assert analysis.task_type == "legal"

    def test_benign_contract_extraction_stays_cheap(self):
        contract = "This agreement is between Party A and Party B. " * 15
        analysis = _classify(f"Extract the effective date and both party names: {contract}")
        assert analysis.task_type != "legal"

    def test_incidental_contract_mention_not_legal(self):
        analysis = _classify("Explain what a smart contract is in blockchain terms.")
        assert analysis.task_type != "legal"

    def test_incidental_court_mention_not_legal(self):
        analysis = _classify("What's the score of the basketball game on Court 2?")
        assert analysis.task_type != "legal"


class TestDomainTierFloor:
    def test_medical_request_routes_to_premium(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req("Am I having a heart attack? What medication should I take right now?")
            )
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"

    def test_legal_request_routes_to_premium(self):
        engine = _engine()
        decision = rr(
            engine.route(_req("Does this non-compete clause violate California law?"))
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"

    def test_benign_summarization_of_medical_text_not_forced_premium(self):
        engine = _engine()
        report = "Patient presents with mild cough and mild fatigue. " * 15
        decision = rr(engine.route(_req(f"Summarize this medical report: {report}")))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier != "premium"

    def test_floor_survives_explicit_confidential_sensitivity(self):
        """Sensitivity restrictions and the domain floor are independent
        constraints — both must hold simultaneously, neither replaces the
        other."""
        engine = _engine()
        req = _req(
            "This is confidential: am I legally liable for this data breach?",
            metadata={"sensitivity_level": "confidential"},
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"
        assert "confidential" in decision.chosen_model.allowed_sensitivity_levels

    def test_restricted_high_stakes_request_still_finds_a_model(self):
        engine = _engine()
        req = _req(
            "This is classified: does releasing this violate the export-control regulations?",
            metadata={"sensitivity_level": "restricted"},
        )
        decision = rr(engine.route(req))
        # restricted -> only anthropic/openai providers; premium floor from
        # the legal domain must still be satisfied by the surviving pool.
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"
        assert "restricted" in decision.chosen_model.allowed_sensitivity_levels

    def test_no_eligible_model_returns_no_candidates_with_floor_reasoning(self):
        """When the catalog has nothing at/above the domain floor, routing
        must return a clear budget/no-model result rather than silently
        picking an under-qualified model."""
        below_floor = [
            ModelOption(
                provider="p-mid",
                model_id="m-mid-only",
                display_name="Mid Only",
                tier="mid",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.01,
                max_context_window=100_000,
                max_output_tokens=4096,
                capabilities=[],
                quality_ratings={"medical": 0.9, "general": 0.9},
            )
        ]
        engine = _engine(below_floor)
        decision = rr(
            engine.route(
                _req("Am I having a heart attack? What medication should I take right now?")
            )
        )
        assert decision.chosen_model is None
        assert decision.routing_rule_matched == "no_candidates"
        assert "domain:medical" in decision.reasoning

    def test_explicit_override_below_floor_falls_through_to_normal_routing(self):
        """An explicit model override naming a model below the mandatory
        floor must not bypass the floor — the override search is restricted
        to the already-floor-filtered candidate set."""
        models = [
            ModelOption(
                provider="p-cheap",
                model_id="m-cheap-underqualified",
                display_name="Cheap",
                tier="cheap",
                cost_per_1k_input=0.0005,
                cost_per_1k_output=0.001,
                max_context_window=100_000,
                max_output_tokens=4096,
                capabilities=[],
                quality_ratings={"medical": 0.9, "general": 0.9},
            ),
            ModelOption(
                provider="p-premium",
                model_id="m-premium-qualified",
                display_name="Premium",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                max_context_window=100_000,
                max_output_tokens=4096,
                capabilities=[],
                quality_ratings={"medical": 0.9, "general": 0.9},
            ),
        ]
        engine = _engine(models)
        req = _req(
            "Am I having a heart attack? What medication should I take right now?",
            metadata={"model": "m-cheap-underqualified"},
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-premium-qualified"
        assert decision.routing_rule_matched != "explicit_model_override"

    def test_always_premium_still_correct_for_medical(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(
                    "Diagnose these symptoms please.",
                    routing_priority="always-premium",
                )
            )
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"
        assert decision.priority_applied == "always-premium"

    def test_quality_max_still_correct_for_legal(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(
                    "Is this contract enforceable given the missing signature?",
                    routing_priority="quality_max",
                )
            )
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"
        assert decision.priority_applied == "quality_max"

    def test_explainability_reports_domain_floor(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req("Am I having a heart attack?"),
                verbose=True,
            )
        )
        assert decision.explanation is not None
        assert any("domain:medical" in r for r in decision.explanation.floors_applied)
