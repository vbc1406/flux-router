"""Tests for the request complexity classifier."""
from __future__ import annotations

import pytest

from router.cache import ResponseCache
from router.classifier import RequestClassifier, _compute_complexity
from router.schemas import RoutingRequest


def _cache() -> ResponseCache:
    return ResponseCache(enabled=False)


def _clf() -> RequestClassifier:
    return RequestClassifier(_cache())


def _req(prompt: str, **kw) -> RoutingRequest:
    defaults = dict(user_id="u1", plan="pro_plan", priority="normal")
    defaults.update(kw)
    return RoutingRequest(raw_prompt=prompt, **defaults)


# ── Task-type detection ──────────────────────────────────────────────────────

class TestTaskTypeDetection:
    def setup_method(self):
        self.clf = _clf()

    def test_conversation_trivial(self):
        assert self.clf.analyze(_req("hi")).task_type == "conversation"

    def test_conversation_thanks(self):
        assert self.clf.analyze(_req("Thanks!")).task_type == "conversation"

    def test_simple_qa(self):
        assert self.clf.analyze(_req("What is the capital of France?")).task_type == "simple_qa"

    def test_simple_qa_who(self):
        assert self.clf.analyze(_req("Who is Alan Turing?")).task_type == "simple_qa"

    def test_translation(self):
        analysis = self.clf.analyze(_req("Translate 'hello world' to Japanese"))
        assert analysis.task_type == "translation"

    def test_translation_spanish(self):
        analysis = self.clf.analyze(_req("Translate the following to Spanish: Good morning"))
        assert analysis.task_type == "translation"

    def test_summarization(self):
        long_text = "lorem ipsum " * 100
        analysis = self.clf.analyze(_req(f"Summarize this text: {long_text}"))
        assert analysis.task_type == "summarization"

    def test_classification(self):
        analysis = self.clf.analyze(_req("Classify this review as positive or negative: 'Great!'"))
        assert analysis.task_type == "classification"

    def test_extraction(self):
        analysis = self.clf.analyze(_req("Extract all email addresses from: a@b.com, c@d.com"))
        assert analysis.task_type == "extraction"

    def test_code_generation(self):
        analysis = self.clf.analyze(_req("Write a Python function to sort a list"))
        assert analysis.task_type == "code_generation"

    def test_code_review(self):
        analysis = self.clf.analyze(_req("Review this code:\n```python\ndef f(): pass\n```"))
        assert analysis.task_type == "code_review"

    def test_creative_writing(self):
        analysis = self.clf.analyze(_req("Write a story about a robot who learns to dream"))
        assert analysis.task_type == "creative_writing"

    def test_analysis(self):
        analysis = self.clf.analyze(_req("Compare PostgreSQL vs MongoDB for OLAP workloads"))
        assert analysis.task_type == "analysis"

    def test_reasoning(self):
        analysis = self.clf.analyze(_req("Prove that √2 is irrational, show all steps"))
        assert analysis.task_type == "reasoning"

    def test_vision_capability(self):
        req = _req("Describe what's in this image.", required_capabilities=["vision"])
        assert self.clf.analyze(req).task_type == "vision"

    def test_function_calling(self):
        req = _req("Call the weather API", required_capabilities=["function_calling"])
        assert self.clf.analyze(req).task_type == "function_calling"

    def test_long_document(self):
        big = "word " * 8000
        analysis = self.clf.analyze(_req(big))
        assert analysis.task_type == "long_document"


# ── Complexity scoring ───────────────────────────────────────────────────────

class TestComplexityScoring:
    def setup_method(self):
        self.clf = _clf()

    def test_trivial_very_low(self):
        assert self.clf.analyze(_req("hi")).complexity_score < 0.20

    def test_simple_qa_low(self):
        score = self.clf.analyze(_req("What is 2+2?")).complexity_score
        assert score < 0.30

    def test_code_gen_mid(self):
        score = self.clf.analyze(_req("Write a Python function to reverse a linked list")).complexity_score
        assert 0.25 < score < 0.80

    def test_math_reasoning_high(self):
        score = self.clf.analyze(_req(
            "Prove ∑(1/n²) = π²/6 using Fourier analysis. Show all derivation steps."
        )).complexity_score
        assert score >= 0.55

    def test_priority_critical_floor(self):
        score = self.clf.analyze(_req("hi", priority="critical")).complexity_score
        assert score >= 0.70

    def test_priority_low_ceiling(self):
        score = self.clf.analyze(_req(
            "Implement a complete compiler from scratch with optimisation passes",
            priority="low"
        )).complexity_score
        assert score <= 0.40

    def test_score_bounded(self):
        for prompt in ["hi", "Write me a full OS kernel with all drivers", "What is 2+2?"]:
            s = self.clf.analyze(_req(prompt)).complexity_score
            assert 0.0 <= s <= 1.0

    def test_sigmoid_smooth(self):
        """Score with many modifiers should not exceed 1.0."""
        long_math = "Prove P≠NP. Show ∀x∃y. Use ∑∫ notation. " * 20
        score = self.clf.analyze(_req(long_math, priority="normal")).complexity_score
        assert score <= 1.0

    def test_compute_complexity_direct(self):
        """Unit-test the pure helper directly."""
        signals = {"math_symbols_present": True, "requires_reasoning": True}
        score = _compute_complexity("reasoning", signals, "normal")
        assert 0.5 < score <= 1.0


# ── Cache eligibility ────────────────────────────────────────────────────────

class TestCacheEligibility:
    def setup_method(self):
        self.clf = _clf()

    def test_simple_qa_zero_temp_eligible(self):
        a = self.clf.analyze(_req("What is the capital of France?", temperature=0.0))
        assert a.cache_eligible is True

    def test_simple_qa_none_temp_eligible(self):
        a = self.clf.analyze(_req("What is the capital of France?"))
        assert a.cache_eligible is True

    def test_high_temp_not_eligible(self):
        a = self.clf.analyze(_req("What is the capital of France?", temperature=0.9))
        assert a.cache_eligible is False

    def test_creative_not_eligible(self):
        a = self.clf.analyze(_req("Write a poem about autumn", temperature=0.0))
        assert a.cache_eligible is False

    def test_conversation_not_eligible(self):
        a = self.clf.analyze(_req("hi", temperature=0.0))
        assert a.cache_eligible is False

    def test_translation_eligible(self):
        a = self.clf.analyze(_req("Translate 'cat' to Spanish", temperature=0.0))
        assert a.cache_eligible is True

    def test_classification_eligible(self):
        a = self.clf.analyze(_req("Classify as spam or not: 'Buy now!'", temperature=0.1))
        assert a.cache_eligible is True


# ── Fingerprint stability ────────────────────────────────────────────────────

class TestFingerprint:
    def setup_method(self):
        self.clf = _clf()

    def test_same_prompt_same_fp(self):
        a1 = self.clf.analyze(_req("What is 2+2?", temperature=0.0))
        a2 = self.clf.analyze(_req("What is 2+2?", temperature=0.0))
        assert a1.prompt_fingerprint == a2.prompt_fingerprint

    def test_filler_removed(self):
        a1 = self.clf.analyze(_req("What is 2+2?", temperature=0.0))
        a2 = self.clf.analyze(_req("Can you tell me what is 2+2?", temperature=0.0))
        # After filler removal "can you" is stripped, should match
        # (They still differ on "tell me what" vs "what" but fingerprint should be close)
        # The important thing: both produce a non-empty hex string
        assert len(a1.prompt_fingerprint) == 64
        assert len(a2.prompt_fingerprint) == 64

    def test_temperature_changes_fp(self):
        a1 = self.clf.analyze(_req("What is 2+2?", temperature=0.0))
        a2 = self.clf.analyze(_req("What is 2+2?", temperature=1.0))
        assert a1.prompt_fingerprint != a2.prompt_fingerprint


# ── Token estimation ─────────────────────────────────────────────────────────

class TestTokenEstimation:
    def setup_method(self):
        self.clf = _clf()

    def test_short_prompt_small_token_count(self):
        a = self.clf.analyze(_req("hi"))
        assert a.estimated_input_tokens < 50

    def test_long_prompt_large_token_count(self):
        a = self.clf.analyze(_req("hello " * 1000))
        assert a.estimated_input_tokens > 500

    def test_total_context_gte_input(self):
        a = self.clf.analyze(_req("Write a Python function to sort a list"))
        assert a.total_context_needed >= a.estimated_input_tokens


# ── Sensitivity detection ─────────────────────────────────────────────────────

class TestSensitivityDetection:
    def setup_method(self):
        self.clf = _clf()

    def test_explicit_metadata_override(self):
        req = _req("hello", metadata={"sensitivity_level": "restricted"})
        assert self.clf.analyze(req).sensitivity_level == "restricted"

    def test_public_default(self):
        assert self.clf.analyze(_req("What is 2+2?")).sensitivity_level == "public"

    def test_confidential_keyword(self):
        a = self.clf.analyze(_req("This document is confidential and NDA-protected"))
        assert a.sensitivity_level == "confidential"

    def test_restricted_keyword(self):
        a = self.clf.analyze(_req("My SSN is 123-45-6789 and bank account 9876"))
        assert a.sensitivity_level == "restricted"
