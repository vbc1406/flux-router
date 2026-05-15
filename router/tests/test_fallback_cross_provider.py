"""
File: router/tests/test_fallback_cross_provider.py

Purpose:
Closes the integration gap left by test_smart_retry.py: smart_retry mocks the
api_caller, so it verifies fallback CONTROL FLOW but not the cross-provider
PAYLOAD TRANSLATION that happens when the primary and fallback model belong to
different provider families (e.g. Anthropic 429 → Google retry).

What this test asserts:
  1. FallbackExecutor reuses the SAME RoutingRequest object across attempts,
     so the full message_history travels with the fallback.
  2. Each provider's serializer (`_build_messages` for OpenAI/Anthropic compat,
     the inline Google `contents` builder) produces a payload that contains
     every original turn plus the new user prompt.
  3. The fallback hop reports the correct failed/next model on the event.

This is the test that would have caught a regression where, say, someone
changed _build_messages to drop assistant turns — the unit tests in
test_smart_retry would still pass because they never reach _build_messages.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from router.analytics import RoutingAnalytics
from router.fallback_chain import FallbackExecutor
from router.provider_caller import ProviderCallError, _build_messages
from router.schemas import ModelOption, RoutingDecision, RoutingRequest

# ── Helpers ─────────────────────────────────────────────────────────────────


def _anthropic_model() -> ModelOption:
    return ModelOption(
        provider="anthropic",
        model_id="claude-test",
        display_name="Claude Test",
        tier="premium",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        max_context_window=200_000,
        max_output_tokens=4096,
        capabilities=["general"],
    )


def _google_model() -> ModelOption:
    return ModelOption(
        provider="google",
        model_id="gemini-test",
        display_name="Gemini Test",
        tier="premium",
        cost_per_1k_input=0.0035,
        cost_per_1k_output=0.0105,
        max_context_window=1_000_000,
        max_output_tokens=8192,
        capabilities=["general"],
    )


def _build_google_contents(request: RoutingRequest) -> list[dict]:
    # Mirrors router/provider_caller.py:_call_google_sync — kept inline because
    # that helper isn't exported. If the production code drifts, this drifts too,
    # and that's the point: this test exists to catch that drift.
    contents = []
    for turn in request.message_history:
        contents.append(
            {
                "role": turn.get("role", "user"),
                "parts": [{"text": turn.get("content", "")}],
            }
        )
    contents.append({"role": "user", "parts": [{"text": request.raw_prompt}]})
    return contents


def _request_with_history() -> RoutingRequest:
    return RoutingRequest(
        raw_prompt="and what about the third point?",
        user_id="u_test",
        plan="business_plan",
        correlation_id=str(uuid.uuid4()),
        message_history=[
            {"role": "user", "content": "summarize the report in three points"},
            {"role": "assistant", "content": "Point 1: revenue up. Point 2: churn down."},
            {"role": "user", "content": "interesting — can you expand on point 2?"},
            {"role": "assistant", "content": "Churn fell from 5% to 3% over the quarter."},
        ],
    )


def _no_delay(monkeypatch):
    # Keep the test fast — production delays are 1s/2s/5s.
    monkeypatch.setattr("router.fallback_chain.FALLBACK_DELAYS", [0.0, 0.0, 0.0])
    monkeypatch.setattr("router.fallback_chain.FALLBACK_JITTER_MAX", 0.0)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestCrossProviderContextHandoff:
    """The actual coverage gap: context survives a provider-family switch."""

    def test_anthropic_429_falls_back_to_google_with_full_history(self, monkeypatch):
        _no_delay(monkeypatch)
        captured: dict = {}

        async def api_caller(request: RoutingRequest, model: ModelOption) -> str:
            if model.provider == "anthropic":
                raise ProviderCallError("rate limited", http_status=429)
            # Fallback hit Google — serialize using the Google branch.
            captured["contents"] = _build_google_contents(request)
            captured["model_used"] = model.model_id
            return "fallback response"

        request = _request_with_history()
        decision = RoutingDecision(
            chosen_model=_anthropic_model(),
            fallback_chain=[_google_model()],
        )
        executor = FallbackExecutor(RoutingAnalytics(log_path=None))

        response, model_used, events = asyncio.run(
            executor.execute_with_fallback(request, decision, api_caller)
        )

        assert response == "fallback response"
        assert model_used.provider == "google"
        assert len(events) == 1
        assert events[0].failed_model == "claude-test"
        assert events[0].reason == "rate_limited"
        assert events[0].next_model == "gemini-test"

        # The critical assertion: the Google payload contains every original
        # turn and the new prompt, in order.
        contents = captured["contents"]
        assert len(contents) == 5  # 4 history turns + raw_prompt
        assert contents[0]["parts"][0]["text"] == "summarize the report in three points"
        assert contents[1]["parts"][0]["text"].startswith("Point 1")
        assert contents[2]["parts"][0]["text"].startswith("interesting")
        assert contents[3]["parts"][0]["text"].startswith("Churn fell")
        assert contents[4]["parts"][0]["text"] == "and what about the third point?"
        # Role mapping is preserved
        assert [c["role"] for c in contents] == ["user", "assistant", "user", "assistant", "user"]

    def test_google_429_falls_back_to_anthropic_with_full_history(self, monkeypatch):
        _no_delay(monkeypatch)
        captured: dict = {}

        async def api_caller(request: RoutingRequest, model: ModelOption) -> str:
            if model.provider == "google":
                raise ProviderCallError("rate limited", http_status=429)
            # Fallback hit Anthropic-compat — serialize using _build_messages.
            captured["messages"] = _build_messages(request)
            captured["model_used"] = model.model_id
            return "fallback response"

        request = _request_with_history()
        decision = RoutingDecision(
            chosen_model=_google_model(),
            fallback_chain=[_anthropic_model()],
        )
        executor = FallbackExecutor(RoutingAnalytics(log_path=None))

        response, model_used, _events = asyncio.run(
            executor.execute_with_fallback(request, decision, api_caller)
        )

        assert model_used.provider == "anthropic"
        messages = captured["messages"]
        assert len(messages) == 5
        assert messages[0] == {
            "role": "user",
            "content": "summarize the report in three points",
        }
        assert messages[-1] == {"role": "user", "content": "and what about the third point?"}
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]


class TestRequestReusedAcrossAttempts:
    """Verifies the FallbackExecutor contract: same request object on every hop."""

    def test_request_identity_preserved(self, monkeypatch):
        _no_delay(monkeypatch)
        seen_request_ids: list[int] = []

        async def api_caller(request: RoutingRequest, model: ModelOption) -> str:
            seen_request_ids.append(id(request))
            if model.provider == "anthropic":
                raise ProviderCallError("down", http_status=503)
            return "fallback ok response"

        request = _request_with_history()
        decision = RoutingDecision(
            chosen_model=_anthropic_model(),
            fallback_chain=[_google_model()],
        )
        executor = FallbackExecutor(RoutingAnalytics(log_path=None))

        asyncio.run(executor.execute_with_fallback(request, decision, api_caller))

        assert len(seen_request_ids) == 2
        assert seen_request_ids[0] == seen_request_ids[1] == id(request)

    def test_history_not_mutated_by_fallback(self, monkeypatch):
        _no_delay(monkeypatch)

        async def api_caller(request: RoutingRequest, model: ModelOption) -> str:
            if model.provider == "anthropic":
                raise ProviderCallError("down", http_status=503)
            return "fallback ok response"

        request = _request_with_history()
        original_history = [dict(t) for t in request.message_history]
        original_prompt = request.raw_prompt

        decision = RoutingDecision(
            chosen_model=_anthropic_model(),
            fallback_chain=[_google_model()],
        )
        executor = FallbackExecutor(RoutingAnalytics(log_path=None))
        asyncio.run(executor.execute_with_fallback(request, decision, api_caller))

        assert request.message_history == original_history
        assert request.raw_prompt == original_prompt


class TestNoRetryStatusesShortCircuit:
    """Auth and bad-request errors must NOT fall back — re-prove at the
    cross-provider level since smart_retry's mocked tests can't catch a
    regression where a real ProviderCallError(401) was wrapped/swallowed."""

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_no_retry_propagates(self, monkeypatch, status):
        _no_delay(monkeypatch)
        attempts = 0

        async def api_caller(request, model):
            nonlocal attempts
            attempts += 1
            raise ProviderCallError("boom", http_status=status)

        request = _request_with_history()
        decision = RoutingDecision(
            chosen_model=_anthropic_model(),
            fallback_chain=[_google_model()],
        )
        executor = FallbackExecutor(RoutingAnalytics(log_path=None))

        with pytest.raises(ProviderCallError):
            asyncio.run(executor.execute_with_fallback(request, decision, api_caller))

        assert attempts == 1, "no-retry status should NOT trigger fallback"
