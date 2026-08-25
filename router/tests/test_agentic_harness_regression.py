"""
File: router/tests/test_agentic_harness_regression.py

Offline regression coverage for the live end-to-end agentic-trajectory
harness (scripts/agentic_e2e_harness.py). No real HTTP requests to any
provider are made — provider I/O is mocked at Flux._call_model, matching
the conventions in test_tool_calling.py/test_server.py.

Covers the two mechanics the live run actually exercises:
  1. x-flux-run-id is echoed identically across every step of one trajectory,
     and the required x-flux-* headers are present on each response — this
     is what the harness's "routing validation" section asserts live.
  2. Cumulative run-budget enforcement returns HTTP 429 once a run-scoped
     RunLimits ceiling is crossed — this is what the harness's tiny-budget
     trajectory proves live against a real server process.

Also covers two real defects the live run surfaced and that were then fixed:
  - server.py's `_messages_to_request_fields()` used to always take the LAST
    message in an incoming `messages` array as the new `raw_prompt` string,
    discarding its role/tool_call_id/name — so a request whose last message
    is a role="tool" observation (exactly what step 4 of the spec'd
    trajectory sends) had that message silently flattened into an unlabeled
    user-shaped prompt before it ever reached message_history. Fixed by
    keeping a trailing role="tool" message in history instead of popping it
    (and provider_caller._build_messages() no longer appends a spurious
    empty trailing user turn after it).
  - provider_caller.py's Anthropic caller used to forward the universal
    OpenAI-shaped message history (assistant `tool_calls`, role="tool" +
    `tool_call_id`) to Anthropic's /v1/messages verbatim, which Anthropic
    doesn't understand (it uses tool_use/tool_result content blocks) — any
    step after a real tool round trip routed to Anthropic got a genuine 400.
    Fixed by `_openai_messages_to_anthropic()`.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("fastapi")

from unittest.mock import AsyncMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import router.server as server  # noqa: E402
from router.provider_caller import ProviderResult, _openai_messages_to_anthropic  # noqa: E402
from router.run_budget import RunLimits  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8000")


def _ok_result(text: str = "ok") -> ProviderResult:
    return ProviderResult(text=text, input_tokens=10, output_tokens=5, usage_source="provider")


class TestRunIdAndHeadersAcrossTrajectory:
    def test_run_id_echoed_and_headers_present_across_steps(self, client, monkeypatch):
        monkeypatch.setattr(server._flux, "_call_model", AsyncMock(return_value=_ok_result()))
        run_id = f"regress-{uuid.uuid4().hex[:8]}"
        required = [
            "x-flux-model",
            "x-flux-task-type",
            "x-flux-complexity-score",
            "x-flux-estimated-cost-usd",
            "x-flux-decision-latency-ms",
            "x-flux-budget-state",
        ]
        for step_type in ("plan", "tool_select", "tool_result_summarize", "reflect", "final_answer"):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "flux-auto", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Flux-Run-Id": run_id, "X-Flux-Step-Type": step_type},
            )
            assert resp.status_code == 200, resp.text
            assert resp.headers.get("x-flux-run-id") == run_id
            assert "x-flux-run-id-missing" not in resp.headers
            for h in required:
                assert h in resp.headers, f"missing {h} on step {step_type}"

    def test_missing_run_id_header_is_flagged(self, client, monkeypatch):
        monkeypatch.setattr(server._flux, "_call_model", AsyncMock(return_value=_ok_result()))
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "flux-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.headers.get("x-flux-run-id-missing") == "true"


class TestCumulativeRunBudget429:
    def test_tiny_run_budget_stops_run_with_429(self, client, monkeypatch):
        monkeypatch.setattr(server._flux, "_call_model", AsyncMock(return_value=_ok_result()))
        run_id = f"regress-tiny-{uuid.uuid4().hex[:8]}"
        server._flux._engine._run_budget.start(
            run_id, limits=RunLimits(max_cost_usd=1_000.0, max_steps=1, max_tokens=1_000_000)
        )
        body = {"model": "flux-auto", "messages": [{"role": "user", "content": "hi"}]}
        headers = {"X-Flux-Run-Id": run_id, "X-Flux-Step-Type": "plan"}

        first = client.post("/v1/chat/completions", json=body, headers=headers)
        assert first.status_code == 200
        assert first.headers.get("x-flux-run-id") == run_id

        second = client.post("/v1/chat/completions", json=body, headers=headers)
        assert second.status_code == 429
        assert second.headers.get("x-flux-run-id") == run_id
        assert second.json()["error"]["type"] == "run_budget_exceeded"

        server._flux._engine._run_budget.finish(run_id)


class TestToolRoleLastMessageDefect:
    """Regression coverage for a defect the live agentic harness surfaced:
    _messages_to_request_fields() used to take messages[-1] as raw_prompt
    unconditionally, dropping role/tool_call_id when the last message was
    itself role="tool". Fixed in server.py (keep a trailing tool message in
    history instead of popping it) and provider_caller.py (Anthropic's
    caller now translates tool_calls/role="tool" into tool_use/tool_result
    content blocks instead of forwarding the OpenAI shape verbatim)."""

    def test_trailing_tool_message_preserved_in_history(self, client, monkeypatch):
        captured = {}
        real_route = server._flux.route

        async def spy_route(request, verbose=False):
            captured["history"] = request.message_history
            return await real_route(request, verbose=verbose)

        monkeypatch.setattr(server._flux, "route", spy_route)
        monkeypatch.setattr(server._flux, "_call_model", AsyncMock(return_value=_ok_result()))

        body = {
            "model": "flux-auto",
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "sunny"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200

        history = captured["history"]
        assert history, "message_history should not be empty"
        last = history[-1]
        assert last.get("role") == "tool"
        assert last.get("tool_call_id") == "call_1"


class TestAnthropicToolHistoryTranslation:
    def test_tool_use_and_tool_result_blocks_produced(self):
        universal = [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "sunny"},
        ]
        out = _openai_messages_to_anthropic(universal)

        assert out[0] == {"role": "user", "content": "weather in Paris?"}

        assistant_turn = out[1]
        assert assistant_turn["role"] == "assistant"
        tool_use = next(b for b in assistant_turn["content"] if b["type"] == "tool_use")
        assert tool_use["id"] == "call_1"
        assert tool_use["name"] == "get_weather"
        assert tool_use["input"] == {"location": "Paris"}

        tool_result_turn = out[2]
        assert tool_result_turn["role"] == "user"
        assert tool_result_turn["content"] == [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}
        ]

    def test_consecutive_tool_results_merge_into_one_user_turn(self):
        universal = [
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "call_2", "content": "72F"},
        ]
        out = _openai_messages_to_anthropic(universal)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        assert [b["tool_use_id"] for b in out[0]["content"]] == ["call_1", "call_2"]
