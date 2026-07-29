"""
server.py — OpenAI-compatible HTTP proxy for Flux (Task 1: Agent Cost Governance Pivot).

Lets any OpenAI SDK client point `base_url` at this server and get routed
completions without touching a line of Python. `model` in the request body is
a routing directive, not a literal:

    "flux-auto"    → route normally (balanced priority)
    "flux-cheap"   → route with cost-optimized priority
    "flux-quality" → route with quality-first priority
    "gpt-4o" (etc) → bypass routing; call that exact model verbatim

Run with `make serve` or `uvicorn router.server:app`. Requires the optional
`server` extra: `pip install flux-router[server]`.

Not a hard dependency of the core package — importing this module without
fastapi/uvicorn installed raises ImportError with a clear message, not a
traceback deep in a missing-module chain.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import structlog

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "router.server requires the 'server' extra. Install with: pip install flux-router[server]"
    ) from exc

from .config import SERVER_MAX_BODY_BYTES, SERVER_REQUIRE_AUTH
from .errors import FluxAPIError
from .flux import Flux, make_flux
from .provider_caller import (
    STREAMING_NATIVE_PROVIDERS,
    ProviderCallError,
    stream_openai_compat_lines,
)
from .run_budget import RunBudgetExceeded
from .schemas import RoutingRequest

log = structlog.get_logger(__name__)

# Reserved `model` values that are routing directives rather than literal model IDs.
_ROUTING_DIRECTIVES: dict[str, str] = {
    "flux-auto": "balanced",
    "flux-cheap": "cost-optimized",
    "flux-quality": "quality-first",
}

_SERVER_TOKEN = os.environ.get("FLUX_SERVER_TOKEN")

if not SERVER_REQUIRE_AUTH:
    log.warning(
        "flux_server_no_auth",
        msg=(
            "FLUX_SERVER_TOKEN is not set — the proxy is running without authentication "
            "and is bound to localhost only. Set FLUX_SERVER_TOKEN to enable auth and "
            "allow non-loopback binding."
        ),
    )

app = FastAPI(title="Flux Router", version="1.0.0")

# 🔧 EXTENSION POINT: swap for a per-process pool / DI container if you need
# multiple Flux instances (e.g. per-tenant provider keys) behind one server.
_flux: Flux = make_flux()


def _check_auth(authorization: str | None) -> None:
    if not SERVER_REQUIRE_AUTH:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != _SERVER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


async def _read_bounded_body(request: Request) -> bytes:
    """Read the request body, rejecting it once it exceeds SERVER_MAX_BODY_BYTES.

    Checked against Content-Length up front AND while streaming the body, so a
    missing or dishonest Content-Length header can't be used to bypass the cap.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > SERVER_MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError:
            pass

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > SERVER_MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _messages_to_request_fields(
    messages: list[dict[str, Any]],
) -> tuple[str | None, str, list[dict]]:
    """Split an OpenAI `messages` array into (system_prompt, raw_prompt, message_history)."""
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if not non_system:
        raise HTTPException(
            status_code=400, detail="messages must include at least one non-system message"
        )
    system_prompt = "\n".join(system_parts) if system_parts else None
    raw_prompt = non_system[-1].get("content", "")
    history = [
        {"role": m.get("role", "user"), "content": m.get("content", "")} for m in non_system[:-1]
    ]
    return system_prompt, raw_prompt, history


def _build_routing_request(body: dict[str, Any], run_id: str) -> tuple[RoutingRequest, bool]:
    """Translate an OpenAI chat-completion request body into a RoutingRequest.

    Returns (request, is_literal_model) — is_literal_model is True when `model`
    named a concrete registered model rather than a flux-* routing directive.
    """
    if "messages" not in body or not isinstance(body["messages"], list):
        raise HTTPException(status_code=400, detail="'messages' is required and must be a list")

    model_field = body.get("model", "flux-auto")
    system_prompt, raw_prompt, history = _messages_to_request_fields(body["messages"])

    kwargs: dict[str, Any] = {
        "user_id": body.get("user") or "flux-server-anonymous",
        "system_prompt": system_prompt,
        "message_history": history,
        "temperature": body.get("temperature"),
        "max_tokens_requested": body.get("max_tokens"),
        "run_id": run_id,
        # Task 6: passed through verbatim for step_type inference + capability
        # filtering. Not forwarded to the provider call itself in this proxy —
        # tool-calling over the proxy is a routing-only signal for now.
        "tools": body.get("tools") or [],
        "response_format": body.get("response_format"),
    }

    is_literal_model = False
    if model_field in _ROUTING_DIRECTIVES:
        kwargs["routing_priority"] = _ROUTING_DIRECTIVES[model_field]
    else:
        if _flux._engine._registry.get_model(model_field) is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown model '{model_field}'. Use a routing directive "
                    f"({', '.join(sorted(_ROUTING_DIRECTIVES))}) or a registered model ID "
                    "(see GET /v1/models)."
                ),
            )
        kwargs["metadata"] = {"model": model_field}
        is_literal_model = True

    try:
        request = RoutingRequest(raw_prompt=raw_prompt, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return request, is_literal_model


def _flux_headers(decision, decision_latency_ms: float) -> dict[str, str]:
    task_type = decision.explanation.task_type if decision.explanation else ""
    complexity = decision.explanation.complexity_score if decision.explanation else 0.0
    headers = {
        "x-flux-model": decision.chosen_model.model_id if decision.chosen_model else "",
        "x-flux-task-type": task_type,
        "x-flux-complexity-score": f"{complexity:.3f}",
        "x-flux-estimated-cost-usd": f"{decision.estimated_cost:.6f}",
        "x-flux-decision-latency-ms": f"{decision_latency_ms:.2f}",
        "x-flux-run-id": decision.run_id or "",
        "x-flux-budget-state": decision.budget_state,
    }
    if decision.budget_warning:
        headers["x-flux-budget-warning"] = decision.budget_warning
    return headers


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe. Never requires auth."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Return the model registry in OpenAI's /v1/models shape."""
    _check_auth(authorization)
    registry = _flux._engine._registry
    return {
        "object": "list",
        "data": [
            {"id": m.model_id, "object": "model", "owned_by": m.provider}
            for m in registry.all_available_models()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_flux_run_id: str | None = Header(default=None, alias="X-Flux-Run-Id"),
) -> Any:
    _check_auth(authorization)
    raw_body = await _read_bounded_body(request)
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # Task 3: every request belongs to a run — X-Flux-Run-Id groups repeated
    # calls into one budgeted trajectory; a request with no header gets its
    # own single-step run (harmless: checked against the generous global
    # RUN_MAX_* defaults, evicted quickly by the RunStore's TTL/LRU).
    run_id = x_flux_run_id or str(uuid.uuid4())
    routing_request, _ = _build_routing_request(body, run_id)
    stream = bool(body.get("stream", False))

    start = time.monotonic()
    try:
        decision = await _flux.route(routing_request, verbose=True)
    except RunBudgetExceeded as exc:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": str(exc), "type": "run_budget_exceeded", **exc.summary}},
            headers={"x-flux-run-id": run_id},
        )
    decision_latency_ms = (time.monotonic() - start) * 1000

    if decision.chosen_model is None:
        raise HTTPException(status_code=502, detail="No model available to route this request")

    headers = _flux_headers(decision, decision_latency_ms)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_id = decision.chosen_model.model_id

    if stream:
        # Recorded up front (using the cost estimate) rather than after the
        # stream closes — StreamingResponse's body can be abandoned by the
        # client mid-stream, which would otherwise leave this step unrecorded
        # and let a run dodge its own budget by disconnecting early.
        _flux._engine._run_budget.record_step(
            run_id, model_id, decision.estimated_cost, routing_request.max_tokens_requested or 1024
        )
        return StreamingResponse(
            _stream_completion(routing_request, decision, completion_id, created, headers),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        text = await _flux._call_model(decision.chosen_model, routing_request)
    except FluxAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _flux._engine._run_budget.record_step(
        run_id, model_id, decision.estimated_cost, max(len(text) // 4, 1)
    )

    response_body = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        # Estimated: provider_caller returns text only, not provider-reported usage.
        "usage": {
            "prompt_tokens": max(len(routing_request.raw_prompt) // 4, 1),
            "completion_tokens": max(len(text) // 4, 1),
            "total_tokens": max((len(routing_request.raw_prompt) + len(text)) // 4, 1),
        },
    }
    return JSONResponse(content=response_body, headers=headers)


async def _stream_completion(routing_request, decision, completion_id, created, headers):
    """Yield SSE chunks for a chat completion.

    For providers that speak OpenAI-native SSE (openai/groq/mistral), each
    chunk is reshaped in place and forwarded as it arrives — genuinely
    incremental, not buffered. For providers without a native OpenAI SSE
    format (anthropic/google), we fall back to a single synthesized chunk
    carrying the full response, since translating each provider's own event
    schema chunk-by-chunk is out of scope here.
    """
    model = decision.chosen_model
    model_id = model.model_id

    def _chunk(delta: dict, finish_reason: str | None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    try:
        if model.provider.lower() in STREAMING_NATIVE_PROVIDERS:
            api_key = _flux._resolve_api_key(model, routing_request)
            yield _chunk({"role": "assistant"}, None)
            async for line in stream_openai_compat_lines(model, routing_request, api_key):
                text = line.decode("utf-8", errors="replace").strip()
                if not text or text == "data: [DONE]":
                    continue
                if text.startswith("data: "):
                    yield f"{text}\n\n".encode()
        else:
            text = await _flux._call_model(model, routing_request)
            yield _chunk({"role": "assistant", "content": text}, None)
            yield _chunk({}, "stop")
    except (ProviderCallError, FluxAPIError) as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
    yield b"data: [DONE]\n\n"
