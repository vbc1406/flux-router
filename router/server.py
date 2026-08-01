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
    from fastapi import FastAPI, Header, HTTPException, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "router.server requires the 'server' extra. Install with: pip install flux-router[server]"
    ) from exc

from .config import (
    ATTRIBUTION_USAGE_PAGE_MAX,
    SERVER_MAX_BODY_BYTES,
    SERVER_REQUIRE_AUTH,
    SERVER_TOKENS,
)
from .errors import AuthenticationError, FluxAPIError
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
elif not SERVER_TOKENS:
    log.warning(
        "flux_server_shared_token_no_tenant_binding",
        msg=(
            "FLUX_SERVER_TOKEN is set but FLUX_SERVER_TOKENS is not — every caller who "
            "holds the shared token is authenticated as every tenant (X-Flux-Tenant-Id "
            "and the `user` field are self-declared, unverified claims). Set "
            "FLUX_SERVER_TOKENS to bind each bearer token to one tenant_id instead."
        ),
    )

app = FastAPI(title="Flux Router", version="1.0.0")

# 🔧 EXTENSION POINT: swap for a per-process pool / DI container if you need
# multiple Flux instances (e.g. per-tenant provider keys) behind one server.
_flux: Flux = make_flux()


def _check_auth(authorization: str | None) -> str | None:
    """Validate the bearer token and return the tenant_id it's bound to, if
    any (FLUX_SERVER_TOKENS multi-tenant mode). None means either auth is
    disabled or the legacy single-shared-token mode is in effect, where the
    caller's self-declared tenant_id/user fields are trusted as-is — see
    SECURITY_ARCHITECTURE.md."""
    if not SERVER_REQUIRE_AUTH:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if SERVER_TOKENS:
        bound_tenant = SERVER_TOKENS.get(token)
        if bound_tenant is None:
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        return bound_tenant
    if token != _SERVER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return None


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
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
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


def _build_routing_request(
    body: dict[str, Any], run_id: str, tenant_id: str | None = None
) -> tuple[RoutingRequest, bool]:
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
        # Task 7: X-Flux-Tenant-Id, for router/attribution.py aggregation.
        "tenant_id": tenant_id,
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


@app.get("/v1/usage")
async def get_usage(
    authorization: str | None = Header(default=None),
    tenant_id: str | None = None,
    run_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Task 7: paginated cost/metadata usage records — never prompt or
    completion content (see SECURITY_ARCHITECTURE.md). Filter with
    ?tenant_id=... and/or ?run_id=...; paginate with ?limit=&offset=.

    In FLUX_SERVER_TOKENS multi-tenant mode, the bearer token's bound tenant
    always wins over ?tenant_id= — a caller can only ever see their own
    tenant's usage, regardless of what they pass on the query string.
    """
    bound_tenant = _check_auth(authorization)
    if bound_tenant is not None:
        tenant_id = bound_tenant
    limit = max(1, min(limit, ATTRIBUTION_USAGE_PAGE_MAX))
    offset = max(0, offset)
    records, total = _flux._engine._attribution.usage(
        tenant_id=tenant_id, run_id=run_id, limit=limit, offset=offset
    )
    return {
        "object": "list",
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": [
            {
                "tenant_id": r.tenant_id,
                "run_id": r.run_id,
                "task_type": r.task_type,
                "step_type": r.step_type,
                "model_id": r.model_id,
                "cost_usd": r.cost_usd,
                "timestamp": r.timestamp,
            }
            for r in records
        ],
    }


@app.get("/metrics")
async def metrics(authorization: str | None = Header(default=None)) -> Response:
    """Task 7: Prometheus text-exposition metrics — flux_cost_usd_total,
    flux_run_steps, flux_budget_exceeded_total, labelled by tenant/model."""
    _check_auth(authorization)
    body = _flux._engine._attribution.render_prometheus()
    return Response(content=body, media_type="text/plain; version=0.0.4")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_flux_run_id: str | None = Header(default=None, alias="X-Flux-Run-Id"),
    x_flux_tenant_id: str | None = Header(default=None, alias="X-Flux-Tenant-Id"),
) -> Any:
    bound_tenant = _check_auth(authorization)
    raw_body = await _read_bounded_body(request)
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # In FLUX_SERVER_TOKENS multi-tenant mode, the bearer token's bound
    # tenant always wins over the client-supplied X-Flux-Tenant-Id header —
    # otherwise any caller could attribute spend/usage to an arbitrary
    # tenant_id regardless of which token they authenticated with.
    tenant_id = bound_tenant if bound_tenant is not None else x_flux_tenant_id

    # Task 3: every request belongs to a run — X-Flux-Run-Id groups repeated
    # calls into one budgeted trajectory; a request with no header gets its
    # own single-step run (harmless: checked against the generous global
    # RUN_MAX_* defaults, evicted quickly by the RunStore's TTL/LRU).
    run_id = x_flux_run_id or str(uuid.uuid4())
    routing_request, _ = _build_routing_request(body, run_id, tenant_id)
    stream = bool(body.get("stream", False))

    start = time.monotonic()
    try:
        decision = await _flux.route(routing_request, verbose=True)
    except RunBudgetExceeded as exc:
        _flux._engine._attribution.record_budget_exceeded(tenant_id)
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
        # Budget/attribution recording happens inside _stream_completion,
        # once the provider call is confirmed to have actually produced
        # output — see its docstring for why that beats recording up front.
        return StreamingResponse(
            _stream_completion(routing_request, decision, completion_id, created, headers),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        # Shared with Flux.complete(): retries the typed fallback chain on
        # transient errors, and records plan/daily-cap spend, run-budget steps,
        # and attribution — all of which this path used to skip by calling
        # _call_model() directly.
        text, used_model, _fallback_used, _fallback_reason = await _flux._dispatch_with_fallback(
            decision, routing_request, max_retries=2
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except FluxAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # The dispatched model may differ from decision.chosen_model if a fallback
    # fired — reflect the model that actually served the request.
    model_id = used_model.model_id
    headers["x-flux-model"] = model_id

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

    Budget/attribution is recorded as soon as the provider call is confirmed
    to have produced output (first chunk for native streaming, full text for
    the synthesized-chunk path) rather than before the call starts — a call
    that fails outright (ProviderCallError/FluxAPIError, e.g. a 429 from the
    provider) never gets this far, so it isn't charged. Once recorded, it
    stays recorded even if the client disconnects mid-stream afterward: the
    provider was already paid for those tokens.
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

    def _record_usage(tokens: int) -> None:
        cost_usd = decision.estimated_cost
        engine = _flux._engine
        engine._budget.record_spend(
            user_id=routing_request.user_id,
            amount=cost_usd,
            model_id=model_id,
            correlation_id=routing_request.correlation_id,
            task_type="unknown",
            plan=routing_request.plan or "free_plan",
        )
        if routing_request.max_daily_cost is not None:
            engine._daily_budget.record_spend(
                customer_id=routing_request.customer_id or routing_request.user_id,
                amount=cost_usd,
                model_id=model_id,
                correlation_id=routing_request.correlation_id,
                task_type="unknown",
            )
        if routing_request.run_id:
            engine._run_budget.record_step(routing_request.run_id, model_id, cost_usd, tokens)
        engine._attribution.record(
            tenant_id=routing_request.tenant_id,
            run_id=routing_request.run_id,
            task_type=decision.task_type,
            step_type=decision.step_type,
            model_id=model_id,
            cost_usd=cost_usd,
        )

    try:
        if model.provider.lower() in STREAMING_NATIVE_PROVIDERS:
            api_key = _flux._resolve_api_key(model, routing_request)
            yield _chunk({"role": "assistant"}, None)
            recorded = False
            async for line in stream_openai_compat_lines(model, routing_request, api_key):
                if not recorded:
                    _record_usage(routing_request.max_tokens_requested or 1024)
                    recorded = True
                text = line.decode("utf-8", errors="replace").strip()
                if not text or text == "data: [DONE]":
                    continue
                if text.startswith("data: "):
                    yield f"{text}\n\n".encode()
        else:
            text = await _flux._call_model(model, routing_request)
            _record_usage(max(len(text) // 4, 1))
            yield _chunk({"role": "assistant", "content": text}, None)
            yield _chunk({}, "stop")
    except (ProviderCallError, FluxAPIError) as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
    yield b"data: [DONE]\n\n"
