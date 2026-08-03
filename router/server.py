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

import hmac
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
    RUN_STORE_BACKEND,
    SERVER_MAX_BODY_BYTES,
    SERVER_REQUIRE_AUTH,
    SERVER_TOKENS,
    SERVER_WORKERS,
    TENANT_DAILY_CAP_USD,
    ServerTokenBinding,
)
from .errors import AuthenticationError, FluxAPIError
from .flux import Flux, make_flux
from .provider_caller import (
    STREAMING_NATIVE_PROVIDERS,
    ProviderCallError,
    compute_actual_cost,
    stream_openai_compat_lines,
)
from .provider_caller import _safe_usage_int as _safe_stream_usage_int
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

def _warn_if_unsafe_run_store(workers: int, run_store_backend: str) -> None:
    """Multiple workers + the process-local InMemoryRunStore is a silent
    under-enforcement bug (see run_budget.RedisRunStore's docstring): each
    worker only sees the run steps it personally handled. We still fall back
    to in-memory storage rather than refusing to start — better a loud
    warning than a hard failure over a budget-accuracy tradeoff."""
    if workers > 1 and run_store_backend != "redis":
        log.warning(
            "flux_multi_worker_no_redis_run_store",
            workers=workers,
            run_store_backend=run_store_backend,
            msg=(
                f"FLUX_SERVER_WORKERS={workers} but FLUX_RUN_STORE is not 'redis' — each "
                "worker enforces run-scoped budgets against only the steps it personally "
                "handles, silently under-counting a run's true cumulative spend. Falling "
                "back to in-memory run storage. Set FLUX_RUN_STORE=redis (+ FLUX_REDIS_URL) "
                "to share run state across workers."
            ),
        )


_warn_if_unsafe_run_store(SERVER_WORKERS, RUN_STORE_BACKEND)

app = FastAPI(title="Flux Router", version="1.0.0")

# 🔧 EXTENSION POINT: swap for a per-process pool / DI container if you need
# multiple Flux instances (e.g. per-tenant provider keys) behind one server.
_flux: Flux = make_flux()


def _tokens_equal(presented: str, expected: str) -> bool:
    """Constant-time bearer-token comparison.

    Compares UTF-8 bytes rather than str: hmac.compare_digest raises TypeError
    on a str containing non-ASCII, and header values reach us as arbitrary
    text, so `Authorization: Bearer ké` would otherwise be a 500 instead of a
    401. Encoding first keeps every rejection on the same 401 path.

    Note that compare_digest is only constant-time with respect to CONTENT, not
    length — it leaks whether the two differ in length. That's inherent to the
    primitive and not worth papering over: token length isn't the secret.
    """
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def _check_auth(authorization: str | None) -> ServerTokenBinding | None:
    """Validate the bearer token and return the ServerTokenBinding (tenant_id
    + plan) it's bound to, if any (FLUX_SERVER_TOKENS multi-tenant mode).
    None means either auth is disabled or the legacy single-shared-token mode
    is in effect, where the caller's self-declared tenant_id/user/plan fields
    are trusted as-is — see SECURITY_ARCHITECTURE.md."""
    if not SERVER_REQUIRE_AUTH:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if SERVER_TOKENS:
        # Constant-time lookup rather than SERVER_TOKENS.get(token): a dict
        # lookup finishes as soon as it knows the answer, so how long the
        # miss took is a (weak, but free to remove) signal about the token.
        # Every candidate is compared, and the loop deliberately does NOT
        # break on a hit — an early return would reintroduce exactly the
        # timing difference this is here to erase. SERVER_TOKENS is read at
        # call time so tests/operators can still swap it wholesale.
        matched: ServerTokenBinding | str | None = None
        for candidate, candidate_binding in SERVER_TOKENS.items():
            if _tokens_equal(token, candidate):
                matched = candidate_binding
        if matched is None:
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        # Accept a bare tenant_id string too (legacy shorthand / tests that
        # construct SERVER_TOKENS by hand) — keeps the pre-existing
        # "pro_plan" default rather than silently tightening it for callers
        # who never asked for per-tenant plan enforcement.
        if isinstance(matched, str):
            matched = ServerTokenBinding(tenant_id=matched)
        return matched
    # Same reasoning for the single-shared-token mode. `_SERVER_TOKEN` is None
    # only if SERVER_REQUIRE_AUTH was somehow set without either env var; guard
    # so this fails closed rather than raising TypeError inside compare_digest.
    if _SERVER_TOKEN is None or not _tokens_equal(token, _SERVER_TOKEN):
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
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise HTTPException(
                status_code=400, detail=f"messages[{i}] must be an object, got {type(m).__name__}"
            )
    # Bugfix: `m.get("content", "")` only supplies the default when the key
    # is absent — an explicit `"content": null` (or any other non-string
    # JSON type) survived as None/non-str here and crashed the "\n".join()
    # below with an uncaught TypeError (500) instead of a clean 400.
    system_parts: list[str] = []
    for i, m in enumerate(messages):
        if m.get("role") != "system":
            continue
        content = m.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"messages[{i}].content must be a string for role=system, "
                    f"got {type(content).__name__}"
                ),
            )
        system_parts.append(content)
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
    body: dict[str, Any], run_id: str, tenant_id: str | None = None, plan: str | None = None
) -> tuple[RoutingRequest, bool]:
    """Translate an OpenAI chat-completion request body into a RoutingRequest.

    Returns (request, is_literal_model) — is_literal_model is True when `model`
    named a concrete registered model rather than a flux-* routing directive.

    `plan` is never taken from the request body — the client cannot grant
    itself a higher budget tier. It comes from the bearer token's
    ServerTokenBinding (FLUX_SERVER_TOKENS multi-tenant mode) via the caller,
    when the operator has opted into per-tenant plan binding (the
    {"tenant_id":..., "plan":...} token form — see config.ServerTokenBinding).
    Bugfix: this used to be left unset entirely regardless of token, so an
    operator had no way to actually cap a tenant at free_plan through this
    API even if they wanted to. Defaults to "pro_plan" — the same default
    RoutingRequest always had — whenever no explicit plan binding is
    configured (auth disabled, legacy shared-token mode, or a legacy
    string-shorthand FLUX_SERVER_TOKENS entry), so existing deployments that
    haven't opted in see no behavior change.
    """
    if "messages" not in body or not isinstance(body["messages"], list):
        raise HTTPException(status_code=400, detail="'messages' is required and must be a list")

    model_field = body.get("model", "flux-auto")
    if not isinstance(model_field, str):
        # Bugfix: an unhashable JSON type here (list/dict) used to blow up
        # `model_field in _ROUTING_DIRECTIVES` with an uncaught TypeError
        # ("unhashable type") — an uncaught-500 instead of a 400.
        raise HTTPException(
            status_code=400, detail=f"'model' must be a string, got {type(model_field).__name__}"
        )
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
        "plan": plan or "pro_plan",
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
    if decision.run_id_missing:
        headers["x-flux-run-id-missing"] = "true"
    return headers


def _record_tenant_daily_spend(bound_tenant: str | None, cost_usd: float, model_id: str) -> None:
    """Record spend toward the tenant-level daily cap (see
    config.TENANT_DAILY_CAP_USD). No-op unless both the caller authenticated
    into a bound tenant AND the cap is configured — this tracker is
    independent of (and in addition to) request.max_daily_cost/customer_id,
    which stay keyed by the client-supplied, rotatable `user` field."""
    if bound_tenant is None or TENANT_DAILY_CAP_USD is None:
        return
    _flux._engine._daily_budget.record_spend(
        customer_id=bound_tenant,
        amount=cost_usd,
        model_id=model_id,
        correlation_id=str(uuid.uuid4()),
        task_type="unknown",
    )


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
    binding = _check_auth(authorization)
    if binding is not None:
        tenant_id = binding.tenant_id
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
                "usage_source": r.usage_source,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }
            for r in records
        ],
    }


@app.get("/metrics")
async def metrics(authorization: str | None = Header(default=None)) -> Response:
    """Task 7: Prometheus text-exposition metrics — flux_cost_usd_total,
    flux_run_steps, flux_budget_exceeded_total, labelled by tenant/model.

    In FLUX_SERVER_TOKENS multi-tenant mode, the bearer token's bound tenant
    restricts the rendered series to that tenant only — otherwise any
    authenticated caller could read every tenant's spend/usage. Outside that
    mode (auth disabled, or the legacy shared-token mode where tenant_id is
    self-declared and unverified) all series are rendered, same as before.
    """
    binding = _check_auth(authorization)
    body = _flux._engine._attribution.render_prometheus(
        tenant_filter=binding.tenant_id if binding is not None else None
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_flux_run_id: str | None = Header(default=None, alias="X-Flux-Run-Id"),
    x_flux_tenant_id: str | None = Header(default=None, alias="X-Flux-Tenant-Id"),
) -> Any:
    binding = _check_auth(authorization)
    bound_tenant = binding.tenant_id if binding is not None else None
    bound_plan = binding.plan if binding is not None else None
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

    # Every other budget in this proxy is keyed by the client-supplied `user`
    # field (RoutingRequest.user_id), which an authenticated caller can
    # rotate per-request to mint itself a fresh, untouched budget bucket —
    # see config.TENANT_DAILY_CAP_USD's docstring. This check is keyed by the
    # bearer token's bound tenant instead, which the caller cannot rotate.
    if bound_tenant is not None and TENANT_DAILY_CAP_USD is not None:
        daily_budget = _flux._engine._daily_budget
        if daily_budget.is_cap_exceeded(bound_tenant, TENANT_DAILY_CAP_USD):
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": (
                            f"Tenant '{bound_tenant}' has reached its daily spend cap "
                            f"(${TENANT_DAILY_CAP_USD:.2f})."
                        ),
                        "type": "tenant_daily_cap_exceeded",
                    }
                },
            )

    # Task 3: every request belongs to a run — X-Flux-Run-Id groups repeated
    # calls into one budgeted trajectory. Bugfix: a request with no header
    # used to silently get its own single-step run, which makes run-scoped
    # budget enforcement a no-op for it (a run of one step never crosses any
    # threshold) — indistinguishable from a caller who forgot to propagate
    # the header across their agent loop. Loud now: warn here, and surface
    # it on the response (decision.run_id_missing / x-flux-run-id-missing)
    # so callers can detect they aren't getting cross-step enforcement.
    run_id_missing = x_flux_run_id is None
    run_id = x_flux_run_id or str(uuid.uuid4())
    if run_id_missing:
        log.warning(
            "flux_run_id_auto_generated",
            run_id=run_id,
            tenant_id=tenant_id,
            reason="no X-Flux-Run-Id header on request; run-scoped budget "
            "enforcement will not span multiple steps for this call",
        )
    routing_request, _ = _build_routing_request(body, run_id, tenant_id, plan=bound_plan)
    stream = bool(body.get("stream", False))
    # Whether the CLIENT'S OWN request asked for the OpenAI stream_options
    # usage chunk — independent of whether WE ask the upstream provider for
    # it (provider_caller.py always does, for OpenAI, so billing can use
    # actual usage). If the client didn't ask, that extra chunk is stripped
    # before forwarding rather than surprising a client that never opted in.
    stream_options = body.get("stream_options")
    client_wants_usage = (
        isinstance(stream_options, dict) and stream_options.get("include_usage") is True
    )

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
    decision.run_id_missing = run_id_missing

    if decision.chosen_model is None:
        if routing_request.run_id:
            _flux._engine._run_budget.release_reservation(
                routing_request.run_id, tenant_id=routing_request.tenant_id
            )
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
            _stream_completion(
                routing_request,
                decision,
                completion_id,
                created,
                headers,
                bound_tenant,
                client_wants_usage,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    try:
        # Shared with Flux.complete(): retries the typed fallback chain on
        # transient errors, and records plan/daily-cap spend, run-budget steps,
        # and attribution — all of which this path used to skip by calling
        # _call_model() directly. `usage` carries what was ACTUALLY billed —
        # provider-reported tokens/cost when the provider returned them,
        # the pre-dispatch estimate otherwise. See flux.DispatchUsage.
        text, used_model, _fallback_used, _fallback_reason, usage = (
            await _flux._dispatch_with_fallback(decision, routing_request, max_retries=2)
        )
    except AuthenticationError as exc:
        # errors.AuthenticationError means the OPERATOR's upstream provider key
        # is missing/invalid/revoked — a server-side misconfiguration. Bugfix:
        # this used to surface as 401 with the provider's message attached
        # ({"detail": "HTTP 403 from google"}), which is exactly what
        # _check_auth() returns for a bad client bearer token. A caller holding
        # a perfectly valid token saw 401 and rotated its own credentials while
        # the real fault was on our side, and the detail string leaked which
        # upstream provider had been routed to. It's an upstream failure like
        # any other FluxAPIError below: 502, with the provider detail kept in
        # the logs rather than the response body.
        log.error(
            "provider_authentication_failed",
            cid=routing_request.correlation_id,
            model=model_id,
            error=str(exc),
            msg="upstream provider rejected our API key — check the provider "
            "credentials for this deployment",
        )
        raise HTTPException(
            status_code=502, detail="Upstream provider authentication failed"
        ) from exc
    except FluxAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # The dispatched model may differ from decision.chosen_model if a fallback
    # fired — reflect the model that actually served the request.
    model_id = used_model.model_id
    headers["x-flux-model"] = model_id
    headers["x-flux-usage-source"] = usage.usage_source
    if usage.usage_source == "provider":
        headers["x-flux-actual-cost-usd"] = f"{usage.cost_usd:.6f}"
    # Bugfix: this used to record decision.estimated_cost (the pre-dispatch
    # estimate) against the tenant daily cap regardless of what was actually
    # billed — now uses the same actual-when-available cost as every other
    # tracker (BudgetTracker, DailyBudgetTracker, attribution) above.
    _record_tenant_daily_spend(bound_tenant, usage.cost_usd, model_id)

    if usage.input_tokens is not None and usage.output_tokens is not None:
        usage_block = {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        }
    else:
        # Fallback: provider didn't report usage — same char-math estimate
        # as before.
        usage_block = {
            "prompt_tokens": max(len(routing_request.raw_prompt) // 4, 1),
            "completion_tokens": max(len(text) // 4, 1),
            "total_tokens": max((len(routing_request.raw_prompt) + len(text)) // 4, 1),
        }

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
        "usage": usage_block,
    }
    return JSONResponse(content=response_body, headers=headers)


async def _stream_completion(
    routing_request,
    decision,
    completion_id,
    created,
    headers,
    bound_tenant=None,
    client_wants_usage=False,
):
    """Yield SSE chunks for a chat completion.

    For providers that speak OpenAI-native SSE (openai/groq/mistral), each
    chunk is forwarded as it arrives — genuinely incremental, not buffered.
    For providers without a native OpenAI SSE format (anthropic/google), we
    fall back to a single synthesized chunk carrying the full response,
    since translating each provider's own event schema chunk-by-chunk is
    out of scope here. That synthesized path gets exact usage for free,
    since _flux._call_model() already returns a ProviderResult.

    For OpenAI specifically, provider_caller.py always requests
    stream_options.include_usage on the upstream call (regardless of
    whether the CLIENT asked us for it) so billing can use actual usage;
    the resulting usage chunk is only forwarded to the client if THEIR
    request also set stream_options.include_usage (client_wants_usage) —
    otherwise it's captured for billing and stripped before forwarding, so
    a client that never opted in doesn't see a surprise empty-choices chunk.

    Budget/attribution recording is deferred to stream end rather than fired
    on the first chunk with a guessed token count: a usage chunk seen along
    the way means the recorded amount is actual; none seen (Groq/Mistral,
    or an OpenAI stream that for some reason didn't include one) falls back
    to the pre-dispatch estimate, as before. A call that fails outright
    (ProviderCallError/FluxAPIError, e.g. a 429 from the provider) never
    produces any content, so it isn't charged either way. A client that
    disconnects mid-stream (after some content but before the stream would
    have ended naturally) still gets billed at the estimate in the `finally`
    below — the provider was already asked to generate those tokens — and
    the run-budget reservation is resolved by that recording rather than
    left dangling.
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

    def _record_usage(
        cost_usd: float,
        tokens: int,
        usage_source: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
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
            engine._run_budget.record_step(
                routing_request.run_id,
                model_id,
                cost_usd,
                tokens,
                tenant_id=routing_request.tenant_id,
            )
        engine._attribution.record(
            tenant_id=routing_request.tenant_id,
            run_id=routing_request.run_id,
            task_type=decision.task_type,
            step_type=decision.step_type,
            model_id=model_id,
            cost_usd=cost_usd,
            usage_source=usage_source,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        _record_tenant_daily_spend(bound_tenant, cost_usd, model_id)

    def _record_estimated() -> None:
        _record_usage(
            decision.estimated_cost,
            routing_request.max_tokens_requested or 1024,
            "estimated",
        )

    # check_before_dispatch() (Step 0 of route(), already run before this
    # generator starts) reserved one run-budget step slot for this request.
    # `recorded` tracks whether the reservation has been resolved — either
    # by _record_usage() (a bill was recorded) or by release_reservation()
    # in the except block below (the call failed outright, nothing to bill).
    # Exactly one of those two must happen; the `finally` below exists for
    # the third case neither of them covers: the client disconnecting mid-
    # stream before either one ran.
    recorded = False
    try:
        if model.provider.lower() in STREAMING_NATIVE_PROVIDERS:
            api_key = _flux._resolve_api_key(model, routing_request)
            yield _chunk({"role": "assistant"}, None)
            captured_usage: dict | None = None
            async for line in stream_openai_compat_lines(model, routing_request, api_key):
                text = line.decode("utf-8", errors="replace").strip()
                if not text or text == "data: [DONE]":
                    continue
                if not text.startswith("data: "):
                    continue
                try:
                    payload = json.loads(text[len("data: ") :])
                except json.JSONDecodeError:
                    payload = None
                usage_obj = payload.get("usage") if isinstance(payload, dict) else None
                if usage_obj:
                    captured_usage = usage_obj
                    if not client_wants_usage:
                        # Our own request to the provider asked for this
                        # (see provider_caller._open_openai_compat_stream),
                        # the client's didn't — capture it for billing below
                        # but don't forward the extra chunk.
                        continue
                yield f"{text}\n\n".encode()
            # Stream ended naturally (not a disconnect) — record now.
            input_tokens = output_tokens = None
            if captured_usage:
                input_tokens = _safe_stream_usage_int(captured_usage.get("prompt_tokens"))
                output_tokens = _safe_stream_usage_int(captured_usage.get("completion_tokens"))
            if input_tokens is not None and output_tokens is not None:
                _record_usage(
                    compute_actual_cost(model, input_tokens, output_tokens),
                    input_tokens + output_tokens,
                    "provider",
                    input_tokens,
                    output_tokens,
                )
            else:
                _record_estimated()
            recorded = True
        else:
            result = await _flux._call_model(model, routing_request)
            text = result.text
            if result.input_tokens is not None and result.output_tokens is not None:
                _record_usage(
                    compute_actual_cost(model, result.input_tokens, result.output_tokens),
                    result.input_tokens + result.output_tokens,
                    "provider",
                    result.input_tokens,
                    result.output_tokens,
                )
            else:
                _record_usage(decision.estimated_cost, max(len(text) // 4, 1), "estimated")
            recorded = True
            yield _chunk({"role": "assistant", "content": text}, None)
            yield _chunk({}, "stop")
    except (ProviderCallError, FluxAPIError) as exc:
        if not recorded and routing_request.run_id:
            _flux._engine._run_budget.release_reservation(
                routing_request.run_id, tenant_id=routing_request.tenant_id
            )
        recorded = True  # resolved via release, not a bill — see comment above
        # Same operator-vs-caller distinction as the non-streaming path above:
        # a rejected upstream API key is our misconfiguration, so it's logged
        # with the provider detail and reported generically here rather than
        # echoing "HTTP 403 from <provider>" into the client's stream.
        if isinstance(exc, AuthenticationError):
            log.error(
                "provider_authentication_failed",
                cid=routing_request.correlation_id,
                model=model_id,
                error=str(exc),
                msg="upstream provider rejected our API key — check the provider "
                "credentials for this deployment",
            )
            error_text = "Upstream provider authentication failed"
        else:
            error_text = str(exc)
        yield f"data: {json.dumps({'error': error_text})}\n\n".encode()
    finally:
        if not recorded:
            _record_estimated()
    yield b"data: [DONE]\n\n"
