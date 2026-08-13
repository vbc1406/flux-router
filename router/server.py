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

import asyncio
import hmac
import ipaddress
import json
import os
import time
import uuid
from collections.abc import MutableMapping
from importlib import resources
from typing import Any, AsyncIterator

import structlog

try:
    from fastapi import FastAPI, Header, HTTPException, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "router.server requires the 'server' extra. Install with: pip install flux-router[server]"
    ) from exc

from .config import (
    ATTRIBUTION_DB_PATH,
    ATTRIBUTION_USAGE_PAGE_MAX,
    BUDGET_LIMITS,
    BUDGET_STORE_BACKEND,
    DASHBOARD_ENABLED,
    DATA_DIR,
    RATE_LIMIT_BURST,
    RATE_LIMIT_RPM,
    RUN_DEGRADE_THRESHOLD,
    RUN_MAX_COST_USD,
    RUN_MAX_DURATION_SECONDS,
    RUN_MAX_STEPS,
    RUN_MAX_TOKENS,
    RUN_STORE_BACKEND,
    RUN_WARN_THRESHOLD,
    SERVER_ALLOWED_HOSTS,
    SERVER_HOST,
    SERVER_MAX_BODY_BYTES,
    SERVER_PORT,
    SERVER_REQUIRE_AUTH,
    SERVER_TOKENS,
    SERVER_WORKERS,
    TENANT_DAILY_CAP_USD,
    ServerTokenBinding,
)
from .errors import AuthenticationError, BudgetExceededError, FluxAPIError, UnsupportedFeatureError
from .flux import _PROVIDER_ENV_VARS, Flux, _routing_telemetry, make_flux
from .provider_caller import (
    STREAMING_NATIVE_PROVIDERS,
    ProviderCallError,
    compute_actual_cost,
    stream_openai_compat_lines,
)
from .provider_caller import _safe_usage_int as _safe_stream_usage_int
from .rate_limit import RateLimiter
from .run_budget import RunBudgetExceeded
from .schemas import RoutingDecision, RoutingRequest

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

def _warn_if_unsafe_budget_store(workers: int, budget_store_backend: str) -> None:
    """Multiple workers + the process-local BudgetTracker is a silent
    over-spend bug (see budget_tracker.RedisBudgetTracker's docstring): each
    worker enforces daily/monthly plan budgets against only the spend IT
    personally handled, so the effective ceiling is roughly workers x the
    configured plan limit. Distinct from _warn_if_unsafe_run_store below —
    plan-level budgets (budget_tracker.py) and run-scoped agent budgets
    (run_budget.py) are independent systems with independent backends. We
    still fall back to in-memory storage rather than refusing to start —
    better a loud warning than a hard failure over a budget-accuracy
    tradeoff."""
    if workers > 1 and budget_store_backend != "redis":
        log.warning(
            "flux_multi_worker_no_redis_budget_store",
            workers=workers,
            budget_store_backend=budget_store_backend,
            msg=(
                f"FLUX_SERVER_WORKERS={workers} but FLUX_BUDGET_STORE is not 'redis' — each "
                "worker enforces daily/monthly plan budgets against only the spend it "
                "personally handles, so the effective budget ceiling is roughly "
                f"{workers}x the configured plan limit. Falling back to in-memory budget "
                "storage. Set FLUX_BUDGET_STORE=redis (+ FLUX_REDIS_URL) to share budget "
                "state across workers."
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


def _warn_if_rate_limit_is_per_worker(workers: int, rpm: int) -> None:
    """Buckets are process-local (see rate_limit.RateLimiter), so N workers
    enforce roughly N x the configured rate globally. Same class of caveat as
    the in-memory run store above, and warned about on the same terms."""
    if workers > 1 and rpm > 0:
        log.warning(
            "flux_rate_limit_per_worker",
            workers=workers,
            rate_limit_rpm=rpm,
            effective_global_rpm=workers * rpm,
            msg=(
                f"FLUX_SERVER_WORKERS={workers} with FLUX_RATE_LIMIT_RPM={rpm} — rate-limit "
                f"buckets are process-local, so the effective global ceiling is about "
                f"{workers * rpm}/min, not {rpm}/min. Divide the setting by the worker "
                "count, or enforce the limit at your ingress instead."
            ),
        )


_warn_if_unsafe_run_store(SERVER_WORKERS, RUN_STORE_BACKEND)
_warn_if_unsafe_budget_store(SERVER_WORKERS, BUDGET_STORE_BACKEND)
_warn_if_rate_limit_is_per_worker(SERVER_WORKERS, RATE_LIMIT_RPM)

app = FastAPI(title="Flux Router", version="1.0.0")

# 🔧 EXTENSION POINT: swap for a Redis-backed limiter to get an exact global
# ceiling across workers, the same way FLUX_RUN_STORE=redis does for run state.
_rate_limiter = RateLimiter()

# 🔧 EXTENSION POINT: swap for a per-process pool / DI container if you need
# multiple Flux instances (e.g. per-tenant provider keys) behind one server.
_flux: Flux = make_flux()


def _rate_limit_key(request: Request, bound_tenant: str | None) -> str:
    """Pick the bucket a request counts against.

    The bound tenant when there is one: it comes from the bearer token map, so
    a caller cannot rotate it to mint itself a fresh bucket. This is exactly
    why the limiter is NOT keyed on the `user` field or X-Flux-Tenant-Id — both
    are self-declared, and a limiter keyed on either is one a caller escapes by
    incrementing a counter (the same rotation hole TENANT_DAILY_CAP_USD exists
    to close for spend).

    Otherwise the peer IP. That's weaker — it's shared by everything behind one
    NAT, and it's absent for ASGI transports with no client info — but the
    unauthenticated case is localhost-only by default (see SERVER_HOST), so the
    fallback is a backstop rather than the main line of defence. Requests with
    no identifiable peer share one bucket rather than going unlimited.

    Deliberately NOT read from X-Forwarded-For: that header is caller-supplied
    and trivially spoofed, so trusting it here would hand every client an
    unlimited supply of fresh buckets. A deployment behind a trusted proxy
    should rate limit at that proxy.
    """
    if bound_tenant is not None:
        return f"tenant:{bound_tenant}"
    client = request.client
    if client is not None and client.host:
        return f"ip:{client.host}"
    return "anonymous"


def _enforce_rate_limit(request: Request, bound_tenant: str | None) -> None:
    """Raise 429 if the caller is over its per-key request rate."""
    key = _rate_limit_key(request, bound_tenant)
    retry_after = _rate_limiter.check(key)
    if retry_after is None:
        return
    log.warning("flux_rate_limited", key=key, retry_after_s=round(retry_after, 2))
    raise HTTPException(
        status_code=429,
        detail=f"Rate limit exceeded ({RATE_LIMIT_RPM} requests/minute). Retry shortly.",
        headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
    )


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
    None means authentication is disabled. A legacy single shared token is
    bound to one immutable synthetic tenant so callers cannot forge tenant
    identities, enumerate another tenant's usage, or rotate budget buckets.
    """
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
    return ServerTokenBinding(tenant_id="shared-token", plan="pro_plan")


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
) -> tuple[str | None, str, list[dict[str, Any]]]:
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
    history = []
    for m in non_system[:-1]:
        turn: dict[str, Any] = {"role": m.get("role", "user"), "content": m.get("content", "")}
        # Preserve tool-calling linkage (Item 1): an assistant turn that made
        # tool calls, and a tool-role turn answering one, both need these
        # fields to survive into message_history or the provider-side
        # conversation loses which call a tool result answers.
        if m.get("tool_calls"):
            turn["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            turn["tool_call_id"] = m["tool_call_id"]
        if m.get("name"):
            turn["name"] = m["name"]
        history.append(turn)
    return system_prompt, raw_prompt, history


def _build_routing_request(
    body: dict[str, Any],
    run_id: str,
    tenant_id: str | None = None,
    plan: str | None = None,
    step_type: str | None = None,
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

    `step_type` (Item 3) comes from the X-Flux-Step-Type header. An
    unrecognized value is rejected by RoutingRequest's own Literal validator
    below (ValueError -> 400), same as any other invalid field — no separate
    header-specific validation needed. None means "no header sent";
    RoutingRequest.step_type stays unset and RequestClassifier infers it as
    before, so absence of the header is a strict no-op.
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
        # Task 6 / Item 1: used for step_type inference + capability filtering
        # AND forwarded to the provider (see provider_caller.py's per-provider
        # translation). tool_choice is only meaningful when tools is set.
        "tools": body.get("tools") or [],
        "tool_choice": body.get("tool_choice"),
        "response_format": body.get("response_format"),
    }
    if step_type is not None:
        kwargs["step_type"] = step_type

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


def _flux_headers(decision: RoutingDecision, decision_latency_ms: float) -> dict[str, str]:
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


def _refuse_remote_spend_data(request: Request) -> None:
    """Withhold spend/config data from a non-loopback caller when no auth is set.

    Same rule, and the same reasoning, as the dashboard: this data is every
    tenant's spend and the deployment's configuration, which is fine for the
    single-operator case and not fine once the port is reachable from
    elsewhere. Gating the dashboard alone was theatre — /dashboard returned
    404 while /v1/stats/summary handed the identical numbers to anyone on the
    network, and the page is only a renderer for these endpoints.

    Loopback keeps working, so the local operator and a same-host reverse proxy
    are unaffected. To read any of this from another machine, configure
    FLUX_SERVER_TOKEN (or FLUX_SERVER_TOKENS) — including for a Prometheus
    scrape of /metrics, which is the one legitimate remote reader likely to
    notice this.
    """
    if SERVER_REQUIRE_AUTH:
        return
    client = request.client
    # No peer address means it cannot be shown to be local — refuse rather
    # than fall through to _is_loopback(""), which is True for a bind address.
    peer = client.host if client is not None and client.host else None
    host_header = request.headers.get("host")
    # Both must hold: the connection came from this box AND it was addressed to
    # this box. The second is what a rebound DNS name fails — see
    # _is_local_host_header().
    if peer is None or not _is_loopback(peer) or not _is_local_host_header(host_header):
        log.warning(
            "flux_remote_spend_data_refused",
            peer=peer,
            host_header=host_header,
            path=request.url.path,
            msg=(
                "Refused a non-local request for spend/configuration data. The server "
                "is reachable off-box with no FLUX_SERVER_TOKEN set. Set a token to allow "
                "remote reads, or FLUX_ALLOWED_HOSTS if a same-host proxy passes its own "
                "hostname through."
            ),
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "Spend and configuration data is served to loopback clients only when no "
                "authentication is configured. Set FLUX_SERVER_TOKEN to read it remotely."
            ),
        )


@app.get("/v1/usage")
async def get_usage(
    request: Request,
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
    _refuse_remote_spend_data(request)
    binding = _check_auth(authorization)
    if binding is not None:
        tenant_id = binding.tenant_id
    limit = max(1, min(limit, ATTRIBUTION_USAGE_PAGE_MAX))
    offset = max(0, offset)
    records, total = await asyncio.to_thread(
        _flux._engine._attribution.usage,
        tenant_id=tenant_id,
        run_id=run_id,
        limit=limit,
        offset=offset,
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
                # Routing telemetry. Persisted since the dashboard work but
                # never projected here, so the console's activity feed had no
                # latency to show. Additive: older rows read back as null.
                "latency_ms": r.latency_ms,
                "cache_hit": r.cache_hit,
                "fallback_used": r.fallback_used,
            }
            for r in records
        ],
    }


# Named windows accepted by the /v1/stats endpoints, in seconds. A closed set
# rather than a free-form duration string: these values reach a SQL comparison,
# and an allowlist means the parsing can't be the weak point. "all" means no
# lower bound on timestamp.
_STATS_WINDOWS: dict[str, float | None] = {
    "1h": 3600.0,
    "24h": 86400.0,
    "7d": 604800.0,
    "30d": 2592000.0,
    "all": None,
}

# Bucket width per window, chosen so a chart has a useful number of points
# without the caller having to pick one.
_STATS_BUCKETS: dict[str, int] = {
    "1h": 300,  # 5 min
    "24h": 3600,  # 1 hour
    "7d": 21600,  # 6 hours
    "30d": 86400,  # 1 day
    "all": 86400,
}


def _stats_since(window: str) -> float | None:
    """Resolve a named window to a lower-bound timestamp, or None for 'all'."""
    if window not in _STATS_WINDOWS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown window '{window}'. Valid: {', '.join(_STATS_WINDOWS)}",
        )
    span = _STATS_WINDOWS[window]
    return None if span is None else time.time() - span


def _stats_scope(request: Request, authorization: str | None) -> str | None:
    """Auth + tenant scoping shared by every /v1/stats endpoint.

    Same rule as /v1/usage and /metrics: in FLUX_SERVER_TOKENS multi-tenant
    mode the bearer token's bound tenant is the only data the caller can see.
    Outside that mode there is no verified tenant identity, so the stats span
    every tenant — which is exactly what the single-operator, self-hosted
    dashboard wants.
    """
    _refuse_remote_spend_data(request)
    binding = _check_auth(authorization)
    if not _flux._engine._attribution.stats_available:
        raise HTTPException(
            status_code=501,
            detail=(
                "The configured usage store does not support aggregate stats. "
                "These endpoints require the built-in SqliteUsageStore."
            ),
        )
    return binding.tenant_id if binding is not None else None


@app.get("/v1/stats/summary")
async def stats_summary(
    request: Request, authorization: str | None = Header(default=None), window: str = "24h"
) -> dict[str, Any]:
    """Headline spend/savings/latency numbers for the selected window."""
    tenant_id = _stats_scope(request, authorization)
    return {
        "window": window,
        **await asyncio.to_thread(
            _flux._engine._attribution.stats_summary,
            since=_stats_since(window),
            tenant_id=tenant_id,
        ),
    }


@app.get("/v1/stats/timeseries")
async def stats_timeseries(
    request: Request,
    authorization: str | None = Header(default=None),
    window: str = "24h",
    bucket_seconds: int | None = None,
) -> dict[str, Any]:
    """Cost and request count per time bucket, oldest first."""
    tenant_id = _stats_scope(request, authorization)
    since = _stats_since(window)
    # Clamp rather than reject: an absurd bucket width is a caller mistake,
    # not an attack, and one-second buckets over 30 days is a lot of JSON.
    bucket = bucket_seconds or _STATS_BUCKETS[window]
    bucket = max(60, min(int(bucket), 86400))
    return {
        "window": window,
        "bucket_seconds": bucket,
        "data": await asyncio.to_thread(
            _flux._engine._attribution.stats_timeseries,
            since=since,
            bucket_seconds=bucket,
            tenant_id=tenant_id,
        ),
    }


@app.get("/v1/stats/models")
async def stats_models(
    request: Request, authorization: str | None = Header(default=None), window: str = "24h"
) -> dict[str, Any]:
    """Per-model traffic, cost, and latency for the selected window."""
    tenant_id = _stats_scope(request, authorization)
    return {
        "window": window,
        "data": await asyncio.to_thread(
            _flux._engine._attribution.stats_by_model,
            since=_stats_since(window),
            tenant_id=tenant_id,
        ),
    }


@app.get("/v1/stats/tasks")
async def stats_tasks(
    request: Request, authorization: str | None = Header(default=None), window: str = "24h"
) -> dict[str, Any]:
    """Per-task-type traffic and cost for the selected window."""
    tenant_id = _stats_scope(request, authorization)
    return {
        "window": window,
        "data": await asyncio.to_thread(
            _flux._engine._attribution.stats_by_task_type,
            since=_stats_since(window),
            tenant_id=tenant_id,
        ),
    }


@app.get("/v1/stats/tenants")
async def stats_tenants(
    request: Request, authorization: str | None = Header(default=None), window: str = "24h"
) -> dict[str, Any]:
    """Per-tenant spend, savings, and traffic for the selected window.

    Scoped exactly like every other stats endpoint: in FLUX_SERVER_TOKENS mode
    _stats_scope() returns the bearer token's bound tenant and this collapses
    to that tenant's single row, so it can't be used to enumerate other
    tenants' spend.
    """
    tenant_id = _stats_scope(request, authorization)
    return {
        "window": window,
        "data": await asyncio.to_thread(
            _flux._engine._attribution.stats_by_tenant,
            since=_stats_since(window),
            tenant_id=tenant_id,
        ),
    }


@app.get("/v1/stats/registry")
async def stats_registry(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """The model registry with pricing and capabilities.

    Deliberately separate from GET /v1/models, which is OpenAI-shaped
    ({id, object, owned_by}) and must stay that way for client compatibility.
    """
    _refuse_remote_spend_data(request)
    _check_auth(authorization)
    registry = _flux._engine._registry
    return {
        "data": [
            {
                "model_id": m.model_id,
                "provider": m.provider,
                "display_name": m.display_name,
                "tier": m.tier,
                "cost_per_1k_input": m.cost_per_1k_input,
                "cost_per_1k_output": m.cost_per_1k_output,
                "max_context_window": m.max_context_window,
                "max_output_tokens": m.max_output_tokens,
                "capabilities": list(m.capabilities),
                "supports_streaming": m.supports_streaming,
                "supports_tools": m.supports_tools,
                "avg_latency_ms": m.avg_latency_ms,
                "rate_limit_rpm": m.rate_limit_rpm,
                "current_load_rpm": m.current_load_rpm,
                "is_available": m.is_available,
            }
            for m in registry.all_available_models()
        ]
    }


@app.get("/v1/stats/config")
async def stats_config(
    request: Request, authorization: str | None = Header(default=None)
) -> dict[str, Any]:
    """The deployment's effective configuration, read-only.

    SECURITY: this endpoint must never echo a secret. No bearer token, no
    provider API key, and no Redis URL (which can embed credentials) appears
    below — auth is reported as a mode name, and providers as a configured
    yes/no. Anything added here later must clear the same bar; see
    test_stats.py, which asserts no configured secret's value appears in the
    response body.

    Read-only by design. Every value comes from a module-level constant read
    from the environment at import time, so there is nothing to write back to;
    making budgets runtime-mutable needs a config file and a reload path,
    which is its own piece of work rather than a dashboard side effect.
    """
    _refuse_remote_spend_data(request)
    _check_auth(authorization)
    if SERVER_TOKENS:
        auth_mode = "bound-tokens"
    elif SERVER_REQUIRE_AUTH:
        auth_mode = "shared-token"
    else:
        auth_mode = "none"
    return {
        "server": {
            "host": SERVER_HOST,
            "port": SERVER_PORT,
            "workers": SERVER_WORKERS,
            "auth_mode": auth_mode,
            "tenant_count": len({b.tenant_id for b in SERVER_TOKENS.values()}),
            "max_body_bytes": SERVER_MAX_BODY_BYTES,
            "run_store_backend": RUN_STORE_BACKEND,
            "budget_store_backend": BUDGET_STORE_BACKEND,
            "data_dir": DATA_DIR,
            "usage_db": ATTRIBUTION_DB_PATH,
            "usage_db_persistent": ATTRIBUTION_DB_PATH != ":memory:",
        },
        "providers": {
            provider: bool(_flux._provider_keys.get(provider) or _flux._api_key)
            for provider in sorted(_PROVIDER_ENV_VARS)
        },
        "budgets": {
            "plans": BUDGET_LIMITS,
            "tenant_daily_cap_usd": TENANT_DAILY_CAP_USD,
            "env_var": "FLUX_TENANT_DAILY_CAP_USD",
        },
        "run_limits": {
            "max_cost_usd": RUN_MAX_COST_USD,
            "max_steps": RUN_MAX_STEPS,
            "max_tokens": RUN_MAX_TOKENS,
            "max_duration_seconds": RUN_MAX_DURATION_SECONDS,
            "degrade_threshold": RUN_DEGRADE_THRESHOLD,
            "warn_threshold": RUN_WARN_THRESHOLD,
        },
        "rate_limit": {
            "rpm": RATE_LIMIT_RPM,
            "burst": RATE_LIMIT_BURST,
            "env_vars": ["FLUX_RATE_LIMIT_RPM", "FLUX_RATE_LIMIT_BURST"],
        },
    }


@app.get("/metrics")
async def metrics(
    request: Request, authorization: str | None = Header(default=None)
) -> Response:
    """Task 7: Prometheus text-exposition metrics — flux_cost_usd_total,
    flux_run_steps, flux_budget_exceeded_total, labelled by tenant/model.

    In FLUX_SERVER_TOKENS multi-tenant mode, the bearer token's bound tenant
    restricts the rendered series to that tenant only — otherwise any
    authenticated caller could read every tenant's spend/usage. Outside that
    mode (auth disabled, or the legacy shared-token mode where tenant_id is
    self-declared and unverified) all series are rendered, same as before.

    Scraping this from another host is the one legitimate remote read of spend
    data, and it needs FLUX_SERVER_TOKEN — without auth configured the series
    are served to loopback only, like every other spend endpoint.
    """
    _refuse_remote_spend_data(request)
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
    x_flux_step_type: str | None = Header(default=None, alias="X-Flux-Step-Type"),
) -> Any:
    binding = _check_auth(authorization)
    bound_tenant = binding.tenant_id if binding is not None else None
    bound_plan = binding.plan if binding is not None else None
    # Before the body is read, let alone routed or dispatched: a rejected
    # request should cost as little as possible, and reading the body first
    # would mean a flood still gets to push SERVER_MAX_BODY_BYTES at us per
    # request. After _check_auth so that bound_tenant is available as the key.
    _enforce_rate_limit(request, bound_tenant)
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
    routing_request, _ = _build_routing_request(
        body, run_id, tenant_id, plan=bound_plan, step_type=x_flux_step_type
    )
    if binding is not None:
        # Budget and rate-limit identity must come from authenticated context,
        # never the caller-controlled OpenAI `user` field.
        routing_request.user_id = binding.tenant_id
        routing_request.customer_id = binding.tenant_id
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
    # Stamp it on the decision so the dispatch path can persist it against the
    # usage row — it used to be measured here, put in a response header, and
    # then thrown away, so the dashboard had no way to show routing overhead.
    decision.decision_latency_ms = decision_latency_ms
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
        text, used_model, _fallback_used, _fallback_reason, usage, tool_calls, finish_reason = (
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
    except UnsupportedFeatureError as exc:
        # The request asked for a provider/feature combination Flux can't
        # bridge (e.g. response_format routed to Anthropic) — a caller error,
        # not an upstream failure, so 400 rather than the 502 below.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
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

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    response_body = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage_block,
    }
    return JSONResponse(content=response_body, headers=headers)


def _is_provider_auth_failure(exc: BaseException) -> bool:
    """True for upstream provider auth failures, however they reach us.

    Flux._call_model() translates a ProviderCallError with status_code in
    (401, 403) into AuthenticationError before it ever reaches server.py —
    but _stream_completion()'s native-streaming branch calls
    stream_openai_compat_lines() directly, bypassing that translation, so a
    raw ProviderCallError(401/403) can also show up here. Both shapes must be
    recognized so the streaming and non-streaming client-visible errors stay
    equivalent (see the non-streaming AuthenticationError handler above).
    """
    return isinstance(exc, AuthenticationError) or (
        isinstance(exc, ProviderCallError) and exc.status_code in (401, 403)
    )


async def _stream_completion(
    routing_request: RoutingRequest,
    decision: RoutingDecision,
    completion_id: str,
    created: int,
    headers: dict[str, str],
    bound_tenant: str | None = None,
    client_wants_usage: bool = False,
) -> AsyncIterator[bytes]:
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
    # The caller (chat_completions) already returns a 502 before ever
    # invoking this generator when decision.chosen_model is None — this
    # narrows that same invariant for the type checker.
    assert decision.chosen_model is not None
    model = decision.chosen_model
    model_id = model.model_id
    reservation_id = _flux._engine._budget.reserve_spend(
        routing_request.user_id,
        decision.estimated_cost,
        routing_request.plan or "free_plan",
    )
    if reservation_id is None:
        if routing_request.run_id:
            _flux._engine._run_budget.release_reservation(
                routing_request.run_id, tenant_id=routing_request.tenant_id
            )
        yield f"data: {json.dumps({'error': 'Budget exceeded'})}\n\n".encode()
        yield b"data: [DONE]\n\n"
        return

    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> bytes:
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
        engine._budget.reconcile_spend(
            reservation_id,
            cost_usd,
            model_id=model_id,
            correlation_id=routing_request.correlation_id,
            task_type="unknown",
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
            # On a stream, "latency" is the whole time spent producing the
            # response, not time-to-first-token — this is called once the
            # stream has finished, which is also the only point at which
            # actual usage is known. The streaming path has no fallback
            # chain (it dispatches the chosen model directly), so
            # fallback_used is always False here.
            **_routing_telemetry(
                decision,
                routing_request,
                latency_ms=(time.monotonic() - stream_started) * 1000,
                fallback_used=False,
            ),
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
    stream_started = time.monotonic()
    try:
        if model.provider.lower() in STREAMING_NATIVE_PROVIDERS:
            api_key = _flux._resolve_api_key(model, routing_request)
            yield _chunk({"role": "assistant"}, None)
            captured_usage: dict[str, Any] | None = None
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
            _flux._engine._circuit_breaker.record_success(model.provider)
            _flux._engine.record_prompt_cache_success(routing_request, model)
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
            _flux._engine._circuit_breaker.record_success(model.provider)
            _flux._engine.record_prompt_cache_success(routing_request, model)
            yield _chunk({"role": "assistant", "content": text}, None)
            yield _chunk({}, "stop")
    except (ProviderCallError, FluxAPIError) as exc:
        _flux._engine._budget.release_reservation(reservation_id)
        _flux._engine._circuit_breaker.record_failure(model.provider)
        if not recorded and routing_request.run_id:
            _flux._engine._run_budget.release_reservation(
                routing_request.run_id, tenant_id=routing_request.tenant_id
            )
        recorded = True  # resolved via release, not a bill — see comment above
        # Same operator-vs-caller distinction as the non-streaming path above:
        # a rejected upstream API key is our misconfiguration, so it's logged
        # with the provider detail and reported generically here rather than
        # echoing "HTTP 403 from <provider>" into the client's stream.
        if _is_provider_auth_failure(exc):
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


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL OPERATOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════


def _dashboard_dir() -> str | None:
    """Locate the bundled dashboard assets.

    Resolved through importlib.resources rather than __file__-relative paths so
    it works the same from an installed wheel as from a source checkout.
    Returns None if the assets aren't present (a trimmed install), which the
    mount below treats as "no dashboard" rather than an error.
    """
    try:
        path = resources.files("router").joinpath("dashboard")
        index = path.joinpath("index.html")
        if index.is_file():
            return str(path)
    except (ModuleNotFoundError, AttributeError, TypeError):  # pragma: no cover
        pass
    return None


def _dashboard_refusal_reason(host: str, require_auth: bool, enabled: bool) -> str | None:
    """Why the dashboard must not be served, or None if it's safe to mount.

    The dashboard shows every tenant's spend and the deployment's configuration
    to whoever can load the page. That is exactly right for the single-operator
    self-hosted case it's built for — the operator owns the box — and exactly
    wrong the moment the port is reachable from elsewhere with no token
    required. So: loopback binding is fine unauthenticated, any other bind
    needs auth configured, and FLUX_DASHBOARD=0 turns it off outright.

    Note this checks the CONFIGURED bind address, which is what `flux serve`
    and the Makefile use. A reverse proxy that forwards to a loopback-bound
    server is deliberately out of scope for this check — that deployment has
    already taken responsibility for its own edge, and should set a token.
    """
    if not enabled:
        return "FLUX_DASHBOARD=0"
    if not _is_loopback(host) and not require_auth:
        return (
            f"the server is bound to {host} (not loopback) and no FLUX_SERVER_TOKEN or "
            "FLUX_SERVER_TOKENS is set, so the dashboard would expose every tenant's "
            "spend and this deployment's configuration to anything that can reach the "
            "port. Set a token, or bind to 127.0.0.1"
        )
    return None


def _is_loopback(host: str) -> bool:
    """Whether a bind address only accepts connections from this machine."""
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        # A hostname we can't classify without resolving it. Treat as
        # non-loopback: the failure mode of being too cautious is a logged
        # warning, the other way round is exposing spend data.
        return False


def _is_local_host_header(host_header: str | None) -> bool:
    """Whether a request's Host header names this machine.

    The peer-address check answers "did this connection come from this box",
    which is not the same question as "did something on this box mean to send
    it". A page on evil.example the operator visits can lower its TTL, rebind
    evil.example to 127.0.0.1, and fetch http://evil.example:8000/v1/stats/*
    from the operator's own browser: the peer address is loopback, the guard
    passes, and because the page's origin IS that host:port the responses come
    back same-origin and fully readable. No CORS header is involved, so
    withholding one prevents nothing.

    The Host header is what separates the two cases, because the browser puts
    the attacker's hostname in it and cannot be talked out of that. Requiring
    it to be a loopback name closes the rebinding path; FLUX_ALLOWED_HOSTS
    reopens exactly the names an operator names.
    """
    if not host_header:
        # HTTP/1.1 requires Host. Absent means we cannot tell, so refuse.
        return False
    host = host_header.strip().lower()
    # Strip the port. IPv6 literals are bracketed ("[::1]:8000"), so split
    # after the bracket; otherwise a bare "::1" would split on its own colons.
    if host.startswith("["):
        host = host.partition("]")[0].lstrip("[")
    elif host.count(":") == 1:
        host = host.partition(":")[0]
    if host in SERVER_ALLOWED_HOSTS or host_header.strip().lower() in SERVER_ALLOWED_HOSTS:
        return True
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _LoopbackOnlyDashboard:
    """Wraps the dashboard's StaticFiles app in a request-time origin check.

    _dashboard_refusal_reason() reads the CONFIGURED bind address, which is
    only the truth when the bind address came from the environment. Pass it on
    the command line instead — `uvicorn router.server:app --host 0.0.0.0`, as
    the Dockerfile did — and config still reports 127.0.0.1, the mount is
    allowed, and the dashboard is served to the whole network unauthenticated.

    So the mount decision is not the only line of defence: every request is
    also checked against the peer address it actually arrived from. That holds
    however the server was started. A reverse proxy on the same host still
    reaches the dashboard (its peer address is loopback), which is the
    documented stance — that deployment owns its own edge and should set a
    token.

    Only engaged when no auth is configured; with a token set, the token is
    the control and remote access is intentional.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(
        self, scope: MutableMapping[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] == "http" and not SERVER_REQUIRE_AUTH:
            client = scope.get("client")
            # No peer address (some transports, e.g. a unix socket) means it
            # cannot be shown to be local. _is_loopback("") is True — that is
            # correct for a BIND address, where empty means "unset", and wrong
            # for a peer, so the absent case is refused here rather than
            # falling through to it.
            peer = client[0] if client else None
            # The Host header is checked too: a loopback peer is also what a
            # DNS-rebinding page gets when it drives the operator's own
            # browser. See _is_local_host_header().
            host_header = next(
                (
                    v.decode("latin-1")
                    for k, v in scope.get("headers", ())
                    if k == b"host"
                ),
                None,
            )
            if peer is None or not _is_loopback(peer) or not _is_local_host_header(host_header):
                log.warning(
                    "flux_dashboard_remote_request_refused",
                    peer=peer,
                    host_header=host_header,
                    msg=(
                        "Refused a non-local request for the dashboard. The server is "
                        "reachable off-box with no FLUX_SERVER_TOKEN set — serving it would "
                        "expose every tenant's spend and this deployment's configuration."
                    ),
                )
                response = JSONResponse(
                    {
                        "detail": (
                            "The dashboard is only served to loopback clients when no "
                            "authentication is configured. Set FLUX_SERVER_TOKEN, or reach "
                            "it from the host itself."
                        )
                    },
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


def _mount_dashboard() -> bool:
    """Mount the dashboard at /dashboard unless it would be unsafe to.

    Refusing is never fatal — the proxy and its API keep serving normally, and
    the reason is logged loudly enough to act on.
    """
    reason = _dashboard_refusal_reason(SERVER_HOST, SERVER_REQUIRE_AUTH, DASHBOARD_ENABLED)
    if reason is not None:
        log.warning(
            "flux_dashboard_not_mounted",
            host=SERVER_HOST,
            msg=f"Dashboard is not being served: {reason}.",
        )
        return False

    directory = _dashboard_dir()
    if directory is None:  # pragma: no cover - only on a trimmed install
        log.warning(
            "flux_dashboard_assets_missing",
            msg="Dashboard assets were not found in the installed package; skipping mount.",
        )
        return False

    app.mount(
        "/dashboard",
        _LoopbackOnlyDashboard(StaticFiles(directory=directory, html=True)),
        name="dashboard",
    )
    log.info(
        "flux_dashboard_mounted",
        path="/dashboard",
        auth_required=SERVER_REQUIRE_AUTH,
        msg=f"Dashboard available at http://{SERVER_HOST}:{SERVER_PORT}/dashboard",
    )
    return True


DASHBOARD_MOUNTED = _mount_dashboard()
