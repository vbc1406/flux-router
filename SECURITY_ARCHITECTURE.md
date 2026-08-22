# Flux Security Architecture

This document explains exactly what data Flux handles, where it stays, and where it goes.

If you're evaluating Flux for your stack and want to verify our claims, see the "How to Verify" section at the bottom — every guarantee here can be checked by reading the code.

---

## TL;DR

- **Flux control-plane data stays in your infrastructure by default.** Routing
  policy, budgets, attribution, and operational telemetry are processed locally.
- **Model traffic goes only to providers you configure.** Provider credentials,
  prompts, and conversation context are sent directly from Flux to the selected
  provider over HTTPS; responses return directly to Flux.
- **No Flux-hosted telemetry by default.** Flux does not send routing telemetry
  to a Flux service unless you explicitly configure an external destination.

---

## Data Flow

| Data Type | Where It Lives | Leaves Your Infrastructure? |
|-----------|---------------|----------------------------|
| Provider API keys (OpenAI, Anthropic, etc.) | Your environment and request headers in memory | Sent only to the selected configured provider |
| User prompts | Your application and Flux process memory | Sent to the selected configured provider |
| Model responses | Selected provider, then Flux/application memory | Returned from the selected configured provider |
| Conversation history | Your application and Flux process memory | Sent when included in the selected provider request |
| Customer/user IDs | Your application memory | Never (default) |
| Routing decisions | Local logs | Never (default) |
| Cost estimates | Local logs | Never (default) |
| Latency measurements | Local logs | Never (default) |
| Quality scores | Local memory | Never (default) |

---

## Code-Level Guarantees

### API Keys Are Treated as Secrets

In `router/schemas.py`, `provider_api_key` is declared as a Pydantic field with `repr=False, exclude=True`. This means:

- Keys are excluded from `.json()` serialization
- Keys are excluded from `repr()` output
- Keys do not appear in error messages or stack traces from Pydantic

### Error Messages Don't Leak Provider Responses

In `router/provider_caller.py`, when a provider API call fails, the error message contains only `HTTP {code} from {provider_name}` — no URL paths, no response body, no headers.

If you need response bodies for debugging, they are written to a separate `log.debug("provider_error_body", ...)` call. Production deployments running at INFO/WARN log level never see provider response bodies.

### Prompts Are Never Logged at INFO/WARN Level

The router logs only metadata: token counts, task types, model decisions, latencies. Raw prompts and responses are never passed to log statements at INFO or WARN level.

A `LOG_PROMPTS` config flag (default: False) exists for development debugging. Setting it True is your explicit choice to accept prompt visibility in DEBUG-level logs.

### Cache Keys Are Hashed

Response cache fingerprints use SHA-256 over a normalized prompt. The cache stores hashes, not raw prompts. Cache keys cannot be reversed to reveal prompt content.

## Response Caching

Flux has an optional response cache. It is **disabled by default**.

When disabled (default):
- Every request goes through full routing logic
- No prompt fingerprints are computed
- No responses are stored

When enabled (`FLUX_ENABLE_RESPONSE_CACHE=true`):
- Identical normalized prompts may return previously generated responses
- Cache entries **are** scoped by tenant, user, plan, and sensitivity level, so a
  cache hit cannot cross any of those boundaries. The scope key is built in
  `router/classifier.py::_cache_scope_key()` and mixed into the fingerprint
- Plan is part of the scope deliberately: a response bought on one budget must
  not be served "for free" under another
- Sensitivity is part of the scope deliberately: a confidential answer must
  never surface from a public-labelled lookup, or vice versa

One caveat, and it is the only way to lose that isolation: the scoping is
applied on the **classifier path**. Any code calling `cache.fingerprint()`,
`ResponseCache.get()`, or `ResponseCache.set()` directly — bypassing the
classifier — gets no isolation unless it passes its own `scope_key`. If you
have written a custom integration against `router/cache.py`, check that it
does.

Verify it yourself:

```python
from router.cache import fingerprint
from router.classifier import _cache_scope_key
from router.schemas import RoutingRequest

a = RoutingRequest(raw_prompt="same prompt", user_id="alice", tenant_id="acme")
b = RoutingRequest(raw_prompt="same prompt", user_id="bob", tenant_id="evilcorp")
fa = fingerprint("same prompt", None, [], None, scope_key=_cache_scope_key(a, "public"))
fb = fingerprint("same prompt", None, [], None, scope_key=_cache_scope_key(b, "public"))
assert fa != fb  # identical prompts, different tenants → different cache keys
```

Recommendation: the cache is safe to enable in a multi-tenant deployment. It
stays off by default because response caching changes behaviour (repeat prompts
stop hitting the model), not because of an isolation gap.

### Adaptive Learning Uses Metadata Only

The adaptive weights system records only:
- Model ID (string)
- Task type (string)
- Quality score (float in [0, 1])
- Customer ID (string, optional)

It never has access to prompt or response text.

### Identity and Budget Fields Must Come From Authenticated Context

`RoutingRequest.customer_id` and `RoutingRequest.user_id` are plain string fields on the request schema. Flux treats them as **trusted identifiers** — the per-customer adaptive EMA, per-customer routing profile, and budget ledger are all keyed off of them.

The same applies to the fields that select or raise a request's spending limits: **`plan`**, **`max_daily_cost`**, and **`max_cost_per_request`**. `plan` maps directly to a daily budget ceiling (e.g. `business_plan` → the highest tier), and the two cost-cap fields override the per-request and per-day limits.

**In a multi-tenant deployment, you MUST populate all of these fields server-side from your authenticated session, not from anything the client controls.** If a client can set them freely on the request body, they can:
- Read another customer's learned routing preferences (via `get_customer_routing_profile`)
- Pollute another customer's adaptive weights with bad-quality signals
- Charge their spend against another customer's budget (via `customer_id`/`user_id`)
- Grant themselves a higher spending limit by claiming a more expensive `plan` or raising `max_daily_cost` / `max_cost_per_request`

Flux does not authenticate. Your application layer must.

**Exception — `router/server.py`'s HTTP proxy in `FLUX_SERVER_TOKENS` mode.**
The proxy's default auth (`FLUX_SERVER_TOKEN`, a single shared bearer token)
does not bind identity: any caller holding the token can set `X-Flux-Tenant-Id`
and `user` to whatever they like, so the risks above still apply verbatim to
proxy traffic. Set `FLUX_SERVER_TOKENS` (a JSON map of `token -> tenant_id`,
e.g. `{"tok-acme": "acme", "tok-globex": "globex"}`) instead to bind each
bearer token to exactly one tenant server-side — the proxy then ignores any
client-supplied `X-Flux-Tenant-Id` and forces both `/v1/chat/completions`
attribution and `/v1/usage` queries to the token's bound tenant. This closes
the tenant-identity gap; it does **not** cover `user_id`/`customer_id`
(per-user budget spoofing within a tenant) — those still require your own
authenticated-session mapping, same as any other Flux deployment.

### File Writes Are Atomic

State files (adaptive weights, analytics) are written via temp file + atomic rename. A process crash mid-write cannot corrupt your data.

### File Paths Are Validated

State file paths are resolved and verified to stay under an allowed base directory. Directory traversal attempts (e.g. `../../etc/passwd`) are rejected.

### Resource Limits Prevent Exhaustion

The router enforces caps on:
- Maximum concurrent customers tracked
- Maximum unique (model, task) keys tracked
- Maximum prompt length accepted
- Maximum metadata size per request

These caps protect against memory exhaustion attacks.

### Inbound Requests Are Rate Limited (`router/rate_limit.py`)

`POST /v1/chat/completions` is rate limited per caller with a token bucket,
checked before the request body is read and before anything is dispatched
upstream. Configured by `FLUX_RATE_LIMIT_RPM` (default 600/min, `0` disables)
and `FLUX_RATE_LIMIT_BURST`. Over-limit callers get `429` with `Retry-After`.

The bucket key is the bearer token's **bound tenant** when running in
`FLUX_SERVER_TOKENS` mode, and the peer IP otherwise. It is deliberately never
keyed on the `user` field or `X-Flux-Tenant-Id` — both are self-declared, and a
limiter keyed on either is one a caller escapes by incrementing a counter.
`X-Forwarded-For` is ignored for the same reason; rate limit at your proxy if
you terminate connections there.

Two limits worth being explicit about:

- **This is an availability control, not a cost control.** It bounds request
  rate, not spend. `FLUX_TENANT_DAILY_CAP_USD` is what caps spend, and it is
  off by default.
- **Buckets are process-local.** With `FLUX_SERVER_WORKERS > 1` each worker
  enforces its own share, so the effective global ceiling is roughly
  `FLUX_RATE_LIMIT_RPM x workers` (the server warns about this at startup).
  Use an ingress limiter if you need an exact global bound.

### Cost Attribution Stores Metadata Only

`router/attribution.py` (`UsageRecord`, `SqliteUsageStore`) records, per
dispatch: `tenant_id`, `run_id`, `task_type`, `step_type`, `model_id`,
`cost_usd`, a timestamp, and — since actual (provider-reported) usage
recording was added — `usage_source` (`"provider"` or `"estimated"`) plus
`input_tokens`/`output_tokens` (counts only). **It never has access to
prompt or response text** — the recording call sites in `flux.py`,
`routing_engine.py`, and `server.py` only ever pass cost/token-count/metadata
values, never `text`/`response`/`prompt`. This is enforced structurally:
`UsageRecord`'s dataclass fields have no slot that could hold arbitrary
text, so there's no field to accidentally populate — `input_tokens`/
`output_tokens` are integer counts extracted from a provider's `usage`/
`usageMetadata` object (see `provider_caller.py::_extract_usage`), never
the token contents themselves.

`GET /v1/usage` on the HTTP proxy exposes this same data, filterable by
`tenant_id`/`run_id`. As with `customer_id`/`user_id` above, `tenant_id` is a
**trusted identifier** — if your deployment lets a client set
`X-Flux-Tenant-Id` freely, they can read another tenant's cost data via
`GET /v1/usage?tenant_id=...`. Populate it server-side from your
authenticated session, same as `customer_id`.

The default `SqliteUsageStore` is `:memory:` (no disk write, no
cross-restart persistence) unless `FLUX_ATTRIBUTION_DB` is set to a file
path. `GET /metrics` (Prometheus) caps label cardinality
(`ATTRIBUTION_METRICS_MAX_LABEL_COMBOS`) so a client minting fresh
`tenant_id` values per request cannot grow the metrics registry without
bound.

---

## The Dashboard and the Stats API

`flux serve` serves a local operator console at `/dashboard` and the
`GET /v1/stats/*` endpoints behind it. Both are read-only. What they expose is
the deployment's spend and configuration, so the access model is deliberately
narrow.

**What it exposes.** Aggregated costs, model and task breakdowns, latency
percentiles, the model registry, and the effective configuration. Not prompts,
not completions — the `usage` table never receives request or response text
(see Data Flow above), so there is none for the dashboard to leak.

Most TEXT columns in that table are enums or registry identifiers
(`task_type`, `step_type`, `model_id`, `usage_source`, `routing_priority`).
Two are not: `tenant_id` and `run_id` carry caller-supplied header values
(`X-Flux-Tenant-Id`, `X-Flux-Run-Id`) whenever identity is not bound to a
token, so a caller can write arbitrary strings there. They are stored and
compared as SQL parameters, escaped before reaching `/metrics`
(`attribution._escape_label`), and rendered with `textContent` rather than
`innerHTML` in the dashboard — but they are caller-controlled, not validated,
and a deployment treating `tenant_id` as trustworthy in shared-token mode is
making the mistake the startup warning describes.

**Never a secret.** `GET /v1/stats/config` reports auth as a mode name and
providers as a configured yes/no. No bearer token, no provider API key, and no
Redis URL (which can embed credentials) appears in the response.
`router/tests/test_stats.py` asserts that no configured secret's value appears
anywhere in the body.

**Not a secret is not the same as not sensitive.** In `FLUX_SERVER_TOKENS`
mode a bearer token belongs to one customer, so `/v1/stats/config` withholds
the operator-only half of the payload from it: the bind address and port, the
worker count, the data directory and usage-database path, how many tenants are
bound, the run/budget store backends, and every plan's budget but the caller's
own. That is the operator's deployment shape, and a customer holding a tenant
token has no business reading it. The caller still gets what it needs to reason
about its own requests — auth mode, body cap, configured providers, its own
plan's budget, and the run and rate limits that apply to it. Loopback callers
and the shared-token operator see the full payload, so the dashboard is
unchanged for the operator; the dashboard omits the operator sections when the
fields are absent rather than rendering blanks.

**Who can reach it.** This data is *every* tenant's spend and the deployment's
configuration — correct for the single-operator case it is built for, wrong the
moment the port is reachable from elsewhere. When no auth is configured, it is
served to **loopback callers only**. That rule covers the whole surface, not
just the page:

```
/dashboard         /v1/stats/summary     /v1/stats/tasks      /v1/usage
                   /v1/stats/timeseries  /v1/stats/tenants    /metrics
                   /v1/stats/models      /v1/stats/registry
                                         /v1/stats/config
```

Gating the dashboard alone would be theatre: the page is only a renderer for
those endpoints, so anyone refused at `/dashboard` could read the identical
numbers from `/v1/stats/summary`. `server._refuse_remote_spend_data()` is the
single gate, and `router/tests/test_stats.py` asserts every path in the list
above refuses a remote peer, serves a loopback one, and allows a remote read
once a token is set.

**Loopback means both the peer and the `Host` header.** The peer address alone
answers "did this connection come from this box", which is not the same
question as "did something on this box mean to send it". A page on
`evil.example` that the operator visits can lower its DNS TTL, rebind
`evil.example` to `127.0.0.1`, and `fetch("http://evil.example:8000/v1/stats/config")`
from the operator's own browser. The peer address is loopback, so a peer-only
check passes — and because the page's origin *is* that host and port, the
responses come back same-origin and fully readable, with no CORS step to
withhold. The `Host` header is the part of that request the attacker cannot
forge: the browser puts their hostname in it. `server._is_local_host_header()`
requires it to name this machine, which closes the rebinding path.

If a same-host reverse proxy passes its own public hostname through
(`Host: flux.internal`) and you have deliberately not set a token, name it in
`FLUX_ALLOWED_HOSTS` (comma-separated). Do not put a name an attacker can
resolve to `127.0.0.1` in that list — it restores the hole exactly.

For the dashboard specifically there is also a **mount-time** check on the
configured bind address, which refuses to mount at all rather than serving a
page whose data calls will fail. That check alone is not sufficient — it reads
the *configured* address, which is only the truth when the bind came from the
environment. Start the server with `uvicorn router.server:app --host 0.0.0.0`
and config still reports `127.0.0.1`. The request-time peer check is what
holds regardless of how the server was started.

None of this applies when `FLUX_SERVER_TOKEN` or `FLUX_SERVER_TOKENS` is set:
the token is then the control and remote access is intentional — including a
Prometheus scrape of `/metrics` from another host, which needs a token. In
`FLUX_SERVER_TOKENS` mode a caller sees only the tenant their token is bound
to.

The proxy API itself (`/v1/chat/completions`, `/v1/models`, `/health`) is not
covered by this rule. It is what the deployment exists for and has its own auth
rules.

**A container never has a loopback client.** Requests arrive from the container
bridge, so the unauthenticated localhost allowance never applies there, even
when the port is published to `127.0.0.1`. A token is required to use the
dashboard under Docker.

**A same-host reverse proxy still reaches it**, since its peer address is
loopback — provided the `Host` it forwards is a loopback name or is listed in
`FLUX_ALLOWED_HOSTS`. That is intentional: such a deployment has taken
responsibility for its own edge, and should configure a token.

---

## What Flux Does NOT Protect Against

We are honest about our threat model. Flux does NOT protect against:

- Compromise of the host machine where Flux runs
- A malicious developer with code-execution access to your application
- Vulnerabilities in your LLM provider (OpenAI, Anthropic, etc.)
- Network attacks on the connection between your application and the provider
- Side-channel attacks (timing, cache, etc.)

For these threats, you need broader infrastructure security — Flux is not a replacement for proper secrets management, network hardening, or supply-chain verification.

---

## How to Verify

Every claim above can be verified by reading the code:

1. **API key handling:** Search the codebase for `provider_api_key`. You will find it has `repr=False, exclude=True` in schemas.py and is never logged.
2. **Error sanitization:** Read `router/provider_caller.py` `_post_json()`. The exception message contains only HTTP code and provider name.
3. **Prompt logging:** Search for `log.info` and `log.warn` near prompt handling. You will find only metadata.
4. **Cache hashing:** Read `router/cache.py` `fingerprint()`. It uses SHA-256.
5. **Atomic writes:** Read `router/adaptive_weights.py` `_flush()`. It uses tempfile + os.replace.
6. **Resource limits:** Read `router/config.py` for MAX_CUSTOMERS, MAX_ADAPTIVE_KEYS, etc.
7. **Dashboard exposure:** Read `router/server.py` `_dashboard_refusal_reason()` (mount-time) and `_LoopbackOnlyDashboard` (request-time). Or check it live — start the server bound off-loopback with no token and confirm `/dashboard` is not served:

   ```bash
   FLUX_SERVER_HOST=0.0.0.0 flux serve --port 8000
   curl -s -o /dev/null -w '%{http_code}\n' http://<your-lan-ip>:8000/dashboard/   # 404
   curl -s -o /dev/null -w '%{http_code}\n' http://<your-lan-ip>:8000/health       # 200
   ```

8. **No prompt text on disk:** Route a request containing a distinctive string, then `strings ~/.local/share/flux/flux.db | grep <that string>` — no match.

You can also run the security audit yourself:

If you find a discrepancy between this document and the code, please report it to fluxllmdev@gmail.com — that is a security issue.

---

## Reporting Issues

Report security issues to fluxllmdev@gmail.com. See SECURITY.md for our full disclosure process.
