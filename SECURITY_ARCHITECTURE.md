# Flux Security Architecture

This document explains exactly what data Flux handles, where it stays, and where it goes.

If you're evaluating Flux for your stack and want to verify our claims, see the "How to Verify" section at the bottom — every guarantee here can be checked by reading the code.

---

## TL;DR

- **Your API keys never leave your infrastructure.** They live in your environment variables and pass directly to provider APIs.
- **Your prompts and responses never leave your infrastructure.** Flux makes routing decisions on metadata only.
- **No telemetry by default.** Flux runs entirely locally unless you explicitly configure it otherwise.

---

## Data Flow

| Data Type | Where It Lives | Leaves Your Infrastructure? |
|-----------|---------------|----------------------------|
| Provider API keys (OpenAI, Anthropic, etc.) | Your environment, in memory only | Never |
| User prompts | Your application memory | Never |
| Model responses | Your application memory | Never |
| Conversation history | Your application memory | Never |
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
- The current cache implementation does NOT segment by user, plan, or sensitivity level
- This means **enabling the cache in a multi-tenant deployment can cause cross-tenant response bleed**
- Tenant-scoped caching is planned for a future release

Recommendation: keep the cache disabled until tenant-scoped caching ships.

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

### Cost Attribution Stores Metadata Only (Task 7)

`router/attribution.py` (`UsageRecord`, `SqliteUsageStore`) records, per
dispatch: `tenant_id`, `run_id`, `task_type`, `step_type`, `model_id`,
`cost_usd`, and a timestamp. **It never has access to prompt or response
text** — the recording call sites in `flux.py` and `routing_engine.py` only
ever pass cost/metadata values, never `text`/`response`/`prompt`. This is
enforced structurally: `UsageRecord`'s dataclass fields have no slot that
could hold arbitrary text, so there's no field to accidentally populate.

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

You can also run the security audit yourself:

If you find a discrepancy between this document and the code, please report it to fluxllmdev@gmail.com — that is a security issue.

---

## Reporting Issues

Report security issues to fluxllmdev@gmail.com. See SECURITY.md for our full disclosure process.
