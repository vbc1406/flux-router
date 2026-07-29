# Codebase Map

## Start Here First

New engineer? Do this in order:

1. Read this file (5 min)
2. Read `router/config.py` — every tunable value lives there (10 min)
3. Read `router/schemas.py` — understand the data contracts (10 min)
4. Read the `routing_engine.py` header comment, then skim the 13-step `route()` method (15 min)
5. Run the tests: `pytest -v` (2 min)
6. Try the keyless demo: `python -m router.demo` (5 min)
7. Read `DEBUG.md` so you know how to diagnose problems (5 min)

Total: ~50 minutes before writing your first line of code.

---

## Directory Structure

```
flux/
├── router/                        ← Main package (all production logic lives here)
│   ├── __init__.py                ← Public API surface: RoutingEngine, RoutingRequest, RoutingDecision
│   ├── config.py                  ← ALL tunable constants — the single source of truth
│   ├── schemas.py                 ← Pydantic data models (RoutingRequest, RoutingDecision, ModelOption, …)
│   ├── routing_engine.py          ← The brain: 13-step routing algorithm
│   ├── classifier.py              ← Heuristic request classifier (task type + complexity score)
│   ├── adaptive_weights.py        ← Per-(model, task) quality learning with EMA + rollback
│   ├── model_registry.py          ← Static model catalog + runtime load tracking
│   ├── models.json                ← Model definitions loaded by ModelRegistry at startup
│   ├── flux.py                    ← High-level facade: route → call → retry → return
│   ├── fallback_chain.py          ← Fallback chain construction + FallbackExecutor
│   ├── cache.py                   ← LRU response cache with prompt fingerprinting
│   ├── budget_tracker.py          ← Per-user daily/monthly spend tracking
│   ├── analytics.py               ← Append-only JSONL decision log with query API
│   ├── circuit_breaker.py         ← Per-provider circuit breaker (open/closed/half-open)
│   ├── context_compressor.py      ← Trim conversation history when context window is full
│   ├── quality_scorer.py          ← Post-response heuristic quality scorer (feeds AdaptiveWeights)
│   ├── provider_caller.py         ← HTTP callers for Anthropic / OpenAI / Google / Groq / Mistral
│   ├── errors.py                  ← Typed exception hierarchy (FluxAPIError and subtypes)
│   ├── benchmark.py               ← Routing decision benchmarks
│   ├── demo.py                    ← Standalone demo script
│   ├── server.py                  ← OpenAI-compatible HTTP proxy (POST /v1/chat/completions); optional `[server]` extra
│   ├── run_budget.py              ← Run-scoped budget enforcement for agent loops (Task 3)
│   ├── prompt_cache.py            ← Cache-aware routing: tracks which provider holds a warm prefix (Task 5)
│   ├── cascade.py                 ← Local response verifiers + cost accounting for cascade routing (Task 8)
│   ├── attribution.py             ← Per-run/per-tenant cost attribution: SQLite usage log + Prometheus counters (Task 7)
│   └── tests/
│       ├── test_routing.py        ← End-to-end routing engine tests (13 change areas)
│       ├── test_server.py         ← HTTP proxy tests (directives, passthrough, streaming, auth, body cap)
│       ├── test_run_budget.py     ← Run-budget ladder, eviction at scale, agent-loop integration
│       ├── test_step_type.py      ← step_type inference, STEP_TYPE_FLOORS, tool-capability filter
│       ├── test_cache_aware_routing.py ← PromptCacheTracker + cache-stickiness routing behavior
│       ├── test_cascade.py        ← Cascade escalation ladder, verifiers, net-savings accounting
│       ├── test_attribution.py    ← UsageStore, cardinality-capped Prometheus counters, wiring
│       ├── test_adaptive_guardrails.py  ← AdaptiveWeights guardrail tests (6 issue areas)
│       ├── test_adaptive_weights.py     ← AdaptiveWeights unit tests with metrics
│       ├── test_cache.py          ← ResponseCache + fingerprinting tests
│       ├── test_classifier.py     ← RequestClassifier unit tests
│       ├── test_circuit_breaker.py      ← CircuitBreaker state machine tests
│       ├── test_context_penalty.py      ← Context window penalty tests
│       ├── test_explainability.py       ← Verbose routing explanation tests
│       ├── test_model_registry.py       ← ModelRegistry tests
│       ├── test_priority_tags.py        ← routing_priority parameter tests
│       ├── test_smart_retry.py          ← Retry and fallback tests
│       └── test_sticky_model.py         ← Conversation sticky bias tests
│
├── CODEBASE_MAP.md                ← This file
├── FEATURES.md                    ← How to add new features
├── DEBUG.md                       ← How to diagnose common problems
├── MIGRATIONS.md                  ← How to handle schema and config migrations
└── README.md                      ← Project overview + quick start
```

---

## Critical Files and Their Purpose

| File | Purpose | Read when… |
|------|---------|------------|
| `config.py` | Every threshold, limit, weight | Before changing any behaviour |
| `schemas.py` | Data contracts between all modules | Before adding fields or new request types |
| `routing_engine.py` | The 13-step routing algorithm | Before touching routing logic |
| `adaptive_weights.py` | EMA learning, rollback, per-customer weights | Before touching quality learning |
| `model_registry.py` + `models.json` | Model catalog + live load | Before adding a new model |
| `classifier.py` | Task type + complexity score | Before adding a new task type |
| `fallback_chain.py` | How fallback chains are built | Before changing fallback behaviour |
| `flux.py` | The public-facing entry point | Before integrating the router |
| `errors.py` | Exception types callers catch | Before adding new error conditions |

---

## Where to Find Things

### Configuration
All tunable constants → `router/config.py`
Every value has a comment explaining what it does, why it exists, and how to tune it.

### Tests
Unit and integration tests → `router/tests/test_*.py`
Run all tests: `pytest -v`
Run one file: `pytest -v router/tests/test_routing.py`

### Model Definitions
Static model catalog → `router/models.json`
Registry loader → `router/model_registry.py` (`ModelRegistry` class)
To add a model: edit `models.json`, no code change needed.

### Routing Logic
Main algorithm → `router/routing_engine.py` (`RoutingEngine.route()`)
The method is structured as 13 numbered steps with header comments.

### Cache-Aware Routing
→ `router/prompt_cache.py` (`PromptCacheTracker`) + a block inside
`routing_engine.py` Step 9. Keyed by `conversation_id` (falling back to
`run_id`); only engages when `system_prompt` is at least
`CACHE_PREFIX_MIN_TOKENS`. A soft `CACHE_STICKINESS_WEIGHT` score bonus goes
to models on the provider already holding a warm prefix; a hard constraint
then blocks switching away from that provider unless the switch's cost
savings clear `CACHE_SWITCH_MARGIN`. Result surfaces on
`RoutingDecision.prompt_cache_status` (`cold` / `warm` / `would_lose_cache`).
Not the same thing as `router/cache.py`'s `ResponseCache` (whole-response
caching) — this tracks provider-side *prefix* caching state only, no prompt
or response content.

### Cascade / Escalation
→ `router/cascade.py` (`verify_response`, `Verifier` protocol,
`estimate_step_cost`) + `Flux._complete_cascade()` in `flux.py` +
`_route_cascade_initial()` in `routing_engine.py`. `routing_priority="cascade"`
starts at the cheapest capable tier (mirrors `always-premium`'s shortcut, but
inverted); `Flux.complete()` then walks `decision.fallback_chain` as an
escalation ladder, running local verifiers (no LLM judge in the hot path)
after each attempt and stopping at the first pass. Surfaces
`RoutingDecision.cascade_attempts` / `cascade_net_savings` — the latter can
go negative when escalation cost more than skipping straight to the priciest
tier tried, by design (see FEATURES.md honesty requirement). Python API
(`Flux.complete(routing_priority="cascade")`) only for now — `router/server.py`
calls `Flux._call_model()` directly rather than `Flux.complete()`, so the
HTTP proxy does not yet expose a `flux-cascade` directive.

### Cost Attribution
→ `router/attribution.py` (`CostAttribution`, `SqliteUsageStore`) — "which
customer/workflow is eating my margin." Recorded at every successful
dispatch point (`RoutingEngine._proxy_execute()`, `Flux.complete()`,
`Flux._complete_cascade()`, and `router/server.py`'s two response paths):
`tenant_id`, `run_id`, `task_type`, `step_type`, `model_id`, `cost_usd` — no
prompt or response content, ever (see SECURITY_ARCHITECTURE.md). Exposed via
`GET /v1/usage` (paginated, filterable) and `GET /metrics` (Prometheus text,
label-cardinality-capped at `ATTRIBUTION_METRICS_MAX_LABEL_COMBOS`). Default
storage is SQLite `:memory:`; set `FLUX_ATTRIBUTION_DB` for a persistent file.

### Step-Type Classification (agent trajectories)
→ `RoutingRequest.step_type` / `TaskAnalysis.step_type`, inferred by
`RequestClassifier._infer_step_type()` when unset (from `tools`,
tool-result messages, `response_format`). Enforced as a hard constraint in
`_passes_hard_constraints()` via `STEP_TYPE_FLOORS` (config.py) — applied
BEFORE scoring, so a cheap model can never win a `plan`/`tool_select`/
`final_answer` step on cost alone. `request.tools` also hard-filters
candidates to `supports_tools=True` models; `response_format` filters to
`supports_structured_output=True`.

### Adaptive Learning / Quality Tracking
→ `router/adaptive_weights.py` (`AdaptiveWeights` class)
Quality is fed back via `AdaptiveWeights.record()` after each response.

### Analytics / Decision Logging
→ `router/analytics.py` (`RoutingAnalytics` class)
Writes to `router/routing_analytics.jsonl` (append-only JSONL).

### Provider Integrations
HTTP callers → `router/provider_caller.py`
Error types → `router/errors.py`
Supported: Anthropic, OpenAI, Google (Gemini), Groq, Mistral

### HTTP Proxy (OpenAI-compatible)
→ `router/server.py` — `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.
`model` in the request body is a routing directive (`flux-auto` / `flux-cheap` /
`flux-quality`) unless it names a real registered model, in which case routing is
bypassed and that model is called verbatim. Requires the `[server]` extra
(fastapi/uvicorn) — not a core dependency. Run with `make serve`.

### Run-Scoped Budgets (agent loops)
→ `router/run_budget.py` (`RunBudget`, `RunLimits`, `RunBudgetExceeded`).
A `run_id` groups N routing decisions into one multi-step trajectory. Checked
BEFORE each step dispatches (`RoutingEngine.route()`'s Step 0): forces
cost-optimized routing once a run crosses `RUN_DEGRADE_THRESHOLD`, sets
`RoutingDecision.budget_warning` at `RUN_WARN_THRESHOLD`, and raises
`RunBudgetExceeded` (with a per-step cost breakdown) once a limit is met —
never after a step has already spent. Entry points: `Flux.start_run()` /
`flux.complete(..., run_id=...)`, or the proxy's `X-Flux-Run-Id` header.
See `examples/agent_loop.py`.

### Fallback and Retry Logic
Chain construction → `router/fallback_chain.py` (`build_fallback_chain`, `build_typed_fallback_chains`)
Execution with retry → `router/fallback_chain.py` (`FallbackExecutor`)
Circuit breaker → `router/circuit_breaker.py` (`CircuitBreaker`)

### Budget Enforcement
Plan-level daily/monthly limits → `router/budget_tracker.py` (`BudgetTracker`)
Per-request daily caps → `router/budget_tracker.py` (`DailyBudgetTracker`)
Limits defined in → `config.py` (`BUDGET_LIMITS`)

### Cache
Response cache → `router/cache.py` (`ResponseCache`)
Fingerprinting → `router/cache.py` (`fingerprint()`)

### Context Management
Compression when context window is full → `router/context_compressor.py`

---

## Key Classes at a Glance

```
RoutingEngine          routing_engine.py    The 13-step decision algorithm
  └─ route(request)                         Call this to get a routing decision

Flux                   flux.py              High-level facade (route + call + retry)
  └─ complete(prompt)                       Call this for a full request cycle

RequestClassifier      classifier.py        Prompt → TaskAnalysis (task type + complexity)
  └─ analyze(request)

AdaptiveWeights        adaptive_weights.py  EMA quality tracking per (model, task)
  ├─ record(model, task, score)             Feed quality signal back
  └─ get_adjusted_score(model, task, base)  Get routing-adjusted quality

ModelRegistry          model_registry.py    Static catalog + runtime load
  └─ all_available_models()

ResponseCache          cache.py             LRU prompt cache
  ├─ get(fingerprint)
  └─ set(fingerprint, response, model, cost)

BudgetTracker          budget_tracker.py    Plan-level spend enforcement
  └─ would_exceed_budget(user_id, cost)

CircuitBreaker         circuit_breaker.py   Per-provider open/closed/half-open
  ├─ is_available(provider)
  ├─ record_success(provider)
  └─ record_failure(provider)

RoutingAnalytics       analytics.py         Append-only decision log
  ├─ log_decision(decision)
  └─ update_actual(correlation_id, ...)     Patch with post-call actuals

FallbackExecutor       fallback_chain.py    Walk the fallback chain on failure
  └─ execute_with_fallback(request, decision, api_caller)
```

---

## Data Flow (one request, end to end)

```
Caller
  │
  ▼
Flux.complete(prompt)          ← flux.py — public entry point
  │
  ▼
RoutingEngine.route(request)   ← routing_engine.py
  │
  ├─ Step 1: RequestClassifier.analyze()    → TaskAnalysis (task_type, complexity_score, tokens)
  ├─ Step 2: ResponseCache.get()            → cache hit shortcut
  ├─ Step 3: cost ceiling check
  ├─ Step 4: hard constraint filtering      → candidate ModelOption list
  ├─ Step 4b: DailyBudgetTracker check
  ├─ Step 5: ContextCompressor.compress()   → shrink if needed
  ├─ Step 6: explicit model override
  ├─ Step 7: special rules (trivial / anti-gaming / always-premium)
  ├─ Step 8: AdaptiveWeights.get_adjusted_score()  → per-(model,task) quality
  ├─ Step 9: tier selection + multi-criteria scoring
  ├─ Step 9b: confidence threshold check    → upgrade to premium if score too low
  ├─ Step 10: A/B exploration               → blocked when confidence_fallback=True
  ├─ Step 11: build fallback chains
  ├─ Step 12: BudgetTracker tier walk-down
  └─ Step 13: log + return RoutingDecision
  │
  ▼
provider_caller.call_provider()  ← proxy mode only
  │
  ▼
RoutingDecision (+ proxy_response if proxy mode)
```

---

## Extension Points (where to add new features)

| What you want to add | Where to add it |
|---------------------|----------------|
| New routing rule | `routing_engine.py` Step 7 or Step 9 |
| New task type | `config.py` → `COMPLEXITY_BASE_SCORES`, then `classifier.py` |
| New provider | `provider_caller.py`, `model_registry.py`, `models.json` |
| New quality metric | `quality_scorer.py` |
| New model | `models.json` (no code change required) |
| New budget enforcement | `budget_tracker.py` + `routing_engine.py` Step 4b |
| New fallback strategy | `fallback_chain.py` `build_typed_fallback_chains()` |
| New config constant | `config.py` (always add a comment explaining it) |
| New analytics field | `analytics.py` `_decision_to_dict()` |
| New request field | `schemas.py` `RoutingRequest`, then `routing_engine.py` |
