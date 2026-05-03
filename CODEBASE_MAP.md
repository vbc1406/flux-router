# Codebase Map

## Start Here First

New engineer? Do this in order:

1. Read this file (5 min)
2. Read `router/config.py` — every tunable value lives there (10 min)
3. Read `router/schemas.py` — understand the data contracts (10 min)
4. Read the `routing_engine.py` header comment, then skim the 13-step `route()` method (15 min)
5. Run the tests: `pytest -v` (2 min)
6. Try the interactive tester: `python testing/router_tester.py` (5 min)
7. Read `DEBUG.md` so you know how to diagnose problems (5 min)

Total: ~50 minutes before writing your first line of code.

---

## Directory Structure

```
vibecode/
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
│   └── tests/
│       ├── test_routing.py        ← End-to-end routing engine tests (13 change areas)
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
├── testing/                       ← Manual and batch testing tools (separate from unit tests)
│   ├── router_tester.py           ← Interactive REPL for live routing decisions
│   ├── batch_runner.py            ← Batch mode: run many prompts, output table + CSV
│   ├── large_benchmark.py         ← Large-scale performance benchmark
│   ├── test_prompts.txt           ← 20+ diverse prompts used by batch tools
│   ├── benchmark_results.json     ← Latest benchmark output
│   ├── benchmark_summary.csv      ← CSV summary of benchmark run
│   ├── benchmark_report.txt       ← Human-readable benchmark report
│   └── README.md                  ← How to use the testing tools
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
Manual/interactive tests → `testing/`
Run all tests: `pytest -v`
Run one file: `pytest -v router/tests/test_routing.py`

### Model Definitions
Static model catalog → `router/models.json`
Registry loader → `router/model_registry.py` (`ModelRegistry` class)
To add a model: edit `models.json`, no code change needed.

### Routing Logic
Main algorithm → `router/routing_engine.py` (`RoutingEngine.route()`)
The method is structured as 13 numbered steps with header comments.

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
