# Changelog

All notable changes to Flux will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cost-vs-quality eval harness (`python -m router.evals`, `make evals`): runs GSM8K / MMLU / HumanEval / MT-Bench through the flux/premium/cheapest/mid strategies and reports cost-savings % and quality-retention % vs an always-premium baseline. Hybrid grading (objective + LLM-as-judge), offline simulated mode by default with a `--live` path; see [EVALS.md](./EVALS.md). Optional `flux-router[evals]` extra pulls the real datasets.
- Interactive terminal chat example (`examples/chat.py`)
- Per-provider API key loading via env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`)
- Cross-provider fallback integration test
- 9 new models with benchmark-derived quality ratings
- LRU eviction for per-customer adaptive state (bounded memory under high tenant counts)
- Per-customer `record()` rate limiter to bound adaptive-state mutation cost
- `MIGRATIONS.md` link in README documentation index
- CI job that validates the clean-installed package and the keyless demo

### Changed
- Tier-based candidate filtering replaced with a complexity → quality floor
- Scoring loop reduced from O(M²) to O(M) via precomputed normalization vectors
- Adaptive `get_adjusted_score` read now stays inside the lock
- `RoutingAnalytics(log_path=None)` is in-memory only, never writes
- Repo URLs and packaging metadata point at `github.com/vbc1406/flux`
- Contact addresses consolidated to `fluxllmdev@gmail.com`
- README adds a comparison table, surfaces licensing, and corrects the P50 latency claim
- Circuit breaker is probed at most once per `route()` invocation during Step 4 filtering
- `rich` is now a declared runtime dependency
- Minimum supported Python raised to 3.10 (dropped 3.9)

### Fixed
- Proxy-mode provider calls now record spend (previously skipped)
- Double-spend recording on the non-proxy path removed
- Budget re-checked on every fallback attempt
- `BudgetTracker` rejects empty/`None` `user_id`; the TOCTOU window is documented
- Replaced deprecated `datetime.utcnow()` with timezone-aware UTC equivalent
- Replaced deprecated `asyncio.get_event_loop().run_until_complete()` with `asyncio.run()` in tests
- Chat example handles request failures without crashing the REPL
- Conftest autouse fixture zeroes fallback delays in tests
- OpenAI token-field handling, cache safety, and lint regressions
- Packaging launch blockers identified in the 2026-05-15 audit

### Documentation
- `customer_id` / `user_id` documented as trusted, server-set inputs (must come from authenticated context)
- `plan` and `max_daily_cost` documented as trusted, server-set fields
- Internal launch-readiness docs moved out of the public tree

### Security
- Per-customer rate limiter for adaptive `record()` adds a defense-in-depth bound on memory and CPU under hostile traffic

## [1.0.0] - 2026-05-03

### Added
- Intelligent LLM routing engine with cost, quality, and latency scoring
- Adaptive weight learning via exponential moving average feedback
- Per-customer routing state with configurable isolation
- Fallback chain with circuit breaker per provider
- Structured logging via `structlog` with no prompt content at INFO+
- Model registry with JSON-driven provider configuration
- Context compression to stay within model token limits
- Budget tracker with per-user and per-team spending caps
- Benchmark harness for comparing routing strategies
- Input validation with length caps on all request fields (C1)
- Sanitized error messages — API response bodies never reach logs at INFO+ (C2)
- Atomic JSON state writes with `fsync` (C3)
- Path traversal protection on all file-path inputs (C4)
- Resource exhaustion caps on adaptive state dicts (C5)

[1.0.0]: https://github.com/vbc1406/flux/releases/tag/v1.0.0
