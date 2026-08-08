# Changelog

All notable changes to Flux will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- The loopback gate on spend and configuration data now checks the `Host` header as well as the peer address, closing a DNS-rebinding path. A page the operator visits could re-point its own hostname at `127.0.0.1` and have the operator's browser fetch `/v1/stats/*` and `/dashboard`: the peer address was loopback so the gate passed, and because the page's origin *was* that host and port the responses were same-origin and fully readable, with no CORS step involved. The `Host` header is the part the attacker cannot forge. `FLUX_ALLOWED_HOSTS` (comma-separated) names extra hosts for a same-host reverse proxy running deliberately without a token.
- Spend and configuration data is served to loopback callers only when no authentication is configured. This covers `/v1/stats/*`, `/v1/usage`, and `/metrics` as well as `/dashboard` — gating the page alone was ineffective, since it is only a renderer for those endpoints and `/v1/stats/summary` returned the identical numbers to any caller that could reach the port. Set `FLUX_SERVER_TOKEN` to read any of it remotely, including a Prometheus scrape of `/metrics`. The proxy API is unaffected.
- The dashboard now refuses non-loopback requests at request time, not only at mount time. The mount decision reads the *configured* bind address, so passing the host on the command line instead (`uvicorn router.server:app --host 0.0.0.0`, which is what the Dockerfile did) left the server believing it was on loopback while serving every tenant's spend and the deployment configuration to anything that could reach the port. The check now also runs per request against the peer address, which holds however the server was started. Unaffected when `FLUX_SERVER_TOKEN` is set, and a reverse proxy on the same host still reaches it.

### Added
- Local operator dashboard (`/dashboard`) served from the FastAPI app, backed by new `/v1/stats` aggregate endpoints, with usage/latency/savings/routing detail persisted to the usage table. Restyled to the flux-llm.com brand palette; adds a tenant breakdown, a live feed, and auto-refresh. `flux` CLI (`flux serve`, `flux version`) creates `FLUX_DATA_DIR` and runs the proxy plus dashboard with no configuration.
- `docker-compose.yml` for the self-hosted stack, with a named volume for `FLUX_DATA_DIR` so the usage database survives rebuilds (`make docker-up` / `make docker-down`). The image now installs the package and runs `flux serve` rather than invoking uvicorn directly, so containers get a data directory and a persistent database by default. Note a container never has a loopback client, so `FLUX_SERVER_TOKEN` is required to use the dashboard under Docker.
- `flux` command-line entry point (`flux serve`, `flux version`; also `python -m router serve`). `serve` creates a data directory (`FLUX_DATA_DIR`, default `$XDG_DATA_HOME/flux`), points the usage database at a file inside it, and runs the proxy plus the local dashboard — so a self-hosted instance persists across restarts with no configuration. Flags: `--host --port --data-dir --db --no-dashboard --workers`, each equivalent to the matching `FLUX_*` env var. `make serve` now uses it.
- Record actual provider-reported token usage/cost (`usage_source`, `input_tokens`, `output_tokens`) instead of always billing the pre-dispatch estimate — OpenAI/Groq/Mistral `usage`, Anthropic `usage`, Google `usageMetadata`, plus the OpenAI streaming `stream_options.include_usage` chunk. Falls back to the estimate only when a provider genuinely doesn't report usage. Threaded through `BudgetTracker`, `DailyBudgetTracker`, run-budget steps, `GET /v1/usage`, and a new `flux_actual_cost_usd_total` Prometheus counter (`flux_cost_usd_total` unchanged). `FluxResponse.usage` (`DispatchUsage`) exposes it on the SDK path; `x-flux-usage-source` / `x-flux-actual-cost-usd` response headers on the HTTP proxy. See the README "Cost attribution" section and MIGRATIONS.md.
- Cost-vs-quality eval harness (`python -m router.evals`, `make evals`): runs GSM8K / MMLU / HumanEval / MT-Bench through the flux/premium/cheapest/mid strategies and reports cost-savings % and quality-retention % vs an always-premium baseline. Hybrid grading (objective + LLM-as-judge), offline simulated mode by default with a `--live` path; see [EVALS.md](./EVALS.md). Optional `flux-router[evals]` extra pulls the real datasets.
- Interactive terminal chat example (`examples/chat.py`)
- Per-provider API key loading via env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`)
- Cross-provider fallback integration test
- 9 new models with benchmark-derived quality ratings
- LRU eviction for per-customer adaptive state (bounded memory under high tenant counts)
- Per-customer `record()` rate limiter to bound adaptive-state mutation cost
- `MIGRATIONS.md` link in README documentation index
- CI job that validates the clean-installed package and the keyless demo
- CI job that builds and smoke-tests the container image
- `gpt-oss-20b` (free-tier Groq entry, replacing two Llama models Groq retired)
- `CircuitBreaker.reset()`, and a test-isolation fixture that resets it between tests alongside the existing RPM-window reset

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
- `router/models.json` catalog cleaned up: 7 retired/deprecated models removed (`claude-opus-4-20250514`, `claude-sonnet-4-20250514`, `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-2.0-flash-free`, `llama-3.3-70b`, `llama-3.1-8b-instant` — all either shut down by their provider or imminently retiring), plus 2 model IDs that were never real, separate provider SKUs removed (`claude-opus-4-20250514-extended`, `gemini-2.5-pro-thinking` — "extended thinking" and "thinking mode" are request-level parameters on the base model, not their own model IDs). The hardcoded fallback catalog in `model_registry.py` (used only if `models.json` fails to load) was pruned to match. 38 → 30 models.
- `gpt-oss-120b`'s `reasoning`/`analysis`/`general`/`unknown` quality ratings lowered to reflect a real, sourced GPQA gap (~90% for `gemini-3-flash-preview` vs ~80% for `gpt-oss-120b`) that was too narrow to survive normal registry composition changes — removing enough other models from the candidate pool could flip Step 9's cost/latency normalization and silently downgrade routing quality on reasoning/analysis/planning tasks with no code change involved. See `benchmark_verification_report.md` (not tracked; independent benchmark verification against public sources, not this repo's own ratings) for the investigation.
- Eval harness's `default_anthropic` baseline (`router/evals/strategies.py`) and default `--judge-model` (`router/evals/runner.py`, `__main__.py`) repinned from the now-retired `claude-opus-4-20250514`/`claude-sonnet-4-20250514` to `claude-opus-4-7`/`claude-sonnet-4-6`

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
- Complexity scoring was inverted for fenced code blocks: math-symbol detection matched `+=`/`/` inside code fences, so a trivial bug scored higher complexity than a genuine concurrency defect. Fences are now stripped before math detection, and four magnitude-based modifiers were added so within-task complexity actually discriminates.
- `priority=critical` requests were routed through `prefer_speed` weights, so an urgent request could land on a free-tier model with no availability guarantee. Critical/urgent priorities now use quality-led weights and exclude the free tier on reliability grounds.
- Output-token estimates were flat regardless of requested length, so cost (and therefore routing) didn't distinguish a one-line reply from a long one unless the caller passed `max_tokens`. Estimates now use explicit length cues from the prompt.
- `STEP_TYPE_FLOORS` defined a `plan: mid` floor that no code path could trigger — planning steps fell to `unknown` (no floor) and could route to a free-tier model. Planning language is now detected and floored, gated on more than "first step of a run" (a request arriving with no `X-Flux-Run-Id` gets an auto-generated `run_id`, which would otherwise floor all plain proxy chat to mid tier).
- The free-tier RPM counter had no upper bound, so a sustained burst against free-tier models could grow it unboundedly rather than reflecting the true 60-second window; documented cache-isolation claims were also out of date with the code.
- The dashboard's chart went blank when all usage fell into a single time bucket; the persistence hint referenced a stale/incorrect condition.

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
