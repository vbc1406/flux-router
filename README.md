# Flux

**Fast, cost-aware heuristic LLM router. Cut spend without sacrificing quality.**

Flux is a heuristic router that picks a low-cost model satisfying configured quality and capability constraints for each request. Routing is pure Python logic — no LLM calls in the decision path. Routing overhead is a few milliseconds or less, and the adaptive weights learn from every response.

---

## Why Flux

Most teams default to a single premium model for every request. That's expensive and often unnecessary — a short summarization or classification call usually doesn't need a top-tier model.

Flux routes each request to a low-cost model that satisfies the configured quality/capability constraints, based on:

- **Task classification** — code, summarization, reasoning, creative, etc.
- **Complexity scoring** — short greeting vs. multi-step proof
- **Adaptive learning** — models that consistently perform well get prioritized
- **Cost ceilings** — never burn budget on a single request
- **Fallback chains** — automatic retry on rate limits, timeouts, or content filters

The model registry includes current-generation models from OpenAI, Anthropic, Google, Mistral, and Groq; the router selects per-request based on the configured quality floor and constraints, not a fixed mapping.

---

## Quickstart

### Option 1: `base_url` swap (no code changes)

Run Flux as a local HTTP proxy and point your existing OpenAI SDK client at it —
no call-site rewrites required.

```bash
pip install -e ".[server]"
export OPENAI_API_KEY=sk-...        # + whichever other provider keys you use
make serve                          # or: uvicorn router.server:app
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="flux-auto",              # or "flux-cheap" / "flux-quality"
    messages=[{"role": "user", "content": "Explain backpropagation in two sentences."}],
)
print(resp.choices[0].message.content)
print("routed to:", resp.model)
```

`model` is a routing directive, not a literal model name: `flux-auto` routes
normally, `flux-cheap` forces cost-optimized routing, `flux-quality` forces
quality-first routing. Passing a concrete model ID (e.g. `gpt-4o`) bypasses
routing entirely and calls that model verbatim. Streaming (`stream: true`) is
supported. Routing metadata (chosen model, task type, complexity, estimated
cost, decision latency) comes back on `x-flux-*` response headers.

By default the server binds to `127.0.0.1` only and logs a warning. Set
`FLUX_SERVER_TOKEN` to require `Authorization: Bearer <token>` on every
request and allow non-loopback binding — see `router/config.py` for the rest
of the `SERVER_*` settings.

### Option 2: Python import

```bash
pip install flux-router
```

Or from a clone, editable:

```bash
pip install -e .
```

Try it with **no API keys** — the demo routes 25 sample prompts through the full
stack with mocked provider calls and prints what it picked for each:

```bash
python -m router.demo
```

For an interactive prompt that routes what you type (needs real keys, see below):

```bash
python examples/chat.py
```

Export the keys for the providers you want Flux to route to. Each provider has
its own variable — Flux dispatches the right one based on the model it picks:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export GROQ_API_KEY=gsk_...
export MISTRAL_API_KEY=...
```

You only need keys for the providers whose models you want eligible — missing
keys mean those providers' models will fail at call time, so restrict the
candidate set via routing constraints if you're only using a subset.

Then:

```python
import asyncio
from router.flux import make_flux

flux = make_flux()  # reads keys from the env vars above
resp = asyncio.run(flux.complete("Explain backpropagation in two sentences."))
print(resp.text)
print("routed to:", resp.model.display_name)
```

Programmatic alternatives:

```python
# One key for all providers (e.g., an OpenAI-compatible gateway like OpenRouter)
flux = make_flux(api_key="sk-...")

# Explicit per-provider keys (overrides env vars)
flux = make_flux(api_keys={
    "openai": "sk-...",
    "anthropic": "sk-ant-...",
})
```

Resolution order per request: explicit `api_keys=` / env var → per-request
`provider_api_key` → legacy single `api_key=`.

### Run-scoped budgets (for agent loops)

Per-request cost ceilings don't stop a runaway multi-step agent loop — by the
time any single step looks expensive, the loop has already made 40 of them.
`flux.start_run()` caps a whole trajectory instead of one call:

```python
from router import RunBudgetExceeded

with flux.start_run(max_cost_usd=0.10, max_steps=50) as run_id:
    for step in agent_steps:
        try:
            resp = await flux.complete(step.prompt, run_id=run_id)
        except RunBudgetExceeded as exc:
            # exc.summary: steps_taken, total_cost_usd, per-step breakdown
            break
```

As the run's spend approaches the cap, Flux automatically forces
cost-optimized routing for the rest of the run (`RoutingDecision.budget_state
== "degraded"`), then flags `budget_warning` so the caller can choose to wrap
up — and only raises `RunBudgetExceeded` once a limit is actually hit,
**before** the next step dispatches, never after it spends. On the HTTP
proxy, tag repeated calls with the same `X-Flux-Run-Id` header to get the
same enforcement without any Python. See `examples/agent_loop.py`.

---

## Cost attribution: actual vs. estimated

Every routing decision carries a pre-dispatch cost **estimate**
(`RoutingDecision.estimated_cost`) — Flux has to know a request's likely
cost before it picks a model and before it's dispatched, so budget checks
and run-budget reservations are estimates by definition and always will be.

What gets **recorded** as spend after dispatch is a different matter. Flux
uses the provider's own reported token usage whenever the provider returns
it (OpenAI/Groq/Mistral's `usage` object on non-streaming responses and the
`stream_options.include_usage` chunk on streaming ones; Anthropic's
`usage`; Google's `usageMetadata`) — the cost recorded to `BudgetTracker`,
`DailyBudgetTracker`, run-budget steps, and `/v1/usage`/`/metrics` is
computed from those actual tokens at the dispatched model's rates
(`provider_caller.compute_actual_cost`), not the pre-dispatch guess. Only
when a provider genuinely doesn't report usage (or reports something
untrustworthy — zero, negative, or the wrong type) does recording fall back
to the pre-dispatch estimate.

Every recorded row — `GET /v1/usage`, the `flux_cost_usd_total` /
`flux_actual_cost_usd_total` Prometheus counters, and the non-streaming
response's `usage` block / `x-flux-usage-source` header — is labeled
`usage_source: "provider"` or `"estimated"` so you always know which one
you're looking at. `flux_cost_usd_total` is every recorded dollar, actual or
estimated; `flux_actual_cost_usd_total` is the subset backed by real
provider usage — the gap between the two is your exposure to estimate drift.

```python
resp = await flux.complete("Explain backpropagation")
print(resp.usage.usage_source)   # "provider" or "estimated"
print(resp.usage.cost_usd)       # what was actually billed
print(resp.usage.input_tokens, resp.usage.output_tokens)  # None if estimated
```

See [MIGRATIONS.md](./MIGRATIONS.md) for the `usage` table's `usage_source`/
`input_tokens`/`output_tokens` columns and how an existing on-disk
`usage.db` migrates automatically.

---

## How Flux compares

Flux overlaps with LiteLLM, OpenRouter, and similar tools. The differences that matter:

| | Flux | LiteLLM | OpenRouter |
|---|---|---|---|
| Routing decision | Pure-Python heuristic, sub-millisecond (~0.23 ms P50) | Config-driven, no automatic per-task selection | Server-side, network round trip |
| Per-task model selection | Yes (15 task types, complexity scoring) | Manual | Manual |
| Adaptive learning from response quality | Yes (per-(model, task) EMA, optional per-customer) | No | No |
| Typed fallback chains (rate-limit / timeout / content-filter) | Yes, separate per failure mode | Yes (single chain) | Yes |
| Runs in your infrastructure | Yes | Yes | No (hosted) |
| License | AGPL-3.0 (or commercial) | MIT | Proprietary |

Use LiteLLM if you want a permissively-licensed SDK and you're happy choosing models yourself. Use OpenRouter if you want a managed gateway and don't mind sending traffic through a third party. Use Flux if you want per-request automatic model selection that learns from outcomes, running entirely in your own infrastructure.

---

## Documentation

- [SECURITY_ARCHITECTURE.md](./SECURITY_ARCHITECTURE.md) — data flows, code-level guarantees, multi-tenant caveats
- [CODEBASE_MAP.md](./CODEBASE_MAP.md) — directory layout and per-file purpose
- [FEATURES.md](./FEATURES.md) — extension points for adding providers, models, or task types
- [EVALS.md](./EVALS.md) — cost-vs-quality eval harness (`python -m router.evals`)
- [DEBUG.md](./DEBUG.md) — troubleshooting
- [MIGRATIONS.md](./MIGRATIONS.md) — schema and config migration guide
- [CHANGELOG.md](./CHANGELOG.md) — release notes

---

## Licensing

Flux is licensed under **AGPL-3.0**. See [LICENSE](./LICENSE) for the full text. AGPL is fine if you're self-hosting Flux for your own use or shipping it as part of an open-source project.

If you want to embed Flux in a closed-source product, offer Flux as a hosted service, or otherwise distribute it without releasing your own source under AGPL, you need a commercial license. Contact **fluxllmdev@gmail.com**.

---

## What Flux Is Not

- Not a hosted API. Flux runs in your infrastructure. Flux never calls an LLM to make routing decisions. In proxy mode, prompts are sent only to the selected provider.
- Not a community project. Flux is built and maintained by the Flux team. Issues are welcome; pull requests are not currently accepted.

---

## Security

Flux is designed so that:

- API keys never appear in logs, error messages, or serialized output
- Prompts and responses are never logged at INFO/WARN level
- All file writes are atomic (no corruption on crash)
- Resource caps prevent memory exhaustion under load

For full guarantees and code-level verification steps, see [SECURITY_ARCHITECTURE.md](./SECURITY_ARCHITECTURE.md).

To report a vulnerability, see [SECURITY.md](./SECURITY.md).

---

## Contact

- Bug reports: GitHub Issues
- Security: fluxllmdev@gmail.com
- Licensing: fluxllmdev@gmail.com
- Everything else: fluxllmdev@gmail.com
