# Flux

**A self-hosted proxy that routes each request to the cheapest model that meets your quality bar — and stops an agent loop from spending past a budget you set.**

Flux is two things at once:

1. **A router.** Every request is scored for task type, complexity, and required capabilities, then sent to the cheapest model that clears those bars. Routing is pure Python — no LLM call in the decision path — and takes a few milliseconds.
2. **Spend control for agents.** A single expensive call is easy to notice; a 40-step agent loop that quietly triples its budget is not. `flux.start_run(max_cost_usd=..., max_steps=...)` caps a whole trajectory, not one call, and Flux downgrades to cheaper routing automatically as a run approaches its cap instead of just erroring out at the end.

Both run through the same proxy, so you get cost savings and a spending ceiling from one deployment, not two separate tools.

---

## Why Flux

Most teams default to a single premium model for every request — expensive, and usually unnecessary for a short summarization or classification call.

Flux fixes that from two directions:

**Cheaper routing, automatically:**
- **Task classification** — code, summarization, reasoning, creative, etc.
- **Complexity scoring** — short greeting vs. multi-step proof
- **Adaptive learning** — models that consistently perform well get prioritized
- **Cost ceilings** — never burn budget on a single request
- **Fallback chains** — automatic retry on rate limits, timeouts, or content filters

**A hard ceiling on agent spend:**
- **Run-scoped budgets** — cap cost and step count for an entire agent trajectory, enforced before each step dispatches, not after
- **Per-tenant budget plans** — give each caller (or customer) its own daily cap, server-bound so it can't be spoofed by the caller
- **Rate limiting** — bound request volume per tenant, not just per request

The model registry includes current-generation models from OpenAI, Anthropic, Google, Mistral, and Groq; the router selects per-request based on the configured quality floor and constraints, not a fixed mapping.

---

## Quickstart

The fastest path is running Flux as a local HTTP proxy — point your existing OpenAI SDK client at it and nothing else in your code changes.

**Step 1 — Check your Python version.** Flux needs 3.10, 3.11, or 3.12.

```bash
python3 --version
```

**Step 2 — Clone and install.** Flux isn't on PyPI, so it installs straight from the repo.

```bash
git clone https://github.com/vbc1406/flux.git
cd flux
pip install -e ".[server]"
```

**Step 3 — Set a provider API key.** Only export keys for providers you actually want Flux routing to.

```bash
export OPENAI_API_KEY=sk-...
```

**Step 4 — Start the server.**

```bash
flux serve
```

You'll see something like this — the important lines are the API and dashboard URLs:

```
Flux 1.0.0
  data dir:  /home/you/.local/share/flux
  usage db:  /home/you/.local/share/flux/flux.db
  API:       http://127.0.0.1:8000/v1
  dashboard: http://127.0.0.1:8000/dashboard
```

Spend history is written to that usage database automatically, so it survives a restart. `flux serve --help` lists every flag; each one also has a `FLUX_*` environment variable if you'd rather configure it that way (see [SELF_HOSTING.md](./SELF_HOSTING.md)).

**Step 5 — Point your existing OpenAI client at it.** No other code changes needed.

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

`model` is a routing directive, not a literal model name: `flux-auto` routes normally, `flux-cheap` forces cost-optimized routing, `flux-quality` forces quality-first routing. Passing a concrete model ID (e.g. `gpt-4o`) bypasses routing entirely and calls that model verbatim. Streaming (`stream: true`) is supported. Routing metadata (chosen model, task type, complexity, estimated cost, decision latency) comes back on `x-flux-*` response headers.

By default the server only listens on `127.0.0.1` (your machine, nobody else). If you want to reach it from another machine or a Docker container, set `FLUX_SERVER_TOKEN` first — that requires an `Authorization: Bearer <token>` header on every request and unlocks non-loopback binding. See `router/config.py` for the rest of the `SERVER_*` settings.

### Prefer Docker?

```bash
FLUX_SERVER_TOKEN=$(openssl rand -hex 32) docker compose up -d   # or: make docker-up
```

This mounts a volume so the usage database survives `docker compose down` and image rebuilds. A token is required for the dashboard under Docker — see [SELF_HOSTING.md](./SELF_HOSTING.md#docker) for why.

### Prefer importing Flux directly into Python instead of running the proxy?

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

Multi-worker deployments (`flux serve --workers N`) run budgets
**under-enforce by default** — each worker only sees the steps it personally
handled, since run state is process-local. Set `FLUX_RUN_STORE=redis` (plus
`FLUX_REDIS_URL`) to share it across workers; the server warns at startup if
you're running multiple workers without it. See SELF_HOSTING.md's "Multiple
workers" section for the rate-limit caveat that goes with it.

### Agent step types (`X-Flux-Step-Type`)

Different points in an agent trajectory warrant different quality floors — a
planning step shouldn't get routed to a model too weak to plan with, even
while the rest of a budget-constrained run is on cost-optimized routing.
`RoutingRequest.step_type` (SDK) / the `X-Flux-Step-Type` header (HTTP proxy)
tags a request with one of:

```
plan · tool_select · tool_result_summarize · reflect · extract · format · final_answer · unknown
```

An explicit value always wins over inference and gets its configured floor
(`STEP_TYPE_FLOORS` in `router/config.py` — e.g. `plan` and `tool_select`
floor at `mid`, `tool_result_summarize` has no floor at all). Leave it unset
and Flux infers it from the request shape (presence of `tools`,
`response_format`, a `role: "tool"` message in history, planning language in
the prompt) — an invalid explicit value is rejected with `400`. See
`examples/agent_loop.py` for the SDK form (`step_type="plan"` /
`step_type="final_answer"`).

```python
resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"X-Flux-Step-Type": "plan", "X-Flux-Run-Id": run_id},
    json={"model": "flux-auto", "messages": [...]},
)
```

Agent steps now also carry an optional **step-specific quality signal**:
`ModelOption.step_quality_ratings` (keyed by the same step-type strings above)
lets a catalog entry document that it's specifically strong or weak at, say,
`tool_select` — independent of its general per-task-type `quality_ratings`.
When a request's `step_type` is known and the chosen model has an entry for
it, routing scores on that figure instead of the flat task-type rating,
falling back to `quality_ratings[task_type]` → `quality_ratings["general"]` →
`0.5` when absent (`routing_engine._resolve_quality()`). No catalog entry
ships with fabricated step ratings today — the field is opt-in, and every
model routes on its existing task-type rating until real step-level data is
added.

---

## High-stakes domain routing and hard-complexity escalation

Two more mandatory **minimum-tier floors** compose with the step-type floor
above, all enforced before cost scoring ever runs (`_passes_hard_constraints`
in Step 4 of the routing pipeline — see `routing_engine._composed_min_tier()`
for how they combine):

- **Legal / medical judgment calls.** A prompt asking Flux to render a
  substantive legal or medical judgment — diagnosis, treatment, medication,
  emergency triage, contractual liability, regulatory compliance, or advice
  affecting someone's legal rights — classifies as `task_type="legal"` /
  `"medical"` (`classifier.py`'s `_LEGAL_SUBSTANTIVE_RE` /
  `_MEDICAL_SUBSTANTIVE_RE`) and is floored to `DOMAIN_TIER_FLOORS`'s tier
  (`premium` by default; override with `FLUX_LEGAL_TIER_FLOOR` /
  `FLUX_MEDICAL_TIER_FLOOR`). A bare mention of "doctor," "contract," or
  "court" does **not** trigger this — detection requires a judgment-seeking
  phrase, so pure transformations of supplied text ("summarize this medical
  report," "extract the parties from this contract") keep routing on their
  normal, cheaper task_type. This is independent of and additional to
  sensitivity/confidentiality filtering (`allowed_sensitivity_levels`) — the
  domain floor controls *capability*, sensitivity controls *which providers
  may see the data*; both apply simultaneously.

  Detection is a conservative, rule-based routing guard—not a medical or
  legal safety system and not a substitute for application-level review.
  Deployments serving these domains should keep the premium floors enabled,
  test their own real user phrasing, and provide appropriate professional-
  advice disclaimers and emergency handling outside Flux.
- **Hard-complexity escalation.** A request whose `complexity_score` clears
  `HARD_COMPLEXITY_TIER_FLOORS`'s threshold (`0.85` by default, override with
  `FLUX_HARD_COMPLEXITY_PREMIUM_THRESHOLD`) is floored to `premium`
  regardless of task_type — catching very hard coding/proof requests that a
  static per-task-type quality rating alone might still leave mid-tier.

All floors compose by taking the **strongest** applicable one (highest tier),
never by overriding each other, and are always enforced ahead of
`always-premium`/`quality_max`/`cascade`/explicit-overrides/budget
walk-down — every routing priority selects from the same already-floor-
filtered candidate set, so a floor can't be silently bypassed downstream. If
a mandatory floor leaves zero eligible models (e.g. after budget/plan/context
filtering), routing returns a `chosen_model=None` decision with
`routing_rule_matched="no_candidates"` and a `reasoning` string naming the
floor that caused it — never an under-qualified selection. Pass
`verbose=True` to see exactly which floors applied on a given decision via
`RoutingDecision.explanation.floors_applied` (e.g.
`["agent_step:plan → mid", "domain:medical → premium",
"complexity:0.91 → premium"]`).

### The `long_document` capability

`required_capabilities=["long_document"]` is a **derived** capability, not a
catalog tag — no `models.json` entry lists it explicitly. A model satisfies
it when its `max_context_window` covers the request's calculated context
need plus a safety margin (`LONG_DOCUMENT_CONTEXT_SAFETY_MARGIN`, 2000 tokens
by default, override with `FLUX_LONG_DOCUMENT_SAFETY_MARGIN`) — not merely
equal it. This works for both short prompts (trivially satisfied by nearly
every model) and genuinely large ones (only satisfied by models with enough
headroom), and composes normally with every other hard constraint. A request
whose input exceeds every eligible model's context still returns a clean
`no_candidates` decision. Automatic long-document detection — a prompt over
~2000 words classifying as `task_type="long_document"` — is unrelated and
unaffected; that's a `task_type`, not a capability.

---

## Tool calling

`tools`, `tool_choice`, and `response_format` are forwarded to the routed
provider and translated into that provider's native request/response shape
— not just used as a routing signal. Assistant `tool_calls` (with `id`,
`function.name`, `function.arguments`) and tool-role messages round-trip
through both the Python SDK (`FluxResponse.tool_calls` /
`FluxResponse.finish_reason`) and the HTTP proxy's OpenAI-compatible response
body, including streamed `tool_calls` deltas for OpenAI-compatible providers.

| | OpenAI | Groq | Mistral | Google | Anthropic |
|---|---|---|---|---|---|
| `tools` / `tool_choice` | ✅ | ✅ | ✅ | ✅ (translated) | ✅ (translated) |
| `response_format` | ✅ | ✅ | ✅ | ✅ (translated) | ❌ — no native equivalent |
| Streamed `tool_calls` deltas | ✅ | ✅ | ✅ | — (no native streaming caller) | — (no native streaming caller) |

Anthropic has no JSON-mode/`response_format` equivalent — a request that sets
`response_format` and routes to an Anthropic model is **rejected** with a
clear `400` (`UnsupportedFeatureError`) rather than silently ignoring the
field or guessing at an emulation. Route around it (a different provider, or
drop `response_format`) if you need both on the same request. Google and
Anthropic have no native streaming caller in `provider_caller.py`, so the
proxy falls back to a synthesized single, non-incremental chunk for them —
a tool call still comes through, just not as incremental deltas.

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

This section is about **cost**: actual vs. estimated dollars. A separate,
easy-to-conflate distinction is about **quality**: `RoutingDecision.
estimated_savings` and the dashboard's `estimated_savings_usd` are cost
projections (this request vs. the registry's most expensive model — nobody
ran that counterfactual), not a claim that quality was measured to be
unaffected. Whether a cheaper model actually holds up on real answers is a
separate question, and it's the one [EVALS.md](./EVALS.md)'s harness exists
to answer — with its own explicit **SIMULATED** (mock, catalog-adjacent)
vs. **MEASURED** (`--live`, real graded completions) labeling. Don't cite a
mock-mode eval number as a quality claim; don't cite `estimated_savings_usd`
as a quality claim either — they're both estimates of different things.

---

## The dashboard

`flux serve` also serves a local operator console at
`http://127.0.0.1:8000/dashboard` — plain HTML and JavaScript, no build step,
no CDN, served by the same process as the API.

It answers the question a router should be able to answer: where did the money
go? Spend and savings over time, a per-model and per-task breakdown, latency
percentiles, cache-hit and fallback rates, the live model registry with
pricing, and the deployment's effective configuration. The window selector
covers 1h through all-time.

Everything it renders comes from `GET /v1/stats/*`, which you can query
directly — there is nothing available to the dashboard that isn't available to
you.

The usage database behind it holds **costs and metadata only**; no prompt or
completion text is ever written to disk. And the dashboard shows every tenant's
spend, so it is served to loopback clients only unless you configure a token.

See [SELF_HOSTING.md](./SELF_HOSTING.md) for the full guide: flags, the data
directory, the stats API reference, Docker, multi-worker caveats, and the
access model.

---

## How Flux compares

Flux overlaps with LiteLLM, OpenRouter, and Portkey. The differences that matter:

| | Flux | LiteLLM | OpenRouter | Portkey |
|---|---|---|---|---|
| Routing decision | Pure-Python heuristic, sub-millisecond (~0.23 ms P50) | Config-driven, no automatic per-task selection | Server-side, network round trip | Server-side, network round trip |
| Per-task model selection | Yes (15 task types, complexity scoring) | Manual | Manual | Manual / rule-based |
| Adaptive learning from response quality | Yes (per-(model, task) EMA, optional per-customer) | No | No | No |
| Typed fallback chains (rate-limit / timeout / content-filter) | Yes, separate per failure mode | Yes (single chain) | Yes | Yes |
| Runs in your infrastructure | Yes | Yes | No (hosted) | Gateway self-hostable; observability is hosted |
| Where your API keys live | Your machine, only | Your machine | Their platform | Their vault |
| Cost dashboard | Local, in-process, no account | Separate hosted/self-hosted UI | Hosted | Hosted |
| License | AGPL-3.0 (or commercial) | MIT | Proprietary | Proprietary (OSS gateway) |

The structural difference is where the data ends up. A hosted gateway sees
every prompt you route through it, and a hosted observability layer sees your
spend, your traffic shape, and your model mix. Flux is a process on your
machine writing to a SQLite file on your disk: the dashboard is served by the
same process, over loopback, with no account, no telemetry, and no callback
home. The only outbound network calls Flux makes are to the model providers
you configured.

That matters most when prompts are the sensitive asset — regulated data,
proprietary agent traces, customer content under a DPA that doesn't cover a
third-party gateway.

Use **LiteLLM** if you want a permissively-licensed SDK and you're happy
choosing models yourself. Use **OpenRouter** if you want a managed gateway and
don't mind third-party traffic. Use **Portkey** if you want hosted
observability and guardrails as a product. Use **Flux** if you want per-request
automatic model selection that learns from outcomes, and you want the routing,
the keys, and the spend history to stay on hardware you control.

---

## Documentation

- [SELF_HOSTING.md](./SELF_HOSTING.md) — running `flux serve`: flags, persistence, the dashboard, the stats API, Docker
- [OPERATIONS.md](./OPERATIONS.md) — deployment, rollback, backups, key rotation, alerts, and incident response
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
