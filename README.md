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

Install (from a clone):

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

---

## How Flux compares

Flux overlaps with LiteLLM, OpenRouter, and similar tools. The differences that matter:

| | Flux | LiteLLM | OpenRouter |
|---|---|---|---|
| Routing decision | Pure-Python heuristic, sub-millisecond (~0.6 ms P50) | Config-driven, no automatic per-task selection | Server-side, network round trip |
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
- [CHANGELOG.md](./CHANGELOG.md) — release notes
- [DEBUG.md](./DEBUG.md) — troubleshooting

---

## Licensing

Flux is licensed under **AGPL-3.0**. See [LICENSE](./LICENSE) for the full text. AGPL is fine if you're self-hosting Flux for your own use or shipping it as part of an open-source project.

If you want to embed Flux in a closed-source product, offer Flux as a hosted service, or otherwise distribute it without releasing your own source under AGPL, you need a commercial license. Contact **licensing@flux.dev**.

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
- Security: security@flux.dev
- Licensing: licensing@flux.dev
- Everything else: hello@flux.dev
