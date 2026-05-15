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

## What Flux Is Not

- Not a hosted API. Flux runs in your infrastructure. Flux never calls an LLM to make routing decisions. In proxy mode, prompts are sent only to the selected provider.
- Not a community project. Flux is built and maintained by the Flux team. Issues are welcome; pull requests are not currently accepted.
- Not a free SaaS. AGPL-3.0 licensed for self-hosting. For commercial licensing or hosted offerings, contact licensing@flux.dev.

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

## License

Flux is licensed under **AGPL-3.0**. See [LICENSE](./LICENSE) for the full text.

If you want to embed Flux in a closed-source product or offer Flux as a hosted service, contact licensing@flux.dev for a commercial license.

---

## Contact

- Bug reports: GitHub Issues
- Security: security@flux.dev
- Licensing: licensing@flux.dev
- Everything else: hello@flux.dev
