# Flux

**Intelligent LLM routing. Save 30–50% on AI costs. Automatically.**

Flux picks the optimal model for each request — cheap models for simple tasks, premium for complex ones — without any LLM calls in the routing path. Pure logic, sub-5ms latency, learns from every response.

---

## Why Flux

Most teams default to GPT-4 or Claude Opus for everything. That's expensive and unnecessary. A summarization request doesn't need a $0.01/1k-token model when a $0.001/1k-token model gives the same answer.

Flux routes each request to the cheapest model that can handle it, based on:

- **Task classification** — code, summarization, reasoning, creative, etc.
- **Complexity scoring** — short greeting vs. multi-step proof
- **Adaptive learning** — models that consistently perform well get prioritized
- **Cost ceilings** — never burn budget on a single request
- **Fallback chains** — automatic retry on rate limits, timeouts, or content filters

Result: same quality output, half the cost.

---

## What Flux Is Not

- Not a hosted API. Flux runs in your infrastructure. Your API keys never leave your environment.
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
