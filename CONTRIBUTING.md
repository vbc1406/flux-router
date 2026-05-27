# Contributing to Flux

Thanks for your interest in Flux. Please read this before opening anything.

## Pull requests are not currently accepted

Flux is built and maintained by the Flux team. It is not a community-developed
project. **We do not accept pull requests at this time** — PRs will be closed
unread. This keeps the licensing and provenance of the codebase unambiguous
(Flux is dual-licensed AGPL-3.0 / commercial).

If you have a fix or feature in mind, please open an issue describing it rather
than sending code.

## Issues are welcome

Bug reports, questions, and feature requests are genuinely useful — please file
them as [GitHub Issues](https://github.com/flux-ai/flux/issues).

A good bug report includes:

- **What you did** — the smallest snippet that reproduces it (the
  `RoutingRequest` / `make_flux(...)` call, the prompt, the priority).
- **What you expected** vs. **what happened** — include the routing decision
  or error message. Run with `verbose=True` if it's a routing question.
- **Environment** — Flux version, Python version (3.10+), OS.
- **Logs** — relevant structlog output. Do **not** paste API keys; Flux
  redacts them, but double-check before sharing.

## Security issues

Do **not** open a public issue for vulnerabilities. Email security@flux.dev —
see [SECURITY.md](./SECURITY.md) for the disclosure process.

## Trying it locally

```bash
pip install -e ".[dev]"
python -m router.demo      # keyless, mocked — no API keys needed
make test                  # run the suite
make lint                  # ruff
```

See [CODEBASE_MAP.md](./CODEBASE_MAP.md) for the directory layout and
[FEATURES.md](./FEATURES.md) for extension points.
