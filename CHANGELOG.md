# Changelog

All notable changes to Flux will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
