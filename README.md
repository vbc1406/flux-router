# Vibecode Router

Intelligent LLM routing engine. Selects the best model for each request based on task type, complexity, cost, latency, quality history, and budget constraints — without any LLM calls in the routing path itself.

---

## Quick Start for New Engineers

**Read these in order (total ~50 minutes):**

1. `CODEBASE_MAP.md` — full directory structure and where to find everything
2. `router/config.py` — every tunable threshold lives here; skim all the comments
3. `router/schemas.py` — understand `RoutingRequest`, `RoutingDecision`, `ModelOption`
4. `router/routing_engine.py` — read the file header and skim the 13 numbered steps in `route()`
5. Run the tests (see below)
6. Try the interactive REPL (see below)
7. `DEBUG.md` — bookmark this for when things go wrong

**Then pick up a small task from `FEATURES.md`.**

---

## Where to Find Things

| Thing | Location |
|-------|----------|
| All config / constants | `router/config.py` |
| Routing algorithm (13 steps) | `router/routing_engine.py` → `RoutingEngine.route()` |
| Adaptive quality learning | `router/adaptive_weights.py` |
| Model definitions | `router/models.json` |
| Task classification | `router/classifier.py` |
| Provider HTTP callers | `router/provider_caller.py` |
| Error types | `router/errors.py` |
| Unit tests | `router/tests/test_*.py` |
| Manual testing tools | `testing/` |
| Debug help | `DEBUG.md` |
| How to add a feature | `FEATURES.md` |
| Schema/config migrations | `MIGRATIONS.md` |
| Full file map | `CODEBASE_MAP.md` |

---

## Running Tests

```bash
# Install dependencies
pip install pydantic structlog pytest

# Run all tests
pytest -v

# Run a specific test file
pytest -v router/tests/test_routing.py

# Run a specific test class
pytest -v router/tests/test_routing.py::TestConfidenceThreshold
```

---

## Interactive Testing

```bash
# Interactive REPL — type any prompt, get a full routing decision
python testing/router_tester.py

# Verbose mode — shows all 13 steps, filter reasons, sigmoid math
python testing/router_tester.py --verbose

# Batch mode — run all prompts in a file
python testing/router_tester.py --batch testing/test_prompts.txt

# Batch mode with CSV output
python testing/batch_runner.py testing/test_prompts.txt --csv results.csv
```

---

## Architecture in One Paragraph

A `RoutingRequest` enters `Flux.complete()` or `RoutingEngine.route()`. The engine runs 13 ordered steps: classify the request (task type + complexity score), check the cache, enforce cost ceilings, filter models by hard constraints, compress context if needed, apply routing priority rules, apply adaptive quality adjustments (learned from past responses), score and rank models within the target tier, check confidence threshold, optionally A/B explore, build fallback chains, check budget, then log and return a `RoutingDecision`. In proxy mode, it then calls the chosen provider and returns the response too.

---

## Key Design Principles

- **No LLM calls in the routing path.** Classification is purely heuristic (regexes + token counting). Target latency: < 5 ms.
- **All constants in `config.py`.** Nothing is magic-numbered in logic files.
- **Thread-safe.** All shared state uses `threading.Lock`. Safe for concurrent `route()` calls.
- **Dependency injection throughout.** `RoutingEngine` receives all its collaborators in `__init__`. This makes unit testing trivial — inject mocks.
- **Safety overrides experimentation.** Confidence fallback (upgrade to premium) always blocks A/B exploration.
