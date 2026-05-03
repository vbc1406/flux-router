# Adding New Features

## Checklist

Every feature, no matter how small, should go through this list:

- [ ] Add or update config constant(s) in `config.py` with a comment (what/why/how to tune/what breaks)
- [ ] Update `schemas.py` if you are adding a new request or response field
- [ ] Write the implementation — surgical, no unrelated changes
- [ ] Add at least one test for the new behaviour (happy path + one failure/edge case)
- [ ] Add docstring to any new public method (one line is fine; just describe what it does)
- [ ] Mark extension points with `# 🔧 EXTENSION POINT:` if the feature is meant to be extended later
- [ ] Update `CODEBASE_MAP.md` if you added a new file
- [ ] Run the full test suite: `pytest -v`
- [ ] Update this file if the process has changed

---

## Worked Examples

### Example 1 — Add model cooldown (prevent re-selecting a recently failed model)

**Step 1: Config constant**
```python
# config.py
# Seconds to skip a model after it fails before trying it again.
# Increase if providers need longer to recover; decrease for faster recovery probing.
# Setting to 0 disables cooldown entirely.
MODEL_COOLDOWN_SECONDS: int = 300
```

**Step 2: No new schema fields needed** (cooldown is internal state)

**Step 3: Implementation** — add a `_cooldown_store` dict to `RoutingEngine.__init__`,
populate it in `_post_route()` on failure, filter in `_passes_hard_constraints()`.

**Step 4: Test**
```python
def test_cooled_down_model_is_skipped():
    # Arrange: mark a model as recently failed
    # Act: route a request
    # Assert: the cooled-down model is not chosen
```

**Step 5: Docstring** on the new filter helper.

**Step 6: Extension point marker** (if cooldown strategy is meant to be swappable):
```python
# 🔧 EXTENSION POINT: swap this dict for a Redis client to share cooldown state across instances
```

**Step 7:** `CODEBASE_MAP.md` — no new file, no update needed.

**Step 8:** `pytest -v` — all green.

---

### Example 2 — Add a new task type: `data_analysis`

**Step 1: Config constant**
```python
# config.py — COMPLEXITY_BASE_SCORES
"data_analysis": 0.55,  # complex reasoning over structured data; mid/premium tier
```

**Step 2: No schema change** — `task_type` is a free string in `TaskAnalysis`.

**Step 3: Classifier** — add a regex in `classifier.py` `_detect_task_type()`:
```python
if re.search(r"\b(pivot\s+table|group\s+by|correlation|regression|dataframe|pandas)\b", prompt, re.IGNORECASE):
    return "data_analysis", 0.8
```
Add output token estimate in `_estimate_output_tokens()`:
```python
"data_analysis": 800,
```

**Step 4: Test**
```python
def test_data_analysis_task_type():
    req = _req("Build a pandas pivot table to analyze sales by region")
    analysis = classifier.analyze(req)
    assert analysis.task_type == "data_analysis"
```

**Steps 5-8:** as above.

---

### Example 3 — Add a new provider: Cohere

**Step 1: No config constant** — providers are defined by their models.

**Step 2:** No schema change needed.

**Step 3: Implementation**
- `provider_caller.py` — add a `_call_cohere()` async function following the pattern of `_call_openai()`.
- Wire it into `call_provider()` dispatch dict.
- `models.json` — add Cohere model entries with `"provider": "cohere"`.
- `model_registry.py` — no code change needed (registry loads from `models.json`).

**Step 4: Test**
```python
def test_cohere_provider_call_formats_correctly():
    # Mock the HTTP layer, verify the request body matches Cohere's API spec
```

**Step 5:** Docstring on the new `_call_cohere()` function.

**Step 6:** No extension point marker needed.

**Step 7:** `CODEBASE_MAP.md` — update the "Provider Integrations" section.

**Step 8:** `pytest -v` — all green.

---

### Example 4 — Add a new routing_priority value: `latency-first`

**Step 1: Config constants**
```python
# config.py
VALID_ROUTING_PRIORITIES: frozenset[str] = frozenset({
    "always-premium", "quality-first", "balanced", "cost-optimized",
    "latency-first",   # NEW: minimise latency above all else
})

# Add to SCORING_WEIGHTS:
"latency-first": {"quality": 0.20, "cost": 0.20, "latency": 0.60},
```

**Step 2:** `schemas.py` — no change needed (routing_priority is a free string validated at call time).

**Step 3: Implementation** — `routing_engine.py` `_get_weights_for_priority()`: add `"latency-first"` to the dispatch, same pattern as `"quality-first"`.

**Step 4: Test**
```python
def test_latency_first_routes_to_fastest_model():
    d = rr(engine.route(_req("hello", routing_priority="latency-first")))
    assert d.chosen_model.avg_latency_ms <= some_threshold
```

**Steps 5-8:** as above.

---

## Common Mistakes to Avoid

- **Magic numbers in logic files.** Always define a constant in `config.py` first.
- **Changing `routing_engine.py` step order.** The 13 steps have deliberate dependencies (e.g., Step 1 must precede Step 2; Step 9b must precede Step 10). Document why if you reorder.
- **Forgetting to update `VALID_ROUTING_PRIORITIES`** when adding a priority tag. The validator will raise at route time, but your tests will catch it first.
- **Mutating registry `ModelOption` objects directly.** The engine works on `.model_copy()` instances. Mutating the registry object leaks state across requests.
- **Adding a new field to `RoutingDecision` without a default.** Existing callers that don't set the new field will break. Always provide a default value.
- **Skipping the test.** Even a one-line assertion is better than nothing — it documents the intended behaviour and prevents silent regression.
