# Debugging Guide

## Quick Commands

```bash
# Run full test suite
pytest -v

# Run a specific test file
pytest -v router/tests/test_routing.py

# Run interactive router REPL (test any prompt manually)
python testing/router_tester.py

# Run with verbose mode (shows all 13 steps, filter reasons, sigmoid math)
python testing/router_tester.py --verbose

# Run a batch of prompts
python testing/router_tester.py --batch testing/test_prompts.txt

# Tail the live analytics log
tail -f router/routing_analytics.jsonl | python -m json.tool

# Check what models are loaded
python -c "from router.model_registry import ModelRegistry; m=ModelRegistry(); print([x.model_id for x in m.all_available_models()])"

# Check current adaptive weights state
python -c "import json; d=json.load(open('router/adaptive_state.json')); print(json.dumps(d, indent=2))"
```

---

## Problem: Adaptive Weights Not Updating

**Symptoms:** Routing decisions never seem to improve even after many requests. Adjusted quality always equals base quality.

**Where to check:**
- `router/adaptive_state.json` — does it exist? Is it being written to?
- `AdaptiveWeights._dirty` — is it incrementing? It flushes every 50 updates (`_WRITE_INTERVAL`).
- `ADAPTIVE_MIN_SAMPLES` in `config.py` — currently 20. Weights don't kick in until this many samples are recorded for a given `(model_id, task_type)` pair.

**Common causes:**
1. `state_file=None` was passed to `AdaptiveWeights()` (in-memory only, no persistence — intentional in tests).
2. The `quality_score` being fed to `record()` is always being rejected as an outlier. Check the `adaptive_signal_rejected_outlier` log line — if it appears constantly, the running mean may be corrupted. Check `_signal_stats` for the key.
3. `ADAPTIVE_LEARNING_ENABLED` is `False` in config (check `config.py`).
4. The `(model_id, task_type)` key used in `record()` doesn't match the key used in `get_adjusted_score()` — they must be identical strings.

**How to safely debug:**
```python
aw = AdaptiveWeights()
print(aw._state)               # current EMA state
print(aw._signal_stats)        # running mean/variance per key
print(aw._dirty)               # updates since last flush
print(aw._total_signals)       # lifetime signal count
```

---

## Problem: Rollback Triggering Too Often

**Symptoms:** `adaptive_weights_rolled_back` appears frequently in logs. Quality scores reset repeatedly.

**Where to check:**
- `_ROLLBACK_DROP_THRESHOLD` in `adaptive_weights.py` — currently 0.90 (roll back if avg drops below 90% of snapshot baseline).
- `_SNAPSHOT_INTERVAL` — currently 1000 signals. If volume is low, the first snapshot may take a long time, leaving `_last_snapshot_avg = None` and blocking rollbacks entirely.
- Look for `adaptive_quality_clamped` log events — repeated out-of-range scores (from buggy quality scorer output) can corrupt the EMA and push the average below the threshold.

**Common causes:**
1. The quality scorer is returning scores outside [0, 1]. Fixed by the clamping guard in `record()` but check `adaptive_quality_clamped` in logs.
2. The snapshot baseline was taken during an unusually good period, making the threshold unreachable under normal conditions. Tune `_ROLLBACK_DROP_THRESHOLD` upward (e.g., 0.80) if rollbacks are too sensitive.
3. A burst of legitimate bad results (e.g., a provider degraded) is correctly triggering rollback — this is the intended behaviour. Check provider health.

**Log lines to inspect:**
```
adaptive_weights_rolled_back      current_avg=X snapshot_avg=Y
adaptive_weights_snapshot_taken   avg_quality=X snapshot_count=Y
adaptive_quality_clamped          key=model:task original=X
```

---

## Problem: Bad Routing Decisions

**Symptoms:** Requests are routed to obviously wrong models (too cheap for complex tasks, too expensive for trivial ones).

**Where to check:**
1. **Classifier output** — run the REPL with `--verbose` to see the full `TaskAnalysis`. Is the `task_type` and `complexity_score` sensible?
   ```
   python testing/router_tester.py --verbose
   > Your prompt here
   ```
2. **Tier boundaries** — `TIER_BOUNDARIES` in `config.py`. Is the complexity score landing in the right tier?
3. **Scoring weights** — `SCORING_WEIGHTS` in `config.py`. A `cost-optimized` routing priority will always prefer cheaper models.
4. **Adaptive quality** — check `router/adaptive_state.json`. Has the adaptive system learned a bad score for this model/task pair? Reset it by deleting the file (in-memory state is lost on restart; persistence restores it on next startup).
5. **Routing rule** — `decision.routing_rule_matched` tells you exactly which path was taken. Common values: `tier_selection`, `trivial_request`, `always_premium`, `ab_exploration`, `budget_downgraded`, `confidence_fallback`.

**Log lines to inspect:**
```
classified     cid=X task=Y score=Z
```

---

## Problem: Confidence Fallback Triggering Too Often

**Symptoms:** Many decisions have `confidence_fallback=True` and route to premium unexpectedly.

**Where to check:**
- `MIN_CONFIDENCE_THRESHOLD` in `config.py` — currently 0.60 (configurable via env var `MIN_CONFIDENCE_THRESHOLD`).
- The `routing_score` of the winning model — check the verbose REPL output.

**Common causes:**
1. The threshold is too high for your model mix. Lower it (e.g., 0.50) if premium fallback is too aggressive.
2. Adaptive weights have learned poor scores for mid/cheap models, depressing their `routing_score`. Check `adaptive_state.json`.
3. A specific task type has a very low base score and no premium alternative exists in the candidate set (e.g., the plan is `free_plan` — premium models are filtered out in Step 4, so confidence fallback silently does nothing).

**How to tune:** Set `MIN_CONFIDENCE_THRESHOLD=0.50` via env var and observe. Or set to `0.0` to disable it entirely.

---

## Problem: A/B Exploration Behaving Unexpectedly

**Symptoms:** Some requests are routed to cheaper/different models unexpectedly.

**Where to check:**
- `exploration_rate` on the request — default is `0.10` (10% of requests explore).
- `AB_MAX_EXPLORATION_RATE` in `config.py` — hard ceiling of 0.25.
- `AB_ALLOWED_PRIORITIES` — A/B only fires on `low` and `normal` priority.
- `AB_MAX_COMPLEXITY_SCORE` — A/B is blocked for requests with complexity > 0.70.
- `AB_BLOCKED_SENSITIVITY_LEVELS` — A/B is blocked for `confidential` and `restricted`.
- `confidence_fallback` — A/B is always blocked when the confidence fallback fired (safety override).

**To disable A/B entirely:**
```python
request = RoutingRequest(..., exploration_rate=0.0)
```

**Log lines to inspect:**
```
ab_exploration_triggered   cid=X model=Y
```

---

## Problem: Provider Failures / All Models Failing

**Symptoms:** Requests fail with `"All N model(s) failed"` or provider errors appear in logs.

**Where to check:**
- Circuit breaker state:
  ```python
  from router.circuit_breaker import CircuitBreaker
  cb = engine._circuit_breaker
  print(cb.get_state("openai"))   # "closed", "open", or "half-open"
  print(cb.get_failure_count("openai"))
  ```
- `CIRCUIT_BREAKER_FAILURE_THRESHOLD` in `config.py` — currently 5. Lower it to open faster; raise it to tolerate more transient failures.
- `CIRCUIT_BREAKER_RECOVERY_TIMEOUT` — currently 60s. How long before a half-open probe is sent.

**Log lines to inspect:**
```
circuit_opened               provider=X failures=5
circuit_half_open            provider=X elapsed_s=62
circuit_closed               provider=X
circuit_reopened_after_probe_failure  provider=X
proxy_call_failed            model=X status=429 error=...
```

**Common causes:**
1. API key expired or wrong — check for `AuthenticationError` / HTTP 401.
2. Provider is down (5xx) — check provider status pages. Circuit will reopen automatically after `CIRCUIT_BREAKER_RECOVERY_TIMEOUT`.
3. Rate limit (429) — check `rate_limit_rpm` in `models.json` for the affected model. Reduce load or add same-tier alternatives.

---

## Problem: Budget Downgrade Confusion

**Symptoms:** Requests route to a cheaper model than expected, with `budget_downgraded` in the rule.

**Where to check:**
- `BUDGET_LIMITS` in `config.py` — daily/monthly limits per plan.
- User's current daily spend: `budget_tracker.get_daily_spend(user_id)`.
- The routing rule: `decision.routing_rule_matched` will contain `budget_downgraded`.

**The tier walk-down logic (Step 12):**
The engine steps down one tier at a time (premium → mid → cheap → free) until it finds a model whose estimated cost fits within the remaining budget. If even the free tier exceeds the budget, `budget_exhausted=True` is set and the cheapest available model is used.

**To debug a specific user's budget:**
```python
from router.budget_tracker import BudgetTracker
bt = engine._budget
print(bt.get_daily_spend("user_id_here"))
print(bt.get_remaining_budget("user_id_here", plan="business_plan"))
```

---

## Understanding the `routing_rule_matched` Field

Every `RoutingDecision` has a `routing_rule_matched` string. Here is what each value means:

| Rule | Meaning |
|------|---------|
| `cache_hit` | Identical request served from cache; no model called |
| `cost_ceiling_blocked` | Estimated cost exceeded `MAX_COST_PER_REQUEST`; no model selected |
| `no_candidates` | No models passed hard constraints |
| `daily_budget_cap_free_tier` | Customer hit their daily cap; routed to free tier |
| `daily_budget_exhausted` | Customer hit their daily cap and no free tier available |
| `explicit_model_override` | Caller specified `metadata.model`; that model was used |
| `always_premium (tier=X)` | `routing_priority="always-premium"` bypassed scoring |
| `trivial_request` | Short casual conversation; free tier used |
| `tier_selection` | Normal scoring path; won by score |
| `tier_selection \| confidence_fallback` | Normal scoring + upgraded to premium (low confidence) |
| `ab_exploration` | A/B: a cheaper tier was selected for experimentation |
| `tier_selection \| budget_downgraded` | Scored normally then downgraded due to budget |
| `tier_selection \| budget_exhausted` | Scored normally then all tiers exceeded budget |
