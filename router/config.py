"""
File: router/config.py

Purpose:
Single source of truth for every tunable threshold, weight, limit, and cap in the
routing system.  Nothing should be magic-numbered in logic files — import from here.

How to use:
  Read this file before touching any routing behaviour.  Every constant has a comment
  explaining what it does, why it exists, how to tune it, and what breaks if changed.

Things NOT to change without discussion:
  - TIER_BOUNDARIES (changing ranges reclassifies all requests system-wide)
  - BUDGET_LIMITS (affects billing and user trust)
  - TIER_ORDER (canonical ordering used by fallback chain and routing engine)
"""

import os

# ══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY SCORING
# ══════════════════════════════════════════════════════════════════════════════

# Base complexity score for each task type, before modifiers are applied.
# Range: 0.0 – 1.0.  This maps roughly to the tier boundary midpoints so that
# a plain request of that type lands in the intended tier by default.
# How to tune: if a task type is consistently over- or under-tiered, adjust
# its base score by ±0.05 and re-run the benchmark.
# What breaks if changed: every request of that task type shifts tier, which
# affects cost and quality across all users.
COMPLEXITY_BASE_SCORES: dict[str, float] = {
    "simple_qa": 0.08,  # fact lookups, definitions — free tier
    "conversation": 0.05,  # casual chat — free tier
    "translation": 0.12,  # language translation — free tier
    "classification": 0.15,  # sentiment/category labelling — cheap tier
    "extraction": 0.25,  # structured data extraction — cheap tier
    "summarization": 0.30,  # document summarisation — cheap/mid tier
    "code_generation": 0.55,  # write new code — mid tier
    "code_review": 0.60,  # review/debug existing code — mid/premium tier
    "creative_writing": 0.45,  # stories, poems, essays — mid tier
    "analysis": 0.55,  # compare, evaluate, assess — mid tier
    "reasoning": 0.70,  # proofs, derivations, step-by-step — premium tier
    "function_calling": 0.40,  # structured tool use — mid tier
    "vision": 0.50,  # image understanding — mid tier
    "long_document": 0.50,  # large document processing — mid tier
    "unknown": 0.40,  # fallback when classifier is uncertain — mid tier
    "general": 0.40,  # broad catch-all — mid tier
}

# Additive modifiers applied on top of the base score when a signal is detected.
# Positive values increase complexity (→ higher tier); negative values decrease it.
# How to tune: inspect verbose REPL output (`python testing/router_tester.py --verbose`)
# to see which modifiers are firing for a given prompt, then adjust.
# What breaks if changed: global complexity distribution shifts for all affected prompts.
COMPLEXITY_MODIFIERS: dict[str, float] = {
    "input_tokens_gt_2000": +0.10,  # large context → more demanding
    "input_tokens_gt_8000": +0.20,  # very large context → probably needs premium
    "conversation_turns_gt_5": +0.10,  # long conversation → more context to handle
    "requires_structured_output": +0.10,  # JSON/YAML output → precision required
    "requires_reasoning": +0.20,  # explicit step-by-step / proof → high reasoning
    "multilingual": +0.10,  # cross-language content → harder
    "domain_specific": +0.15,  # technical/legal/medical jargon → harder
    "complex_system_prompt": +0.10,  # long or detailed system prompt → more context
    "multiple_questions_detected": +0.12,  # compound questions → more to address
    "math_symbols_present": +0.15,  # mathematical notation → reasoning required
    "high_code_fence_density": +0.10,  # lots of code → technical precision needed
    "high_bullet_density": +0.05,  # structured list → mild complexity signal
    "repetitive_templated": -0.10,  # fill-in-the-blank → lower complexity
    "expected_short_response": -0.10,  # short prompt, likely short answer → simpler
}

# ══════════════════════════════════════════════════════════════════════════════
# TIER BOUNDARIES
# ══════════════════════════════════════════════════════════════════════════════

# Maps a complexity score range to a model tier.
# The upper bound of "premium" is 1.01 so that score == 1.0 maps to "premium"
# and there is no gap or overlap between tiers.
# How to tune: widen a tier's range to make more requests land there.
# What breaks if changed: ALL requests are reclassified system-wide.
#   Changing tier boundaries is a breaking change — discuss with the team first.
# Note: "ultra" tier was merged into "premium" (Change 8).  Do not re-add "ultra".
TIER_BOUNDARIES: dict[str, tuple[float, float]] = {
    "free": (0.00, 0.15),  # simple Q&A, casual conversation
    "cheap": (0.15, 0.30),  # classification, extraction, summarisation
    "mid": (0.30, 0.60),  # code generation, analysis, creative writing
    "premium": (0.60, 1.01),  # reasoning, complex code review, domain-expert tasks
}

# Canonical tier ordering from cheapest to most capable.
# Used by fallback_chain.py and routing_engine.py for tier walk-ups/walk-downs.
# How to tune: do not reorder — the index position is used for arithmetic.
# What breaks if changed: fallback chain construction and budget walk-down break.
TIER_ORDER: list[str] = ["free", "cheap", "mid", "premium"]

# ══════════════════════════════════════════════════════════════════════════════
# BUDGET LIMITS (USD)
# ══════════════════════════════════════════════════════════════════════════════

# Per-plan daily and monthly spend limits.
# The routing engine's BudgetTracker enforces these in Step 12 (tier walk-down).
# How to tune: adjust limits to match your pricing tiers and cost targets.
# What breaks if changed: users on affected plans may be downgraded more or less
#   aggressively, impacting response quality and cost recovery.
BUDGET_LIMITS: dict[str, dict[str, float]] = {
    "free_plan": {"daily": 5.00, "monthly": 100.00},
    "pro_plan": {"daily": 50.00, "monthly": 1000.00},
    "business_plan": {"daily": 500.00, "monthly": 10000.00},
}

# ══════════════════════════════════════════════════════════════════════════════
# COST SAFETY
# ══════════════════════════════════════════════════════════════════════════════

# Hard ceiling on estimated cost per request (USD).
# Requests whose worst-case cost exceeds this are rejected before model selection.
# How to tune: raise if you have legitimate use cases with very large contexts;
#   lower if you need a tighter cost safety net.
# What breaks if changed: very large requests may be blocked or allowed through.
MAX_COST_PER_REQUEST: float = 5.00

# Log a warning when a single request's estimated cost exceeds this (USD).
# Does not block the request — purely informational.
# How to tune: set below MAX_COST_PER_REQUEST to get early warning before the ceiling.
COST_WARNING_THRESHOLD: float = 1.00

# ══════════════════════════════════════════════════════════════════════════════
# QUALITY FLOOR
# ══════════════════════════════════════════════════════════════════════════════

# Minimum adjusted quality score for a model to be included in the candidate set.
# Models scoring below this after adaptive adjustment are excluded in Step 8.
# How to tune: raise to force higher-quality models; lower to allow cheaper models
#   even when their adaptive score has degraded.
# What breaks if changed: a very high floor may eliminate all candidates for some
#   plans, falling back to _relaxed_filter().
MIN_QUALITY_THRESHOLD: float = 0.35

# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLD (Change 2)
# ══════════════════════════════════════════════════════════════════════════════

# If the winning model's routing score is below this floor after Step 9, the engine
# automatically upgrades to the best available premium model (confidence fallback).
# This is a safety net: low score = low confidence in the model's ability to handle
# the task → escalate to a capable model rather than return a bad response.
# How to tune: lower (e.g., 0.50) if premium fallback triggers too often for your
#   model mix; raise (e.g., 0.70) for stricter quality guarantees.
# Configurable at runtime via the MIN_CONFIDENCE_THRESHOLD environment variable.
# What breaks if changed: higher value → more requests fall back to premium → higher cost.
MIN_CONFIDENCE_THRESHOLD: float = float(os.environ.get("MIN_CONFIDENCE_THRESHOLD", "0.60"))

# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

# Response cache TTL in seconds.  Cached entries expire after this time.
# How to tune: raise for stable factual tasks (translations, simple Q&A);
#   lower if freshness matters more than cache hit rate.
CACHE_TTL_SECONDS: int = 3600  # 1 hour

# Maximum number of entries in the in-memory LRU cache.
# When full, the least-recently-used entry is evicted.
# How to tune: raise to improve hit rate at the cost of memory; lower to reduce
#   memory footprint.
CACHE_MAX_ENTRIES: int = 10000

# Master switch for the response cache.  Set to False to disable globally.
# How to tune: disable in development to ensure deterministic routing decisions;
#   re-enable in production.
CACHE_ENABLED: bool = True

# Task types that MAY be cached (subject to temperature and other conditions).
# How to tune: add a task type here when its responses are deterministic and reusable.
CACHEABLE_TASK_TYPES: set[str] = {
    "simple_qa",
    "translation",
    "classification",
    "extraction",
    "summarization",
}

# Task types that must NEVER be cached — outputs are expected to vary per request.
# How to tune: add a task type here when users expect fresh results every time.
NON_CACHEABLE_TASK_TYPES: set[str] = {
    "creative_writing",
    "conversation",
    "vision",
    "function_calling",
}

# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK AND RETRY
# ══════════════════════════════════════════════════════════════════════════════

# Base wait times (seconds) between fallback attempts.
# Attempt 1 waits FALLBACK_DELAYS[0], attempt 2 waits FALLBACK_DELAYS[1], etc.
# Random jitter in [0, FALLBACK_JITTER_MAX] is added to each delay.
# How to tune: increase delays if providers need more recovery time after failures.
FALLBACK_DELAYS: list[float] = [1.0, 2.0, 5.0]

# Maximum random jitter added to each fallback delay (seconds).
# Prevents thundering-herd effect when many requests fail simultaneously.
FALLBACK_JITTER_MAX: float = 0.5

# Timeout for simple requests (complexity < 0.5).
# How to tune: lower if fast responses are critical; raise if simple tasks sometimes
#   take longer than expected.
TIMEOUT_SIMPLE_SECONDS: int = 30

# Timeout for complex requests (complexity >= 0.5).
# How to tune: raise for requests that generate very long responses (creative writing,
#   long code generation); lower to fail fast and trigger fallback sooner.
TIMEOUT_COMPLEX_SECONDS: int = 120

# Per-HTTP-call timeout for provider_caller (urllib.request.urlopen). This bounds
# a single outbound call to a provider.
# Why 30s: Anthropic and OpenAI typically respond in 1–30s. A longer timeout means
# hung threads in the executor pool block other requests; the previous 120s value
# was effectively unbounded for healthy traffic and starved the pool under failure.
# Fallback retries handle transient slowness — the per-call timeout should be tight.
PROVIDER_CALL_TIMEOUT_SECONDS: int = 30

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING KILL SWITCH
# ══════════════════════════════════════════════════════════════════════════════

# DEFAULT FALSE. Setting True makes you responsible for downstream PII handling.
# Only enable for development debugging. When False, no log call in the routing
# layer should emit prompt content; gate any such call with `if LOG_PROMPTS:`.
LOG_PROMPTS: bool = os.getenv("FLUX_LOG_PROMPTS", "false").lower() == "true"

# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMIT PRE-EMPTION
# ══════════════════════════════════════════════════════════════════════════════

# A model is considered "near rate limit" when its current_load_rpm exceeds
# rate_limit_rpm * this factor.  It is then excluded from the candidate set (Step 4).
# How to tune: lower (e.g., 0.75) for more aggressive pre-emption; raise (e.g., 0.95)
#   to use more of the available capacity before pre-empting.
RATE_LIMIT_SAFETY_MARGIN: float = 0.85

# ══════════════════════════════════════════════════════════════════════════════
# ANTI-GAMING
# ══════════════════════════════════════════════════════════════════════════════

# Users exceeding this free-tier RPM within a 60-second sliding window are
# temporarily blocked from routing to the free tier (Step 7 anti-gaming rule).
# How to tune: lower if free-tier abuse is a problem; raise to be more permissive.
FREE_TIER_ABUSE_THRESHOLD_RPM: int = 100

# ══════════════════════════════════════════════════════════════════════════════
# A/B EXPLORATION (Change 6)
# ══════════════════════════════════════════════════════════════════════════════

# Default A/B exploration rate applied when the request does not set exploration_rate.
# At 0.10, roughly 10% of eligible requests are routed to a one-tier-cheaper model.
# How to tune: raise to gather more quality data; lower to reduce experimentation.
AB_EXPLORATION_RATE: float = 0.10

# Hard ceiling on per-request exploration_rate.  Requests attempting to set a higher
# value receive a validation error.
# How to tune: lower if you want to limit maximum experimentation per request.
AB_MAX_EXPLORATION_RATE: float = 0.25

# A/B exploration is only allowed when request.priority is in this set.
# High and critical priority requests should not be downgraded for experiments.
AB_ALLOWED_PRIORITIES: set[str] = {"low", "normal"}

# A/B exploration is blocked when the complexity score exceeds this.
# High-complexity requests are too risky to route to cheaper models experimentally.
AB_MAX_COMPLEXITY_SCORE: float = 0.70

# A/B exploration is blocked for requests with these sensitivity levels.
# Sensitive data must always go to the most trusted model, not an experimental one.
AB_BLOCKED_SENSITIVITY_LEVELS: set[str] = {"confidential", "restricted"}

# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE LEARNING
# ══════════════════════════════════════════════════════════════════════════════

# Master switch for adaptive quality learning.
# When False, get_adjusted_score() always returns the static base score.
# How to tune: disable during initial deployment or when debugging routing issues.
ADAPTIVE_LEARNING_ENABLED: bool = True

# EMA decay factor for the running quality average.
# Range: (0.0, 1.0).  Higher = longer memory (older signals matter more).
# Lower = faster adaptation (recent signals matter more).
# At 0.95, a signal from 14 updates ago still has ~50% influence.
# How to tune: lower (e.g., 0.80) after a model upgrade so stale pre-upgrade
#   scores wash out quickly.  Per-key overrides via AdaptiveWeights.set_decay_factor().
ADAPTIVE_DECAY_FACTOR: float = 0.95

# Minimum number of quality signals before adaptive scores override static ratings.
# Below this threshold, get_adjusted_score() returns the static base score unchanged.
# Why 20: fewer samples = high variance; the adaptive adjustment would be noisy.
# How to tune: lower (e.g., 10) if you need faster adaptation at the cost of variance;
#   raise (e.g., 50) for more stability.
ADAPTIVE_MIN_SAMPLES: int = 20

# Minimum per-customer quality samples before per-customer EMA overrides global EMA.
# Below this, the global adaptive score is used as the fallback.
# Lowered from 20 → 10: per-customer quality varies enough from the global baseline
# that faster adaptation outweighs the small increase in variance at 10 samples.
# How to tune: raise if per-customer scores are too noisy; lower for faster
#   personalisation at the cost of variance.
PER_CUSTOMER_MIN_SAMPLES: int = 10

# Path to the adaptive state file.  Relative to the working directory.
# In tests, AdaptiveWeights is instantiated with state_file=None (in-memory only).
ADAPTIVE_WEIGHT_FILE: str = "router/adaptive_state.json"

# ══════════════════════════════════════════════════════════════════════════════
# RESOURCE LIMITS (DoS protection)
# ══════════════════════════════════════════════════════════════════════════════
# Hard caps on adaptive-state cardinality. A malicious or buggy caller passing
# fresh (model_id, task_type, customer_id) tuples on every request would
# otherwise grow these dicts without bound until OOM. When a cap is reached,
# new keys are rejected with a structured warning; existing keys continue to
# update normally.

MAX_CUSTOMERS: int = 100_000  # _customer_state size cap
MAX_ADAPTIVE_KEYS: int = 50_000  # _state and _signal_stats size cap
MAX_DECAY_OVERRIDES: int = 1_000  # _decay_overrides size cap
RECORD_RATE_PER_CUSTOMER_PER_S: int = 100  # adaptive record() rate-limit budget

# Hard cap on persisted state-file size at load time. A malformed, corrupted, or
# adversarially-large state/log file would otherwise be slurped into memory and
# OOM the process during startup. 100 MB is far above any legitimate steady-state
# footprint (analytics JSONL is bounded by ANALYTICS_MAX_STARTUP_ENTRIES, adaptive
# JSON by the per-customer/key caps above). When the cap is exceeded, the loader
# logs an error and starts with empty state rather than crashing.
MAX_STATE_FILE_BYTES: int = 100_000_000

# ══════════════════════════════════════════════════════════════════════════════
# LATENCY PREFERENCES
# ══════════════════════════════════════════════════════════════════════════════

# Maps request.priority to a latency scoring mode used in SCORING_WEIGHTS.
# "prefer_quality" → quality-heavy weights; "prefer_speed" → latency-heavy weights.
# How to tune: change the mapping if your latency expectations per priority differ.
LATENCY_PRIORITY_MAP: dict[str, str] = {
    "low": "prefer_quality",  # low-priority requests can take their time
    "normal": "balanced",  # default balanced weighting
    "high": "prefer_speed",  # high-priority requests want fast responses
    "critical": "prefer_speed",  # critical requests need the fastest available model
}

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT COMPRESSION
# ══════════════════════════════════════════════════════════════════════════════

# Compress context when total_context_needed exceeds this fraction of the largest
# candidate model's context window.  At 0.80, compression fires when the request
# fills more than 80% of the available window.
# How to tune: lower (e.g., 0.70) to compress earlier and leave more headroom;
#   raise (e.g., 0.90) to compress only when nearly full.
CONTEXT_COMPRESSION_THRESHOLD: float = 0.80

# Target token count for the compressed context summary.
# How to tune: raise if important context is being lost; lower to compress more aggressively.
CONTEXT_SUMMARY_TARGET_TOKENS: int = 2000

# ══════════════════════════════════════════════════════════════════════════════
# MODEL SCORING WEIGHTS (Change 1)
# ══════════════════════════════════════════════════════════════════════════════

# Per-mode scoring weights applied in Step 9 of the routing algorithm.
# Keys are the latency mode (from LATENCY_PRIORITY_MAP) or routing_priority value.
# Weights must sum to 1.0.
# How to tune: adjust the balance between quality, cost, and latency for each mode.
# What breaks if changed: routing decisions for ALL requests using that mode shift.
SCORING_WEIGHTS: dict[str, dict[str, float]] = {
    # Latency-mode entries (driven by request.priority via LATENCY_PRIORITY_MAP)
    "prefer_quality": {"quality": 0.70, "cost": 0.25, "latency": 0.05},
    "balanced": {"quality": 0.50, "cost": 0.30, "latency": 0.20},
    "prefer_speed": {"quality": 0.30, "cost": 0.25, "latency": 0.45},
    # routing_priority overrides (Change 1) — these override the latency-mode entry
    "quality-first": {"quality": 0.70, "cost": 0.20, "latency": 0.10},
    "cost-optimized": {"quality": 0.30, "cost": 0.60, "latency": 0.10},
}

# Valid values for RoutingRequest.routing_priority.
# Validated at the start of RoutingEngine.route() — invalid values raise ValueError.
# How to tune: add new priority tags here AND add their weights to SCORING_WEIGHTS.
VALID_ROUTING_PRIORITIES: frozenset[str] = frozenset(
    {"always-premium", "quality-first", "balanced", "cost-optimized"}
)

# ══════════════════════════════════════════════════════════════════════════════
# BUDGET DOWNGRADE WEIGHTS (legacy reference)
# ══════════════════════════════════════════════════════════════════════════════

# Kept for reference / rollback only.  The live routing code uses the linear
# tier walk-down (Change 7) rather than these weighted values.
BUDGET_DOWNGRADE_QUALITY_WEIGHT: float = 0.65
BUDGET_DOWNGRADE_COST_WEIGHT: float = 0.35

# ══════════════════════════════════════════════════════════════════════════════
# STICKY MODEL BIAS (Fix 1)
# ══════════════════════════════════════════════════════════════════════════════

# Bonus added to the routing score of the model used in the previous conversation turn.
# Encourages the same model to be reused across turns for consistency.
# SHALLOW applies when the conversation has fewer than CONVERSATION_DEPTH_THRESHOLD messages.
# DEEP applies when the conversation is longer — deeper conversations need stronger continuity.
# How to tune: raise if conversations are switching models too often; lower if you want
#   the router to be more willing to upgrade/downgrade mid-conversation.
CONVERSATION_STICKY_BIAS_SHALLOW: float = 0.15
CONVERSATION_STICKY_BIAS_DEEP: float = 0.20

# Messages before the "deep" bias applies.
CONVERSATION_DEPTH_THRESHOLD: int = 6

# A conversation entry is considered expired after this many seconds of inactivity.
# After expiry, no sticky bias is applied on the next message.
# How to tune: raise if users frequently resume conversations after a long pause;
#   lower to reduce memory usage.
CONVERSATION_EXPIRY_SECONDS: int = 1800  # 30 minutes

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT LENGTH PENALTY (Fix 2)
# ══════════════════════════════════════════════════════════════════════════════

# Models are dropped from the candidate set entirely when the input fills more than
# this fraction of their context window (Step 4 hard constraint).
# At 0.90, a model with a 128k window is dropped when input exceeds ~115k tokens.
# How to tune: lower (e.g., 0.80) to leave more headroom for output; raise (e.g., 0.95)
#   to use models up to their practical limit.
CONTEXT_PENALTY_HARD_CUTOFF: float = 0.90

# Above this fill ratio, a severe penalty is applied to the routing score.
# Models are not dropped but their score is reduced to discourage selection.
CONTEXT_PENALTY_HIGH_RATIO: float = 0.50
CONTEXT_PENALTY_HIGH_FACTOR: float = 0.40  # score -= (ratio - HIGH_RATIO) * this

# Above this fill ratio but below HIGH_RATIO, a mild penalty applies.
CONTEXT_PENALTY_MID_RATIO: float = 0.30
CONTEXT_PENALTY_MID_FACTOR: float = 0.15  # score -= (ratio - MID_RATIO) * this

# ══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER (Fix 4)
# ══════════════════════════════════════════════════════════════════════════════

# Number of consecutive failures before a provider's circuit opens.
# Once open, that provider's models are excluded from routing until recovery.
# How to tune: lower (e.g., 3) for faster protection at the cost of more false-opens;
#   raise (e.g., 10) to tolerate more transient failures before opening.
CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5

# Seconds to wait before sending one probe request to a failed provider.
# If the probe succeeds, the circuit closes; if it fails, the circuit reopens.
# How to tune: raise if providers need more time to recover (e.g., after a 5xx outage);
#   lower for faster recovery detection.
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = 60.0  # seconds

# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE DETECTION THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

# Used by classifier._is_templated() to detect fill-in-the-blank / repetitive prompts.
# A prompt is considered templated when TEMPLATE_SIMILARITY_RATIO of its lines have
# lengths within TEMPLATE_LINE_VARIATION_THRESHOLD fraction of the average line length.
# Templated prompts receive a -0.10 complexity modifier (COMPLEXITY_MODIFIERS).
# How to tune: raise TEMPLATE_LINE_VARIATION_THRESHOLD to be less strict about what
#   counts as "similar"; lower to require tighter uniformity.
TEMPLATE_LINE_VARIATION_THRESHOLD: float = 0.20
TEMPLATE_SIMILARITY_RATIO: float = 0.65

# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE WEIGHTS CUSTOMER LOG CAP
# ══════════════════════════════════════════════════════════════════════════════

# Maximum routing events kept per customer in memory (used by AdaptiveWeights).
# When the cap is reached, the oldest entry is silently dropped (deque wraps).
# A debug log event "adaptive_customer_log_wrapped" is emitted when this happens.
# How to tune: raise if you need longer customer routing history for profiles/analytics;
#   lower to reduce memory usage for high-volume deployments.
ADAPTIVE_CUSTOMER_LOG_MAX: int = 500

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

# Maximum number of JSONL entries replayed into memory from the analytics log on startup.
# Prevents OOM on deployments with large log files.
# Only the most recent N entries are loaded; older entries remain in the file for
# archival querying but are not available for in-process aggregation queries.
# How to tune: raise if your in-process analytics queries need more history;
#   lower if startup time or memory usage is a concern.
ANALYTICS_MAX_STARTUP_ENTRIES: int = 50_000

# ══════════════════════════════════════════════════════════════════════════════
# BUDGET LEDGER CAP
# ══════════════════════════════════════════════════════════════════════════════

# Maximum spend records kept per user/customer in the in-memory ledger.
# When the cap is reached, the oldest record is evicted.
# Since budget checks only look at the current day/month, very old records have no
# effect on budget enforcement — eviction is safe.
# How to tune: raise if you need a longer ledger window; lower to reduce memory.
BUDGET_LEDGER_MAX_PER_USER: int = 10_000
