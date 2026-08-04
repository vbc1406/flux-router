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

import json
import os
from dataclasses import dataclass

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
# How to tune: instantiate RequestClassifier and inspect which modifiers
# fire for a given prompt (see DEBUG.md "Bad Routing Decisions"), then adjust.
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
# COMPLEXITY → QUALITY FLOOR (Change 10: global scoring)
# ══════════════════════════════════════════════════════════════════════════════
#
# Replaces tier-based filtering in Step 9.  Maps a complexity score to the
# minimum per-task-type quality_rating a model must have to be considered.
#
# Why: tier labels were assigned by price, but benchmark-grounded quality
# ratings show some "mid" models match or beat some "premium" models on
# specific tasks.  Locking selection to tier prevented the scorer from
# picking a cheaper-but-capable model when one existed.
#
# Each entry is (max_complexity, min_quality).  For a request with complexity
# c, the floor is the min_quality of the first entry whose max_complexity > c.
# How to tune: raise floors to enforce stricter quality on harder tasks; lower
# to allow more cost-optimization at the cost of routing quality.
# What breaks if changed: shifts the candidate pool for every request.
COMPLEXITY_QUALITY_FLOOR: list[tuple[float, float]] = [
    (0.15, 0.40),  # trivial:    free-tier-grade quality is enough
    (0.30, 0.60),  # cheap:      require modestly capable models
    (0.60, 0.75),  # mid:        require generally capable models
    (1.01, 0.85),  # premium:    require frontier-grade quality
]

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

# Maximum bytes to read from a provider HTTP response.
# Protects against OOM if a provider returns malformed/oversized data.
# 50MB is generous — typical provider responses are <1MB.
MAX_PROVIDER_RESPONSE_BYTES: int = 50 * 1024 * 1024  # 50MB

# Per-message content size cap. A chat message dict's "content" field
# (or any string within it) cannot exceed this.
MAX_MESSAGE_CONTENT_BYTES: int = 1 * 1024 * 1024  # 1MB per message

# Total serialized request size cap (the entire RoutingRequest as JSON).
# Bigger than this is rejected before we even classify the prompt.
MAX_REQUEST_BYTES: int = 10 * 1024 * 1024  # 10MB total

# Metadata depth cap. Nested dicts/lists deeper than this are rejected.
MAX_METADATA_DEPTH: int = 5

# Maximum entries in any list inside metadata or required_capabilities.
MAX_METADATA_LIST_LEN: int = 100

# Per-HTTP-call timeout for provider_caller (urllib.request.urlopen). This bounds
# a single outbound call to a provider.
# Why 30s: Anthropic and OpenAI typically respond in 1–30s. A longer timeout means
# hung threads in the executor pool block other requests; the previous 120s value
# was effectively unbounded for healthy traffic and starved the pool under failure.
# Fallback retries handle transient slowness — the per-call timeout should be tight.
PROVIDER_CALL_TIMEOUT_SECONDS: int = 30

# Hard ceiling on RoutingRequest.max_tokens_requested. Matches the largest
# max_output_tokens across router/models.json. A caller cannot make routing,
# budget, and cost-attribution treat a request as cheap while asking the
# provider itself for far more output than that — every model-specific cap
# is applied on top of this (see routing_engine._estimate_cost).
MAX_TOKENS_REQUESTED_CEILING: int = 128_000

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING KILL SWITCH
# ══════════════════════════════════════════════════════════════════════════════

# DEFAULT FALSE. Setting True makes you responsible for downstream PII handling.
# Only enable for development debugging. When False, no log call in the routing
# layer should emit prompt content; gate any such call with `if LOG_PROMPTS:`.
LOG_PROMPTS: bool = os.getenv("FLUX_LOG_PROMPTS", "false").lower() == "true"

# Response cache disabled by default for privacy reasons.
# When enabled, identical prompts return cached responses — but the cache
# does NOT segment by user_id, plan, required_capabilities, or sensitivity
# level. Enabling shared cache in multi-tenant deployments can cause
# cross-tenant response bleed. See SECURITY_ARCHITECTURE.md.
# Set FLUX_ENABLE_RESPONSE_CACHE=true to opt in.
ENABLE_RESPONSE_CACHE: bool = (
    os.getenv("FLUX_ENABLE_RESPONSE_CACHE", "false").lower() == "true"
)

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
    # Task 8: cascade is a shortcut priority (like always-premium) that
    # bypasses Step 9 scoring entirely — see _route_cascade_initial(). This
    # entry exists for explainability/tooling consistency (every
    # VALID_ROUTING_PRIORITIES value should be describable in weight terms),
    # heavily cost-weighted since the whole point is to start cheap.
    "cascade": {"quality": 0.20, "cost": 0.70, "latency": 0.10},
    # quality_max is also a shortcut priority (like always-premium/cascade)
    # that bypasses Step 9 scoring entirely — see _route_quality_max().
    # Entry exists purely for explainability/tooling consistency; quality=1.0
    # documents that cost/latency play no role in the selection.
    "quality_max": {"quality": 1.0, "cost": 0.0, "latency": 0.0},
}

# Valid values for RoutingRequest.routing_priority.
# Validated at the start of RoutingEngine.route() — invalid values raise ValueError.
# How to tune: add new priority tags here AND add their weights to SCORING_WEIGHTS.
VALID_ROUTING_PRIORITIES: frozenset[str] = frozenset(
    {"always-premium", "quality-first", "balanced", "cost-optimized", "cascade", "quality_max"}
)

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

# Maximum distinct user_ids/customer_ids tracked in BudgetTracker /
# DailyBudgetTracker at once. Unlike BUDGET_LEDGER_MAX_PER_USER (which bounds
# each user's own record count), nothing previously bounded the number of
# distinct users — a growing user/tenant base, or a caller minting fresh
# user_ids, grew this dict forever. Same LRU-eviction pattern as
# RUN_STORE_MAX_ENTRIES: the least-recently-touched user is evicted once the
# cap is hit. How to tune: raise for legitimately high-cardinality user bases.
BUDGET_TRACKER_MAX_USERS: int = 100_000

# ══════════════════════════════════════════════════════════════════════════════
# HTTP PROXY SERVER (router/server.py)
# ══════════════════════════════════════════════════════════════════════════════

# Bind address for `router.server`. Defaults to loopback-only when no auth token
# is configured, so an unauthenticated server is never reachable off-box by
# accident. Set FLUX_SERVER_HOST to override explicitly (e.g. "0.0.0.0" behind
# a reverse proxy that terminates auth itself).
# What breaks if misused: setting this to 0.0.0.0 without FLUX_SERVER_TOKEN
# exposes an unauthenticated proxy — that spends real provider API budget —
# to anything that can reach the port.
SERVER_HOST: str = os.environ.get(
    "FLUX_SERVER_HOST",
    (
        "0.0.0.0"  # noqa: S104 # nosec B104 - only reached when FLUX_SERVER_TOKEN(S) is set, see comment above
        if os.environ.get("FLUX_SERVER_TOKEN") or os.environ.get("FLUX_SERVER_TOKENS")
        else "127.0.0.1"
    ),
)

# Port for `router.server`. How to tune: change FLUX_SERVER_PORT if 8000 collides
# with something else in your stack.
SERVER_PORT: int = int(os.environ.get("FLUX_SERVER_PORT", "8000"))

# Whether the proxy requires `Authorization: Bearer <FLUX_SERVER_TOKEN>` on every
# request (except /health). Derived from whether the token env var is set at all —
# there is no way to run with SERVER_REQUIRE_AUTH=True and no token, since that
# would lock every caller out.
SERVER_REQUIRE_AUTH: bool = bool(
    os.environ.get("FLUX_SERVER_TOKEN") or os.environ.get("FLUX_SERVER_TOKENS")
)

@dataclass(frozen=True)
class ServerTokenBinding:
    """What a bearer token is authenticated as, server-side: a tenant_id and
    a budget plan. Both come from FLUX_SERVER_TOKENS (never the client), so
    a caller can't self-declare either one — see FLUX_SERVER_TOKENS below."""

    tenant_id: str
    plan: str = "pro_plan"


_VALID_SERVER_TOKEN_PLANS: frozenset[str] = frozenset({"free_plan", "pro_plan", "business_plan"})

# Optional multi-tenant token map. Two accepted value shapes per entry:
#   FLUX_SERVER_TOKENS='{"tok-acme":{"tenant_id":"acme","plan":"business_plan"}}'
#   FLUX_SERVER_TOKENS='{"tok-acme":"acme"}'   (legacy shorthand — unchanged
#                                                behavior, see below)
# With a single shared FLUX_SERVER_TOKEN, every caller who holds it is
# authenticated as every tenant — X-Flux-Tenant-Id is a bare, self-declared
# claim (see SECURITY_ARCHITECTURE.md: "identity fields must come from
# authenticated context"). Setting FLUX_SERVER_TOKENS instead binds each
# bearer token to exactly one tenant_id server-side; the proxy then ignores
# any client-supplied X-Flux-Tenant-Id and uses the token's bound tenant, so
# a caller can no longer read another tenant's /v1/usage or rotate identities
# to dodge per-tenant budget caps by forging the header.
# Bugfix: RoutingRequest.plan used to be left at its schema default
# ("pro_plan") for EVERY proxy caller with no way to override it per tenant,
# so a tenant that should be capped at free_plan's tighter daily/monthly caps
# had no way to actually get them enforced through this API. The dict form
# above lets an operator bind a token to a specific plan; the legacy
# string-shorthand form (tenant_id only) and no-FLUX_SERVER_TOKENS mode both
# keep the prior "pro_plan" default UNCHANGED — this is an opt-in tightening,
# not a behavior change for existing deployments that don't ask for it.
# Takes priority over FLUX_SERVER_TOKEN when both are set. Malformed JSON, or
# an entry with an unrecognized `plan` value, is treated as unset / defaulted
# to "pro_plan" (logged, not raised) rather than failing startup.
def _parse_server_tokens() -> dict[str, ServerTokenBinding]:
    raw = os.environ.get("FLUX_SERVER_TOKENS")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    bindings: dict[str, ServerTokenBinding] = {}
    for k, v in parsed.items():
        token = str(k)
        if isinstance(v, str):
            bindings[token] = ServerTokenBinding(tenant_id=v)
        elif isinstance(v, dict) and "tenant_id" in v:
            plan = v.get("plan", "pro_plan")
            if plan not in _VALID_SERVER_TOKEN_PLANS:
                plan = "pro_plan"
            bindings[token] = ServerTokenBinding(tenant_id=str(v["tenant_id"]), plan=plan)
        # Any other shape (missing tenant_id, wrong type) is skipped rather
        # than raised, matching the existing malformed-JSON-is-unset stance.
    return bindings


SERVER_TOKENS: dict[str, ServerTokenBinding] = _parse_server_tokens()

# Hard cap on request body size for the proxy endpoint, checked against
# Content-Length and enforced again while streaming the body off the wire (so a
# missing/lying Content-Length header can't bypass it). Chat completion request
# bodies (prompt + history) are rarely more than a few hundred KB; 2MB leaves
# headroom for large message_history without allowing an unbounded upload.
# How to tune: raise for workloads with very large system prompts / long
# conversation histories; lower to reduce exposure to body-based DoS.
SERVER_MAX_BODY_BYTES: int = 2 * 1024 * 1024  # 2MB

# Optional per-tenant daily spend cap (USD), enforced ONLY in FLUX_SERVER_TOKENS
# multi-tenant mode, keyed by the bearer token's bound tenant_id — never by the
# client-supplied `user` field. Every plan/daily budget elsewhere in this file
# keys spend by RoutingRequest.user_id, which the OpenAI-compatible proxy takes
# straight from the client-supplied `user` field (see server.py::
# _build_routing_request) — an authenticated caller can mint a fresh `user` per
# request and each one gets its own untouched budget, evading per-user caps
# entirely while still hitting the same tenant. This cap closes that gap by
# enforcing a ceiling the caller cannot rotate around. None (default) disables
# it — set FLUX_TENANT_DAILY_CAP_USD to enable.
TENANT_DAILY_CAP_USD = (
    float(os.environ["FLUX_TENANT_DAILY_CAP_USD"])
    if os.environ.get("FLUX_TENANT_DAILY_CAP_USD")
    else None
)

# ── Inbound request rate limiting (router/rate_limit.py) ──────────────────────
# Sustained requests per minute per key on POST /v1/chat/completions, where the
# key is the bearer token's bound tenant when there is one and the client IP
# otherwise (see server.py::_rate_limit_key). 0 disables the limiter entirely.
#
# Unlike every other cap in this file this one is ON by default, because every
# other cap is a SPEND cap and none of them bound request rate: they are opt-in
# (TENANT_DAILY_CAP_USD is None unless set, and only applies in
# FLUX_SERVER_TOKENS mode) and they only stop a caller after the money is
# already spent. Without this, a runaway agent loop holding a valid token had
# no ceiling at all — every request went upstream.
#
# The default is deliberately generous: 600/min is ~10/s per tenant, far above
# normal agent traffic, so it should be invisible to legitimate callers while
# still bounding a loop that has come off the rails. It is a availability
# control, not a cost control — pair it with FLUX_TENANT_DAILY_CAP_USD, which
# is what actually caps spend.
# How to tune: raise for high-throughput multi-agent fleets; lower to tighten;
# set FLUX_RATE_LIMIT_RPM=0 to turn it off and rely on an upstream gateway.
RATE_LIMIT_RPM: int = int(os.environ.get("FLUX_RATE_LIMIT_RPM", "600"))

# Token-bucket capacity: how large an instantaneous spike is absorbed before
# the sustained RATE_LIMIT_RPM rate starts applying. Defaults to 10 seconds'
# worth of the sustained rate, so bursty-but-well-behaved batch traffic isn't
# punished for arriving all at once.
RATE_LIMIT_BURST: int = int(
    os.environ.get("FLUX_RATE_LIMIT_BURST", str(max(1, RATE_LIMIT_RPM // 6)))
)

# Cap on how many distinct rate-limit keys are tracked at once. Keys can be
# client IPs when auth is disabled, so an unbounded table is itself a
# memory-exhaustion vector — same reasoning as
# ATTRIBUTION_METRICS_MAX_LABEL_COMBOS. At the cap the least-recently-used
# bucket is evicted (see RateLimiter's docstring for why eviction resets
# rather than denies).
RATE_LIMIT_MAX_TRACKED_KEYS: int = 10_000

# ══════════════════════════════════════════════════════════════════════════════
# RUN-SCOPED BUDGET ENFORCEMENT (router/run_budget.py)
# ══════════════════════════════════════════════════════════════════════════════

# Global default ceilings for a single "run" — a multi-step agent trajectory
# sharing one run_id. Any of these can be overridden per-run via
# Flux.start_run(max_cost_usd=..., max_steps=..., max_tokens=..., max_duration_seconds=...).
# How to tune: these are deliberately conservative single-run defaults, not
# fleet-wide budgets — raise for workloads with long, expensive trajectories
# (deep research agents); lower to fail fast on runaway loops in dev/CI.
RUN_MAX_COST_USD: float = 1.00
RUN_MAX_STEPS: int = 100
RUN_MAX_TOKENS: int = 500_000
RUN_MAX_DURATION_SECONDS: float = 900.0  # 15 minutes

# Graceful degradation ladder (checked as fractions of whichever limit above
# is closest to being hit). At DEGRADE_THRESHOLD, routing_priority is forced
# to "cost-optimized" for the rest of the run regardless of what the caller
# requested. At WARN_THRESHOLD, the same plus RoutingDecision.budget_warning
# is populated so the caller's agent loop can choose to wrap up early. At
# 1.0, RunBudgetExceeded is raised — before the next step dispatches, not
# after it spends.
# How to tune: lower RUN_DEGRADE_THRESHOLD to downgrade earlier/more
# conservatively; the gap between the two thresholds is the caller's window
# to notice budget_warning and stop voluntarily before a hard stop.
RUN_DEGRADE_THRESHOLD: float = 0.70
RUN_WARN_THRESHOLD: float = 0.90

# Max concurrent run_ids tracked in the default in-memory RunStore. Oldest
# (least-recently-touched) run is evicted first once the cap is hit — mirrors
# the LRU eviction pattern used by AdaptiveWeights' MAX_CUSTOMERS.
# How to tune: raise for high-concurrency multi-tenant deployments; lower to
# bound memory tighter on constrained hosts.
RUN_STORE_MAX_ENTRIES: int = 50_000

# A run with no activity (no check/record call) for this long is considered
# abandoned and is evicted on the next housekeeping sweep, freeing its slot
# even if RUN_STORE_MAX_ENTRIES hasn't been reached. Agent loops that crash
# or hang without ever closing their run must not leak memory.
# How to tune: raise if legitimate runs can go quiet for long stretches
# (e.g. waiting on a human-in-the-loop step); lower to reclaim memory faster.
RUN_TTL_SECONDS: float = 3600.0  # 1 hour

# Proactive housekeeping sweep frequency: every Nth check/record call across
# all runs, sweep for TTL-expired entries. Mirrors RoutingEngine's
# ConversationStore housekeeping cadence (every 500 route() calls).
RUN_HOUSEKEEPING_INTERVAL: int = 500

# Bugfix: which RunStore backend RunBudget() constructs by default when no
# store is injected explicitly. "memory" (InMemoryRunStore) is process-local
# — safe for a single worker, WRONG for a multi-worker/multi-instance
# deployment, where each worker would enforce run budgets against only the
# steps it personally handled, silently under-counting the run's true
# cumulative spend. Set FLUX_RUN_STORE=redis (+ FLUX_REDIS_URL) to share run
# state across workers via RedisRunStore. See router/run_budget.py.
RUN_STORE_BACKEND: str = os.environ.get("FLUX_RUN_STORE", "memory").lower()

# Connection URL for RedisRunStore, only read when RUN_STORE_BACKEND == "redis".
REDIS_URL: str = os.environ.get("FLUX_REDIS_URL", "redis://localhost:6379/0")

# Number of server worker processes, as configured by the deployer (e.g. via
# `uvicorn --workers N` or a WSGI manager). Not auto-detected — the process
# can't see its sibling workers — so this only reflects what's set here.
# Used solely to warn at startup when workers > 1 but RUN_STORE_BACKEND is
# still "memory" (see server.py), since that combination silently
# under-enforces run budgets (each worker sees only its own steps).
SERVER_WORKERS: int = int(
    os.environ.get("FLUX_SERVER_WORKERS", os.environ.get("WEB_CONCURRENCY", "1"))
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP-TYPE CLASSIFICATION (Task 6: router/classifier.py, routing_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

# Minimum tier (by TIER_ORDER index) a model must meet for a given step_type,
# enforced as a hard constraint in _passes_hard_constraints() — BEFORE scoring,
# so a cheap model can never win a plan/tool_select/final_answer step on cost
# alone. step_types not listed here (including "unknown") have no floor.
# How to tune: raise a step_type's floor if you see it fumbling tool schemas
# or looping; lower tool_result_summarize/extract/format's floor (or omit
# them entirely) to push more savings into steps where mistakes are cheap to
# catch downstream.
STEP_TYPE_FLOORS: dict[str, str] = {
    "plan": "mid",
    "tool_select": "mid",
    "final_answer": "mid",
    "reflect": "cheap",
    "tool_result_summarize": "free",
    "extract": "free",
    "format": "free",
}

# ══════════════════════════════════════════════════════════════════════════════
# CACHE-AWARE ROUTING (Task 5: router/prompt_cache.py, routing_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

# Minimum shared-prefix token count before cache stickiness applies at all.
# Below this, provider prompt caches rarely engage anyway (most providers
# have their own minimums around 1024 tokens) so there is nothing to protect.
CACHE_PREFIX_MIN_TOKENS: int = 1024

# A switch away from the provider currently holding a warm prefix is only
# allowed when the candidate's all-in cost (cold, no cache) is cheaper than
# the incumbent's cache-aware cost by more than this fraction. Below it, the
# switch is blocked even if the candidate looks nominally cheaper — losing
# the warm cache typically costs more than it saves.
# How to tune: raise to make the router more conservative about switching
# (protects cache harder); lower to let routing chase small savings more
# aggressively at the risk of thrashing a warm cache.
CACHE_SWITCH_MARGIN: float = 0.15

# Scoring bonus applied to the incumbent cache-holding model's routing_score
# in Step 9, on the same additive scale as CONVERSATION_STICKY_BIAS_*. This is
# the "soft" signal; CACHE_SWITCH_MARGIN above is the hard constraint that
# actually blocks a switch.
CACHE_STICKINESS_WEIGHT: float = 0.08

# How long a provider is considered to be holding a warm prefix after last
# use, absent a more specific model.cache_ttl_seconds. Real provider TTLs are
# typically 5 minutes (ephemeral) or 1 hour (extended); this default assumes
# the shorter, more common case.
CACHE_DEFAULT_TTL_SECONDS: int = 300

# Max distinct (conversation/run key) entries tracked by PromptCacheTracker.
# Same LRU-eviction pattern as RUN_STORE_MAX_ENTRIES.
CACHE_TRACKER_MAX_ENTRIES: int = 50_000

# ══════════════════════════════════════════════════════════════════════════════
# COST ATTRIBUTION (Task 7: router/attribution.py, router/server.py)
# ══════════════════════════════════════════════════════════════════════════════

# Where a self-hosted `flux serve` keeps its local state (currently just the
# SQLite usage database). Follows the XDG base-directory spec so it lands
# somewhere predictable and user-owned rather than in the working directory.
# Nothing in the library reads this — only router/cli.py, which creates the
# directory and points FLUX_ATTRIBUTION_DB at a file inside it. See the
# library-vs-server split documented on ATTRIBUTION_DB_PATH below.
DATA_DIR: str = os.environ.get("FLUX_DATA_DIR") or os.path.join(
    os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "flux"
)

# Filename of the usage database inside DATA_DIR, used by `flux serve`.
DATA_DB_FILENAME: str = "flux.db"

# Path to the SQLite usage database. Defaults to ":memory:" — matching this
# codebase's convention that new engine collaborators default to no disk I/O
# (see ResponseCache(enabled=False), AdaptiveWeights(state_file=None)) — so
# constructing a RoutingEngine never silently creates a file on disk. Set
# FLUX_ATTRIBUTION_DB to a real path for cross-restart persistence; never
# contains prompts or completions, costs and metadata only.
#
# The *library* default stays ephemeral on purpose. The *server* is opinionated
# the other way: `flux serve` (router/cli.py) sets FLUX_ATTRIBUTION_DB to
# DATA_DIR/DATA_DB_FILENAME before importing router.server, so a self-hosted
# instance persists across restarts without the operator configuring anything.
# That ordering matters — every constant in this module is read from the
# environment at import time, and router/server.py builds its Flux instance at
# module scope, so setting the env var after importing the server has no effect.
ATTRIBUTION_DB_PATH: str = os.environ.get("FLUX_ATTRIBUTION_DB", ":memory:")

# Prometheus label cardinality cap: distinct (tenant_id, model_id) label
# combinations tracked before falling back to an "_overflow_" bucket. Prevents
# a hostile or buggy caller from exploding /metrics cardinality by minting a
# fresh tenant_id per request.
# How to tune: raise for legitimately high-tenant-count deployments; the cost
# is proportional memory use in the metrics registry.
ATTRIBUTION_METRICS_MAX_LABEL_COMBOS: int = 10_000

# Page size cap for GET /v1/usage on the proxy.
ATTRIBUTION_USAGE_PAGE_MAX: int = 500

# Bound on how many UsageRecords may sit queued for the background SQLite
# writer thread (see SqliteUsageStore) before record() starts dropping new
# ones rather than blocking the caller. record() is called synchronously from
# async request handlers (the HTTP proxy's hot path); this cap trades a rare
# dropped usage record (logged) for never blocking the event loop on disk I/O.
# How to tune: raise for bursty high-throughput deployments if the writer
# thread ever falls meaningfully behind (watch for
# attribution_write_queue_full log lines).
ATTRIBUTION_WRITE_QUEUE_MAX_SIZE: int = 10_000

# ══════════════════════════════════════════════════════════════════════════════
# CASCADE / ESCALATION (Task 8: router/cascade.py, flux.py)
# ══════════════════════════════════════════════════════════════════════════════

# Escalation tiers tried in order for routing_priority="cascade", cheapest
# first. Escalation stops as soon as a tier's response passes verification.
# How to tune: this must be a subsequence of TIER_ORDER, cheapest-first: do
# not reorder without updating cascade.py's tier-walk logic.
CASCADE_ESCALATION_TIERS: list[str] = ["free", "cheap", "mid", "premium"]

# Hard cap on escalations per request, independent of how many tiers exist
# above. Prevents a systematically-failing verifier from walking every tier
# (and paying for every one) on every request.
CASCADE_MAX_ESCALATIONS: int = 3

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY FRESHNESS (router/model_registry.py)
# ══════════════════════════════════════════════════════════════════════════════

# Bugfix: models.json's "last_updated" field used to be write-only — nothing
# ever read it, so the catalog could silently go stale (missing new model
# SKUs, wrong pricing) with no signal short of a human noticing. If
# last_updated is older than this many days when the registry loads, a
# startup warning is logged (model_registry_json_stale). Purely observational
# — never blocks startup or falls back to the hardcoded registry.
# How to tune: lower for a team that ships new model pricing frequently and
# wants an earlier nudge; raise if 60 days produces noise for a slow-moving
# deployment.
MODELS_JSON_STALE_AFTER_DAYS: int = 60
