"""
Central configuration — every tunable threshold, weight, and limit lives here.
Nothing is magic-numbered in logic files; import from this module instead.
"""

import os

# --- Complexity Scoring Weights ---
COMPLEXITY_BASE_SCORES: dict[str, float] = {
    "simple_qa":        0.08,
    "conversation":     0.05,
    "translation":      0.12,
    "classification":   0.15,
    "extraction":       0.25,
    "summarization":    0.30,
    "code_generation":  0.55,
    "code_review":      0.60,
    "creative_writing": 0.45,
    "analysis":         0.55,
    "reasoning":        0.70,
    "function_calling": 0.40,
    "vision":           0.50,
    "long_document":    0.50,
    "unknown":          0.40,
}

COMPLEXITY_MODIFIERS: dict[str, float] = {
    "input_tokens_gt_2000":       +0.10,
    "input_tokens_gt_8000":       +0.20,
    "conversation_turns_gt_5":    +0.10,
    "requires_structured_output": +0.10,
    "requires_reasoning":         +0.20,
    "multilingual":               +0.10,
    "domain_specific":            +0.15,
    "complex_system_prompt":      +0.10,
    "multiple_questions_detected":+0.12,
    "math_symbols_present":       +0.15,
    "high_code_fence_density":    +0.10,
    "high_bullet_density":        +0.05,
    "repetitive_templated":       -0.10,
    "expected_short_response":    -0.10,
}

# --- Tier Boundaries (complexity score → model tier) ---
# Change 8: "ultra" has been merged into "premium".  4 tiers remain.
# premium upper bound is 1.01 to ensure score == 1.0 resolves to "premium".
TIER_BOUNDARIES: dict[str, tuple[float, float]] = {
    "free":    (0.00, 0.15),
    "cheap":   (0.15, 0.30),
    "mid":     (0.30, 0.60),
    "premium": (0.60, 1.01),
}

# --- Budget Limits (USD) ---
BUDGET_LIMITS: dict[str, dict[str, float]] = {
    "free_plan":     {"daily":  5.00,  "monthly":   100.00},
    "pro_plan":      {"daily": 50.00,  "monthly":  1000.00},
    "business_plan": {"daily": 500.00, "monthly": 10000.00},
}

# --- Cost Safety ---
MAX_COST_PER_REQUEST: float  = 5.00
COST_WARNING_THRESHOLD: float = 1.00

# --- Quality Floor ---
# Models scoring below this (after adaptive adjustment) are excluded from selection.
# Prevents garbage outputs even if the model is cheap.
MIN_QUALITY_THRESHOLD: float = 0.35

# --- Change 2: Confidence Threshold ---
# If the winning model's final routing score is below this floor, the router
# automatically falls back to the highest-tier premium model that passed Step 4.
# Configurable via the MIN_CONFIDENCE_THRESHOLD env var.
MIN_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("MIN_CONFIDENCE_THRESHOLD", "0.60")
)

# --- Cache ---
CACHE_TTL_SECONDS: int  = 3600
CACHE_MAX_ENTRIES: int  = 10000
CACHE_ENABLED: bool     = True
# Task types that are eligible for caching
CACHEABLE_TASK_TYPES: set[str] = {
    "simple_qa", "translation", "classification", "extraction", "summarization"
}
# Task types that should NEVER be cached (output should vary)
NON_CACHEABLE_TASK_TYPES: set[str] = {
    "creative_writing", "conversation", "vision", "function_calling"
}

# --- Fallback ---
FALLBACK_DELAYS: list[float] = [1.0, 2.0, 5.0]
FALLBACK_JITTER_MAX: float   = 0.5
TIMEOUT_SIMPLE_SECONDS: int  = 30
TIMEOUT_COMPLEX_SECONDS: int = 120

# --- Rate Limit Pre-emption ---
RATE_LIMIT_SAFETY_MARGIN: float = 0.85

# --- Anti-Gaming ---
FREE_TIER_ABUSE_THRESHOLD_RPM: int = 100

# --- A/B Exploration ---
AB_EXPLORATION_RATE: float             = 0.10   # default; overridable per-request
AB_MAX_EXPLORATION_RATE: float         = 0.25   # hard ceiling (Change 6)
# A/B testing is ONLY allowed when ALL of these conditions are true:
AB_ALLOWED_PRIORITIES: set[str]        = {"low", "normal"}
AB_MAX_COMPLEXITY_SCORE: float         = 0.70
AB_BLOCKED_SENSITIVITY_LEVELS: set[str] = {"confidential", "restricted"}

# --- Adaptive Learning ---
ADAPTIVE_LEARNING_ENABLED: bool  = True
ADAPTIVE_DECAY_FACTOR: float     = 0.95
ADAPTIVE_MIN_SAMPLES: int        = 10
# Change 3: per-customer weights only kick in once we have this many samples.
PER_CUSTOMER_MIN_SAMPLES: int    = 20
ADAPTIVE_WEIGHT_FILE: str        = "router/adaptive_state.json"

# --- Latency Preferences ---
LATENCY_PRIORITY_MAP: dict[str, str] = {
    "low":      "prefer_quality",
    "normal":   "balanced",
    "high":     "prefer_speed",
    "critical": "prefer_speed",
}

# --- Context Compression ---
CONTEXT_COMPRESSION_THRESHOLD: float = 0.80
CONTEXT_SUMMARY_TARGET_TOKENS: int   = 2000

# --- Model Scoring Weights (by latency mode or routing_priority) ---
# Used in Step 9 to compute the final per-model routing score.
# Change 1: "quality-first" and "cost-optimized" entries added.
SCORING_WEIGHTS: dict[str, dict[str, float]] = {
    # latency-mode entries (driven by request.priority via LATENCY_PRIORITY_MAP)
    "prefer_quality": {"quality": 0.70, "cost": 0.25, "latency": 0.05},
    "balanced":       {"quality": 0.50, "cost": 0.30, "latency": 0.20},
    "prefer_speed":   {"quality": 0.30, "cost": 0.25, "latency": 0.45},
    # routing_priority overrides (Change 1)
    "quality-first":  {"quality": 0.70, "cost": 0.20, "latency": 0.10},
    "cost-optimized": {"quality": 0.30, "cost": 0.60, "latency": 0.10},
}

# Valid routing_priority values (Change 1).
VALID_ROUTING_PRIORITIES: frozenset[str] = frozenset(
    {"always-premium", "quality-first", "balanced", "cost-optimized"}
)

# --- Budget Downgrade Weights ---
# Kept for reference / rollback; the live code now uses tier walk-down (Change 7).
BUDGET_DOWNGRADE_QUALITY_WEIGHT: float = 0.65
BUDGET_DOWNGRADE_COST_WEIGHT: float    = 0.35

# --- Sticky Model Bias (Fix 1) ---
# When a conversation_id is present and a previous model exists, add this bonus
# to that model's routing score so it tends to win unless something else scores
# significantly higher.
# Shallow: 2–5 messages.  Deep: 6+ messages.
CONVERSATION_STICKY_BIAS_SHALLOW: float = 0.15
CONVERSATION_STICKY_BIAS_DEEP: float    = 0.20
CONVERSATION_DEPTH_THRESHOLD: int       = 6     # messages before "deep" bias
CONVERSATION_EXPIRY_SECONDS: int        = 1800  # 30 minutes of inactivity

# --- Context Length Penalty (Fix 2) ---
# Models approaching their context window limit get a scoring penalty.
# Models over the hard cutoff are filtered out entirely in Step 4.
CONTEXT_PENALTY_HARD_CUTOFF: float  = 0.90   # drop model if input_tokens > window * this
CONTEXT_PENALTY_HIGH_RATIO: float   = 0.50   # severe penalty above this fill ratio
CONTEXT_PENALTY_HIGH_FACTOR: float  = 0.40   # (ratio - HIGH_RATIO) * this
CONTEXT_PENALTY_MID_RATIO: float    = 0.30   # mild penalty above this fill ratio
CONTEXT_PENALTY_MID_FACTOR: float   = 0.15   # (ratio - MID_RATIO) * this

# --- Circuit Breaker (Fix 4) ---
CIRCUIT_BREAKER_FAILURE_THRESHOLD: int   = 5    # failures before opening
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float  = 60.0 # seconds before allowing one retry

# --- Tier Ordering (canonical source of truth) ---
# Both routing_engine and fallback_chain import this; never define it locally.
TIER_ORDER: list[str] = ["free", "cheap", "mid", "premium"]

# --- Template Detection Thresholds (classifier._is_templated) ---
TEMPLATE_LINE_VARIATION_THRESHOLD: float = 0.20  # max fractional length deviation to count as "similar"
TEMPLATE_SIMILARITY_RATIO: float         = 0.65  # fraction of lines that must be similar to flag as template

# --- AdaptiveWeights customer log cap ---
ADAPTIVE_CUSTOMER_LOG_MAX: int = 500  # max routing events kept per customer in memory

# --- Analytics startup load limit ---
ANALYTICS_MAX_STARTUP_ENTRIES: int = 50_000  # rows replayed into memory at startup

# --- BudgetTracker ledger cap ---
BUDGET_LEDGER_MAX_PER_USER: int = 10_000  # max spend records kept per user/customer
