"""
Central configuration — every tunable threshold, weight, and limit lives here.
Nothing is magic-numbered in logic files; import from this module instead.
"""

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
TIER_BOUNDARIES: dict[str, tuple[float, float]] = {
    "free":    (0.00, 0.15),
    "cheap":   (0.15, 0.30),
    "mid":     (0.30, 0.60),
    "premium": (0.60, 0.85),
    "ultra":   (0.85, 1.00),
}

# --- Budget Limits (USD) ---
BUDGET_LIMITS: dict[str, dict[str, float]] = {
    "free_plan":     {"daily":  5.00,  "monthly":   100.00},
    "pro_plan":      {"daily": 50.00,  "monthly":  1000.00},
    "business_plan": {"daily": 500.00, "monthly": 10000.00},
}

# --- Cost Safety ---
MAX_COST_PER_REQUEST: float  = 0.50
COST_WARNING_THRESHOLD: float = 0.25

# --- Quality Floor ---
# Models scoring below this (after adaptive adjustment) are excluded from selection.
# Prevents garbage outputs even if the model is cheap.
MIN_QUALITY_THRESHOLD: float = 0.35

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
AB_EXPLORATION_RATE: float             = 0.10
# A/B testing is ONLY allowed when ALL of these conditions are true:
AB_ALLOWED_PRIORITIES: set[str]        = {"low", "normal"}
AB_MAX_COMPLEXITY_SCORE: float         = 0.70
AB_BLOCKED_SENSITIVITY_LEVELS: set[str] = {"confidential", "restricted"}

# --- Adaptive Learning ---
ADAPTIVE_LEARNING_ENABLED: bool  = True
ADAPTIVE_DECAY_FACTOR: float     = 0.95
ADAPTIVE_MIN_SAMPLES: int        = 10
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

# --- Model Scoring Weights (by latency mode) ---
SCORING_WEIGHTS: dict[str, dict[str, float]] = {
    "prefer_quality": {"quality": 0.70, "cost": 0.25, "latency": 0.05},
    "balanced":       {"quality": 0.50, "cost": 0.30, "latency": 0.20},
    "prefer_speed":   {"quality": 0.30, "cost": 0.25, "latency": 0.45},
}

# --- Budget Downgrade Weights ---
# When budget is exhausted and we must pick a cheaper model,
# don't just pick the absolute cheapest — balance quality and cost.
BUDGET_DOWNGRADE_QUALITY_WEIGHT: float = 0.65
BUDGET_DOWNGRADE_COST_WEIGHT: float    = 0.35
