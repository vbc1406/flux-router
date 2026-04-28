"""
File: router/schemas.py

Purpose:
Pydantic data models that define the contracts between every module in the
routing layer.  Every public API boundary uses these types — do not pass raw
dicts between modules.

Main Models:
  RoutingRequest    — input to RoutingEngine.route(); describes a single LLM request
  TaskAnalysis      — output of RequestClassifier.analyze(); drives all routing decisions
  ModelOption       — a single model entry (static metadata + runtime scoring fields)
  RoutingDecision   — output of RoutingEngine.route(); the complete routing result
  RoutingExplanation — verbose breakdown of a routing decision (populated when verbose=True)
  QualityScore      — heuristic quality assessment of a model response
  CachedResponse    — what the ResponseCache stores per fingerprint
  FallbackEvent     — audit record of a single fallback attempt

Key Design Rules:
  - ModelOption has two RUNTIME fields (adjusted_quality, routing_score) that the
    routing engine sets on .model_copy() instances.  NEVER mutate the registry's
    ModelOption objects directly — always work on copies.
  - All new fields on RoutingRequest and RoutingDecision MUST have default values
    so existing callers that don't set them continue to work.
  - The "ultra" tier has been removed (Change 8).  The valid tiers are:
    "free", "cheap", "mid", "premium".  Do not re-add "ultra".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class RoutingExplanation:
    """Human-readable breakdown of a routing decision. Populated when verbose=True."""

    task_type: str = ""
    task_type_confidence: float = 0.0
    complexity_score: float = 0.0
    complexity_modifiers: list[str] = field(default_factory=list)
    tier_selected: str = ""
    candidates_considered: int = 0
    candidates_filtered: int = 0
    filter_reasons: dict[str, str] = field(default_factory=dict)
    scoring_breakdown: list[dict] = field(default_factory=list)
    winner: str = ""
    runner_up: str = ""
    score_gap: float = 0.0
    sticky_bias_applied: bool = False
    confidence_fallback_used: bool = False
    rules_fired: list[str] = field(default_factory=list)

    def explain(self) -> str:
        """Return a human-readable summary of the routing decision."""
        lines: list[str] = [
            f"Task: {self.task_type} (confidence: {self.task_type_confidence:.2f})",
            f"Complexity: {self.complexity_score:.2f}"
            + (f" [{', '.join(self.complexity_modifiers)}]" if self.complexity_modifiers else ""),
            f"Tier: {self.tier_selected}",
            f"Candidates: {self.candidates_considered} considered, {self.candidates_filtered} filtered",
        ]
        if self.filter_reasons:
            reason_summary: dict[str, int] = {}
            for reason in self.filter_reasons.values():
                reason_summary[reason] = reason_summary.get(reason, 0) + 1
            parts = ", ".join(f"{r}={c}" for r, c in reason_summary.items())
            lines[-1] += f" ({parts})"
        lines.append("")
        lines.append("Scoring:")
        for entry in self.scoring_breakdown:
            marker = " ← WINNER" if entry.get("model") == self.winner else ""
            lines.append(f"  {entry['model']}:")
            lines.append(f"    quality={entry.get('quality', 0):.2f}")
            lines.append(f"    cost={entry.get('cost', 0):.2f}")
            lines.append(f"    latency={entry.get('latency', 0):.2f}")
            lines.append(f"    → {entry.get('score', 0):.3f}{marker}")
        lines.append("")
        lines.append(f"Winner: {self.winner}")
        if self.runner_up:
            lines.append(f"Beat runner-up by {self.score_gap:.3f}")
        if self.sticky_bias_applied:
            lines.append("Sticky bias applied (same-conversation model preference)")
        if self.confidence_fallback_used:
            lines.append("Confidence fallback triggered (upgraded to premium)")
        if self.rules_fired:
            lines.append(f"Rules: {', '.join(self.rules_fired)}")
        return "\n".join(lines)


class RoutingRequest(BaseModel):
    """A single request arriving at the router for a model-selection decision."""

    raw_prompt: str
    message_history: list[dict] = Field(default_factory=list)
    system_prompt: str | None = None
    max_tokens_requested: int | None = None
    user_id: str
    team_id: str | None = None
    plan: Literal["free_plan", "pro_plan", "business_plan"] = "pro_plan"
    priority: Literal["low", "normal", "high", "critical"] = "normal"
    required_capabilities: list[str] = Field(default_factory=list)
    prefer_streaming: bool = False
    temperature: float | None = None
    metadata: dict = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ── Change 1: Routing priority tags ────────────────────────────────────────
    # Controls model selection strategy:
    #   always-premium  → skip scoring; pick highest-tier available model
    #   quality-first   → weights quality=0.70, cost=0.20, latency=0.10
    #   balanced        → default behavior (current logic, no change)
    #   cost-optimized  → weights quality=0.30, cost=0.60, latency=0.10
    routing_priority: str = "balanced"

    # ── Change 3: Per-customer adaptive memory ──────────────────────────────────
    # When provided, per-customer adaptive quality scores are used after 20+ samples.
    customer_id: str | None = None

    # ── Change 4: Per-request and per-day cost caps ─────────────────────────────
    # max_cost_per_request: models exceeding this estimate are filtered in Step 4.
    # max_daily_cost: if customer's daily spend >= this cap, force free tier.
    max_cost_per_request: float | None = None
    max_daily_cost: float | None = None

    # ── Change 6: Configurable A/B exploration rate ─────────────────────────────
    # Range [0.0, 0.25]. At 0.0, Step 10 is skipped entirely.
    exploration_rate: float = 0.10

    # ── Change 9: Decision-only vs full-proxy mode ──────────────────────────────
    # "decision" → return routing decision only (default, current behaviour).
    # "proxy"    → make the actual provider API call and return the response too.
    mode: str = "decision"
    # Provider API key required for proxy mode.  Excluded from serialization/repr
    # to prevent it leaking into logs, analytics, or error messages.
    provider_api_key: str | None = Field(default=None, repr=False, exclude=True)

    # ── Fix 1: Sticky model bias ─────────────────────────────────────────────
    # When provided, the router stores a per-conversation model preference so
    # that follow-up messages stay on the same model unless scores diverge by
    # more than the sticky bias threshold.
    conversation_id: str | None = None


class TaskAnalysis(BaseModel):
    """
    Output of the classifier: a complete feature vector describing the request.
    Drives every subsequent routing decision.
    """

    complexity_score: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    total_context_needed: int
    task_type: str
    requires_reasoning: bool = False
    requires_creativity: bool = False
    requires_precision: bool = False
    requires_multilingual: bool = False
    requires_streaming: bool = False
    is_multi_turn: bool = False
    question_count: int = 0
    has_code_fences: bool = False
    has_math_symbols: bool = False
    sensitivity_level: Literal["public", "internal", "confidential", "restricted"] = "public"
    cache_eligible: bool = False
    prompt_fingerprint: str = ""


class ModelOption(BaseModel):
    """
    A single model entry in the registry, including all static metadata and
    two runtime fields (adjusted_quality, routing_score) that the engine populates
    on copies before scoring — never mutate registry objects directly.

    Change 8: "ultra" tier has been merged into "premium".  All ultra models are
    now premium-tier.  The Literal has been narrowed to 4 tiers.
    """

    provider: str
    model_id: str
    display_name: str
    tier: Literal["free", "cheap", "mid", "premium"]
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_context_window: int
    max_output_tokens: int
    capabilities: list[str]
    supports_streaming: bool = True
    quality_ratings: dict[str, float] = Field(default_factory=dict)
    avg_latency_ms: int = 1000
    rate_limit_rpm: int = 100
    current_load_rpm: int = 0
    is_available: bool = True
    allowed_sensitivity_levels: list[str] = Field(
        default_factory=lambda: ["public", "internal", "confidential", "restricted"]
    )

    # Runtime-only fields — set by the routing engine on a .model_copy()
    adjusted_quality: float = 0.5
    routing_score: float = 0.0


class RoutingDecision(BaseModel):
    """The complete output of one routing cycle."""

    chosen_model: ModelOption | None = None

    # ── Primary fallback chain (generic best-effort) ─────────────────────────
    fallback_chain: list[ModelOption] = Field(default_factory=list)

    # ── Change 5: Context-aware typed fallback chains ─────────────────────────
    # Use fallback_on_rate_limit when the primary model returns HTTP 429.
    # Use fallback_on_content_safety when a content-policy refusal is received.
    # Use fallback_on_timeout when the primary model times out mid-response.
    fallback_on_rate_limit: list[ModelOption] = Field(default_factory=list)
    fallback_on_content_safety: list[ModelOption] = Field(default_factory=list)
    fallback_on_timeout: list[ModelOption] = Field(default_factory=list)

    reasoning: str = ""
    estimated_cost: float = 0.0
    estimated_savings: float = 0.0
    confidence: float = 0.0
    routing_rule_matched: str = ""
    cache_hit: bool = False
    context_was_compressed: bool = False
    cost_blocked: bool = False
    correlation_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # ── Change 1: Priority applied ────────────────────────────────────────────
    priority_applied: str = "balanced"

    # ── Change 2: Confidence fallback flag ───────────────────────────────────
    # True when the winning model's score was below MIN_CONFIDENCE_THRESHOLD and
    # the router automatically upgraded to a premium fallback.
    confidence_fallback: bool = False

    # ── Change 4: Budget exhausted flag ──────────────────────────────────────
    budget_exhausted: bool = False

    # ── Change 9: Proxy mode response ────────────────────────────────────────
    # Populated only when request.mode == "proxy".
    proxy_response: str | None = None
    proxy_model_used: ModelOption | None = None

    # ── Fix 1: Sticky model bias ─────────────────────────────────────────────
    # The model_id that was used in the PREVIOUS turn of this conversation,
    # or None if this is the first turn (or no conversation_id was supplied).
    last_model: str | None = None

    # ── Fix 5: Decision explainability ───────────────────────────────────────
    # Populated only when verbose=True is passed to flux.route() or engine.route().
    explanation: RoutingExplanation | None = None

    def explain(self) -> str:
        """Return human-readable routing explanation, or a placeholder if not populated."""
        if self.explanation is None:
            return "No explanation available (pass verbose=True to populate)"
        return self.explanation.explain()


class QualityScore(BaseModel):
    """Heuristic quality assessment of a completed model response."""

    overall: float
    length_appropriateness: float
    format_compliance: float
    refusal_detected: bool
    hallucination_risk: float
    repetition_detected: bool
    latency_rating: float
    cut_off_detected: bool


class CachedResponse(BaseModel):
    """What the cache stores alongside a fingerprinted response."""

    fingerprint: str
    response_text: str
    model_used: ModelOption
    original_cost: float
    quality_score: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FallbackEvent(BaseModel):
    """Records a single fallback attempt for audit and analytics."""

    correlation_id: str
    failed_model: str
    reason: str
    http_status: int | None = None
    latency_ms: int = 0
    next_model: str | None = None
    attempt_number: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
