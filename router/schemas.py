"""
All data models for the routing layer.
Using Pydantic v2 throughout; these are the contracts between every module.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
    """

    provider: str
    model_id: str
    display_name: str
    tier: Literal["free", "cheap", "mid", "premium", "ultra"]
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
    """
    The complete output of one routing cycle.
    Every field must be populated — downstream consumers (analytics, API layer)
    crash on missing fields.
    """

    chosen_model: ModelOption | None = None
    fallback_chain: list[ModelOption] = Field(default_factory=list)
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
