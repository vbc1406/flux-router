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

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from .config import (
    MAX_MESSAGE_CONTENT_BYTES,
    MAX_METADATA_DEPTH,
    MAX_METADATA_LIST_LEN,
    MAX_REQUEST_BYTES,
    MAX_TOKENS_REQUESTED_CEILING,
)

_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-:.]+$")
_MAX_METADATA_BYTES = 10_000

RUN_ID_MAX_LENGTH = 256


def validate_run_id(value: str) -> str:
    """Shared validation for a client-supplied run correlation ID
    (X-Flux-Run-Id / RoutingRequest.run_id), used both by RoutingRequest's
    own field validator below AND by server.py's early header check (which
    runs before the request body is even read) — one rule, checked in two
    places, so they can't drift apart. Raises ValueError on anything unsafe
    to use as a run-budget storage key: blank/whitespace-only, over length,
    or containing characters outside the safe-id charset (this also blocks
    the tenant-scoping separator `\\x00` used internally by
    RunBudget._key(), so a client can never forge a cross-tenant key)."""
    if not value.strip():
        raise ValueError("run_id must not be empty or whitespace-only")
    if len(value) > RUN_ID_MAX_LENGTH:
        raise ValueError(f"run_id exceeds {RUN_ID_MAX_LENGTH} characters")
    if not _SAFE_ID_PATTERN.match(value):
        raise ValueError("run_id contains characters outside [a-zA-Z0-9_-:.]")
    return value


def _max_depth(obj, current: int = 0, limit: int = MAX_METADATA_DEPTH) -> int:
    if current > limit:
        raise ValueError(f"metadata exceeds max depth {limit}")
    if isinstance(obj, dict):
        return max(
            (_max_depth(v, current + 1, limit) for v in obj.values()),
            default=current,
        )
    if isinstance(obj, list):
        if len(obj) > MAX_METADATA_LIST_LEN:
            raise ValueError(f"list exceeds max length {MAX_METADATA_LIST_LEN}")
        return max(
            (_max_depth(item, current + 1, limit) for item in obj),
            default=current,
        )
    return current


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
    # Human-readable minimum-tier floors that applied to this decision, e.g.
    # "agent_step:plan → mid", "domain:medical → premium",
    # "complexity:0.91 → premium". Populated by
    # routing_engine._composed_min_tier() in Step 4; empty when no floor
    # applied. Each entry is a floor that was IN EFFECT, not just the winning
    # (strongest) one — see tier_selected for the model's actual tier.
    floors_applied: list[str] = field(default_factory=list)
    # scoring_breakdown entries carry an optional "quality_source" key
    # describing where each candidate's quality figure came from:
    # "step:<step_type>" (ModelOption.step_quality_ratings hit),
    # "task:<task_type>" (ModelOption.quality_ratings[task_type]), or
    # "fallback:general" (ModelOption.quality_ratings["general"] / 0.5
    # default). See routing_engine._resolve_quality().
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
            f"Candidates: {self.candidates_considered} considered, "
            f"{self.candidates_filtered} filtered",
        ]
        if self.floors_applied:
            lines.append(f"Floors applied: {', '.join(self.floors_applied)}")
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

    raw_prompt: str = Field(..., max_length=1_000_000)
    message_history: list[dict] = Field(default_factory=list, max_length=200)
    system_prompt: str | None = Field(default=None, max_length=100_000)
    max_tokens_requested: int | None = None
    user_id: str = Field(..., min_length=1, max_length=256)
    team_id: str | None = Field(default=None, max_length=256)
    # `plan` selects the budget ceiling (business_plan = highest). Like
    # user_id/customer_id, it is a TRUSTED field: populate it server-side from
    # the authenticated session, never from client input, or a caller can grant
    # itself a higher spending limit. See SECURITY_ARCHITECTURE.md.
    plan: Literal["free_plan", "pro_plan", "business_plan"] = "pro_plan"
    priority: Literal["low", "normal", "high", "critical"] = "normal"
    required_capabilities: list[str] = Field(default_factory=list)
    prefer_streaming: bool = False
    temperature: float | None = None
    metadata: dict = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=256)

    # ── Change 1: Routing priority tags ────────────────────────────────────────
    # Controls model selection strategy:
    #   always-premium  → skip scoring; pick highest-tier available model
    #   quality-first   → weights quality=0.70, cost=0.20, latency=0.10
    #   balanced        → default behavior (current logic, no change)
    #   cost-optimized  → weights quality=0.30, cost=0.60, latency=0.10
    #   cascade         → start at the cheapest capable tier; Flux
    #                     escalates through decision.fallback_chain on
    #                     verification failure. See router/cascade.py.
    #   quality_max     → skip cost-minimization scoring entirely; select the
    #                     highest-capability model that passes all hard
    #                     constraints, regardless of cost. See
    #                     routing_engine.py::_route_quality_max().
    routing_priority: Literal[
        "always-premium",
        "quality-first",
        "balanced",
        "cost-optimized",
        "cascade",
        "quality_max",
    ] = "balanced"

    # ── Change 3: Per-customer adaptive memory ──────────────────────────────────
    # When provided, per-customer adaptive quality scores are used after 20+ samples.
    customer_id: str | None = Field(default=None, max_length=256)

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
    # SecretStr prevents the key from appearing in repr/str/JSON serialization
    # (pydantic emits "**********" instead). repr=False/exclude=True kept as defense
    # in depth. Use .get_secret_value() to extract the raw key for HTTP calls.
    provider_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)

    # ── Fix 1: Sticky model bias ─────────────────────────────────────────────
    # When provided, the router stores a per-conversation model preference so
    # that follow-up messages stay on the same model unless scores diverge by
    # more than the sticky bias threshold.
    conversation_id: str | None = Field(default=None, max_length=256)

    # ── Run-scoped budget enforcement ────────────────────────────────
    # A correlation ID grouping N routing decisions into one "run" (a
    # multi-step agent trajectory). See router/run_budget.py. Auto-generated
    # by Flux.start_run() / the proxy's X-Flux-Run-Id header if not supplied;
    # requests with no run_id are not subject to run-budget enforcement at all.
    run_id: str | None = Field(default=None, max_length=RUN_ID_MAX_LENGTH)

    # ── Per-tenant cost attribution ──────────────────────────────────
    # Which customer/workflow this request belongs to, for router/attribution.py
    # aggregation. Independent of user_id (a tenant may have many users).
    tenant_id: str | None = Field(default=None, max_length=256)

    # ── Step-type classification for agent trajectories ─────────────
    # Orthogonal to task_type (which classifies the PROMPT). step_type
    # classifies the AGENT STEP: what kind of action is being asked for.
    # If left unset, RequestClassifier infers it from tools/response_format/
    # message-role pattern — see classifier.py::_infer_step_type().
    step_type: (
        Literal[
            "plan",
            "tool_select",
            "tool_result_summarize",
            "reflect",
            "extract",
            "format",
            "final_answer",
            "unknown",
        ]
        | None
    ) = None
    # OpenAI-shaped tool definitions, if this step offers the model tools to
    # call. Used both by routing (capability filtering, step_type inference)
    # AND, as of Item 1, forwarded to the provider — see provider_caller.py's
    # per-provider translation functions. Contents are not validated against
    # a JSON schema; malformed tool defs surface as a provider-side error.
    tools: list[dict] = Field(default_factory=list)
    # OpenAI-shaped tool_choice ("auto" | "none" | "required" | a specific
    # {"type": "function", "function": {"name": ...}}). Only meaningful when
    # tools is non-empty. Forwarded to the provider with per-provider
    # translation (see provider_caller.py); not validated against an enum
    # here since Anthropic/Google accept a subset — an unsupported value for
    # the chosen provider raises a clear ProviderCallError at call time
    # rather than being silently dropped.
    tool_choice: dict | str | None = None
    # OpenAI-shaped response_format (e.g. {"type": "json_schema", ...}).
    # Presence signals a structured-output step for step_type inference AND
    # is forwarded to providers that support it (OpenAI/Groq/Mistral/Google).
    # Anthropic has no native equivalent — a request with response_format
    # routed to an Anthropic model is rejected with a clear error rather
    # than silently ignored (see provider_caller.py::_call_anthropic_sync).
    response_format: dict | None = None

    @field_validator("max_tokens_requested")
    @classmethod
    def _max_tokens_requested_bounds(cls, v: int | None) -> int | None:
        if v is not None and not (0 < v <= MAX_TOKENS_REQUESTED_CEILING):
            raise ValueError(
                f"max_tokens_requested must be between 1 and {MAX_TOKENS_REQUESTED_CEILING}"
            )
        return v

    @field_validator("user_id", "team_id", "customer_id", "conversation_id", "run_id", "tenant_id")
    @classmethod
    def _safe_id(cls, v: str | None) -> str | None:
        if v is not None and not _SAFE_ID_PATTERN.match(v):
            raise ValueError("ID contains unsafe characters")
        return v

    @field_validator("metadata")
    @classmethod
    def _metadata_size_limit(cls, v: dict) -> dict:
        if len(json.dumps(v, default=str)) > _MAX_METADATA_BYTES:
            raise ValueError(f"metadata too large (max {_MAX_METADATA_BYTES} bytes)")
        _max_depth(v)
        return v

    @field_validator("message_history")
    @classmethod
    def _message_history_content_limit(cls, v: list[dict]) -> list[dict]:
        for i, entry in enumerate(v):
            content = entry.get("content")
            if isinstance(content, str):
                if len(content.encode("utf-8")) > MAX_MESSAGE_CONTENT_BYTES:
                    raise ValueError(
                        f"message_history[{i}].content exceeds "
                        f"{MAX_MESSAGE_CONTENT_BYTES} bytes"
                    )
            elif isinstance(content, (list, dict)):
                raise ValueError(
                    f"message_history[{i}].content: multimodal content not yet supported"
                )
        return v

    @field_validator("required_capabilities")
    @classmethod
    def _required_capabilities_limit(cls, v: list) -> list:
        if len(v) > MAX_METADATA_LIST_LEN:
            raise ValueError(
                f"required_capabilities exceeds max length {MAX_METADATA_LIST_LEN}"
            )
        for i, entry in enumerate(v):
            if not isinstance(entry, str):
                raise ValueError(f"required_capabilities[{i}] must be a string")
        return v

    @model_validator(mode="after")
    def _total_request_size_limit(self) -> "RoutingRequest":
        size = len(self.model_dump_json().encode("utf-8"))
        if size > MAX_REQUEST_BYTES:
            raise ValueError(
                f"request exceeds max size {MAX_REQUEST_BYTES} bytes (got {size})"
            )
        return self


class TaskAnalysis(BaseModel):
    """
    Output of the classifier: a complete feature vector describing the request.
    Drives every subsequent routing decision.
    """

    complexity_score: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    # Output-token figure used for COST/BUDGET estimation specifically — the
    # larger of the task-type heuristic (estimated_output_tokens) and the
    # caller's request.max_tokens_requested (when set). A client can request
    # a huge completion while the heuristic alone would still look like a
    # cheap short answer; routing_engine._estimate_cost() uses this field
    # (not estimated_output_tokens) so caps/budgets/attribution reflect what
    # the provider could actually be asked to generate and bill for.
    billing_output_tokens: int = 0
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
    # ── Step-type classification ─────────────────────────────────────
    # Resolved step_type: request.step_type if the caller set it explicitly,
    # otherwise inferred by RequestClassifier._infer_step_type(). Always set —
    # "unknown" when no signal is present, never None.
    step_type: str = "unknown"

    @model_validator(mode="after")
    def _default_billing_output_tokens(self) -> "TaskAnalysis":
        # Callers that construct TaskAnalysis directly (tests, older code)
        # without billing_output_tokens fall back to estimated_output_tokens
        # rather than silently under-billing at 0.
        if self.billing_output_tokens <= 0:
            self.billing_output_tokens = self.estimated_output_tokens
        return self


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
    # The literal model string sent to the provider's API. Distinct from
    # model_id (Flux's stable internal/public identifier used for routing,
    # attribution, adaptive-weights keys, and everything client-facing).
    # None (the default) means "same as model_id" — resolved by the
    # validator below so every existing models.json entry and hardcoded
    # ModelOption() call site keeps working without a change.
    provider_model_id: str | None = None
    display_name: str
    tier: Literal["free", "cheap", "mid", "premium"]
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_context_window: int
    max_output_tokens: int
    capabilities: list[str]
    supports_streaming: bool = True
    quality_ratings: dict[str, float] = Field(default_factory=dict)

    # ── Item 3: Agent-step-specific quality data ─────────────────────────────
    # Optional, keyed by agent step_type ("plan", "tool_select",
    # "tool_result_summarize", "reflect", "extract", "format",
    # "final_answer"). When a request's resolved step_type is known and this
    # model has a rating for it, routing_engine._resolve_quality() uses it in
    # place of the flat per-task-type quality_ratings figure for scoring
    # (balanced/quality-first/cost-optimized/quality_max/cascade ordering all
    # go through the same resolver). Absent by default on every catalog
    # entry — when empty or missing a key, resolution falls back to
    # quality_ratings[task_type], then quality_ratings["general"], then 0.5.
    # Populate only with real, documented empirical values; an absent key is
    # the honest default, not an error.
    step_quality_ratings: dict[str, float] = Field(default_factory=dict)

    avg_latency_ms: int = 1000
    rate_limit_rpm: int = 100
    current_load_rpm: int = 0
    is_available: bool = True
    allowed_sensitivity_levels: list[str] = Field(
        default_factory=lambda: ["public", "internal", "confidential", "restricted"]
    )

    # ── Verified capability flags ────────────────────────────────────
    # Honest defaults: False unless models.json explicitly sets them True based
    # on the provider's public docs. A request with tools=[...] is filtered to
    # supports_tools=True models only (hard constraint) — an unverified False
    # here means "excluded from tool-calling steps," not "definitely can't."
    supports_tools: bool = False
    supports_structured_output: bool = False

    # ── Prompt-caching pricing ───────────────────────────────────────
    # None (the default) means "caching not modeled for this model" — treated
    # as always-cold by router/prompt_cache.py. Populate from the provider's
    # actual prompt-caching pricing page; these are NOT derived from
    # cost_per_1k_input automatically because write/read multipliers vary by
    # provider and change independently of base pricing.
    cache_write_cost_per_1m: float | None = None
    cache_read_cost_per_1m: float | None = None
    cache_min_tokens: int | None = None
    cache_ttl_seconds: int | None = None

    # Runtime-only fields — set by the routing engine on a .model_copy()
    adjusted_quality: float = 0.5
    routing_score: float = 0.0

    @model_validator(mode="after")
    def _default_provider_model_id(self) -> "ModelOption":
        if self.provider_model_id is None:
            self.provider_model_id = self.model_id
        return self


class RoutingDecision(BaseModel):
    """The complete output of one routing cycle."""

    chosen_model: ModelOption | None = None

    # ── Cost attribution ─────────────────────────────────────────────
    # Denormalized from TaskAnalysis so callers (and router/attribution.py)
    # don't need verbose=True / explanation populated just to see what kind
    # of request this was.
    task_type: str = ""
    step_type: str = ""

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

    # How long the routing decision itself took, in milliseconds — the caller
    # that measured it stamps it here (router/server.py) so the dispatch path
    # can record it against the usage row without re-measuring. None when
    # nobody timed the call. Distinct from the provider call's own latency.
    decision_latency_ms: float | None = None

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

    # ── Run-scoped budget enforcement ────────────────────────────────
    # Populated only when the request carried a run_id. run_cost_so_far and
    # run_steps_so_far reflect the run's state BEFORE this step (this step's
    # own cost/tokens are recorded separately, after dispatch succeeds).
    # budget_state: "ok" (well under limits), "degraded" (routing_priority
    # was forced to cost-optimized), "warning" (degraded + caller should
    # consider wrapping up), or "exceeded" (only appears inside
    # RunBudgetExceeded.summary — a request in that state never reaches here).
    run_id: str | None = None
    run_cost_so_far: float = 0.0
    run_steps_so_far: int = 0

    # Bugfix: True when the caller didn't supply a run_id (no X-Flux-Run-Id
    # header / no run_id kwarg) and one was auto-generated for this single
    # request. A request in this state gets its own one-step "run" — it is
    # NOT grouped with any other step, so run-scoped budget enforcement is
    # effectively a no-op for it. Set by router/server.py (the only place
    # that currently auto-generates a run_id) so callers who actually intend
    # cross-step enforcement can detect they aren't getting it, instead of
    # this failing silently. See x-flux-run-id-missing response header.
    run_id_missing: bool = False
    budget_state: Literal["ok", "degraded", "warning", "exceeded"] = "ok"
    budget_warning: str | None = None

    # ── Cache-aware routing ──────────────────────────────────────────
    # "cold": no relevant warm prefix known. "warm": stayed on the provider
    # already holding a warm cache for this prefix. "would_lose_cache": a
    # warm prefix existed on another provider but routing switched away from
    # it anyway because the savings cleared CACHE_SWITCH_MARGIN.
    prompt_cache_status: Literal["cold", "warm", "would_lose_cache"] = "cold"

    # ── Cascade / escalation ─────────────────────────────────────────
    # Populated only when routing_priority == "cascade". cascade_attempts is
    # the number of tiers actually dispatched (1 = no escalation needed).
    # cascade_net_savings is vs. always dispatching the top escalation tier
    # directly — negative when escalation made this request MORE expensive
    # than skipping straight to the tier it ended up on.
    cascade_attempts: int = 0
    cascade_net_savings: float = 0.0

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
