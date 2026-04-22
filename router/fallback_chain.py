"""
Fallback + retry logic with exponential backoff and jitter.

When the primary model fails (rate limit, timeout, provider error), the
FallbackExecutor walks the pre-built fallback chain from the RoutingDecision,
introducing per-attempt delays so we don't hammer a recovering provider.

The build_fallback_chain() helper is also used by the routing engine to
populate RoutingDecision.fallback_chain before any call is made.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable

import structlog

from .analytics import RoutingAnalytics
from .config import (
    FALLBACK_DELAYS,
    FALLBACK_JITTER_MAX,
    TIMEOUT_COMPLEX_SECONDS,
    TIMEOUT_SIMPLE_SECONDS,
)
from .schemas import FallbackEvent, ModelOption, RoutingDecision, RoutingRequest, TaskAnalysis

log = structlog.get_logger(__name__)

# HTTP status codes we should NOT retry — the issue is on our side.
_NO_RETRY_STATUSES: frozenset[int] = frozenset({400, 401, 403})

# Tier ordering used to build up/down chains.
# Change 8: "ultra" has been merged into "premium"; 4 tiers remain.
_TIER_ORDER: list[str] = ["free", "cheap", "mid", "premium"]


def build_fallback_chain(
    chosen: ModelOption,
    candidates: list[ModelOption],
    analysis: TaskAnalysis,
    max_chain: int = 3,
) -> list[ModelOption]:
    """
    Build an ordered fallback sequence for the routing decision:

      1. Next-best model in the same tier (same capability, different provider).
      2. Cheapest model one tier UP (more capable — handles edge cases the primary can't).
      3. Most capable available model (safety net of last resort).

    Duplicates and the chosen model are excluded from the chain.
    """
    chain: list[ModelOption] = []
    seen: set[str] = {chosen.model_id}

    # 1. Same-tier alternative (prefer highest routing_score)
    same_tier = [
        m for m in candidates
        if m.tier == chosen.tier and m.model_id not in seen
    ]
    if same_tier:
        # Sort by routing_score then adjusted_quality (both may be 0 for non-scored models)
        same_tier.sort(key=lambda m: (m.routing_score, m.adjusted_quality), reverse=True)
        chain.append(same_tier[0])
        seen.add(same_tier[0].model_id)

    # 2. Cheapest model one tier up
    chosen_idx = _TIER_ORDER.index(chosen.tier) if chosen.tier in _TIER_ORDER else 0
    if chosen_idx + 1 < len(_TIER_ORDER):
        one_up_tier = _TIER_ORDER[chosen_idx + 1]
        one_up = [m for m in candidates if m.tier == one_up_tier and m.model_id not in seen]
        if one_up:
            one_up.sort(key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
            chain.append(one_up[0])
            seen.add(one_up[0].model_id)

    # 3. Most capable overall (highest tier → highest quality within that tier)
    remaining = [m for m in candidates if m.model_id not in seen]
    if remaining:
        remaining.sort(
            key=lambda m: (_TIER_ORDER.index(m.tier) if m.tier in _TIER_ORDER else 0,
                           m.adjusted_quality),
            reverse=True,
        )
        chain.append(remaining[0])

    return chain[:max_chain]


def build_typed_fallback_chains(
    chosen: ModelOption,
    candidates: list[ModelOption],
    analysis: TaskAnalysis,
    max_chain: int = 3,
) -> tuple[list[ModelOption], list[ModelOption], list[ModelOption]]:
    """
    Build three error-type-aware fallback chains from the post-Step-4 candidate set.

    Returns (rate_limit_chain, content_safety_chain, timeout_chain).

    Usage guide (document at the call site):
      - fallback_on_rate_limit    → call when primary returns HTTP 429.
            Same-tier alternatives (different provider) with highest routing score.
      - fallback_on_content_safety → call when primary refuses on content policy.
            One tier up — premium models have broader safety handling.
      - fallback_on_timeout       → call when primary times out mid-stream.
            Fastest models by avg_latency_ms regardless of tier.

    Change 5: Provides richer fallback semantics than the generic fallback_chain.
    Only models that passed Step 4 hard constraints are eligible.
    """
    seen = {chosen.model_id}

    # ── Rate-limit chain: same tier, different provider ──────────────────────
    rate_limit = [
        m for m in candidates
        if m.tier == chosen.tier and m.model_id not in seen
    ]
    rate_limit.sort(key=lambda m: (m.routing_score, m.adjusted_quality), reverse=True)
    rate_limit_chain = rate_limit[:max_chain]

    # ── Content-safety chain: step up one or more tiers ─────────────────────
    content_safety_chain: list[ModelOption] = []
    if chosen.tier in _TIER_ORDER:
        chosen_idx = _TIER_ORDER.index(chosen.tier)
        for up_idx in range(chosen_idx + 1, len(_TIER_ORDER)):
            tier_up = [
                m for m in candidates
                if m.tier == _TIER_ORDER[up_idx] and m.model_id not in seen
            ]
            tier_up.sort(key=lambda m: m.adjusted_quality, reverse=True)
            content_safety_chain.extend(tier_up)
            if len(content_safety_chain) >= max_chain:
                break
    content_safety_chain = content_safety_chain[:max_chain]

    # ── Timeout chain: fastest models by average latency ────────────────────
    timeout_candidates = [m for m in candidates if m.model_id not in seen]
    timeout_candidates.sort(key=lambda m: m.avg_latency_ms)
    timeout_chain = timeout_candidates[:max_chain]

    return rate_limit_chain, content_safety_chain, timeout_chain


class FallbackExecutor:
    """
    Execute an API call against the primary model and fall back on failure.

    The api_caller is injected so this module stays free of real HTTP code.
    Error detection is based on exception types or HTTP status codes returned
    in the exception's attributes.
    """

    def __init__(self, analytics: RoutingAnalytics) -> None:
        self._analytics = analytics

    async def execute_with_fallback(
        self,
        request: RoutingRequest,
        decision: RoutingDecision,
        api_caller: Callable,
    ) -> tuple[Any, ModelOption, list[FallbackEvent]]:
        """
        Try the primary model, then walk the fallback chain on failure.

        Returns (response, model_used, list_of_fallback_events).
        Raises RuntimeError if ALL models in the chain fail.
        """
        models_to_try: list[ModelOption] = []
        if decision.chosen_model:
            models_to_try.append(decision.chosen_model)
        models_to_try.extend(decision.fallback_chain)

        fallback_events: list[FallbackEvent] = []
        timeout = (
            TIMEOUT_SIMPLE_SECONDS
            if hasattr(decision, "_analysis") and getattr(decision, "_analysis").complexity_score < 0.5
            else TIMEOUT_COMPLEX_SECONDS
        )

        for attempt, model in enumerate(models_to_try):
            is_fallback = attempt > 0

            if is_fallback:
                delay_base  = FALLBACK_DELAYS[min(attempt - 1, len(FALLBACK_DELAYS) - 1)]
                jitter      = random.uniform(0, FALLBACK_JITTER_MAX)
                total_delay = delay_base + jitter
                log.info(
                    "fallback_waiting",
                    model=model.model_id,
                    attempt=attempt,
                    delay=round(total_delay, 2),
                )
                await asyncio.sleep(total_delay)

            start_ms = int(time.monotonic() * 1000)
            try:
                response = await asyncio.wait_for(
                    api_caller(request, model),
                    timeout=float(timeout),
                )
                elapsed  = int(time.monotonic() * 1000) - start_ms

                # Treat empty/broken response as a failure for non-trivial tasks
                if isinstance(response, str) and len(response) < 5:
                    raise _ResponseTooShortError(f"Response only {len(response)} chars")

                log.info(
                    "api_call_succeeded",
                    model=model.model_id,
                    attempt=attempt,
                    latency_ms=elapsed,
                )
                return response, model, fallback_events

            except Exception as exc:
                elapsed   = int(time.monotonic() * 1000) - start_ms
                reason    = _classify_error(exc)
                status    = getattr(exc, "status_code", None) or getattr(exc, "status", None)

                # 400/401/403 → our fault or auth issue, do not try fallback
                if status in _NO_RETRY_STATUSES:
                    log.error(
                        "api_call_non_retriable",
                        model=model.model_id,
                        status=status,
                        reason=reason,
                    )
                    raise

                next_model_id = models_to_try[attempt + 1].model_id if attempt + 1 < len(models_to_try) else None
                event = FallbackEvent(
                    correlation_id  = request.correlation_id,
                    failed_model    = model.model_id,
                    reason          = reason,
                    http_status     = status,
                    latency_ms      = elapsed,
                    next_model      = next_model_id,
                    attempt_number  = attempt,
                )
                fallback_events.append(event)

                self._analytics.update_actual(
                    correlation_id       = request.correlation_id,
                    fallback_used        = True,
                    fallback_models_tried= [e.failed_model for e in fallback_events],
                    fallback_reasons     = [e.reason for e in fallback_events],
                )

                log.warning(
                    "api_call_failed_trying_fallback",
                    model=model.model_id,
                    reason=reason,
                    next_model=next_model_id,
                    attempt=attempt,
                )

        raise RuntimeError(
            f"All {len(models_to_try)} model(s) failed for correlation_id={request.correlation_id}"
        )


# ── Error helpers ────────────────────────────────────────────────────────────

class _ResponseTooShortError(Exception):
    pass


def _classify_error(exc: Exception) -> str:
    """Map an exception to a human-readable failure reason string."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return "rate_limited"
    if status in (500, 502, 503):
        return "provider_error"
    if status == 400:
        return "bad_request"
    if status in (401, 403):
        return "auth_error"

    name = type(exc).__name__
    msg  = str(exc).lower()
    if "timeout" in name.lower() or "timeout" in msg:
        return "timeout"
    if isinstance(exc, _ResponseTooShortError):
        return "empty_response"
    if "content" in msg and ("filter" in msg or "policy" in msg):
        return "content_filter"
    return "unknown_error:{}".format(name)
