"""
File: router/cascade.py

Purpose:
Cascade / escalation (Task 8): dispatch the cheapest capable model first,
verify the response locally, and escalate to the next tier only on
verification failure. "Route once, hope it's right" leaves savings on the
table when the cheap tier is usually fine; cascading tries cheap first and
only pays for premium when it's actually needed.

Verification is entirely local (schema/JSON validity, required-field
presence, refusal/empty/truncation detection, length sanity) — no LLM judge
in the hot path, since that just reinvents the latency/cost problem cascade
is trying to solve.

Main entry points:
  verify_response()   — run the default verifier chain (+ an optional
                         caller-supplied one) against a response
  Verifier             — Protocol for a single verifier function

Config Dependencies (all in config.py):
  CASCADE_ESCALATION_TIERS — tier order used by routing_engine.py's
                              _route_cascade_initial() to pick the starting tier
  CASCADE_MAX_ESCALATIONS  — hard cap on how many tiers Flux.complete() walks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import structlog

from .schemas import ModelOption, RoutingRequest

log = structlog.get_logger(__name__)


@dataclass
class VerificationResult:
    """Outcome of running a Verifier against one response."""

    passed: bool
    reason: str = ""


class Verifier(Protocol):
    """A local, synchronous check against a completed response.

    🔧 EXTENSION POINT: pass your own via Flux.complete(cascade_verifier=...).
    It runs AFTER all DEFAULT_VERIFIERS pass, so it only needs to encode
    task-specific checks (e.g. "does this contain a valid SQL query").
    """

    def __call__(self, text: str, request: RoutingRequest) -> VerificationResult: ...


_REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i'm not able to",
    "i am not able to",
    "as an ai",
    "i won't",
    "i will not",
    "i'm sorry, but",
    "i apologize, but i",
)

_TERMINAL_CHARS = ".!?`\"')]}’”"


def verify_not_empty_or_refusal(text: str, request: RoutingRequest) -> VerificationResult:
    """Fails on an empty/whitespace-only body, or an early, unprompted refusal."""
    stripped = text.strip()
    if not stripped:
        return VerificationResult(False, "empty response")
    lowered = stripped.lower()
    # Only flag refusal phrases in SHORT responses — a long response that merely
    # mentions "I cannot" in passing (e.g. discussing limitations) is not a refusal.
    if len(stripped) < 200 and any(p in lowered for p in _REFUSAL_PHRASES):
        return VerificationResult(False, "refusal detected")
    return VerificationResult(True)


def verify_not_truncated(text: str, request: RoutingRequest) -> VerificationResult:
    """Heuristic: a non-trivial response with no terminal punctuation likely
    got cut off mid-sentence (hit max_tokens)."""
    stripped = text.strip()
    if len(stripped) <= 40:
        return VerificationResult(True)  # too short for this heuristic to be meaningful
    if stripped[-1] not in _TERMINAL_CHARS:
        return VerificationResult(False, "response looks truncated (no terminal punctuation)")
    return VerificationResult(True)


def verify_structured_output(text: str, request: RoutingRequest) -> VerificationResult:
    """When request.response_format asked for structured output, the body
    must parse as JSON and contain any schema-declared required fields."""
    if not request.response_format:
        return VerificationResult(True)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return VerificationResult(False, "response_format requested but body is not valid JSON")
    schema = (request.response_format.get("json_schema") or {}).get("schema") or {}
    required = schema.get("required") or []
    missing = [f for f in required if not isinstance(parsed, dict) or f not in parsed]
    if missing:
        return VerificationResult(False, f"missing required field(s): {', '.join(missing)}")
    return VerificationResult(True)


# Order matters: cheapest/fastest checks first so a bad response fails fast.
DEFAULT_VERIFIERS: tuple[Verifier, ...] = (
    verify_not_empty_or_refusal,
    verify_not_truncated,
    verify_structured_output,
)


def verify_response(
    text: str,
    request: RoutingRequest,
    extra_verifier: Verifier | None = None,
) -> VerificationResult:
    """Run DEFAULT_VERIFIERS in order, then `extra_verifier` if all pass.
    Returns the first failure encountered, or a pass if everything clears."""
    for v in DEFAULT_VERIFIERS:
        result = v(text, request)
        if not result.passed:
            return result
    if extra_verifier is not None:
        result = extra_verifier(text, request)
        if not result.passed:
            return result
    return VerificationResult(True)


def estimate_step_cost(
    model: ModelOption,
    reference_model: ModelOption,
    reference_cost: float,
) -> float:
    """
    Approximate a cascade step's cost by scaling `reference_cost` (a real
    _estimate_cost() result for `reference_model`, computed with actual
    token counts) by the two models' relative per-token rates.

    Used instead of calling routing_engine._estimate_cost() directly because
    that needs a TaskAnalysis, which Flux doesn't have — this is precise
    enough for the cascade_net_savings accounting, which is inherently an
    estimate (see CASCADE net-economics honesty requirement in FEATURES.md).
    """
    ref_rate = reference_model.cost_per_1k_input + reference_model.cost_per_1k_output
    rate = model.cost_per_1k_input + model.cost_per_1k_output
    if ref_rate <= 0:
        return reference_cost
    return round(reference_cost * (rate / ref_rate), 6)
