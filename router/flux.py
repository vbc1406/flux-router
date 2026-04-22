"""
flux.py — High-level Flux facade with smart retry (Fix 3).

Flux wraps the RoutingEngine and adds a complete() method that:
  1. Routes the prompt to get a RoutingDecision.
  2. Calls the selected provider.
  3. On failure, walks the appropriate typed fallback chain (rate-limit /
     timeout / content-safety) up to max_retries times.
  4. Returns a FluxResponse with fallback_used / fallback_reason populated.
  5. Never retries on AuthenticationError — the key is wrong, fix it first.
  6. Deduplicates the models_to_try list so the same model is never called twice.

All provider calls go through _call_model(), which is designed to be patched in
tests so no real HTTP requests are needed.

Example:
    engine = RoutingEngine(...)
    flux   = Flux(engine, api_key="sk-...")
    resp   = await flux.complete("Explain backpropagation")
    print(resp.text, resp.model.display_name, resp.fallback_used)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from . import errors as err
from .adaptive_weights import AdaptiveWeights
from .analytics import RoutingAnalytics
from .budget_tracker import BudgetTracker
from .cache import ResponseCache
from .classifier import RequestClassifier
from .context_compressor import ContextCompressor
from .model_registry import ModelRegistry
from .routing_engine import RoutingEngine
from .schemas import ModelOption, RoutingDecision, RoutingRequest

log = structlog.get_logger(__name__)


@dataclass
class FluxResponse:
    """
    Return value of Flux.complete().

    Attributes:
        text            — The model's text response.
        model           — The ModelOption that produced the response.
        decision        — The full RoutingDecision (for inspection / logging).
        fallback_used   — True if the primary model failed and a fallback was used.
        fallback_reason — The error category that triggered fallback, or None.
    """

    text:           str
    model:          ModelOption
    decision:       RoutingDecision
    fallback_used:  bool        = False
    fallback_reason: str | None = None


class Flux:
    """
    High-level client that combines routing and provider calling.

    Parameters:
        engine  — A configured RoutingEngine instance.
        api_key — Provider API key passed to every model call.  Required for
                  complete(); not required for route()-only usage.
    """

    def __init__(self, engine: RoutingEngine, api_key: str | None = None) -> None:
        self._engine  = engine
        self._api_key = api_key

    # ── Public API ──────────────────────────────────────────────────────────

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """Delegate to the underlying RoutingEngine."""
        return await self._engine.route(request)

    async def complete(
        self,
        prompt:      str,
        max_retries: int = 2,
        **request_kwargs,
    ) -> FluxResponse:
        """
        Route ``prompt`` and call the selected model, retrying on transient
        failures up to ``max_retries`` times.

        ``request_kwargs`` are forwarded to RoutingRequest (e.g. user_id, plan,
        priority, conversation_id, routing_priority …).

        Raises:
            AuthenticationError — immediately, never retried.
            FluxAPIError        — after all retries are exhausted.
        """
        request  = self._build_request(prompt, **request_kwargs)
        decision = await self._engine.route(request)

        models_to_try: list[ModelOption] = []
        if decision.chosen_model:
            models_to_try.append(decision.chosen_model)

        seen_ids:   set[str] = {m.model_id for m in models_to_try}
        last_error: str | None = None
        attempts:   int = 0

        for model in models_to_try:
            if attempts > max_retries:
                break

            try:
                text = await self._call_model(model, request)
                log.info(
                    "flux_complete_success",
                    model=model.model_id,
                    attempt=attempts,
                    fallback=attempts > 0,
                )
                return FluxResponse(
                    text           = text,
                    model          = model,
                    decision       = decision,
                    fallback_used  = attempts > 0,
                    fallback_reason= last_error,
                )

            except err.AuthenticationError:
                # Key is wrong — retrying won't help.
                raise

            except err.RateLimitError:
                last_error = "rate_limit"
                for m in decision.fallback_on_rate_limit:
                    if m.model_id not in seen_ids:
                        models_to_try.append(m)
                        seen_ids.add(m.model_id)

            except err.TimeoutError:
                last_error = "timeout"
                for m in decision.fallback_on_timeout:
                    if m.model_id not in seen_ids:
                        models_to_try.append(m)
                        seen_ids.add(m.model_id)

            except err.ContentFilterError:
                last_error = "content_filter"
                for m in decision.fallback_on_content_safety:
                    if m.model_id not in seen_ids:
                        models_to_try.append(m)
                        seen_ids.add(m.model_id)

            except err.ProviderDownError:
                last_error = "provider_down"
                # Use rate-limit chain (same-tier alternatives) for down providers.
                for m in decision.fallback_on_rate_limit:
                    if m.model_id not in seen_ids:
                        models_to_try.append(m)
                        seen_ids.add(m.model_id)

            except err.FluxAPIError:
                last_error = "unknown"

            log.warning(
                "flux_complete_failed_trying_fallback",
                model=model.model_id,
                attempt=attempts,
                reason=last_error,
                remaining_candidates=len(models_to_try) - attempts - 1,
            )
            attempts += 1

        raise err.FluxAPIError(
            f"All models failed after {attempts} attempt(s). Last error: {last_error}"
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_request(self, prompt: str, **kwargs) -> RoutingRequest:
        """Build a RoutingRequest from the prompt and keyword overrides."""
        kwargs.setdefault("user_id", "flux_default")
        return RoutingRequest(raw_prompt=prompt, **kwargs)

    async def _call_model(self, model: ModelOption, request: RoutingRequest) -> str:
        """
        Call the provider API for ``model`` and return the response text.

        Translates ProviderCallError into the appropriate typed Flux error so
        complete() can dispatch to the right fallback chain.

        This method is intentionally small and mockable — patch it in tests to
        simulate any failure scenario without real HTTP calls.
        """
        from .provider_caller import ProviderCallError, call_provider

        api_key = self._api_key or request.provider_api_key or ""
        try:
            return await call_provider(model, request, api_key)
        except ProviderCallError as exc:
            status = exc.status_code
            msg    = str(exc).lower()

            if status in (401, 403):
                raise err.AuthenticationError(str(exc)) from exc
            if status == 429:
                raise err.RateLimitError(str(exc)) from exc
            if status in (500, 502, 503):
                raise err.ProviderDownError(str(exc)) from exc
            if "timeout" in msg:
                raise err.TimeoutError(str(exc)) from exc
            if "content" in msg and ("filter" in msg or "policy" in msg):
                raise err.ContentFilterError(str(exc)) from exc
            raise err.FluxAPIError(str(exc)) from exc


# ── Factory helper ───────────────────────────────────────────────────────────

def make_flux(api_key: str | None = None, **engine_kwargs) -> Flux:
    """
    Convenience factory that wires up a full RoutingEngine with sane defaults
    and returns a ready-to-use Flux instance.

    Keyword arguments are forwarded to RoutingEngine collaborators:
        adaptive_state_file — path for AdaptiveWeights JSON (default: None = in-memory).
        analytics_log_path  — path for analytics JSONL (default: None = disabled).
        cache_enabled       — bool, default True.
    """
    registry   = ModelRegistry()
    cache      = ResponseCache(enabled=engine_kwargs.pop("cache_enabled", True))
    adaptive   = AdaptiveWeights(
        state_file=engine_kwargs.pop("adaptive_state_file", None)
    )
    analytics  = RoutingAnalytics(
        log_path=engine_kwargs.pop("analytics_log_path", None)
    )
    budget     = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    engine     = RoutingEngine(
        model_registry     = registry,
        classifier         = classifier,
        cache              = cache,
        budget_tracker     = budget,
        adaptive_weights   = adaptive,
        context_compressor = compressor,
        analytics          = analytics,
    )
    return Flux(engine, api_key=api_key)
