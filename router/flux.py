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

import contextlib
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import structlog
from pydantic import SecretStr

from . import errors as err
from .adaptive_weights import AdaptiveWeights
from .analytics import RoutingAnalytics
from .budget_tracker import BudgetTracker
from .cache import ResponseCache
from .cascade import Verifier, estimate_step_cost, verify_response
from .classifier import RequestClassifier
from .config import CASCADE_MAX_ESCALATIONS
from .context_compressor import ContextCompressor
from .model_registry import ModelRegistry
from .routing_engine import RoutingEngine
from .run_budget import RunLimits
from .schemas import ModelOption, RoutingDecision, RoutingRequest

# Maps provider name (as used in models.json / ModelOption.provider) to the
# environment variable users set their key in. Order doesn't matter; lookup
# is by key.
_PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _load_keys_from_env() -> dict[str, SecretStr]:
    out: dict[str, SecretStr] = {}
    for provider, var in _PROVIDER_ENV_VARS.items():
        val = os.environ.get(var)
        if val:
            out[provider] = SecretStr(val)
    return out


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

    text: str
    model: ModelOption
    decision: RoutingDecision
    fallback_used: bool = False
    fallback_reason: str | None = None


class Flux:
    """
    High-level client that combines routing and provider calling.

    Parameters:
        engine  — A configured RoutingEngine instance.
        api_key — Provider API key passed to every model call.  Required for
                  complete(); not required for route()-only usage.
    """

    def __init__(
        self,
        engine: RoutingEngine,
        api_key: SecretStr | str | None = None,
        api_keys: dict[str, SecretStr | str] | None = None,
    ) -> None:
        self._engine = engine
        if api_key is None or isinstance(api_key, str):
            self._api_key: SecretStr | None = (
                SecretStr(api_key) if isinstance(api_key, str) else None
            )
        else:
            self._api_key = api_key

        # Per-provider keys: explicit `api_keys=` overrides env vars, which
        # in turn override the legacy single `api_key=` (kept for back-compat
        # and for OpenAI-compatible gateway setups where one key fits all).
        merged: dict[str, SecretStr] = _load_keys_from_env()
        if api_keys:
            for prov, k in api_keys.items():
                merged[prov.lower()] = SecretStr(k) if isinstance(k, str) else k
        self._provider_keys: dict[str, SecretStr] = merged

    # ── Public API ──────────────────────────────────────────────────────────

    async def route(self, request: RoutingRequest, verbose: bool = False) -> RoutingDecision:
        """Delegate to the underlying RoutingEngine."""
        return await self._engine.route(request, verbose=verbose)

    @contextlib.contextmanager
    def start_run(
        self,
        run_id: str | None = None,
        max_cost_usd: float | None = None,
        max_steps: int | None = None,
        max_tokens: int | None = None,
        max_duration_seconds: float | None = None,
    ) -> Iterator[str]:
        """
        Task 3: open a budgeted run and yield its run_id. Pass that run_id to
        every complete() call in the trajectory (as run_id=...) to enable
        run-scoped enforcement: cost-optimized degradation as the budget gets
        close, then RunBudgetExceeded raised before the next step dispatches.

        Any limit left as None falls back to the RUN_MAX_* global default in
        config.py.

        Example:
            with flux.start_run(max_cost_usd=0.10, max_steps=50) as run_id:
                for _ in range(50):
                    resp = await flux.complete(next_prompt, run_id=run_id)
                    ...

        The run's state is dropped when the context exits (whether the loop
        finished, broke early, or raised) — call again with the same run_id
        if you need it to persist across separate start_run() blocks.
        """
        run_id = run_id or str(uuid.uuid4())
        defaults = RunLimits()
        self._engine._run_budget.start(
            run_id,
            RunLimits(
                max_cost_usd=max_cost_usd if max_cost_usd is not None else defaults.max_cost_usd,
                max_steps=max_steps if max_steps is not None else defaults.max_steps,
                max_tokens=max_tokens if max_tokens is not None else defaults.max_tokens,
                max_duration_seconds=(
                    max_duration_seconds
                    if max_duration_seconds is not None
                    else defaults.max_duration_seconds
                ),
            ),
        )
        try:
            yield run_id
        finally:
            self._engine._run_budget.finish(run_id)

    async def complete(
        self,
        prompt: str,
        max_retries: int = 2,
        cascade_verifier: Verifier | None = None,
        **request_kwargs,
    ) -> FluxResponse:
        """
        Route ``prompt`` and call the selected model, retrying on transient
        failures up to ``max_retries`` times.

        ``request_kwargs`` are forwarded to RoutingRequest (e.g. user_id, plan,
        priority, conversation_id, routing_priority …).

        Task 8: when routing_priority="cascade", this delegates to
        _complete_cascade() instead — a different escalation strategy (verify
        locally, escalate tiers on failure) rather than the typed-error
        fallback chain below. ``cascade_verifier`` is only used in that path.

        Raises:
            AuthenticationError — immediately, never retried.
            FluxAPIError        — after all retries are exhausted.
        """
        if request_kwargs.get("routing_priority") == "cascade":
            return await self._complete_cascade(prompt, cascade_verifier, **request_kwargs)

        request = self._build_request(prompt, **request_kwargs)
        decision = await self._engine.route(request)
        text, model, fallback_used, fallback_reason = await self._dispatch_with_fallback(
            decision, request, max_retries
        )
        return FluxResponse(
            text=text,
            model=model,
            decision=decision,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    async def _dispatch_with_fallback(
        self,
        decision: RoutingDecision,
        request: RoutingRequest,
        max_retries: int = 2,
    ) -> tuple[str, ModelOption, bool, str | None]:
        """
        Call ``decision.chosen_model`` and, on a typed transient failure, walk
        the appropriate fallback chain up to ``max_retries`` further attempts.

        Shared by complete() and router.server's HTTP proxy so both paths get
        identical retry, budget-recording, run-budget, and attribution
        behavior — see module docstring.

        Returns (text, model_used, fallback_used, fallback_reason).

        Raises:
            AuthenticationError — immediately, never retried.
            FluxAPIError        — after all retries are exhausted.
        """
        models_to_try: list[ModelOption] = []
        if decision.chosen_model:
            models_to_try.append(decision.chosen_model)

        seen_ids: set[str] = {m.model_id for m in models_to_try}
        last_error: str | None = None
        attempts: int = 0

        for model in models_to_try:
            if attempts > max_retries:
                break

            # Re-check budget on each fallback attempt. The original routing
            # decision passed the budget check, but spend recorded by other
            # in-flight requests (or by the prior failed attempt's own logging,
            # if any) may have pushed the user over the cap since then. Using
            # decision.estimated_cost is a reasonable proxy — fallback models
            # within a tier have similar costs, and we'd rather err toward
            # stopping than blow the cap.
            if attempts > 0:
                plan = request.plan or "free_plan"
                if self._engine._budget.would_exceed_budget(
                    request.user_id, decision.estimated_cost, plan
                ):
                    last_error = "budget_exceeded"
                    log.warning(
                        "flux_fallback_skipped_budget",
                        user_id=request.user_id,
                        model=model.model_id,
                        attempt=attempts,
                    )
                    attempts += 1
                    continue
                if request.max_daily_cost is not None:
                    cust_id = request.customer_id or request.user_id
                    if self._engine._daily_budget.is_cap_exceeded(cust_id, request.max_daily_cost):
                        last_error = "daily_cap_exceeded"
                        log.warning(
                            "flux_fallback_skipped_daily_cap",
                            customer_id=cust_id,
                            model=model.model_id,
                            attempt=attempts,
                        )
                        attempts += 1
                        continue

            try:
                text = await self._call_model(model, request)
                log.info(
                    "flux_complete_success",
                    model=model.model_id,
                    attempt=attempts,
                    fallback=attempts > 0,
                )
                # SS#4: record actual plan-level spend after a successful provider call.
                self._engine._budget.record_spend(
                    user_id=request.user_id,
                    amount=decision.estimated_cost,
                    model_id=model.model_id,
                    correlation_id=request.correlation_id,
                    task_type="unknown",
                    plan=request.plan or "free_plan",
                )
                if request.max_daily_cost is not None:
                    self._engine._daily_budget.record_spend(
                        customer_id=request.customer_id or request.user_id,
                        amount=decision.estimated_cost,
                        model_id=model.model_id,
                        correlation_id=request.correlation_id,
                        task_type="unknown",
                    )
                # Task 3: record this step against the run's cumulative budget so
                # the NEXT call to complete(run_id=...) sees it in check_before_dispatch().
                # Token count is estimated (~4 chars/token) since _call_model returns text only.
                if request.run_id:
                    self._engine._run_budget.record_step(
                        request.run_id,
                        model.model_id,
                        decision.estimated_cost,
                        max(len(text) // 4, 1),
                    )
                # Task 7: cost attribution — costs/metadata only, never `text`.
                self._engine._attribution.record(
                    tenant_id=request.tenant_id,
                    run_id=request.run_id,
                    task_type=decision.task_type,
                    step_type=decision.step_type,
                    model_id=model.model_id,
                    cost_usd=decision.estimated_cost,
                )
                return text, model, attempts > 0, last_error

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

    async def _complete_cascade(
        self,
        prompt: str,
        cascade_verifier: Verifier | None,
        **request_kwargs,
    ) -> FluxResponse:
        """
        Task 8: dispatch the cheapest capable tier, verify locally, escalate
        to the next tier in decision.fallback_chain on verification failure,
        up to CASCADE_MAX_ESCALATIONS. Never raises on verification failure —
        if every tier tried fails verification, the LAST (most capable)
        response tried is returned anyway (a bad-but-present answer beats
        none), with cascade_attempts / cascade_net_savings on the decision so
        the caller can see what happened and decide for itself.
        """
        request = self._build_request(prompt, **request_kwargs)
        decision = await self._engine.route(request)
        if decision.chosen_model is None:
            raise err.FluxAPIError("No model available to route this request")

        tier_ladder: list[ModelOption] = [decision.chosen_model]
        seen_ids = {decision.chosen_model.model_id}
        for m in decision.fallback_chain:
            if m.model_id not in seen_ids:
                tier_ladder.append(m)
                seen_ids.add(m.model_id)
        tier_ladder = tier_ladder[: CASCADE_MAX_ESCALATIONS + 1]

        step_breakdown: list[dict] = []
        last_text = ""
        last_model = decision.chosen_model
        last_reason = "no attempts succeeded"
        passed = False
        got_any_response = False  # distinct from `last_text` truthiness — an
        # empty STRING response still counts as "we got something from the
        # provider," vs. every attempt raising and nothing coming back at all.

        for model in tier_ladder:
            try:
                text = await self._call_model(model, request)
            except err.AuthenticationError:
                raise
            except err.FluxAPIError as exc:
                step_breakdown.append(
                    {"model_id": model.model_id, "verified": False, "reason": str(exc)}
                )
                last_reason = str(exc)
                continue

            got_any_response = True
            result = verify_response(text, request, cascade_verifier)
            step_breakdown.append(
                {"model_id": model.model_id, "verified": result.passed, "reason": result.reason}
            )
            last_text, last_model, last_reason = text, model, result.reason
            if result.passed:
                passed = True
                break

        if not got_any_response:
            raise err.FluxAPIError(
                f"Cascade exhausted all tiers without a usable response: {last_reason}"
            )

        # Task 3: run-budget accounting for the FINAL attempt only — matches
        # the non-cascade path's convention of recording one step per
        # complete() call, not one per internal escalation attempt.
        self._engine._budget.record_spend(
            user_id=request.user_id,
            amount=decision.estimated_cost,
            model_id=last_model.model_id,
            correlation_id=request.correlation_id,
            task_type="unknown",
            plan=request.plan or "free_plan",
        )
        if request.run_id:
            self._engine._run_budget.record_step(
                request.run_id,
                last_model.model_id,
                decision.estimated_cost,
                max(len(last_text) // 4, 1),
            )

        # Task 7: attribution gets ONE record per tier actually dispatched
        # (unlike the plan/run-budget trackers above, which only count the
        # final tier) — cascade's real economics require seeing every attempt.
        for i in range(len(step_breakdown)):
            self._engine._attribution.record(
                tenant_id=request.tenant_id,
                run_id=request.run_id,
                task_type=decision.task_type,
                step_type=decision.step_type,
                model_id=tier_ladder[i].model_id,
                cost_usd=estimate_step_cost(
                    tier_ladder[i], decision.chosen_model, decision.estimated_cost
                ),
            )

        # cascade_net_savings: actual cost of every tier PAID FOR (all
        # attempts, since each one is a real API call) vs. what it would have
        # cost to always dispatch the most expensive tier in the ladder
        # directly. Negative means escalation cost MORE than skipping straight
        # to premium — the honest failure mode this metric exists to surface.
        priciest = max(tier_ladder, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)
        priciest_cost = estimate_step_cost(priciest, decision.chosen_model, decision.estimated_cost)
        actual_spend = sum(
            estimate_step_cost(tier_ladder[i], decision.chosen_model, decision.estimated_cost)
            for i in range(len(step_breakdown))
        )
        decision.cascade_attempts = len(step_breakdown)
        decision.cascade_net_savings = round(priciest_cost - actual_spend, 6)

        log.info(
            "flux_cascade_complete",
            attempts=decision.cascade_attempts,
            final_model=last_model.model_id,
            passed=passed,
            net_savings=decision.cascade_net_savings,
        )
        return FluxResponse(
            text=last_text,
            model=last_model,
            decision=decision,
            fallback_used=decision.cascade_attempts > 1,
            fallback_reason=None if passed else last_reason,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_request(self, prompt: str, **kwargs) -> RoutingRequest:
        """Build a RoutingRequest from the prompt and keyword overrides."""
        kwargs.setdefault("user_id", "flux_default")
        return RoutingRequest(raw_prompt=prompt, **kwargs)

    def _resolve_api_key(self, model: ModelOption, request: RoutingRequest) -> str:
        """Resolve the API key for ``model``'s provider: per-provider > per-request > legacy."""
        provider = model.provider.lower()
        secret = self._provider_keys.get(provider) or request.provider_api_key or self._api_key
        return secret.get_secret_value() if secret is not None else ""

    async def _call_model(self, model: ModelOption, request: RoutingRequest) -> str:
        """
        Call the provider API for ``model`` and return the response text.

        Translates ProviderCallError into the appropriate typed Flux error so
        complete() can dispatch to the right fallback chain.

        This method is intentionally small and mockable — patch it in tests to
        simulate any failure scenario without real HTTP calls.
        """
        from .provider_caller import ProviderCallError, call_provider

        api_key = self._resolve_api_key(model, request)
        try:
            return await call_provider(model, request, api_key)
        except ProviderCallError as exc:
            status = exc.status_code
            msg = str(exc).lower()

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


def make_flux(
    api_key: SecretStr | str | None = None,
    api_keys: dict[str, SecretStr | str] | None = None,
    **engine_kwargs,
) -> Flux:
    """
    Convenience factory that wires up a full RoutingEngine with sane defaults
    and returns a ready-to-use Flux instance.

    Keyword arguments are forwarded to RoutingEngine collaborators:
        adaptive_state_file — path for AdaptiveWeights JSON (default: None = in-memory).
        analytics_log_path  — path for analytics JSONL (default: None = disabled).
        cache_enabled       — bool, default False. The default response cache is
                              process-wide and NOT segmented by user/tenant/sensitivity,
                              so it MUST be opt-in. See SECURITY_ARCHITECTURE.md before
                              enabling in any multi-tenant deployment.
    """
    registry = ModelRegistry()
    cache = ResponseCache(enabled=engine_kwargs.pop("cache_enabled", False))
    adaptive = AdaptiveWeights(state_file=engine_kwargs.pop("adaptive_state_file", None))
    analytics = RoutingAnalytics(log_path=engine_kwargs.pop("analytics_log_path", None))
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    engine = RoutingEngine(
        model_registry=registry,
        classifier=classifier,
        cache=cache,
        budget_tracker=budget,
        adaptive_weights=adaptive,
        context_compressor=compressor,
        analytics=analytics,
    )
    return Flux(engine, api_key=api_key, api_keys=api_keys)
