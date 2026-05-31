"""
strategies.py — map a strategy name to the concrete model it would use.

Each strategy answers "which model handles this request?":
  flux      — let the RoutingEngine decide (the thing under evaluation).
  premium   — always the most expensive model (quality/cost ceiling baseline).
  cheapest  — always the cheapest capable model (quality/cost floor baseline).
  mid       — a representative mid-tier model (a midpoint reference).

Cost is NOT taken from the routing decision; the runner recomputes it uniformly
from token counts so every strategy is compared on the same yardstick.
"""

from __future__ import annotations

from ..model_registry import ModelRegistry
from ..routing_engine import RoutingEngine
from ..schemas import ModelOption, RoutingRequest


def _capable(models: list[ModelOption], request: RoutingRequest) -> list[ModelOption]:
    """Keep only models that satisfy the request's required capabilities."""
    required = set(request.required_capabilities)
    if not required:
        return models
    return [m for m in models if required.issubset(set(m.capabilities))]


def _combined_cost(model: ModelOption) -> float:
    return model.cost_per_1k_input + model.cost_per_1k_output


def _cheapest(registry: ModelRegistry, request: RoutingRequest) -> ModelOption:
    candidates = _capable(registry.all_available_models(), request)
    if not candidates:
        candidates = registry.all_available_models()
    return min(candidates, key=_combined_cost)


def _mid(registry: ModelRegistry, request: RoutingRequest) -> ModelOption:
    candidates = _capable(registry.all_available_models(), request)
    mids = [m for m in candidates if m.tier == "mid"]
    if mids:
        return min(mids, key=_combined_cost)
    # No mid-tier model fits the capabilities — fall back to the cheapest capable one.
    return _cheapest(registry, request)


async def pick_model(
    strategy: str,
    request: RoutingRequest,
    engine: RoutingEngine,
    registry: ModelRegistry,
) -> ModelOption:
    """Return the ModelOption the named strategy would route this request to."""
    if strategy == "flux":
        decision = await engine.route(request)
        if decision.chosen_model is not None:
            return decision.chosen_model
        # Router declined to pick (e.g. cost-blocked). For eval purposes we still
        # need an answer, so fall back to the cheapest capable model.
        return _cheapest(registry, request)
    if strategy == "premium":
        return registry.most_expensive_model()
    if strategy == "cheapest":
        return _cheapest(registry, request)
    if strategy == "mid":
        return _mid(registry, request)
    raise ValueError(f"Unknown strategy '{strategy}'")
