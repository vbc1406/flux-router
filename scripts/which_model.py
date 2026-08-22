#!/usr/bin/env python3
"""
Interactive CLI: type a prompt, see which model Flux's router would pick.

Runs the real routing decision (RoutingEngine.route(), mode="decision") --
no provider API key needed, no model is actually called.

Usage:
    python scripts/which_model.py
    python scripts/which_model.py "gimme a write up about india on independence day"
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import structlog

from router.flux import make_flux
from router.schemas import RoutingRequest

# Quiet the router's internal debug/info logging -- we only want the
# decision summary printed below, not the classifier's internal trace.
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
)


async def which_model(prompt: str, **request_kwargs) -> None:
    flux = make_flux()
    request = RoutingRequest(
        raw_prompt=prompt,
        user_id="demo-user",
        mode="decision",
        **request_kwargs,
    )
    decision = await flux._engine.route(request, verbose=True)

    model = decision.chosen_model
    print()
    print(f"  Prompt:      {prompt!r}")
    name = model.display_name if model else "none"
    model_id = model.model_id if model else "-"
    print(f"  Model:       {name} ({model_id})")
    print(f"  Provider:    {model.provider if model else '-'}")
    print(f"  Task type:   {decision.task_type}")
    print(f"  Reasoning:   {decision.reasoning}")
    print(f"  Est. cost:   ${decision.estimated_cost:.6f}")
    print(f"  Confidence:  {decision.confidence:.2f}")
    print()


def main() -> None:
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        asyncio.run(which_model(prompt))
        return

    print("Flux model picker -- type a prompt (Ctrl+C / Ctrl+D to quit)\n")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        asyncio.run(which_model(prompt))


if __name__ == "__main__":
    main()
