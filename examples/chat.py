"""
chat.py — minimal interactive terminal loop for Flux.

Instead of a hardcoded prompt, this reads whatever the user types and routes
it through Flux to a cost-effective model. Keys are read from the environment
(OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY,
MISTRAL_API_KEY) — set the ones for the providers you want eligible.

Run:
    python examples/chat.py

Type a question and press Enter. Type 'quit' (or Ctrl-D) to exit.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running as `python examples/chat.py` from a fresh clone without
# `pip install -e .` first (mirrors router/demo.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.flux import make_flux


async def main() -> None:
    # Persist adaptive weights so routing improves across runs.
    flux = make_flux(adaptive_state_file="flux_state.json")

    print("Flux chat — ask anything. Type 'quit' or Ctrl-D to exit.\n")

    while True:
        try:
            prompt = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            break

        resp = await flux.complete(
            prompt,
            routing_priority="cost-optimized",
        )

        print(f"\n{resp.text}\n")
        print(
            f"  └─ routed to {resp.model.display_name} "
            f"(est. ${resp.decision.estimated_cost:.4f}"
            f"{', fallback' if resp.fallback_used else ''})\n"
        )


if __name__ == "__main__":
    asyncio.run(main())
