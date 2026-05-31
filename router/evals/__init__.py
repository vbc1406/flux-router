"""
router.evals — Cost-vs-quality evaluation harness for the Flux router.

This subsystem answers the trustworthiness question: when Flux routes a prompt
to a cheaper model, *how much* quality (if any) is actually lost, and how much
cost is saved? It runs a labeled benchmark set through several routing
strategies (flux, always-premium, always-cheapest, …), generates completions
(simulated by default, real with ``--live``), grades the answers (objective
where verifiable, LLM-as-judge for open-ended), and reports the tradeoff per
task type.

Run it with::

    python -m router.evals                 # offline, simulated, CI-safe
    python -m router.evals --live --n 30    # real provider calls (needs keys)

See EVALS.md for the methodology and the latest published numbers.
"""

from .schemas import Completion, EvalSample, GradedResult, StrategyReport

__all__ = ["Completion", "EvalSample", "GradedResult", "StrategyReport"]
