"""
Demo — runs 25 diverse sample requests through the full routing stack and
prints a rich summary table.  No real LLM calls are made; a mock api_caller
is injected for the FallbackExecutor example at the bottom.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# Allow running as `python -m router.demo` OR `python router/demo.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.console import Console
from rich.table import Table

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import RoutingEngine
from router.schemas import RoutingRequest

console = Console()

# ── Sample requests ──────────────────────────────────────────────────────────

_MOCK_EMAIL = (
    "Hi team, just a quick note that the Monday standup has been moved from 9am "
    "to 10am due to a scheduling conflict. Please update your calendars accordingly. "
    "The agenda remains the same: sprint review, blockers, and upcoming milestones. "
    "Let me know if you have any questions. Thanks!"
)

_MOCK_CODE = (
    "```python\n"
    "import asyncio\n"
    "import threading\n\n"
    "shared_lock = threading.Lock()\n"
    "shared_data = []\n\n"
    "async def producer():\n"
    "    while True:\n"
    "        with shared_lock:\n"
    "            shared_data.append(await fetch_data())\n"
    "        await asyncio.sleep(1)\n\n"
    "async def consumer():\n"
    "    while True:\n"
    "        with shared_lock:  # potential deadlock here\n"
    "            if shared_data:\n"
    "                item = shared_data.pop(0)\n"
    "                await process(item)\n"
    "        await asyncio.sleep(0.1)\n"
    "```"
)

_MOCK_CONTRACT = (
    "SERVICE AGREEMENT\n\nThis Service Agreement (the 'Agreement') is entered into as of the "
    "Effective Date between ACME Corp ('Service Provider') and Client Co ('Client'). "
    "1. SERVICES. Provider shall deliver software development services as described in SOW. "
    "2. PAYMENT. Client shall pay $10,000/month net-30. Late payments incur 1.5%/month interest. "
    "3. INTELLECTUAL PROPERTY. All work product is work-for-hire and vests in Client upon payment. "
    "4. INDEMNIFICATION. Each party indemnifies the other for its own gross negligence or willful misconduct. "
    "5. LIMITATION OF LIABILITY. In no event shall either party be liable for indirect, consequential, "
    "or punitive damages. Provider liability capped at 3 months of fees. "
    "6. CONFIDENTIALITY. Both parties agree to keep Confidential Information secret for 5 years. "
    "7. TERMINATION. Either party may terminate with 30-day written notice. Client owes fees through termination. "
    "8. GOVERNING LAW. This Agreement is governed by the laws of Delaware. " * 20
)

_SAMPLES: list[dict] = [
    # FREE tier targets
    {"prompt": "hi", "priority": "normal"},
    {"prompt": "What is 2+2?", "priority": "normal"},
    {"prompt": "Translate 'hello world' to Japanese", "priority": "normal"},
    {"prompt": "What's the capital of France?", "priority": "normal"},
    {"prompt": "Thanks!", "priority": "normal"},
    # CHEAP tier targets
    {
        "prompt": "Classify this review as positive or negative: 'Great product, loved it!'",
        "priority": "normal",
    },
    {
        "prompt": "Extract all email addresses from: john@test.com, jane@corp.io, bob@mail.com",
        "priority": "normal",
    },
    {
        "prompt": f"Summarize this email about a meeting reschedule: {_MOCK_EMAIL}",
        "priority": "normal",
    },
    {"prompt": "What is photosynthesis?", "priority": "normal"},
    {"prompt": "Convert 100 USD to EUR", "priority": "normal"},
    # MID tier targets
    {
        "prompt": "Write a Python function to reverse a linked list with O(1) space complexity",
        "priority": "normal",
    },
    {
        "prompt": "Compare PostgreSQL vs MongoDB for a real-time analytics dashboard",
        "priority": "normal",
    },
    {
        "prompt": "Explain the CAP theorem and its implications for distributed databases",
        "priority": "normal",
    },
    {
        "prompt": "Write a 500-word blog post about the future of renewable energy",
        "priority": "normal",
    },
    {
        "prompt": "Explain quantum entanglement to a 10 year old using simple analogies",
        "priority": "normal",
    },
    # PREMIUM tier targets
    {
        "prompt": "Write a React component for a drag-and-drop kanban board with state management, animations, and persistence",
        "priority": "normal",
    },
    {
        "prompt": "Analyze the legal implications of GDPR Article 17 on cross-border data transfers considering recent ECJ rulings",
        "priority": "normal",
        "sensitivity": "confidential",
    },
    {
        "prompt": f"Debug this async Python program with a subtle deadlock:\n{_MOCK_CODE}",
        "priority": "high",
    },
    {
        "prompt": "Write a comprehensive REST API with auth, rate limiting, pagination, and WebSocket support in FastAPI",
        "priority": "normal",
    },
    {
        "prompt": "Design a database schema for a multi-tenant SaaS platform with row-level security, audit logging, and soft deletes",
        "priority": "normal",
    },
    # ULTRA tier targets
    {
        "prompt": "Prove that the sum of all natural numbers diverges, then explain why Ramanujan's -1/12 result is valid in zeta function regularization. Show all steps with ∑ notation.",
        "priority": "critical",
    },
    {
        "prompt": f"Given this contract, identify every liability clause, cross-reference with indemnification sections, and produce a risk matrix:\n{_MOCK_CONTRACT}",
        "priority": "high",
    },
    # SPECIAL cases
    {
        "prompt": "Write a story about a time traveler who accidentally prevents their own birth",
        "priority": "low",
        "temperature": 1.0,
    },
    {
        "prompt": "What objects are visible in this image? Describe in detail.",
        "priority": "normal",
        "capabilities": ["vision"],
    },
    {
        "prompt": "Solve: minimize f(x,y) = x² + y² subject to x + y ≥ 10, using KKT conditions. Show all derivation steps.",
        "priority": "critical",
    },
]


def _build_request(idx: int, sample: dict) -> RoutingRequest:
    meta: dict = {}
    if "sensitivity" in sample:
        meta["sensitivity_level"] = sample["sensitivity"]

    return RoutingRequest(
        raw_prompt=sample["prompt"],
        user_id=f"demo_user_{(idx % 3) + 1}",
        plan="business_plan",
        priority=sample.get("priority", "normal"),
        temperature=sample.get("temperature"),
        required_capabilities=sample.get("capabilities", []),
        metadata=meta,
        correlation_id=str(uuid.uuid4()),
    )


def _truncate(text: str, max_len: int = 40) -> str:
    return (text[:max_len] + "…") if len(text) > max_len else text


def _fmt_cost(cost: float) -> str:
    return f"${cost:.4f}"


def _fmt_savings(savings: float) -> str:
    return f"${savings:.4f}"


async def run_demo() -> None:
    # ── Wire up all components ──────────────────────────────────────────────
    registry = ModelRegistry()
    cache = ResponseCache()
    adaptive = AdaptiveWeights(state_file=None)  # in-memory only for demo
    analytics = RoutingAnalytics(log_path=None)  # in-memory only for demo
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

    # ── Build rich table ────────────────────────────────────────────────────
    table = Table(
        box=box.ROUNDED,
        show_footer=False,
        title="[bold cyan]LLM Router — 25 Sample Requests[/bold cyan]",
        title_style="bold",
        expand=True,
    )
    table.add_column("#", style="dim", width=3, no_wrap=True)
    table.add_column("Request", style="white", min_width=36, no_wrap=True)
    table.add_column("Type", style="cyan", width=10, no_wrap=True)
    table.add_column("Score", style="yellow", width=6, no_wrap=True)
    table.add_column("Model", style="green", min_width=24, no_wrap=True)
    table.add_column("Tier", style="magenta", width=8, no_wrap=True)
    table.add_column("Est Cost", style="blue", width=9, no_wrap=True)
    table.add_column("Savings", style="bright_green", width=9, no_wrap=True)
    table.add_column("Cache", style="dim", width=5, no_wrap=True)

    # ── Run all requests ─────────────────────────────────────────────────────
    total_cost = 0.0
    total_savings = 0.0
    cache_hits = 0
    tier_counts: dict[str, int] = {"free": 0, "cheap": 0, "mid": 0, "premium": 0, "ultra": 0}
    all_scores: list[float] = []

    # Add a cache-seeding pass so request #3 (translation) can become a cache hit
    # on a re-run.  For a single run, we'll demonstrate by adding one duplicate.
    requests: list[tuple[int, RoutingRequest]] = []
    for i, s in enumerate(_SAMPLES, start=1):
        requests.append((i, _build_request(i, s)))

    # Intentionally route request #3 twice to show a cache hit on the second pass
    # (same prompt, same temperature → same fingerprint → cache hit)
    dup_req = _build_request(99, _SAMPLES[2])  # same as request 3
    # Pre-warm: route it silently so it gets cached
    pre_decision = await engine.route(dup_req)
    if pre_decision.chosen_model and not pre_decision.cost_blocked:
        cache.set(
            fp=dup_req.correlation_id,  # wrong key but illustrative
            response_text="こんにちは世界",
            model_used=pre_decision.chosen_model,
            original_cost=pre_decision.estimated_cost,
        )
        # Seed the REAL fingerprint so the demo translation request hits cache
        from router.cache import fingerprint as fp_fn

        real_fp = fp_fn(_SAMPLES[2]["prompt"], None, [], None)
        cache.set(
            fp=real_fp,
            response_text="こんにちは世界",
            model_used=pre_decision.chosen_model,
            original_cost=pre_decision.estimated_cost,
        )

    decisions = []
    for i, req in requests:
        d = await engine.route(req)
        decisions.append(d)

        model_name = d.chosen_model.display_name if d.chosen_model else "BLOCKED"
        tier_name = d.chosen_model.tier if d.chosen_model else "—"
        task_type = ""
        score = 0.0

        # Re-analyse for display (classifier already ran; re-run is cheap)
        analysis = classifier.analyze(req)
        task_type = analysis.task_type[:9]
        score = analysis.complexity_score

        if d.cost_blocked:
            tier_name = "blocked"
            model_name = "COST CEILING"

        total_cost += d.estimated_cost
        total_savings += d.estimated_savings
        if d.cache_hit:
            cache_hits += 1
        if tier_name in tier_counts:
            tier_counts[tier_name] += 1
        all_scores.append(score)

        # Tier colour coding
        tier_style = {
            "free": "bright_green",
            "cheap": "green",
            "mid": "yellow",
            "premium": "red",
            "ultra": "bright_red",
            "blocked": "dim",
            "—": "dim",
        }.get(tier_name, "white")

        cache_icon = "[bold green]✓[/]" if d.cache_hit else "[dim]✗[/]"

        table.add_row(
            str(i),
            _truncate(req.raw_prompt),
            task_type,
            f"{score:.3f}",
            model_name,
            f"[{tier_style}]{tier_name}[/]",
            _fmt_cost(d.estimated_cost),
            _fmt_savings(d.estimated_savings),
            cache_icon,
        )

    console.print()
    console.print(table)

    # ── Summary footer ───────────────────────────────────────────────────────
    avg_score = sum(all_scores) / max(len(all_scores), 1)
    premium_eq = total_cost + total_savings
    pct_saved = (total_savings / premium_eq * 100) if premium_eq > 0 else 0.0

    tier_str = "  |  ".join(
        f"[bold]{t}[/]: {tier_counts[t]}" for t in ["free", "cheap", "mid", "premium", "ultra"]
    )
    cache_pct = cache_hits / len(requests) * 100

    console.rule("[bold]Summary[/bold]")
    console.print(
        f"  [blue]Cost[/]:      ${total_cost:.4f}    "
        f"[dim]Premium equiv[/]: ${premium_eq:.4f}    "
        f"[green]Saved[/]: ${total_savings:.4f} ({pct_saved:.1f}%)"
    )
    console.print(
        f"  [magenta]Tier breakdown[/]:  {tier_str}    [yellow]Avg complexity[/]: {avg_score:.3f}"
    )
    console.print(
        f"  [cyan]Cache hits[/]: {cache_hits} ({cache_pct:.0f}%)    "
        f"[dim]Total decisions[/]: {len(requests)}"
    )
    console.print()

    # ── Cache stats ──────────────────────────────────────────────────────────
    stats = cache.stats()
    console.print(
        f"  [dim]Cache stats[/] — "
        f"entries: {stats['entries_count']}  |  "
        f"hit rate: {stats['hit_rate']:.1%}  |  "
        f"total hits: {stats['total_hits']}  |  "
        f"misses: {stats['total_misses']}"
    )
    console.print()


if __name__ == "__main__":
    asyncio.run(run_demo())
