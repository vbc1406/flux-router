"""
router/benchmark.py — stress-test the routing engine with 500 synthetic prompts.

Usage:
    python router/benchmark.py
    python -m router.benchmark
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache, is_cache_eligible
from router.cache import fingerprint as fp_fn
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.quality_scorer import QualityScorer
from router.routing_engine import RoutingEngine
from router.schemas import RoutingDecision, RoutingRequest

console = Console()

# ── Static word lists ─────────────────────────────────────────────────────────

_GREETINGS = ["hi", "hello", "hey", "thanks", "ok", "sure", "bye", "okay", "cool", "great"]

_LANGUAGES = ["French", "Spanish", "German", "Japanese", "Korean", "Arabic"]

_SIMPLE_FACTS = [
    "What is 2+2?",
    "Who is Einstein?",
    "Define osmosis",
    "What is gravity?",
    "Who painted the Mona Lisa?",
    "What is DNA?",
    "Define democracy",
    "What is the speed of light?",
    "Who invented the telephone?",
    "What is Pi?",
]

_YES_NO = [
    "Is Python compiled?",
    "Is the earth round?",
    "Is water H2O?",
    "Is Bitcoin decentralized?",
    "Is the sun a star?",
    "Is coffee a stimulant?",
    "Is React a framework?",
    "Is SQL a programming language?",
]

_PHRASES = [
    "good morning",
    "thank you",
    "good night",
    "how are you",
    "see you later",
    "have a nice day",
    "excuse me",
    "you're welcome",
]

_UNITS = [
    ("5 miles", "km"),
    ("100 Fahrenheit", "Celsius"),
    ("10 liters", "gallons"),
    ("50 kg", "pounds"),
    ("100 USD", "EUR"),
    ("3 feet", "meters"),
    ("1 hour", "minutes"),
    ("500 calories", "kilojoules"),
]

_REVIEWS = [
    "This product is amazing, would buy again!",
    "Terrible quality, broke after one day.",
    "Decent but not worth the price.",
    "Absolutely love it, five stars!",
    "Mediocre at best, nothing special.",
    "Great value for money!",
    "Do not buy, complete waste of money!",
    "Pretty good overall, minor complaints.",
]

_PARAGRAPHS = [
    (
        "Climate change is one of the most pressing issues of our time. Rising temperatures, "
        "melting ice caps, and extreme weather events are becoming more common. Scientists "
        "warn that without significant reductions in greenhouse gas emissions, the consequences "
        "could be catastrophic for both human society and natural ecosystems worldwide."
    ),
    (
        "The invention of the printing press by Johannes Gutenberg in the 15th century "
        "revolutionized the spread of information. For the first time, books could be produced "
        "quickly and cheaply, making knowledge accessible to a much wider audience. This helped "
        "fuel the Renaissance, the Protestant Reformation, and the Scientific Revolution."
    ),
    (
        "Machine learning is a subset of artificial intelligence that enables systems to learn "
        "from data without being explicitly programmed. Through exposure to training data, "
        "algorithms can identify patterns and make decisions with minimal human intervention. "
        "Applications range from spam filtering to autonomous vehicles and medical diagnostics."
    ),
]

_TWEETS = [
    "Just had the best coffee ever at this new cafe downtown!",
    "I can't believe how bad the traffic is today. Two hours to go 10 miles.",
    "Really impressed by the new features. Totally worth the upgrade!",
    "Another Monday, another chance to prove everyone wrong. Let's go!",
    "Disappointed with the customer service at this store. Will not return.",
]

_SORT_TASKS = [
    "reverse a linked list with O(1) space complexity",
    "sort a list of dictionaries by a nested key",
    "find the longest common subsequence of two strings",
    "implement binary search on a sorted array",
    "merge two sorted arrays into one sorted array",
    "detect a cycle in a directed graph",
    "find all permutations of a string",
    "implement a min-heap from scratch",
]

_BLOG_TOPICS = [
    "the future of renewable energy",
    "benefits of remote work for productivity",
    "impact of social media on mental health",
    "rise of electric vehicles and charging infrastructure",
    "future of artificial intelligence in healthcare",
    "importance of cybersecurity for small businesses",
    "benefits of meditation and mindfulness for developers",
    "future of space exploration and commercial spaceflight",
]

_COMPARE_PAIRS = [
    ("PostgreSQL", "MongoDB", "a real-time analytics dashboard"),
    ("React", "Vue", "a large-scale enterprise application"),
    ("Docker", "Kubernetes", "container orchestration at scale"),
    ("REST", "GraphQL", "a mobile API backend"),
    ("Python", "Go", "building a high-throughput microservice"),
    ("AWS", "GCP", "machine learning workloads"),
]

_EXPLAIN_TOPICS = [
    "the CAP theorem and its implications for distributed databases",
    "quantum entanglement using simple everyday analogies",
    "how the TCP/IP protocol stack works end-to-end",
    "the difference between a process and a thread in operating systems",
    "how neural networks learn from data through backpropagation",
    "the ACID properties of relational database transactions",
]

_ANALYSIS_TOPICS = [
    "moving the company's infrastructure to the cloud",
    "adopting a microservices architecture from a monolith",
    "implementing a four-day work week policy",
    "expanding into international markets with a new product",
    "switching the data pipeline from batch to streaming processing",
]

_COMPLEX_CODE = [
    ("REST API with authentication, rate limiting, pagination, and WebSocket support", "FastAPI"),
    ("drag-and-drop kanban board with state management, animations, and persistence", "React"),
    ("CLI tool for monitoring system resources with configurable alerts and logging", "Python"),
    (
        "real-time chat application with rooms, typing indicators, and message persistence",
        "Node.js",
    ),
    ("distributed job queue with retry logic, dead letter queues, and metrics", "Python"),
]

_LEGAL_FINANCIAL = [
    (
        "Analyze the legal implications of GDPR Article 17 on cross-border data transfers "
        "considering recent ECJ rulings and the invalidation of Privacy Shield"
    ),
    (
        "Explain the tax implications of converting a C-Corp to an LLC in California, "
        "including treatment of accumulated earnings and built-in gains"
    ),
    (
        "What are the liability risks in an indemnification clause that caps damages at "
        "3 months of fees? Cross-reference with our limitation of liability section"
    ),
    (
        "Analyze the financial risks of taking on venture debt with 15% interest, "
        "warrant coverage at 20%, and a material adverse change clause"
    ),
    (
        "Explain the SOC 2 Type II compliance requirements for a SaaS company that handles "
        "PHI under HIPAA — identify overlaps and conflicts between the two frameworks"
    ),
]

_BUGGY_CODE_1 = """```python
def process_orders(orders):
    results = []
    for order in orders:
        if order["status"] == "pending":
            total = 0
            for item in order["items"]:
                total += item["price"] * item["quantity"]
            if total > 1000:
                order["discount"] = total * 0.1
            results.append(order)
    return results
# Issues: mutates input dicts, no handling for missing keys, no float precision
```"""

_BUGGY_CODE_2 = """```python
import asyncio
import threading

shared_lock = threading.Lock()
shared_data = []

async def producer():
    while True:
        with shared_lock:
            shared_data.append(await fetch_data())
        await asyncio.sleep(1)

async def consumer():
    while True:
        with shared_lock:  # potential deadlock
            if shared_data:
                item = shared_data.pop(0)
                await process(item)
        await asyncio.sleep(0.1)
```"""

_BUGGY_CODE_3 = """```javascript
const handlers = [];
setInterval(() => {
  handlers.push(() => console.log('tick'));
}, 100);
// Memory leak: handlers array grows unbounded
```"""

_ARCH_SCENARIOS = [
    "a multi-tenant SaaS platform with row-level security, audit logging, and soft deletes",
    "a high-frequency trading system that must process 1 million events per second",
    "a distributed cache with consistent hashing, automatic rebalancing, and eviction policies",
    "a recommendation engine for an e-commerce platform with 10 million products",
    "a real-time fraud detection system for a payment processor handling global transactions",
]

_THEOREMS = [
    "Prove that √2 is irrational using proof by contradiction. Show all algebraic steps.",
    "Prove that there are infinitely many prime numbers using Euclid's original proof. Show ∑ notation where applicable.",
    "Derive the quadratic formula from ax² + bx + c = 0 by completing the square. Show every step.",
    "Prove the Pythagorean theorem using geometric dissection methods. Include ∑ and ∫ notation.",
    "Prove and derive Bayes' theorem P(A|B) = P(B|A)P(A)/P(B) from first principles.",
]

_REASONING_PROMPTS = [
    (
        "Prove that the sum of all natural numbers diverges, then explain why Ramanujan's "
        "result of -1/12 is valid under zeta function regularization ζ(s). "
        "Show all steps with ∑ notation."
    ),
    (
        "Solve: minimize f(x,y) = x² + y² subject to x + y ≥ 10, "
        "using KKT conditions. Show all derivation steps including the Lagrangian, "
        "∇L = 0, and complementary slackness conditions."
    ),
    (
        "Using the Hamiltonian formalism, derive the equations of motion for a simple "
        "pendulum. Show all ∂H/∂p and ∂H/∂q steps and verify against the Lagrangian result."
    ),
]

_LONG_CONTRACT = (
    "SERVICE AGREEMENT\n\nThis Service Agreement ('Agreement') is entered into by "
    "ACME Corp ('Provider') and Client Co ('Client'). "
    "1. SERVICES: Provider delivers software development per attached SOW. "
    "2. PAYMENT: $10,000/month net-30. Late payments incur 1.5%/month interest. "
    "3. IP: All work is work-for-hire vesting in Client upon full payment. "
    "4. INDEMNIFICATION: Each party indemnifies the other for gross negligence. "
    "5. LIABILITY CAP: Capped at 3 months fees; no indirect or consequential damages. "
    "6. CONFIDENTIALITY: 5-year NDA on all Confidential Information. "
    "7. TERMINATION: 30-day written notice; fees owed through termination date. "
    "8. GOVERNING LAW: State of Delaware. "
) * 30  # ~900-word simulated document


# ── Dataset generation ─────────────────────────────────────────────────────────


def _req(
    prompt: str,
    category: str,
    priority: str = "normal",
    temperature: Optional[float] = None,
    capabilities: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
    user_id: Optional[str] = None,
    plan: str = "business_plan",
) -> dict:
    return {
        "raw_prompt": prompt,
        "user_id": user_id or "bench_user_{}".format(random.randint(1, 20)),
        "plan": plan,
        "priority": priority,
        "temperature": temperature,
        "required_capabilities": capabilities or [],
        "metadata": dict(metadata or {}),
        "category": category,
    }


def generate_test_dataset() -> List[dict]:
    """
    Return exactly 500 synthetic benchmark requests, seeded for reproducibility.
    Distribution: 100 free + 100 cheap + 100 mid + 100 premium + 50 ultra + 50 edge.
    """
    random.seed(42)
    ds: List[dict] = []

    # ── FREE tier targets (100) ───────────────────────────────────────────────

    # Greetings (20)
    for _ in range(20):
        ds.append(_req(random.choice(_GREETINGS), "free_greeting"))

    # Simple factual (20)
    for q in random.choices(_SIMPLE_FACTS, k=20):
        ds.append(_req(q, "free_factual"))

    # Yes/no (10)
    for q in random.choices(_YES_NO, k=10):
        ds.append(_req(q, "free_yesno"))

    # Translations (25)
    for phrase in random.choices(_PHRASES, k=25):
        lang = random.choice(_LANGUAGES)
        ds.append(_req("Translate '{}' to {}".format(phrase, lang), "free_translation"))

    # Unit conversions (25)
    for _ in range(25):
        amount, unit = random.choice(_UNITS)
        ds.append(_req("Convert {} to {}".format(amount, unit), "free_conversion"))

    # ── CHEAP tier targets (100) ──────────────────────────────────────────────

    # Classification (25)
    for review in random.choices(_REVIEWS, k=25):
        ds.append(
            _req(
                "Classify this review as positive or negative: '{}'".format(review),
                "cheap_classification",
            )
        )

    # Extraction (25)
    for i in range(25):
        email = "user{}@example{}.com".format(i, i)
        text = "Please contact {} for more information about our services.".format(email)
        ds.append(_req("Extract all email addresses from: {}".format(text), "cheap_extraction"))

    # Summarization (25)
    for para in random.choices(_PARAGRAPHS, k=25):
        ds.append(_req("Summarize in one sentence: {}".format(para), "cheap_summarization"))

    # Simple Q&A (12)
    _simple_qa = [
        "What causes rain?",
        "How do vaccines work?",
        "Why is the sky blue?",
        "How does the internet work?",
        "What is inflation?",
    ]
    for q in random.choices(_simple_qa, k=12):
        ds.append(_req(q, "cheap_qa"))

    # Sentiment (13)
    for tweet in random.choices(_TWEETS, k=13):
        ds.append(_req("What's the sentiment of: '{}'".format(tweet), "cheap_sentiment"))

    # ── MID tier targets (100) ────────────────────────────────────────────────

    # Code generation (25)
    for task in random.choices(_SORT_TASKS, k=25):
        lang = random.choice(["Python", "JavaScript", "Go", "Java"])
        ds.append(
            _req(
                "Write a {} function to {} with O(n log n) or better time complexity".format(
                    lang, task
                ),
                "mid_code",
            )
        )

    # Blog posts (20)
    for topic in random.choices(_BLOG_TOPICS, k=20):
        ds.append(_req("Write a 500-word blog post about {}".format(topic), "mid_blog"))

    # Comparisons (20)
    for x, y, use in random.choices(_COMPARE_PAIRS, k=20):
        ds.append(_req("Compare {} vs {} for {}".format(x, y, use), "mid_comparison"))

    # Explanations (20)
    for topic in random.choices(_EXPLAIN_TOPICS, k=20):
        ds.append(
            _req("Explain {} in detail with practical examples".format(topic), "mid_explanation")
        )

    # Analysis (15)
    for topic in random.choices(_ANALYSIS_TOPICS, k=15):
        ds.append(_req("Analyze the pros and cons of {}".format(topic), "mid_analysis"))

    # ── PREMIUM tier targets (100) ────────────────────────────────────────────

    # Complex code (25)
    for desc, fw in random.choices(_COMPLEX_CODE, k=25):
        prio = random.choice(["normal", "high"])
        ds.append(_req("Build a complete {} in {}".format(desc, fw), "premium_code", priority=prio))

    # Legal / financial (25)
    for prompt in random.choices(_LEGAL_FINANCIAL, k=25):
        ds.append(_req(prompt, "premium_legal", metadata={"sensitivity_level": "confidential"}))

    # Debugging (25)
    _debug = [_BUGGY_CODE_1, _BUGGY_CODE_2, _BUGGY_CODE_3]
    for code in random.choices(_debug, k=25):
        ds.append(
            _req(
                "Debug this code and explain every bug you find:\n{}".format(code),
                "premium_debug",
                priority="high",
            )
        )

    # Architecture (25)
    for scenario in random.choices(_ARCH_SCENARIOS, k=25):
        ds.append(
            _req(
                "Design a complete system architecture for {}".format(scenario),
                "premium_architecture",
            )
        )

    # ── ULTRA tier targets (50) ───────────────────────────────────────────────

    # Math proofs (20)
    for theorem in random.choices(_THEOREMS, k=20):
        ds.append(_req(theorem, "ultra_math", priority="critical"))

    # Long-document analysis (15)
    for _ in range(15):
        ds.append(
            _req(
                "Given this contract, identify every liability clause, cross-reference "
                "with indemnification sections, and produce a risk matrix:\n{}".format(
                    _LONG_CONTRACT
                ),
                "ultra_longdoc",
                priority="critical",
            )
        )

    # Multi-step reasoning (15)
    for prompt in random.choices(_REASONING_PROMPTS, k=15):
        ds.append(_req(prompt, "ultra_reasoning", priority="critical"))

    # ── EDGE CASES (50) ──────────────────────────────────────────────────────

    # Empty / near-empty (5)
    ds.append(_req("", "edge_empty"))
    ds.append(_req(" ", "edge_empty"))
    ds.append(_req("??", "edge_empty"))
    ds.append(_req("🤔", "edge_emoji"))
    ds.append(_req("👍", "edge_emoji"))

    # Very long prompts (5)
    _lorem = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    ds.append(_req("Analyze this text: " + _lorem * 150, "edge_long"))
    ds.append(
        _req("Summarize the following: " + ("The quick brown fox jumps. " * 400), "edge_long")
    )
    _code_blob = "```python\n" + ("result = result + compute(x)\n" * 400) + "```"
    ds.append(_req(_code_blob, "edge_long"))
    ds.append(_req("Review this code: " + _code_blob, "edge_long"))
    ds.append(_req("What does this do? " + _code_blob, "edge_long"))

    # Code / SQL / CLI / URL only (5)
    ds.append(_req("```python\ndef foo(x):\n    return x * 2\n```", "edge_code_only"))
    ds.append(_req("```javascript\nconst add = (a, b) => a + b;\n```", "edge_code_only"))
    ds.append(_req("SELECT * FROM users WHERE id = 1 AND active = true;", "edge_sql_only"))
    ds.append(
        _req("git commit -m 'fix: resolve race condition in auth middleware'", "edge_cli_only")
    )
    ds.append(
        _req(
            "curl -X POST https://api.example.com/v1/data -H 'Content-Type: application/json'",
            "edge_url_only",
        )
    )

    # Vision requests (5)
    for _ in range(5):
        ds.append(
            _req(
                "What objects are visible in this image? Describe each one in detail.",
                "edge_vision",
                capabilities=["vision"],
            )
        )

    # Function calling (5)
    for _ in range(5):
        ds.append(
            _req(
                "Get the current weather for San Francisco, CA",
                "edge_function_calling",
                capabilities=["function_calling"],
            )
        )

    # High temperature (should NOT be cached) (5)
    _creative = "Write a short poem about the ocean at sunset"
    for _ in range(5):
        ds.append(_req(_creative, "edge_high_temp", temperature=1.0))

    # Low temperature (SHOULD be cached after first hit) (5)
    _cacheable = "What is the capital of Germany?"
    for _ in range(5):
        ds.append(_req(_cacheable, "edge_low_temp", temperature=0.0))

    # Duplicate prompts for cache hit testing (5)
    _dup = "Translate 'hello' to French"
    for _ in range(5):
        ds.append(_req(_dup, "edge_duplicate"))

    # Explicit model overrides (2)
    ds.append(
        _req("Write a haiku about mountains", "edge_override", metadata={"model": "gpt-4o-mini"})
    )
    ds.append(
        _req(
            "Explain recursion briefly",
            "edge_override",
            metadata={"model": "claude-haiku-4-5-20251001"},
        )
    )

    # Anti-gaming: rapid-fire from same user (5)
    for _ in range(5):
        ds.append(_req(random.choice(_GREETINGS), "edge_antispam", user_id="spammy_user_99"))

    # Budget-limited: free_plan user (3)
    for _ in range(3):
        ds.append(
            _req(
                "Write a detailed analysis of quantum computing applications in cryptography",
                "edge_budget",
                user_id="budget_user_1",
                plan="free_plan",
            )
        )

    assert len(ds) == 500, "Expected 500, got {}".format(len(ds))
    return ds


# ── Results container ─────────────────────────────────────────────────────────


@dataclass
class RouteResult:
    idx: int
    category: str
    prompt: str
    decision: RoutingDecision
    routing_time_ms: float
    task_type: str = "unknown"
    complexity: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    results: List[RouteResult]
    total_time_s: float

    @property
    def successful(self) -> List[RouteResult]:
        return [r for r in self.results if r.error is None]

    @property
    def failed(self) -> List[RouteResult]:
        return [r for r in self.results if r.error is not None]


# ── Benchmark runner ──────────────────────────────────────────────────────────


def _extract_reasoning(reasoning: str) -> Tuple[str, float]:
    """Pull task_type and complexity out of the routing decision reasoning string."""
    task_type = "unknown"
    complexity = 0.0
    if "task=" in reasoning:
        task_type = reasoning.split("task=")[1].split(" |")[0].strip()
    if "complexity=" in reasoning:
        try:
            complexity = float(reasoning.split("complexity=")[1].split(" |")[0].strip())
        except ValueError:
            pass
    return task_type, complexity


async def run_benchmark(dataset: List[dict]) -> BenchmarkResults:
    registry = ModelRegistry()
    cache = ResponseCache()
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    _quality = QualityScorer(adaptive)
    # Separate classifier instance with cache disabled — used to get task_type
    # for every request regardless of cache hit status (cache hits have no task=
    # in their reasoning string, which would otherwise inflate "unknown" count).
    _probe_cache = ResponseCache(enabled=False)
    _probe_clf = RequestClassifier(_probe_cache)

    engine = RoutingEngine(
        model_registry=registry,
        classifier=classifier,
        cache=cache,
        budget_tracker=budget,
        adaptive_weights=adaptive,
        context_compressor=compressor,
        analytics=analytics,
    )

    results: List[RouteResult] = []
    start_all = time.perf_counter()

    for idx, sample in enumerate(dataset, start=1):
        category = sample["category"]
        temp = sample["temperature"]

        try:
            req = RoutingRequest(
                raw_prompt=sample["raw_prompt"],
                user_id=sample["user_id"],
                plan=sample["plan"],
                priority=sample["priority"],
                temperature=temp,
                required_capabilities=sample["required_capabilities"],
                metadata=sample["metadata"],
            )
        except Exception as exc:
            results.append(
                RouteResult(
                    idx=idx,
                    category=category,
                    prompt=sample["raw_prompt"],
                    decision=RoutingDecision(),
                    routing_time_ms=0.0,
                    error=str(exc),
                )
            )
            continue

        t0 = time.perf_counter()
        decision = RoutingDecision()
        error = None
        try:
            decision = await engine.route(req)
        except Exception as exc:
            error = str(exc)
        routing_ms = (time.perf_counter() - t0) * 1000

        # Always classify directly — don't rely on reasoning string which is
        # absent for cache hits.
        try:
            _analysis = _probe_clf.analyze(req)
            task_type = _analysis.task_type
            complexity = _analysis.complexity_score
        except Exception:
            task_type, complexity = _extract_reasoning(decision.reasoning)

        # Seed the response cache after the first successful routing of a cache-eligible
        # request, so that subsequent identical prompts can return cache hits.
        if (
            error is None
            and not decision.cache_hit
            and not decision.cost_blocked
            and decision.chosen_model is not None
            and is_cache_eligible(task_type, temp)
        ):
            real_fp = fp_fn(
                req.raw_prompt,
                req.system_prompt,
                req.message_history,
                req.temperature,
            )
            cache.set(
                fp=real_fp,
                response_text="[benchmark mock response]",
                model_used=decision.chosen_model,
                original_cost=decision.estimated_cost,
            )

        results.append(
            RouteResult(
                idx=idx,
                category=category,
                prompt=sample["raw_prompt"],
                decision=decision,
                routing_time_ms=routing_ms,
                task_type=task_type,
                complexity=complexity,
                error=error,
            )
        )

    total_time = time.perf_counter() - start_all
    return BenchmarkResults(results=results, total_time_s=total_time)


# ── Consistency check ─────────────────────────────────────────────────────────


async def run_consistency_check(prompts: List[str]) -> List[dict]:
    """
    Route each prompt twice with a fresh engine (cache disabled) and return
    any cases where the chosen model differed between runs — excluding A/B
    exploration which is intentionally non-deterministic.
    """
    registry = ModelRegistry()
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
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

    inconsistencies: List[dict] = []
    for prompt in prompts:
        req1 = RoutingRequest(raw_prompt=prompt, user_id="consistency_user", plan="business_plan")
        req2 = RoutingRequest(raw_prompt=prompt, user_id="consistency_user", plan="business_plan")
        d1 = await engine.route(req1)
        d2 = await engine.route(req2)

        m1 = d1.chosen_model.model_id if d1.chosen_model else "none"
        m2 = d2.chosen_model.model_id if d2.chosen_model else "none"

        ab1 = "ab_exploration" in d1.routing_rule_matched
        ab2 = "ab_exploration" in d2.routing_rule_matched

        if m1 != m2 and not ab1 and not ab2:
            inconsistencies.append({"prompt": prompt[:60], "run1": m1, "run2": m2})

    return inconsistencies


# ── Reporting helpers ─────────────────────────────────────────────────────────


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    return sv[min(int(len(sv) * p / 100), len(sv) - 1)]


def _bar(count: int, max_count: int, width: int = 36) -> str:
    if max_count == 0:
        return ""
    return "█" * max(1, int(count / max_count * width)) if count else ""


# ── Report sections ───────────────────────────────────────────────────────────


def print_section_a(bench: BenchmarkResults) -> None:
    console.rule("[bold cyan]A · Overall Performance[/bold cyan]")
    times = [r.routing_time_ms for r in bench.results]
    total_ms = bench.total_time_s * 1000
    under_10 = sum(1 for t in times if t < 10)
    n = len(bench.results)

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column("Metric", style="dim", min_width=28)
    tbl.add_column("Value", style="bold")
    for label, val in [
        ("Total requests", str(n)),
        ("Total routing time", "{:.1f} ms".format(total_ms)),
        ("Avg per request", "{:.3f} ms".format(total_ms / n)),
        ("P50 latency", "{:.3f} ms".format(_pct(times, 50))),
        ("P95 latency", "{:.3f} ms".format(_pct(times, 95))),
        ("P99 latency", "{:.3f} ms".format(_pct(times, 99))),
        ("Max latency", "{:.3f} ms".format(max(times))),
        ("Requests under 10ms", "{}/{} ({:.1f}%)".format(under_10, n, under_10 / n * 100)),
        ("Errors / crashes", str(len(bench.failed))),
    ]:
        tbl.add_row(label, val)
    console.print(tbl)


def print_section_b(bench: BenchmarkResults) -> None:
    console.rule("[bold cyan]B · Tier Distribution[/bold cyan]")
    _tiers = ["free", "cheap", "mid", "premium", "ultra", "blocked/none"]
    tier_data: dict = {t: {"count": 0, "costs": []} for t in _tiers}
    total = len(bench.results)

    for r in bench.results:
        if r.decision.cost_blocked or r.decision.chosen_model is None:
            key = "blocked/none"
        else:
            key = r.decision.chosen_model.tier
        tier_data[key]["count"] += 1
        tier_data[key]["costs"].append(r.decision.estimated_cost)

    _colors = {
        "free": "bright_green",
        "cheap": "green",
        "mid": "yellow",
        "premium": "red",
        "ultra": "bright_red",
        "blocked/none": "dim",
    }
    tbl = Table(box=box.ROUNDED)
    tbl.add_column("Tier", style="bold", width=14)
    tbl.add_column("Count", justify="right", width=7)
    tbl.add_column("% Share", justify="right", width=8)
    tbl.add_column("Avg Cost", justify="right", width=12)
    tbl.add_column("Bar", min_width=20)

    max_cnt = max(d["count"] for d in tier_data.values()) or 1
    for tier in _tiers:
        d = tier_data[tier]
        if d["count"] == 0:
            continue
        cnt = d["count"]
        avg = sum(d["costs"]) / cnt if cnt else 0.0
        color = _colors.get(tier, "white")
        bar = _bar(cnt, max_cnt, width=20)
        tbl.add_row(
            "[{}]{}[/]".format(color, tier),
            str(cnt),
            "{:.1f}%".format(cnt / total * 100),
            "${:.5f}".format(avg),
            "[{}]{}[/]".format(color, bar),
        )
    console.print(tbl)


def print_section_c(bench: BenchmarkResults) -> None:
    console.rule("[bold cyan]C · Task Type Classification[/bold cyan]")

    # Expected task type per category (for misclassification detection)
    _expected: dict = {
        "free_greeting": {"conversation"},
        "free_factual": {"simple_qa"},
        "free_yesno": {"simple_qa"},
        "free_translation": {"translation"},
        "free_conversion": {"simple_qa", "unknown"},
        "cheap_classification": {"classification"},
        "cheap_extraction": {"extraction"},
        "cheap_summarization": {"summarization"},
        "cheap_qa": {"simple_qa"},
        "cheap_sentiment": {"classification"},
        "mid_code": {"code_generation"},
        "mid_blog": {"creative_writing"},
        "mid_comparison": {"analysis"},
        "mid_explanation": {"simple_qa", "reasoning", "analysis", "unknown"},
        "mid_analysis": {"analysis"},
        "premium_code": {"code_generation", "code_review"},
        "premium_legal": {"analysis", "long_document", "unknown"},
        "premium_debug": {"code_review"},
        "premium_architecture": {"analysis", "unknown"},
        "ultra_math": {"reasoning"},
        "ultra_longdoc": {"long_document"},
        "ultra_reasoning": {"reasoning"},
    }

    # Aggregate by actual task type
    type_stats: dict = {}
    for r in bench.results:
        tt = r.task_type
        if tt not in type_stats:
            type_stats[tt] = {"count": 0, "complexities": [], "models": [], "wrong": 0}
        type_stats[tt]["count"] += 1
        type_stats[tt]["complexities"].append(r.complexity)
        if r.decision.chosen_model:
            type_stats[tt]["models"].append(r.decision.chosen_model.display_name)
        # Check misclassification
        expected_for_cat = _expected.get(r.category, set())
        if expected_for_cat and tt not in expected_for_cat:
            type_stats[tt]["wrong"] += 1

    tbl = Table(box=box.ROUNDED)
    tbl.add_column("Task Type", min_width=16)
    tbl.add_column("Count", justify="right", width=7)
    tbl.add_column("Avg Complexity", justify="right", width=15)
    tbl.add_column("Top Model", min_width=22)
    tbl.add_column("Notes", min_width=14)

    for tt, d in sorted(type_stats.items(), key=lambda x: -x[1]["count"]):
        cnt = d["count"]
        avg_c = sum(d["complexities"]) / cnt if cnt else 0.0
        models = d["models"]
        top_m = max(set(models), key=models.count) if models else "—"
        wrong = d["wrong"]
        note = ""
        style = "white"

        if tt == "unknown" and cnt > len(bench.results) * 0.05:
            style = "bold red"
            note = "[bold red]HIGH[/]"
        elif wrong > 0:
            note = "[yellow]{}?[/]".format(wrong)

        tbl.add_row(
            "[{}]{}[/]".format(style, tt),
            str(cnt),
            "{:.3f}".format(avg_c),
            top_m[:28],
            note,
        )
    console.print(tbl)

    unknown_cnt = type_stats.get("unknown", {}).get("count", 0)
    unknown_pct = unknown_cnt / len(bench.results) * 100
    color = "red" if unknown_pct > 5 else "green"
    console.print(
        "  'unknown' task type: [{c}]{n} ({p:.1f}%)[/{c}]".format(
            c=color,
            n=unknown_cnt,
            p=unknown_pct,
        )
    )


def print_section_d(bench: BenchmarkResults) -> None:
    console.rule("[bold cyan]D · Cost Analysis[/bold cyan]")
    costs = [r.decision.estimated_cost for r in bench.results]
    savings = [r.decision.estimated_savings for r in bench.results]
    total_c = sum(costs)
    total_s = sum(savings)
    prem_eq = total_c + total_s
    pct_s = (total_s / prem_eq * 100) if prem_eq > 0 else 0.0
    n = len(bench.results)

    nonzero = [
        (c, r)
        for c, r in zip(costs, bench.results)  # noqa: B905 — Python 3.9 lacks strict=
        if c > 0 and r.decision.chosen_model
    ]
    cheapest = min(nonzero, key=lambda x: x[0]) if nonzero else None
    costliest = max(nonzero, key=lambda x: x[0]) if nonzero else None

    rows = [
        ("Total estimated cost", "${:.5f}".format(total_c)),
        ("Premium-equivalent cost", "${:.5f}".format(prem_eq)),
        ("Total savings", "${:.5f} ({:.1f}%)".format(total_s, pct_s)),
        ("Cost per request (avg)", "${:.7f}".format(total_c / n)),
    ]
    if cheapest:
        m = cheapest[1].decision.chosen_model
        rows.append(
            (
                "Cheapest request",
                "${:.7f}  model: {}".format(cheapest[0], m.display_name if m else "n/a"),
            )
        )
    if costliest:
        m = costliest[1].decision.chosen_model
        rows.append(
            (
                "Most expensive request",
                "${:.5f}  model: {}".format(costliest[0], m.display_name if m else "n/a"),
            )
        )

    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column("Metric", style="dim", min_width=28)
    tbl.add_column("Value", style="bold")
    for label, val in rows:
        tbl.add_row(label, val)
    console.print(tbl)


def print_section_e(bench: BenchmarkResults) -> None:
    console.rule("[bold cyan]E · Cache Performance[/bold cyan]")
    cache_hits = sum(1 for r in bench.results if r.decision.cache_hit)
    total = len(bench.results)
    # Approximate eligible count: task types known to be cacheable
    eligible = sum(
        1
        for r in bench.results
        if r.task_type
        in {"translation", "classification", "extraction", "summarization", "simple_qa"}
    )
    cache_saved = sum(r.decision.estimated_savings for r in bench.results if r.decision.cache_hit)

    rows = [
        ("Approx cache-eligible requests", str(eligible)),
        ("Cache hits", str(cache_hits)),
        ("Cache hit rate (overall)", "{:.1f}%".format(cache_hits / total * 100)),
        (
            "Cache hit rate (eligible only)",
            "{:.1f}%".format(cache_hits / eligible * 100) if eligible else "0.0%",
        ),
        ("Cost saved by cache", "${:.6f}".format(cache_saved)),
    ]
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column("Metric", style="dim", min_width=36)
    tbl.add_column("Value", style="bold")
    for label, val in rows:
        tbl.add_row(label, val)
    console.print(tbl)


def print_section_f(bench: BenchmarkResults) -> None:
    console.rule("[bold cyan]F · Edge Case Report[/bold cyan]")

    _edge_labels = [
        ("edge_empty", "Empty / near-empty prompts"),
        ("edge_emoji", "Single emoji prompts"),
        ("edge_long", "Very long prompts (10K+ chars)"),
        ("edge_code_only", "Code only (no natural language)"),
        ("edge_sql_only", "SQL statement only"),
        ("edge_cli_only", "CLI command only"),
        ("edge_url_only", "URL / curl command only"),
        ("edge_vision", "Vision requests (image capability)"),
        ("edge_function_calling", "Function calling capability"),
        ("edge_high_temp", "High temperature 1.0 (no cache)"),
        ("edge_low_temp", "Low temperature 0.0 (cacheable)"),
        ("edge_duplicate", "Duplicate prompts (cache hit test)"),
        ("edge_override", "Explicit model override"),
        ("edge_antispam", "Anti-gaming (same user flood)"),
        ("edge_budget", "Budget-limited (free plan)"),
    ]

    tbl = Table(box=box.ROUNDED)
    tbl.add_column("Edge Case", min_width=32)
    tbl.add_column("N", width=4, justify="right")
    tbl.add_column("Status", width=6)
    tbl.add_column("Models Chosen", min_width=24)
    tbl.add_column("Notes", min_width=18)

    for cat, label in _edge_labels:
        group = [r for r in bench.results if r.category == cat]
        if not group:
            continue
        errors = [r for r in group if r.error]
        status = "[bold red]FAIL[/]" if errors else "[bold green]PASS[/]"
        model_names = set()
        for r in group:
            if r.decision.chosen_model:
                model_names.add(r.decision.chosen_model.display_name[:18])
            elif r.decision.cost_blocked:
                model_names.add("BLOCKED")
        cache_hits = sum(1 for r in group if r.decision.cache_hit)

        if cat == "edge_duplicate":
            note = (
                "[green]{} cache hit(s)[/]".format(cache_hits)
                if cache_hits
                else "[red]0 cache hits[/]"
            )
        elif cat == "edge_low_temp":
            note = (
                "[green]{} cache hit(s)[/]".format(cache_hits)
                if cache_hits
                else "[yellow]0 hits yet[/]"
            )
        elif cat == "edge_high_temp":
            note = "should not cache (temp=1.0)"
        elif cat == "edge_budget":
            tiers = {r.decision.chosen_model.tier for r in group if r.decision.chosen_model}
            note = "tiers: {}".format(tiers)
        elif cat == "edge_antispam":
            tiers = [r.decision.chosen_model.tier for r in group if r.decision.chosen_model]
            note = "all tiers: {}".format(set(tiers))
        elif cat == "edge_override":
            note = (
                "override honoured"
                if all(
                    r.decision.routing_rule_matched == "explicit_model_override"
                    for r in group
                    if r.decision.chosen_model
                )
                else "[yellow]some not honoured[/]"
            )
        else:
            note = ""

        tbl.add_row(
            label,
            str(len(group)),
            status,
            ", ".join(sorted(model_names)[:2]),
            note,
        )
    console.print(tbl)


def print_section_g(inconsistencies: List[dict]) -> None:
    console.rule("[bold cyan]G · Routing Consistency Check (20 prompts × 2 runs)[/bold cyan]")
    if not inconsistencies:
        console.print("  [bold green]✓ All 20 prompts produced identical routing on both runs[/]")
    else:
        console.print(
            "  [bold red]✗ {} inconsistency/ies detected (excluding expected A/B jitter):[/]".format(
                len(inconsistencies)
            )
        )
        tbl = Table(box=box.SIMPLE)
        tbl.add_column("Prompt", min_width=40)
        tbl.add_column("Run 1 Model", min_width=20)
        tbl.add_column("Run 2 Model", min_width=20)
        for inc in inconsistencies:
            tbl.add_row(inc["prompt"], inc["run1"], inc["run2"])
        console.print(tbl)


def print_section_h(bench: BenchmarkResults) -> None:
    console.rule("[bold cyan]H · Routing Time Distribution[/bold cyan]")
    times = [r.routing_time_ms for r in bench.results]
    _buckets = [
        ("  0–1 ms ", 0, 1),
        ("  1–2 ms ", 1, 2),
        ("  2–5 ms ", 2, 5),
        ("  5–10 ms", 5, 10),
        ("  10ms+  ", 10, float("inf")),
    ]
    counts = [(lbl, sum(1 for t in times if lo <= t < hi)) for lbl, lo, hi in _buckets]
    max_cnt = max(c for _, c in counts) or 1
    for label, cnt in counts:
        bar = _bar(cnt, max_cnt, width=44)
        console.print("  {}: [cyan]{:<44}[/] {}".format(label, bar, cnt))


# ── Validation rules ──────────────────────────────────────────────────────────


def validate(bench: BenchmarkResults) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    times = [r.routing_time_ms for r in bench.results]
    n = len(bench.results)

    # 1. Any single request > 50 ms
    over_50 = [r for r in bench.results if r.routing_time_ms > 50]
    if over_50:
        failures.append(
            "{} request(s) exceeded 50ms (max {:.2f}ms)".format(len(over_50), max(times))
        )

    # 2. Average > 10 ms
    avg_ms = sum(times) / n
    if avg_ms > 10:
        failures.append("Average routing time {:.3f}ms exceeds 10ms limit".format(avg_ms))

    # 3. Any unhandled exception
    if bench.failed:
        samples = "; ".join(r.error or "" for r in bench.failed[:3])
        failures.append("{} request(s) threw exceptions: {}".format(len(bench.failed), samples))

    # 4. Cache hit rate must be > 0% (duplicates were inserted)
    cache_hits = sum(1 for r in bench.results if r.decision.cache_hit)
    if cache_hits == 0:
        failures.append("Cache hit rate is 0% — duplicate prompts should produce cache hits")

    # 5. 'unknown' task type > 5%
    unknown_cnt = sum(1 for r in bench.results if r.task_type == "unknown")
    unknown_pct = unknown_cnt / n * 100
    if unknown_pct > 5.0:
        failures.append(
            "{:.1f}% of requests classified as 'unknown' (limit: 5%)".format(unknown_pct)
        )

    # 6. Trivial requests (≤ 20 chars, non-edge) routing to premium / ultra
    trivial_bad = [
        r
        for r in bench.results
        if len(r.prompt.strip()) <= 20
        and not r.category.startswith("edge_")
        and r.decision.chosen_model is not None
        and r.decision.chosen_model.tier in ("premium", "ultra")
    ]
    if trivial_bad:
        failures.append(
            "{} trivial request(s) (≤20 chars) routed to premium/ultra tier".format(
                len(trivial_bad)
            )
        )

    # 7. Critical reasoning / long_document routed to free tier
    critical_free = [
        r
        for r in bench.results
        if r.task_type in ("reasoning", "long_document")
        and r.decision.chosen_model is not None
        and r.decision.chosen_model.tier == "free"
    ]
    if critical_free:
        failures.append(
            "{} reasoning/long_document request(s) routed to free tier".format(len(critical_free))
        )

    return len(failures) == 0, failures


# ── Entry point ───────────────────────────────────────────────────────────────


async def main() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]LLM Router Benchmark[/bold cyan]\n"
            "[dim]500 synthetic requests · full routing stack · no real API calls[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    console.print("[dim]Generating 500-request dataset (seed=42)...[/dim]")
    dataset = generate_test_dataset()
    console.print("[dim]Generated {} requests across 6 categories.[/dim]".format(len(dataset)))
    console.print()

    console.print("[dim]Running routing benchmark...[/dim]")
    bench = await run_benchmark(dataset)
    console.print(
        "[dim]Benchmark complete: {} decisions in {:.1f}ms.[/dim]".format(
            len(bench.results),
            bench.total_time_s * 1000,
        )
    )
    console.print()

    console.print("[dim]Running consistency check (20 prompts × 2 runs)...[/dim]")
    _consistency_prompts = [
        "What is 2+2?",
        "Translate 'hello' to French",
        "Write a Python function to sort a list",
        "Explain the CAP theorem in detail with examples",
        "Prove that √2 is irrational using proof by contradiction. Show all algebraic steps.",
        "What is the capital of France?",
        "Classify this review as positive or negative: 'Great product!'",
        "Summarize in one sentence: The earth is the third planet from the sun.",
        "Build a complete REST API with authentication, rate limiting, pagination, and WebSocket support in FastAPI",
        "hi",
        "Is Python compiled?",
        "Compare PostgreSQL vs MongoDB for a real-time analytics dashboard",
        "Extract all email addresses from: test@example.com, foo@bar.com",
        "Design a complete system architecture for a multi-tenant SaaS platform with row-level security, audit logging, and soft deletes",
        "What causes rain?",
        "Write a 500-word blog post about the future of renewable energy",
        "Debug this code and explain every bug you find:\n```python\ndef foo(x):\n    return x/0\n```",
        "What is photosynthesis?",
        "Analyze the pros and cons of moving the company's infrastructure to the cloud",
        "Convert 100 USD to EUR",
    ]
    inconsistencies = await run_consistency_check(_consistency_prompts)
    console.print("[dim]Consistency check complete.[/dim]")
    console.print()

    print_section_a(bench)
    console.print()
    print_section_b(bench)
    console.print()
    print_section_c(bench)
    console.print()
    print_section_d(bench)
    console.print()
    print_section_e(bench)
    console.print()
    print_section_f(bench)
    console.print()
    print_section_g(inconsistencies)
    console.print()
    print_section_h(bench)
    console.print()

    passed, failures = validate(bench)
    console.rule("[bold]Final Verdict[/bold]")
    if passed:
        console.print(
            Panel(
                "[bold bright_green]✓  PASS — All {} validation checks passed[/]".format(7),
                border_style="bright_green",
                expand=False,
            )
        )
    else:
        msg = "[bold red]✗  FAIL[/bold red]\n" + "\n".join("  • " + f for f in failures)
        console.print(Panel(msg, border_style="red", expand=False))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
