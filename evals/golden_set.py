"""
golden_set.py — the 25-task golden set, spanning all four step types.

Each GoldenTask carries a `step_type` (drives STEP_TYPE_FLOORS + step_type
inference in the router) and a `task_type` (matches a key in models.json's
quality_ratings, e.g. "reasoning" / "summarization" / "function_calling" — used
to look up how good a candidate model is expected to be at this kind of task).

grader is either:
  "exact"  — compare the model's answer to `expected` (case/whitespace/number
             insensitive). Used for tool_select (which tool name did it pick?)
             and a few deterministic final_answer questions.
  "rubric" — no single right answer; `rubric` is what the judge is told to
             score against, 1-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenTask:
    id: str
    step_type: str  # plan | tool_select | tool_result_summarize | final_answer
    task_type: str  # router quality_ratings key
    prompt: str
    grader: str  # "exact" | "rubric"
    expected: str | None = None
    rubric: str | None = None
    tools: list[dict] = field(default_factory=list)


def _tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


_WEATHER = _tool("get_weather", "Get the current weather for a location")
_STOCK = _tool("get_stock_price", "Get the latest price for a stock ticker")
_EMAIL = _tool("send_email", "Send an email to a recipient")
_SEARCH = _tool("search_web", "Search the web for current information")
_CALENDAR = _tool("create_calendar_event", "Create an event on the user's calendar")
_SQL = _tool("run_sql_query", "Run a read-only SQL query against the app database")

# ── PLAN (6) — rubric-graded; there's no single "correct" plan ──────────────
_PLAN_TASKS = [
    GoldenTask(
        id="plan-01",
        step_type="plan",
        task_type="reasoning",
        prompt=(
            "Plan the steps to migrate a PostgreSQL database from AWS RDS to "
            "Google Cloud SQL with zero downtime."
        ),
        grader="rubric",
        rubric=(
            "A strong plan covers: setting up logical replication or a CDC "
            "pipeline, cutover/rollback strategy, and validating data "
            "integrity before decommissioning the old database. Penalize "
            "plans that skip cutover/rollback or assume downtime is fine."
        ),
    ),
    GoldenTask(
        id="plan-02",
        step_type="plan",
        task_type="reasoning",
        prompt="Create a step-by-step plan to onboard a new backend engineer in their first two weeks.",
        grader="rubric",
        rubric=(
            "A strong plan sequences access/setup, codebase orientation, a "
            "small guided first task, and a check-in cadence. Penalize plans "
            "that are just a flat list with no sequencing or milestones."
        ),
    ),
    GoldenTask(
        id="plan-03",
        step_type="plan",
        task_type="reasoning",
        prompt="Plan how to debug a production API that intermittently returns 500 errors under load.",
        grader="rubric",
        rubric=(
            "A strong plan starts with observability (logs/metrics/traces), "
            "forms hypotheses (resource exhaustion, race condition, "
            "downstream dependency), and only then proposes fixes. Penalize "
            "plans that jump straight to a fix without first localizing the "
            "cause."
        ),
    ),
    GoldenTask(
        id="plan-04",
        step_type="plan",
        task_type="reasoning",
        prompt="Outline a plan to migrate a monolithic Django app to a set of microservices.",
        grader="rubric",
        rubric=(
            "A strong plan identifies bounded contexts before splitting, "
            "proposes an incremental strangler-fig approach, and addresses "
            "shared-database risk. Penalize plans that propose a single big-"
            "bang rewrite."
        ),
    ),
    GoldenTask(
        id="plan-05",
        step_type="plan",
        task_type="reasoning",
        prompt=(
            "Plan the rollout of a feature flag system across a 20-engineer "
            "team, including rollback strategy."
        ),
        grader="rubric",
        rubric=(
            "A strong plan covers a staged rollout (internal -> % of users -> "
            "100%), monitoring during rollout, and an explicit fast rollback "
            "path. Penalize plans that omit a rollback mechanism entirely."
        ),
    ),
    GoldenTask(
        id="plan-06",
        step_type="plan",
        task_type="reasoning",
        prompt="Create a plan to reduce a web app's p95 latency from 800ms to under 200ms.",
        grader="rubric",
        rubric=(
            "A strong plan starts with profiling to find the actual "
            "bottleneck (DB, network, rendering) before proposing fixes, and "
            "sets intermediate milestones. Penalize plans that list generic "
            "performance tips without a measurement step first."
        ),
    ),
]

# ── TOOL_SELECT (6) — exact-match on the tool name the model should call ────
_TOOL_SELECT_TASKS = [
    GoldenTask(
        id="tool-01",
        step_type="tool_select",
        task_type="function_calling",
        prompt="What's the weather like in Tokyo right now?",
        grader="exact",
        expected="get_weather",
        tools=[_WEATHER, _STOCK, _EMAIL, _SEARCH],
    ),
    GoldenTask(
        id="tool-02",
        step_type="tool_select",
        task_type="function_calling",
        prompt="Schedule a meeting with the team for 3pm tomorrow.",
        grader="exact",
        expected="create_calendar_event",
        tools=[_WEATHER, _CALENDAR, _SEARCH, _EMAIL],
    ),
    GoldenTask(
        id="tool-03",
        step_type="tool_select",
        task_type="function_calling",
        prompt="How is Apple's stock doing today?",
        grader="exact",
        expected="get_stock_price",
        tools=[_SEARCH, _STOCK, _EMAIL, _SQL],
    ),
    GoldenTask(
        id="tool-04",
        step_type="tool_select",
        task_type="function_calling",
        prompt="Find out how many users signed up last week from the database.",
        grader="exact",
        expected="run_sql_query",
        tools=[_SEARCH, _SQL, _EMAIL, _WEATHER],
    ),
    GoldenTask(
        id="tool-05",
        step_type="tool_select",
        task_type="function_calling",
        prompt="Look up the latest news about the Federal Reserve interest rate decision.",
        grader="exact",
        expected="search_web",
        tools=[_EMAIL, _SEARCH, _CALENDAR, _WEATHER],
    ),
    GoldenTask(
        id="tool-06",
        step_type="tool_select",
        task_type="function_calling",
        prompt="Email Priya a summary of today's standup notes.",
        grader="exact",
        expected="send_email",
        tools=[_EMAIL, _SEARCH, _CALENDAR, _WEATHER],
    ),
]

# ── TOOL_RESULT_SUMMARIZE (6) — rubric-graded ────────────────────────────────
_TOOL_RESULT_SUMMARIZE_TASKS = [
    GoldenTask(
        id="tsum-01",
        step_type="tool_result_summarize",
        task_type="summarization",
        prompt=(
            "Summarize this weather API response for the user in one "
            'sentence: {"location": "Tokyo", "temp_c": 29, "condition": '
            '"humid, scattered thunderstorms", "wind_kph": 14}'
        ),
        grader="rubric",
        rubric=(
            "A strong summary is one natural sentence covering temperature, "
            "condition, and is user-facing (not a JSON dump). Penalize "
            "answers that just restate the raw JSON."
        ),
    ),
    GoldenTask(
        id="tsum-02",
        step_type="tool_result_summarize",
        task_type="summarization",
        prompt=(
            "Summarize these SQL query results (250 rows of signup data, "
            "columns: signup_date, source, converted) into 3 key takeaways. "
            "Sample: 2026-07-24,organic,true; 2026-07-24,paid_ad,false; "
            "2026-07-25,referral,true; 2026-07-25,paid_ad,false; "
            "2026-07-26,organic,true — repeated with organic/referral "
            "converting ~2x more often than paid_ad across the 250 rows."
        ),
        grader="rubric",
        rubric=(
            "A strong summary surfaces the organic/referral vs. paid_ad "
            "conversion gap as the headline insight, in 3 bullet-style "
            "takeaways. Penalize answers that just describe the schema "
            "without stating the conversion-gap finding."
        ),
    ),
    GoldenTask(
        id="tsum-03",
        step_type="tool_result_summarize",
        task_type="summarization",
        prompt=(
            "Summarize this stock price history tool output into a short "
            "investor-friendly note: AAPL closed at $228.40 today, up 2.3% "
            "on strong iPhone demand guidance, after trading in a $221-$229 "
            "range over the past week."
        ),
        grader="rubric",
        rubric=(
            "A strong note leads with the move (+2.3%, close price) and the "
            "stated driver (iPhone demand guidance), in investor-friendly "
            "prose. Penalize answers that omit the reason for the move."
        ),
    ),
    GoldenTask(
        id="tsum-04",
        step_type="tool_result_summarize",
        task_type="summarization",
        prompt=(
            "Summarize this list of 15 search results about 'renewable "
            "energy tax credits' into a 4-bullet answer: results cover the "
            "US federal Investment Tax Credit (30% through 2032, stepping "
            "down after), state-level rebate stacking, the residential vs. "
            "commercial eligibility split, and recent proposals to extend "
            "the credit timeline."
        ),
        grader="rubric",
        rubric=(
            "A strong answer is exactly a short bulleted list covering the "
            "federal ITC rate/step-down, state stacking, the "
            "residential/commercial split, and the extension proposals. "
            "Penalize answers that ramble in prose instead of bulleting the "
            "distinct points."
        ),
    ),
    GoldenTask(
        id="tsum-05",
        step_type="tool_result_summarize",
        task_type="summarization",
        prompt=(
            "Summarize this calendar tool output (8 upcoming meetings today) "
            "into a short daily digest: 9am standup, 10am 1:1 with manager, "
            "11:30am design review (2hrs), 2pm customer call, 3pm interview "
            "panel, 4pm no-meeting focus block, 5pm team sync, 6pm optional "
            "social."
        ),
        grader="rubric",
        rubric=(
            "A strong digest is concise, groups/orders the meetings "
            "chronologically, and flags the notable ones (the 2hr design "
            "review, the customer call). Penalize answers that just repeat "
            "every meeting with no synthesis (e.g. no flagging the 2hr block "
            "or the focus time)."
        ),
    ),
    GoldenTask(
        id="tsum-06",
        step_type="tool_result_summarize",
        task_type="summarization",
        prompt=(
            "Summarize this log-search tool output (40 error log lines) into "
            "a root-cause hypothesis in 2-3 sentences: all 40 lines are "
            "'connection pool exhausted' errors from the checkout-service, "
            "clustered in a 3-minute window right after a deploy, with pool "
            "size unchanged in that deploy but a new synchronous call to the "
            "inventory-service added to the checkout path."
        ),
        grader="rubric",
        rubric=(
            "A strong hypothesis connects the new synchronous "
            "inventory-service call to pool exhaustion (calls holding "
            "connections longer) rather than treating the pool-exhaustion "
            "errors and the deploy as coincidental. Penalize answers that "
            "just restate 'connection pool exhausted' without proposing why."
        ),
    ),
]

# ── FINAL_ANSWER (7) — 4 exact, 3 rubric ─────────────────────────────────────
_FINAL_ANSWER_TASKS = [
    GoldenTask(
        id="fa-01",
        step_type="final_answer",
        task_type="reasoning",
        prompt="What is 17 * 24? Answer with only the number.",
        grader="exact",
        expected="408",
    ),
    GoldenTask(
        id="fa-02",
        step_type="final_answer",
        task_type="simple_qa",
        prompt="What is the capital of Australia? Answer with only the city name.",
        grader="exact",
        expected="Canberra",
    ),
    GoldenTask(
        id="fa-03",
        step_type="final_answer",
        task_type="reasoning",
        prompt="Convert 100 Fahrenheit to Celsius, rounded to the nearest whole number. Answer with only the number.",
        grader="exact",
        expected="38",
    ),
    GoldenTask(
        id="fa-04",
        step_type="final_answer",
        task_type="simple_qa",
        prompt="Explain why the sky appears blue, in terms a 10-year-old would understand.",
        grader="rubric",
        rubric=(
            "A strong answer mentions sunlight scattering off air molecules "
            "and that blue light scatters more, in simple non-technical "
            "language. Penalize answers using jargon like 'Rayleigh "
            "scattering' without explaining it, or answers that are just "
            "wrong (e.g. 'the sky reflects the ocean')."
        ),
    ),
    GoldenTask(
        id="fa-05",
        step_type="final_answer",
        task_type="code_generation",
        prompt="Write a Python function that returns the nth Fibonacci number, iteratively (not recursive).",
        grader="rubric",
        rubric=(
            "A strong answer is a correct, iterative (loop-based, not "
            "recursive) Fibonacci function with sane handling of n=0/n=1. "
            "Penalize recursive solutions (they don't satisfy 'iteratively') "
            "or solutions with an off-by-one error."
        ),
    ),
    GoldenTask(
        id="fa-06",
        step_type="final_answer",
        task_type="analysis",
        prompt=(
            "Given our Q3 numbers (revenue $2.1M, up 15% QoQ; churn 4.2%, up "
            "from 3.1%), write a 3-sentence executive summary."
        ),
        grader="rubric",
        rubric=(
            "A strong summary is exactly about 3 sentences and surfaces the "
            "tension: revenue growth is good, but churn also rose "
            "meaningfully and should be called out as a risk, not buried. "
            "Penalize answers that only mention the revenue growth and omit "
            "or downplay the churn increase."
        ),
    ),
    GoldenTask(
        id="fa-07",
        step_type="final_answer",
        task_type="simple_qa",
        prompt="What is the chemical symbol for gold? Answer with only the symbol.",
        grader="exact",
        expected="Au",
    ),
]

GOLDEN_SET: list[GoldenTask] = (
    _PLAN_TASKS + _TOOL_SELECT_TASKS + _TOOL_RESULT_SUMMARIZE_TASKS + _FINAL_ANSWER_TASKS
)

assert len(GOLDEN_SET) == 25, f"expected 25 golden tasks, got {len(GOLDEN_SET)}"
assert len({t.id for t in GOLDEN_SET}) == 25, "duplicate task id in GOLDEN_SET"
