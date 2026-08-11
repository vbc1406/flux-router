# Evals — cost vs. quality

Flux saves money by routing each prompt to the cheapest model that can handle
it. The fair question is: **how much quality, if any, does that cost?** This
harness answers it with evidence instead of assertion — it runs labeled
benchmarks through several routing strategies, grades the *actual answers*, and
reports the cost-savings-vs-quality-drop tradeoff per task type. The numbers also
serve as a **North Star** for improving the classifier: change the routing logic,
re-run, and see whether quality retention went up for the same savings.

## What it measures

For every benchmark item it generates an answer under each **strategy** and
grades it:

| Strategy            | Model used                                        | Role                        |
|---------------------|---------------------------------------------------|-----------------------------|
| `premium`           | always the most expensive model                   | quality + cost **ceiling**  |
| `flux`              | whatever the router picks (the system under test) | the thing being evaluated   |
| `mid`               | a representative mid-tier model                   | a midpoint reference        |
| `cheapest`          | always the cheapest capable model                 | quality + cost **floor**    |
| `default_openai`    | `gpt-4o`                                           | OpenAI's general-purpose default |
| `default_anthropic` | `claude-sonnet-4-6`                               | Anthropic's default flagship |
| `default_google`    | `gemini-2.5-pro`                                  | Google's default flagship   |

The `default_*` strategies represent the model a company would **normally call
per provider** if it weren't routing at all — each provider's recommended
general-purpose flagship, not its most expensive option. They power the
per-question drill-down below. The pins live in `_PROVIDER_DEFAULTS`
(`router/evals/strategies.py`) — change one line to re-pin a baseline.

Headline metrics, all relative to the `premium` baseline:

- **Cost savings %** — how much cheaper the strategy was.
- **Quality retention %** — strategy mean quality ÷ premium mean quality.
- **Quality drop** — absolute mean-quality difference.

## Datasets & grading (hybrid)

Grading is objective where answers are verifiable, and an LLM judge where they
aren't:

| Dataset     | Task type        | Grader                                             |
|-------------|------------------|----------------------------------------------------|
| GSM8K       | reasoning/math   | extract final number, exact match                  |
| MMLU        | knowledge Q&A    | extract chosen letter (A–D), exact match           |
| HumanEval   | code generation  | left ungraded — no code execution, see below       |
| MT-Bench    | open-ended       | Claude-as-judge, 1–10 rubric → normalized to [0,1] |
| Agentic     | plan / tool_select / tool_result_summarize / reflect / final_answer | tool_select: exact tool-name match; the other four: Claude-as-judge |

The **Agentic** dataset (`router/evals/fixtures/agentic.json`, fixture-only —
it's a Flux-original set, not a public benchmark) is the answer to "does
routing hold up on agent *steps*, not just one-shot Q&A?" Each sample carries
an explicit `step_type` that's threaded into the `RoutingRequest`
(`runner.py`), so routing for these cases is step-type-floor-aware
(`config.STEP_TYPE_FLOORS`) exactly like a real agent loop passing
`step_type=`, not just task-type-classified like every other dataset here. It
includes a `medical` (`final_answer`) and a `legal` (`plan`) case exercising
the high-stakes domain floor (`config.DOMAIN_TIER_FLOORS`) composed with an
agent-step floor in the same request.

Two more agent steps, `budget_degradation` and `budget_stop`
(`router/evals/fixtures/agentic.json`), are **not** model-quality questions —
they're deterministic system checks (`grader="budget_ladder"`) that call
`router.run_budget.RunBudget.check_before_dispatch()` directly against a
seeded run (75%/100% of a $1.00 ceiling) and assert it returns `"degraded"` /
raises `RunBudgetExceeded`. No model call, no cost, `simulated=False` — this
is exercising real production code, not mocking it.

The `wrapper_tasks` dataset (`router/evals/fixtures/wrapper_tasks.json`,
fixture-only) covers the ordinary-request categories the public benchmarks
above don't: summarization, JSON extraction (`grader="json_schema"` — parses
the completion and checks every required key is present), translation, basic
vs. complex-distributed-systems coding, math proofs, long-document requests
(`required_capabilities=["long_document"]`, a genuinely long prompt so the
router's context-window filtering is actually exercised), wrapper-level tool
calling (reuses `agentic_tool_select`), and benign vs. high-stakes
legal/medical transformations. The benign/high-stakes split is load-bearing:
benign samples ("extract the parties from this contract") deliberately avoid
`classifier._LEGAL_SUBSTANTIVE_RE` / `_MEDICAL_SUBSTANTIVE_RE` so they keep
their normal (cheap) task_type, while high-stakes samples ("am I legally
liable...") are written to trip those regexes for real, so the *actual*
classifier — not a hand-set label — assigns `task_type="legal"/"medical"` and
the domain tier floor engages. `GradedResult.safety_escalated` records
whether the chosen model's tier actually met the floor
(`config.DOMAIN_TIER_FLOORS`) for the flux strategy on these samples; the
report's **high-stakes routing recall** metric is the fraction where it did —
a launch blocker if it's ever below 100%.

Every `GradedResult` also carries `provider`, `input_tokens`, `output_tokens`,
`latency_ms` (the completion), `routing_latency_ms` (routing decision alone,
`flux` strategy only), `tool_call_valid` / `structured_output_valid` (mirror
`correct` for the matching grader), and `domain`. `report.quality_metrics()`
rolls these up into tool-call success rate, structured-output validity rate,
high-stakes routing recall, and p50/p95 routing + end-to-end latency, plus a
`launch_blockers` list — all printed in the console report and written into
the JSON snapshot under `"quality_metrics"`. A separate, one-shot
**fallback-recovery system check** (`runner._check_fallback_recovery`)
exercises the real `FallbackExecutor` end to end with a fake failing primary
call (no network) and reports `system_checks.fallback_recovery_ok`.

## Modes

- **mock (default)** — deterministic *simulated* completions whose correctness is
  drawn from each model's published `quality_ratings` (seeded, reproducible).
  Runs fully offline with the bundled fixtures — no network, no API keys, no
  spend. This is what CI runs. **Mock numbers are illustrative, not real model
  quality** (the report and JSON label them `SIMULATED`).
- **live (`--live`)** — real provider calls (needs the relevant `*_API_KEY` env
  vars) against the full datasets pulled from HuggingFace. Completions and judge
  verdicts are disk-cached so re-runs don't re-pay. This produces the real,
  shareable numbers.

## Running it

```bash
# Offline, simulated, CI-safe (bundled fixtures):
make evals
python -m router.evals

# A subset / larger sample:
python -m router.evals --datasets gsm8k,mmlu --n 100

# Real numbers (needs API keys + the optional 'datasets' extra; costs money):
pip install 'flux-router[evals]'
export ANTHROPIC_API_KEY=...   # plus keys for every provider the strategies touch
python -m router.evals --live --n 30
```

Each run writes a JSON snapshot to `eval_results/<timestamp>.json` for
tracking over time; add `--md PATH` to also emit a markdown table.

> **A note on HumanEval:** Flux provides no sandbox for executing
> model-generated code, so HumanEval samples are graded `(None, None)` —
> skipped, not scored — in every mode. `--allow-code-exec` is not a supported
> flag; passing it exits immediately with an explanation. Run HumanEval
> through a purpose-built isolated code runner outside this harness if you
> need pass@1 numbers.

## Example (mock / SIMULATED, bundled fixtures)

```
| Strategy | N  | Total Cost | Mean Quality | Cost Savings | Quality Retained | Quality Drop |
| premium  | 20 | $0.07105   | 0.879        | —            | —                | —            |
| flux     | 20 | $0.00205   | 0.810        | 97.1%        | 92.1%            | +0.069       |
| mid      | 20 | $0.00190   | 0.748        | 97.3%        | 85.1%            | +0.131       |
| cheapest | 20 | $0.00000   | 0.641        | 100.0%       | 72.9%            | +0.238       |
```

Replace this table with a real `--live` run before citing it anywhere.

## Per-question drill-down (`--per-question`)

The aggregate table answers "how much does flux save overall?" The per-question
view answers "*on this specific question*, what did flux pick, and how does it
compare to just calling ChatGPT / Claude / Gemini?"

```bash
python -m router.evals --per-question
python -m router.evals --per-question --datasets gsm8k,mmlu --n 20 --md results.md
```

With `--per-question` (and no explicit `--strategies`) the run compares
`flux` against the three provider defaults. For every question it prints:

- the **question type** (dataset + MMLU subject / MT-Bench category + task type)
  and the classifier's **complexity** band (low / med / high),
- the **model flux routed to**, with its rated quality and modeled cost,
- the same for each provider default, so you can read flux vs ChatGPT vs Claude
  vs Gemini side by side,

followed by a mean-quality/cost rollup per question type and a headline of flux's
cost savings + quality retention against each provider default. The full
per-question payload (rows + rollup) is also written into the JSON snapshot, and
appended to the `--md` markdown table.

For samples that carry a `step_type` (currently only the Agentic dataset), a
second rollup — **mean quality / cost by agent step type** — prints
immediately below the question-type one, and the payload's `by_step_type` key
carries the same data. This is where `ModelOption.step_quality_ratings`
(when populated) would show up as a quality delta on a specific step, as
opposed to the question-type rollup which reflects the flat task-type
rating. Samples with no step concept (`step_type=""`, i.e. everything outside
Agentic) are excluded rather than lumped under `"unknown"`.

> Quality here is the same **graded** value the aggregate table uses
> (`GradedResult.quality`) — a *simulated* grade in mock mode, a *measured*
> grade of a real completion under `--live`. The payload's top-level `"mode"`
> field says which (`"simulated"` or `"measured"`), and the console/markdown
> output prints an explicit SIMULATED/MEASURED banner — so `--per-question
> --live` gives you real, per-question numbers, not the mock ones with a live
> label slapped on. (Earlier versions of this drill-down always showed each
> model's static catalog `quality_ratings` entry here regardless of mode —
> that bug is fixed; `GradedResult.quality_rating` is still on the data
> model for other uses, just no longer what the per-question view plots.)

## Extending

- **Add a dataset:** add a `_build_<name>()` normalizer + a hub loader branch in
  `router/evals/datasets.py`, a fixture in `router/evals/fixtures/`, and (if
  needed) a grader.
- **Add a grader:** implement it in `router/evals/graders/` and dispatch on
  `sample.grader` in `graders/__init__.py`.
