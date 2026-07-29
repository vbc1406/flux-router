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
| `default_anthropic` | `claude-sonnet-4`                                 | Anthropic's default flagship |
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
| HumanEval   | code generation  | execute against unit tests in a subprocess (pass@1)|
| MT-Bench    | open-ended       | Claude-as-judge, 1–10 rubric → normalized to [0,1] |

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
python -m router.evals --live --n 30 --allow-code-exec
```

`--allow-code-exec` is required to run live model-generated code for HumanEval
(simulated code in mock mode runs without it). Each run writes a JSON snapshot to
`eval_results/<timestamp>.json` for tracking over time; add `--md PATH` to also
emit a markdown table.

> **A note on HumanEval:** grading executes code. In mock mode the code is the
> bundled canonical/broken solutions (trusted). In live mode it is real model
> output, which only runs when you pass `--allow-code-exec`, in an isolated
> subprocess with a timeout.

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

> Quality here is each model's **rated benchmark** for that question's task type
> (from the registry `quality_ratings` tables), not a freshly graded live answer
> — i.e. the drill-down is a mock/analytical comparison, consistent with the
> default mock mode.

## Extending

- **Add a dataset:** add a `_build_<name>()` normalizer + a hub loader branch in
  `router/evals/datasets.py`, a fixture in `router/evals/fixtures/`, and (if
  needed) a grader.
- **Add a grader:** implement it in `router/evals/graders/` and dispatch on
  `sample.grader` in `graders/__init__.py`.
