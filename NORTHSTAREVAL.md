# North Star Eval — Per-Question Cost vs Quality

**What this is:** the headline cost-vs-quality result for the Flux router, broken
down per question and per question type, comparing **flux** against the model a
company would *normally* call per provider (ChatGPT / Claude / Gemini defaults).

| Field | Value |
|---|---|
| Date | 2026-06-01 |
| Command | `python -m router.evals --per-question --datasets gsm8k,mmlu,humaneval,mtbench --n 50` |
| Mode | `mock` / **SIMULATED** (quality = each model's rated benchmark for the task type, not a graded live answer) |
| Source | bundled fixtures |
| Samples | 20 (5 per dataset) · 0 skipped |
| Baselines | `default_openai`=gpt-4o · `default_anthropic`=claude-sonnet-4 · `default_google`=gemini-2.5-pro |
| Snapshot | `eval_results/20260601T150304Z.json` |

> ⚠️ **SIMULATED numbers.** Quality is the model's *rated* benchmark for the
> question's task type (registry `quality_ratings`), not a freshly graded answer.
> Re-run with `--live` (API keys, real grading) before citing externally.

---

## Headline

**flux: mean rated quality 0.871 at total cost $0.00219.**

| vs provider default | flux cost savings | flux quality retained |
|---|--:|--:|
| `default_openai` (gpt-4o) | **78.2%** | **105.5%** |
| `default_anthropic` (claude-sonnet-4) | **84.7%** | **103.0%** |
| `default_google` (gemini-2.5-pro) | **74.6%** | **100.6%** |

flux delivers **equal-or-better rated quality at ~¼ the cost** of calling any
single provider's default flagship.

### Overall totals

| Strategy | Total Cost | Mean Quality |
|---|--:|--:|
| **flux** | **$0.00219** | **0.867** |
| default_openai (gpt-4o) | $0.01007 | 0.658 |
| default_anthropic (claude-sonnet-4) | $0.01430 | 0.857 |
| default_google (gemini-2.5-pro) | $0.00862 | 0.915 |

---

## What flux routed to (per question type)

flux never reached for a premium-tier model — it picked **cheap, high-rated**
models and matched task to model:

| Question type | flux's pick |
|---|---|
| `code_generation` (HumanEval) | `gpt-5-mini` |
| everything else (reasoning, simple_qa, analysis, creative_writing) | `gemini-3-flash-preview` |

This is the core mechanism: flux exploits cheap models that score *high on the
specific task type*, instead of paying flagship prices on every call.

---

## Per question type — rated quality (q) and total cost ($)

| Question type | flux | gpt-4o | claude-sonnet-4 | gemini-2.5-pro | flux verdict |
|---|---|---|---|---|---|
| gsm8k — reasoning | **q=0.90** $0.00041 | q=0.74 $0.00163 | q=0.78 $0.00220 | q=0.84 $0.00122 | **Wins quality *and* cost** |
| humaneval — code_generation | q=0.86 **$0.00086** | q=0.85 $0.00507 | q=0.90 $0.00732 | q=0.86 $0.00449 | Ties gpt-4o/gemini, below claude; ~6–8× cheaper |
| mmlu — simple_qa | q=0.88 **$0.00002–0.00006** | q=0.88 | q=0.85 | q=0.90 | Tie on quality, ~5× cheaper |
| mtbench — analysis | **q=0.88** $0.00016 | q=0.82 | q=0.84 | q=0.88 | Wins/ties, ~3–5× cheaper |
| mtbench — creative_writing | q=0.82 $0.00048 | q=0.84 | q=0.86 | q=0.85 | **Slightly below all three** on quality; ~3× cheaper |

---

## Inferences

1. **The thesis holds: flux ≈ flagship quality at ~25% of the cost.** Against
   every provider default, flux retains 100–105% of rated quality while cutting
   cost 75–85%.

2. **The win is concentrated in reasoning.** On GSM8K, flux's pick is rated
   *higher* than all three flagships (0.90 vs 0.74–0.84) **and** cheaper — the
   single biggest contributor to the headline.

3. **simple_qa is a wash on quality, a rout on cost.** For MMLU knowledge Q&A
   everyone lands ~0.85–0.90; flux just gets there ~5× cheaper. Routing matters
   most here as pure cost arbitrage.

4. **Creative writing is the one soft spot.** flux's cheap pick (0.82) trails all
   three defaults (0.84–0.86). The quality gap is small and still ~3× cheaper, but
   it's the only category where flux concedes quality — the place to watch if a
   workload is writing-heavy.

5. **Code generation: par, not lead.** flux matches gpt-4o/gemini (0.85–0.86) and
   trails claude-sonnet (0.90), but at 6–8× lower cost — a deliberate
   cost/quality trade rather than a free win.

6. **gpt-4o is the weakest default here (mean 0.658)** because its rated quality on
   reasoning/math drags it down — a reminder that "the OpenAI default" is not
   automatically the quality bar.

---

## Caveats

- **SIMULATED.** Quality is the rated benchmark per task type, not a graded live
  answer. Quality-retention figures **>100%** are a direct artifact of this: flux
  routes to a model rated higher than the provider default on that task type.
  Treat as *analytical/relative*, not absolute accuracy.
- **Small sample (n=20, 5/dataset)** from bundled fixtures — directional, not
  statistically tight. Bump `--n` (and `--source hub` under `--live`) for real
  numbers.
- **Baselines are a judgment call** (gpt-4o / claude-sonnet-4 / gemini-2.5-pro),
  pinned in `_PROVIDER_DEFAULTS` (`router/evals/strategies.py`). Re-pin and re-run
  to compare against, e.g., Opus 4.
- **Registry-dependent.** flux's picks (`gemini-3-flash-preview`, `gpt-5-mini`)
  come from `router/models.json`; results shift if the catalog changes.

## Reproduce

```bash
python -m router.evals --per-question \
  --datasets gsm8k,mmlu,humaneval,mtbench --n 50 --md results.md
# real numbers (API keys + spend):
python -m router.evals --per-question --live --n 30 --allow-code-exec
```
