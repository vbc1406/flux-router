# North Star Eval — Per-Question Cost vs Quality

**What this is:** the headline cost-vs-quality result for the Flux router, broken
down per question and per question type, comparing **flux** against the model a
company would *normally* call per provider (ChatGPT / Claude / Gemini defaults).

| Field | Value |
|---|---|
| Date | 2026-08-08 |
| Command | `python -m router.evals --per-question --datasets gsm8k,mmlu,humaneval,mtbench --n 50` |
| Mode | `mock` / **SIMULATED** (quality = each model's rated benchmark for the task type, not a graded live answer) |
| Source | bundled fixtures |
| Samples | 20 (5 per dataset) · 0 skipped |
| Baselines | `default_openai`=gpt-4o · `default_anthropic`=claude-sonnet-4-6 · `default_google`=gemini-2.5-pro |
| Snapshot | `eval_results/20260808T144905Z.json` |

> ⚠️ **SIMULATED numbers.** Quality is the model's *rated* benchmark for the
> question's task type (registry `quality_ratings`), not a freshly graded answer.
> Re-run with `--live` (API keys, real grading) before citing externally.
>
> ⚠️ **Re-run note (2026-08-08):** this snapshot supersedes the 2026-06-01 one.
> `router/models.json` was cleaned up between the two runs (8 dead/retired/fake
> models removed, `gpt-oss-20b` added, 38→30 models), which shifted flux's picks
> for `code_generation` and `creative_writing` below — a direct illustration of
> the "registry-dependent" caveat this doc already carried. See
> `router/tests/test_routing.py` and the routing-defect writeups in git history
> for how these picks are kept honest against real benchmark data.

---

## Headline

**flux: mean rated quality 0.880 at total cost $0.00153.**

| vs provider default | flux cost savings | flux quality retained |
|---|--:|--:|
| `default_openai` (gpt-4o) | **84.8%** | **106.6%** |
| `default_anthropic` (claude-sonnet-4-6) | **89.3%** | **98.8%** |
| `default_google` (gemini-2.5-pro) | **82.2%** | **101.7%** |

In this simulated snapshot, Flux retained **98.8–106.6% of catalog-rated
quality** while costing **10.7–17.8% of the compared provider defaults**. These
are registry-derived estimates, not live answer-quality measurements.

### Overall totals

| Strategy | Total Cost | Mean Quality |
|---|--:|--:|
| **flux** | **$0.00153** | **0.880** |
| default_openai (gpt-4o) | $0.01007 | 0.658 |
| default_anthropic (claude-sonnet-4-6) | $0.01430 | 0.891 |
| default_google (gemini-2.5-pro) | $0.00862 | 0.865 |

---

## What flux routed to (per question type)

| Question type | flux's pick |
|---|---|
| `code_generation` (HumanEval) | `gpt-oss-120b` |
| `reasoning`, `simple_qa`, `analysis` | `gemini-3-flash-preview` |
| `creative_writing` (mtbench/writing) | split: `gemini-3-flash-preview` / `gpt-5.6-luna` |

This is the core mechanism: flux exploits cheap models that score *high on the
specific task type*, instead of paying flagship prices on every call.

---

## Per question type — rated quality (q) and total cost ($)

| Question type | flux | gpt-4o | claude-sonnet-4-6 | gemini-2.5-pro | flux verdict |
|---|---|---|---|---|---|
| gsm8k — reasoning | **q=0.90** $0.00041 | q=0.74 $0.00163 | q=0.84 $0.00220 | q=0.84 $0.00122 | **Wins quality *and* cost** |
| humaneval — code_generation | q=0.90 **$0.00030** | q=0.85 $0.00507 | q=0.95 $0.00732 | q=0.86 $0.00449 | Beats gpt-4o/gemini, below claude; ~15–24× cheaper |
| mmlu — simple_qa | q=0.88 **$0.00013** | q=0.88 $0.00059 | q=0.88 $0.00073 | q=0.90 $0.00032 | Ties gpt-4o/claude, ~5× cheaper |
| mtbench — analysis | **q=0.88** $0.00032 | q=0.82 $0.00112 | q=0.88 $0.00163 | q=0.88 $0.00104 | Ties/wins, ~3–5× cheaper |
| mtbench — creative_writing | q=0.81 $0.00038 | q=0.84 $0.00166 | q=0.90 $0.00243 | q=0.85 $0.00155 | **Below all three** on quality; ~4–6× cheaper |

---

## Inferences

1. **The thesis holds: flux ≈ or exceeds flagship quality at ~⅕–⅙ the cost.**
   Against every provider default, flux retains 99–107% of rated quality while
   cutting cost 82–89%.

2. **The win is concentrated in reasoning.** On GSM8K, flux's pick is rated
   *higher* than all three flagships (0.90 vs 0.74–0.84) **and** cheaper — the
   single biggest contributor to the headline.

3. **Code generation flipped from a "par" result to a clear win** since the last
   snapshot: flux now routes to `gpt-oss-120b` (q=0.90) instead of `gpt-5-mini`,
   beating both gpt-4o and gemini-2.5-pro and trailing only claude-sonnet-4-6,
   at roughly a fifth of gpt-4o's cost and a twenty-fourth of claude's.

4. **simple_qa is a wash on quality, a rout on cost.** For MMLU knowledge Q&A
   everyone lands ~0.88–0.90; flux just gets there ~5× cheaper. Routing matters
   most here as pure cost arbitrage.

5. **Creative writing is still the one soft spot**, and it's the category that
   moved most: flux's picks (split between `gemini-3-flash-preview` and the newer
   `gpt-5.6-luna`) average q=0.81, trailing all three defaults (0.84–0.90). Same
   conclusion as before — the place to watch if a workload is writing-heavy —
   but the gap widened slightly against claude-sonnet-4-6 specifically.

6. **gpt-4o is still the weakest default here (mean 0.658)** because its rated
   quality on reasoning/math drags it down — a reminder that "the OpenAI default"
   is not automatically the quality bar.

---

## Caveats

- **SIMULATED.** Quality is the rated benchmark per task type, not a graded live
  answer. Quality-retention figures **>100%** are a direct artifact of this: flux
  routes to a model rated higher than the provider default on that task type.
  Treat as *analytical/relative*, not absolute accuracy.
- **Small sample (n=20, 5/dataset)** from bundled fixtures — directional, not
  statistically tight. Bump `--n` (and `--source hub` under `--live`) for real
  numbers.
- **Baselines are a judgment call** (gpt-4o / claude-sonnet-4-6 / gemini-2.5-pro),
  pinned in `_PROVIDER_DEFAULTS` (`router/evals/strategies.py`). Re-pin and re-run
  to compare against, e.g., Opus.
- **Registry-dependent — confirmed, not just theoretical.** flux's picks
  (`gemini-3-flash-preview`, `gpt-oss-120b`, `gpt-5.6-luna`) come from
  `router/models.json`, and this run demonstrably shifted from the prior
  snapshot's picks (`gemini-3-flash-preview`, `gpt-5-mini`) after an unrelated
  catalog cleanup changed the relative cost/quality normalization. Re-run this
  eval after any `models.json` change before citing headline numbers externally
  — don't assume the last snapshot still holds. See `benchmark_verification_report.md`
  (gitignored, kept locally for reference) for the deeper investigation behind
  this caveat.
- **Rated ≠ independently verified.** This eval checks flux against its own
  `quality_ratings` table, so a "win" here proves flux optimizes against its own
  beliefs, not that those beliefs are correct. An independent benchmark-sourced
  audit (GPQA/SWE-bench/etc., not this repo's own ratings) is a separate,
  periodic exercise — do that before making external quality claims, not this
  mock eval alone.

## Reproduce

```bash
python -m router.evals --per-question \
  --datasets gsm8k,mmlu,humaneval,mtbench --n 50 --md results.md
# real numbers (API keys + spend):
python -m router.evals --per-question --live --n 30 --allow-code-exec
```
