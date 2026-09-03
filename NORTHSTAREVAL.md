# North Star Eval — Per-Question Cost vs Quality

**What this is:** the headline cost-vs-quality result for the Flux router, broken
down per question and per question type, comparing **flux** against the model a
company would *normally* call per provider (ChatGPT / Claude / Gemini defaults).

| Field | Value |
|---|---|
| Date | 2026-09-03 |
| Command | `python -m router.evals --per-question --datasets gsm8k,mmlu,humaneval,mtbench --n 50 --seed 42` |
| Mode | `mock` / **SIMULATED** (objective datasets are graded exact-match on the mock completion; MT-Bench is graded via the mock LLM-judge path — see `EVALS.md`) |
| Source | bundled fixtures |
| Samples | 20 (5 per dataset) · humaneval left ungraded (no code execution, 5 skipped) |
| Baselines | `default_openai`=gpt-4o · `default_anthropic`=claude-sonnet-4-6 · `default_google`=gemini-2.5-pro |
| Snapshot | `eval_results/20260903T162823Z.json` |

> ⚠️ **SIMULATED numbers.** Not a freshly graded live answer. Re-run with
> `--live` (API keys, real grading) before citing externally.
>
> ⚠️ **Re-run note (2026-09-03):** this snapshot supersedes the 2026-08-08 one.
> `router/models.json` was refreshed against providers' current model/pricing
> docs (Claude Sonnet 5's introductory price became permanent at $2/$10 —
> corrected down from a defensive $3/$15; `claude-fable-5` → `claude-fable-5-1`;
> `gemini-3.6-flash` → `gemini-3.8-flash`; GPT-5.6 Sol corrected to $4/$20; see
> `MODEL_CATALOG_AUDIT.md`). This run is pinned with `--seed 42` for
> reproducibility going forward — the previous snapshot's run wasn't seeded, so
> its numbers aren't directly comparable run-over-run, only directionally.

---

## Headline

**flux: mean quality 0.951 at total cost $0.00157.**

| vs provider default | flux cost savings | flux quality retained |
|---|--:|--:|
| `default_openai` (gpt-4o) | **68.7%** | **117.0%** |
| `default_anthropic` (claude-sonnet-4-6) | **77.6%** | **114.6%** |
| `default_google` (gemini-2.5-pro) | **62.1%** | **100.2%** |

In this simulated snapshot, Flux retained **100.2–117.0% of quality** while
costing **22.4–37.9% of the compared provider defaults**. These are
registry-derived estimates, not live answer-quality measurements.

### Overall totals

| Strategy | Total Cost | Mean Quality |
|---|--:|--:|
| **flux** | **$0.00157** | **0.951** |
| default_openai (gpt-4o) | $0.00501 | 0.813 |
| default_anthropic (claude-sonnet-4-6) | $0.00698 | 0.830 |
| default_google (gemini-2.5-pro) | $0.00413 | 0.949 |

---

## What flux routed to (per question type)

| Question type | flux's pick |
|---|---|
| `code_generation` (HumanEval) | left ungraded (no code execution in mock mode) |
| `reasoning` (GSM8K) | `o4-mini` |
| `simple_qa` (MMLU) | `gemini-3-flash-preview` (3 of 4), `o4-mini` (1 high-complexity item) |
| `analysis` (mtbench) | `gemini-3-flash-preview` |
| `creative_writing` (mtbench/writing) | split: `gemini-3-flash-preview` / `gpt-5.6-luna` |

This is the core mechanism: flux exploits cheap models that score *high on the
specific task type*, instead of paying flagship prices on every call.

---

## Per question type — quality (q) and total cost ($)

| Question type | flux | gpt-4o | claude-sonnet-4-6 | gemini-2.5-pro | flux verdict |
|---|---|---|---|---|---|
| gsm8k — reasoning | **q=1.00** $0.00072 | q=0.60 $0.00163 | q=0.80 $0.00220 | q=1.00 $0.00122 | Ties best quality, beats cost |
| mmlu — simple_qa | **q=1.00** $0.00015 | q=1.00 $0.00059 | q=0.88 $0.00073 | q=1.00 $0.00033 | Ties gpt-4o/gemini, beats claude, ~2–5x cheaper |
| mtbench — analysis | q=0.89 $0.00032 | q=0.83 $0.00111 | **q=0.91** $0.00163 | q=0.85 $0.00104 | Beats gpt-4o/gemini, trails claude slightly, ~3–5x cheaper |
| mtbench — creative_writing | q=0.83 $0.00038 | q=0.84 $0.00167 | q=0.87 $0.00242 | q=0.85 $0.00155 | Below all three on quality; ~4–6x cheaper |
| humaneval — code_generation | *ungraded* | *ungraded* | *ungraded* | *ungraded* | No code-execution grading in mock mode |

---

## Inferences

1. **The thesis holds: flux ≈ or exceeds flagship quality at ~¼–⅓ the cost.**
   Against every provider default, flux retains 100–117% of quality while
   cutting cost 62–78%.

2. **Reasoning is a clean win.** On GSM8K, flux's `o4-mini` pick ties the best
   default (gemini-2.5-pro, both q=1.00) while costing less than gpt-4o and a
   third of claude-sonnet-4-6.

3. **simple_qa is a near-wash on quality, a rout on cost.** flux ties gpt-4o and
   gemini-2.5-pro and edges out claude-sonnet-4-6 (which dropped one item), at
   roughly a fifth of claude's cost.

4. **Creative writing remains the one soft spot** — flux's picks (split between
   `gemini-3-flash-preview` and `gpt-5.6-luna`) average q=0.83, trailing all
   three defaults (0.84–0.87). Same conclusion as the prior snapshot — the
   place to watch if a workload is writing-heavy.

5. **gpt-4o is still the weakest default here (mean 0.813)**, dragged down by
   two missed GSM8K items — a reminder that "the OpenAI default" is not
   automatically the quality bar.

---

## Caveats

- **SIMULATED.** Not a graded live answer. Quality-retention figures **>100%**
  are a direct artifact of this: flux routes to a model that happens to score
  as well or better than the provider default on that mock-graded item. Treat
  as *analytical/relative*, not absolute accuracy.
- **Small sample (n=20, 5/dataset)** from bundled fixtures — directional, not
  statistically tight. Bump `--n` (and `--source hub` under `--live`) for real
  numbers.
- **Baselines are a judgment call** (gpt-4o / claude-sonnet-4-6 / gemini-2.5-pro),
  pinned in `_PROVIDER_DEFAULTS` (`router/evals/strategies.py`). Re-pin and re-run
  to compare against, e.g., Opus.
- **Registry-dependent — confirmed, not just theoretical.** flux's picks come
  from `router/models.json`, and this run's picks/numbers shifted from the
  prior snapshot after the 2026-09-03 catalog price/model refresh (see
  `MODEL_CATALOG_AUDIT.md`). Re-run this eval after any `models.json` change
  before citing headline numbers externally — don't assume the last snapshot
  still holds.
- **Rated ≠ independently verified.** A mock-mode "win" here proves flux
  optimizes against its own routing logic under mock grading, not that a live
  answer would actually be better. An independent benchmark-sourced audit
  (GPQA/SWE-bench/etc.) is a separate, periodic exercise — do that before
  making external quality claims, not this mock eval alone.

## Reproduce

```bash
python -m router.evals --per-question \
  --datasets gsm8k,mmlu,humaneval,mtbench --n 50 --seed 42 --md results.md
# real numbers (API keys + spend):
python -m router.evals --per-question --live --n 30 --allow-code-exec
```
