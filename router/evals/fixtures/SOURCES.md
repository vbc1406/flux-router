# Fixture sources

These tiny fixtures (≈5 items per dataset) exist so the eval harness can run
fully offline — in CI, in tests, and in the default simulated mode — **without
any network access or HuggingFace download**. They mirror the exact field
schema of the upstream `datasets` records so a single set of builders in
`datasets.py` normalizes both the fixtures and the real downloads.

For real numbers, run with `--live --source hub`, which pulls the full datasets:

| dataset    | hub id                              | license        |
|------------|-------------------------------------|----------------|
| gsm8k      | `gsm8k` (config `main`)             | MIT            |
| mmlu       | `cais/mmlu` (config `all`)          | MIT            |
| humaneval  | `openai_humaneval`                  | MIT            |
| mtbench    | `HuggingFaceH4/mt_bench_prompts`    | Apache-2.0/CC  |

The `gsm8k.json` and `humaneval.json` fixtures contain a few genuine upstream
items (both datasets are MIT-licensed). The `mmlu.json` and `mtbench.json`
fixtures are hand-authored items in the upstream format to avoid redistributing
those corpora; the real items load via `--source hub`.
