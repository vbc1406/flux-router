# Model catalog audit

Last verified: 2026-09-03

The routable catalog was compared with the providers' official model and
pricing documentation. Flux stores standard synchronous token prices in USD
per 1,000 tokens. It does not attempt to encode batch, flex, priority,
regional-processing, cached-token, long-context, or tool-call surcharges in a
single base price.

## Corrections made (2026-09-03)

- Claude Sonnet 5's introductory pricing ($2/$10 per MTok) ended 2026-08-31
  and Anthropic confirmed the scheduled increase to $3/$15 will **not**
  occur — $2/$10 is now the permanent standard price. The catalog had
  defensively priced it at $3/$15 in anticipation of the increase; corrected
  back down to $2/$10, with cache read/write repriced off the new base.
- Replaced `claude-fable-5` with its successor `claude-fable-5-1` (same
  $10/$50 price and tier; cheaper prompt-cache reads at 0.025x base input
  instead of 0.1x, per Anthropic's published multiplier for the 5.1 line).
- Replaced `gemini-3.6-flash` with its successor `gemini-3.8-flash` (same
  current promotional price, $0.75/$3.75/MTok through 2026, better coding/
  agentic quality per Google's own comparison).
- Corrected GPT-5.6 Sol standard pricing from $5/$30 to the documented
  $4/$20 per MTok, with cache pricing rederived off the corrected base.
- Corrected stale prompt-cache read/write prices on Claude Opus 5 (was
  1.1x the documented $0.50/$6.25 read/write rate; corrected to $0.50/$6.25).

## Corrections made (2026-08-11)

- Replaced deprecated Mistral Medium 3 (`mistral-medium-2505`) with Mistral
  Medium 3.5 (`mistral-medium-3-5`) and updated its standard price and limits.
- Disabled the ambiguous `mistral-small-latest` alias. Flux already carries the
  pinned Mistral Small 4 endpoint, so routing does not need a moving alias.
- Corrected GPT-OSS 20B pricing and the documented Groq output limits for
  GPT-OSS 20B/120B and Qwen 3.6 27B.
- Corrected GPT-5 and GPT-5 mini output limits and the reduced o3 standard
  token price.
- Corrected Gemini 3.1 Flash-Lite, Gemini 3 Flash Preview, Gemini 2.5 Pro, and
  Gemini 3.1 Pro Preview context/output limits.
- Corrected Claude Haiku 4.5, Sonnet 4.6, Opus 4.7, Fable 5, Sonnet 5, and
  Opus 5 context/output limits.

## Pricing qualifications

- Gemini 2.5 Pro and Gemini 3.1 Pro charge higher rates when a prompt exceeds
  200k tokens. The catalog records the standard price at or below 200k.
- GPT-5.5 charges long-context premiums above 272k input tokens. The catalog
  records its base standard price.
- Provider rate limits depend on the customer's account, usage tier, region,
  and model permissions. Catalog RPM values are routing defaults, not a claim
  about the quota attached to a specific API key. Account-specific limits must
  be confirmed during live provider validation.

## Upcoming

- OpenAI has scheduled `o4-mini` for shutdown 2026-10-23, recommending
  `gpt-5.6-terra` as the replacement. Not yet retired as of this audit;
  re-check before that date.

## Official sources

- OpenAI model catalog and individual model pages:
  https://developers.openai.com/api/docs/models
- Anthropic model overview, pricing, migration guide, and release notes:
  https://platform.claude.com/docs/en/about-claude/models/overview
- Google Gemini model and pricing pages:
  https://ai.google.dev/gemini-api/docs/models
- Groq supported-model and model-specific pages:
  https://console.groq.com/docs/models
- Mistral model overview, model cards, and known limitations:
  https://docs.mistral.ai/models/overview
