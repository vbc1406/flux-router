# Model catalog audit

Last verified: 2026-08-11

The routable catalog was compared with the providers' official model and
pricing documentation. Flux stores standard synchronous token prices in USD
per 1,000 tokens. It does not attempt to encode batch, flex, priority,
regional-processing, cached-token, long-context, or tool-call surcharges in a
single base price.

## Corrections made

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
- Claude Sonnet 5 has introductory pricing through 2026-08-31. The catalog
  records the published standard price so estimates remain conservative after
  the promotion ends.
- Provider rate limits depend on the customer's account, usage tier, region,
  and model permissions. Catalog RPM values are routing defaults, not a claim
  about the quota attached to a specific API key. Account-specific limits must
  be confirmed during live provider validation.

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
