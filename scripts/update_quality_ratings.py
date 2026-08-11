"""
One-shot: update router/models.json quality_ratings using public benchmark
data fetched 2026-05-10. Mappings:
  simple_qa, classification        <- MMLU
  general, conversation, summarization, extraction <- composite (MMLU + IFEval proxy)
  translation                      <- multilingual MMLU proxy
  code_generation                  <- HumanEval / LiveCodeBench
  code_review                      <- SWE-bench Verified
  analysis                         <- MMLU-Pro
  reasoning                        <- GPQA Diamond / AIME / ARC-AGI-2
  function_calling                 <- BFCL
  vision                           <- MMMU-Pro (0.0 if no vision capability)
  long_document                    <- RULER / NIH
  creative_writing                 <- Arena writing Elo proxy
  unknown                          <- conservative average
"""
import json
from pathlib import Path

# Hand-curated per model from May 2026 benchmark data
RATINGS = {
    "gemini-2.0-flash-lite": {
        "simple_qa": 0.75, "conversation": 0.80, "translation": 0.75,
        "classification": 0.70, "extraction": 0.65, "summarization": 0.65,
        "code_generation": 0.50, "code_review": 0.45, "creative_writing": 0.55,
        "analysis": 0.55, "reasoning": 0.45, "function_calling": 0.50,
        "vision": 0.60, "long_document": 0.60, "unknown": 0.60, "general": 0.60,
    },
    "gemini-2.0-flash-free": {
        "simple_qa": 0.80, "conversation": 0.82, "translation": 0.80,
        "classification": 0.80, "extraction": 0.78, "summarization": 0.78,
        "code_generation": 0.72, "code_review": 0.68, "creative_writing": 0.70,
        "analysis": 0.72, "reasoning": 0.68, "function_calling": 0.70,
        "vision": 0.80, "long_document": 0.78, "unknown": 0.72, "general": 0.72,
    },
    "llama-3.3-70b": {
        # MMLU 86.0, HumanEval 88.4, BFCL 77.3, MMLU-Pro 68.9, GPQA ~50
        "simple_qa": 0.86, "conversation": 0.82, "translation": 0.80,
        "classification": 0.84, "extraction": 0.80, "summarization": 0.80,
        "code_generation": 0.84, "code_review": 0.72, "creative_writing": 0.74,
        "analysis": 0.76, "reasoning": 0.62, "function_calling": 0.78,
        "vision": 0.00, "long_document": 0.74, "unknown": 0.76, "general": 0.78,
    },
    "llama-4-scout": {
        # 10M ctx, vision, weaker than Maverick (MMLU 85.5)
        "simple_qa": 0.83, "conversation": 0.82, "translation": 0.80,
        "classification": 0.83, "extraction": 0.80, "summarization": 0.82,
        "code_generation": 0.78, "code_review": 0.72, "creative_writing": 0.74,
        "analysis": 0.78, "reasoning": 0.70, "function_calling": 0.74,
        "vision": 0.76, "long_document": 0.86, "unknown": 0.76, "general": 0.78,
    },
    "mistral-medium-3.5": {
        # No public bench; vendor claims ~90% of Sonnet 3.7
        "simple_qa": 0.84, "conversation": 0.83, "translation": 0.88,
        "classification": 0.84, "extraction": 0.82, "summarization": 0.82,
        "code_generation": 0.82, "code_review": 0.78, "creative_writing": 0.78,
        "analysis": 0.82, "reasoning": 0.78, "function_calling": 0.80,
        "vision": 0.00, "long_document": 0.78, "unknown": 0.80, "general": 0.82,
    },
    "mistral-small-latest": {
        "simple_qa": 0.80, "conversation": 0.80, "translation": 0.85,
        "classification": 0.80, "extraction": 0.76, "summarization": 0.76,
        "code_generation": 0.72, "code_review": 0.68, "creative_writing": 0.70,
        "analysis": 0.72, "reasoning": 0.66, "function_calling": 0.70,
        "vision": 0.00, "long_document": 0.70, "unknown": 0.72, "general": 0.72,
    },
    "claude-haiku-4-5-20251001": {
        # Cheap-tier 4.5 family; strong code, decent reasoning
        "simple_qa": 0.85, "conversation": 0.86, "translation": 0.84,
        "classification": 0.86, "extraction": 0.84, "summarization": 0.83,
        "code_generation": 0.87, "code_review": 0.85, "creative_writing": 0.78,
        "analysis": 0.82, "reasoning": 0.78, "function_calling": 0.84,
        "vision": 0.78, "long_document": 0.80, "unknown": 0.82, "general": 0.82,
    },
    "gpt-5-mini": {
        # Cheap-tier GPT-5 variant
        "simple_qa": 0.86, "conversation": 0.86, "translation": 0.82,
        "classification": 0.86, "extraction": 0.84, "summarization": 0.82,
        "code_generation": 0.86, "code_review": 0.78, "creative_writing": 0.80,
        "analysis": 0.84, "reasoning": 0.80, "function_calling": 0.84,
        "vision": 0.80, "long_document": 0.80, "unknown": 0.82, "general": 0.83,
    },
    "gpt-4o-mini": {
        "simple_qa": 0.78, "conversation": 0.82, "translation": 0.78,
        "classification": 0.80, "extraction": 0.78, "summarization": 0.78,
        "code_generation": 0.78, "code_review": 0.65, "creative_writing": 0.74,
        "analysis": 0.72, "reasoning": 0.62, "function_calling": 0.78,
        "vision": 0.74, "long_document": 0.74, "unknown": 0.74, "general": 0.76,
    },
    "gemini-3.1-flash-lite": {
        # MMLU-Pro 83.0, MATH-500 85.6, GPQA 72.2
        "simple_qa": 0.85, "conversation": 0.84, "translation": 0.84,
        "classification": 0.85, "extraction": 0.82, "summarization": 0.82,
        "code_generation": 0.80, "code_review": 0.72, "creative_writing": 0.76,
        "analysis": 0.83, "reasoning": 0.78, "function_calling": 0.80,
        "vision": 0.80, "long_document": 0.84, "unknown": 0.80, "general": 0.82,
    },
    "gemini-2.0-flash": {
        "simple_qa": 0.82, "conversation": 0.84, "translation": 0.82,
        "classification": 0.82, "extraction": 0.80, "summarization": 0.80,
        "code_generation": 0.75, "code_review": 0.70, "creative_writing": 0.74,
        "analysis": 0.78, "reasoning": 0.72, "function_calling": 0.76,
        "vision": 0.82, "long_document": 0.82, "unknown": 0.76, "general": 0.78,
    },
    "claude-sonnet-4-6": {
        # SWE-bench 79.6, GPQA 74.1, ARC-AGI-2 58.3, HumanEval ~97
        "simple_qa": 0.88, "conversation": 0.90, "translation": 0.86,
        "classification": 0.88, "extraction": 0.87, "summarization": 0.88,
        "code_generation": 0.95, "code_review": 0.92, "creative_writing": 0.90,
        "analysis": 0.88, "reasoning": 0.84, "function_calling": 0.88,
        "vision": 0.84, "long_document": 0.86, "unknown": 0.86, "general": 0.88,
    },
    "claude-sonnet-4-20250514": {
        # Older Sonnet 4 (May 2025)
        "simple_qa": 0.85, "conversation": 0.87, "translation": 0.84,
        "classification": 0.86, "extraction": 0.84, "summarization": 0.84,
        "code_generation": 0.90, "code_review": 0.86, "creative_writing": 0.86,
        "analysis": 0.84, "reasoning": 0.78, "function_calling": 0.84,
        "vision": 0.80, "long_document": 0.82, "unknown": 0.82, "general": 0.84,
    },
    "gpt-5": {
        # MMLU ~91, SWE-bench 74.9, GPQA 88.4, AIME 94.6, HumanEval ~96
        "simple_qa": 0.91, "conversation": 0.90, "translation": 0.88,
        "classification": 0.90, "extraction": 0.88, "summarization": 0.88,
        "code_generation": 0.92, "code_review": 0.86, "creative_writing": 0.88,
        "analysis": 0.90, "reasoning": 0.90, "function_calling": 0.90,
        "vision": 0.86, "long_document": 0.86, "unknown": 0.88, "general": 0.90,
    },
    "gpt-4o": {
        # GPQA 70.1, MMLU ~88
        "simple_qa": 0.88, "conversation": 0.88, "translation": 0.86,
        "classification": 0.88, "extraction": 0.84, "summarization": 0.84,
        "code_generation": 0.85, "code_review": 0.78, "creative_writing": 0.84,
        "analysis": 0.82, "reasoning": 0.74, "function_calling": 0.86,
        "vision": 0.84, "long_document": 0.82, "unknown": 0.82, "general": 0.84,
    },
    "gemini-3-flash-preview": {
        # GPQA 90.4, MMMU-Pro 81.2 (frontier flash)
        "simple_qa": 0.88, "conversation": 0.88, "translation": 0.86,
        "classification": 0.88, "extraction": 0.86, "summarization": 0.86,
        "code_generation": 0.86, "code_review": 0.80, "creative_writing": 0.82,
        "analysis": 0.88, "reasoning": 0.90, "function_calling": 0.86,
        "vision": 0.88, "long_document": 0.88, "unknown": 0.86, "general": 0.88,
    },
    "gemini-2.5-pro": {
        # GPQA 84.0, Global MMLU 89.8, LiveCodeBench 70.4
        "simple_qa": 0.90, "conversation": 0.88, "translation": 0.88,
        "classification": 0.90, "extraction": 0.86, "summarization": 0.88,
        "code_generation": 0.86, "code_review": 0.82, "creative_writing": 0.85,
        "analysis": 0.88, "reasoning": 0.84, "function_calling": 0.84,
        "vision": 0.86, "long_document": 0.92, "unknown": 0.85, "general": 0.88,
    },
    "claude-opus-4-7": {
        # SWE-bench ~88+, GPQA 94.2, MMMU big jump, frontier
        "simple_qa": 0.92, "conversation": 0.93, "translation": 0.90,
        "classification": 0.92, "extraction": 0.92, "summarization": 0.93,
        "code_generation": 0.96, "code_review": 0.95, "creative_writing": 0.95,
        "analysis": 0.94, "reasoning": 0.94, "function_calling": 0.92,
        "vision": 0.94, "long_document": 0.92, "unknown": 0.92, "general": 0.94,
    },
    "claude-opus-4-20250514": {
        # SWE-bench 80.8, GPQA 91.3 (Opus 4 May 2025)
        "simple_qa": 0.90, "conversation": 0.91, "translation": 0.88,
        "classification": 0.90, "extraction": 0.90, "summarization": 0.91,
        "code_generation": 0.92, "code_review": 0.90, "creative_writing": 0.92,
        "analysis": 0.90, "reasoning": 0.91, "function_calling": 0.88,
        "vision": 0.86, "long_document": 0.88, "unknown": 0.88, "general": 0.90,
    },
    "gpt-5.5": {
        # MMLU 92.4, SWE-bench 88.7, GPQA 93.6, ARC-AGI-2 85.0
        "simple_qa": 0.92, "conversation": 0.91, "translation": 0.89,
        "classification": 0.92, "extraction": 0.90, "summarization": 0.90,
        "code_generation": 0.94, "code_review": 0.92, "creative_writing": 0.89,
        "analysis": 0.92, "reasoning": 0.94, "function_calling": 0.92,
        "vision": 0.88, "long_document": 0.88, "unknown": 0.90, "general": 0.92,
    },
    "gpt-4.5-preview": {
        "simple_qa": 0.89, "conversation": 0.92, "translation": 0.88,
        "classification": 0.90, "extraction": 0.88, "summarization": 0.90,
        "code_generation": 0.86, "code_review": 0.82, "creative_writing": 0.92,
        "analysis": 0.86, "reasoning": 0.80, "function_calling": 0.86,
        "vision": 0.84, "long_document": 0.84, "unknown": 0.86, "general": 0.88,
    },
    "gemini-3.1-pro-preview": {
        # GPQA 94.3 (top), MMMU-Pro 81.0, Arena #1
        "simple_qa": 0.92, "conversation": 0.91, "translation": 0.90,
        "classification": 0.92, "extraction": 0.90, "summarization": 0.90,
        "code_generation": 0.90, "code_review": 0.86, "creative_writing": 0.88,
        "analysis": 0.93, "reasoning": 0.94, "function_calling": 0.90,
        "vision": 0.92, "long_document": 0.94, "unknown": 0.90, "general": 0.92,
    },
    "gemini-2.5-pro-thinking": {
        "simple_qa": 0.90, "conversation": 0.88, "translation": 0.88,
        "classification": 0.90, "extraction": 0.88, "summarization": 0.88,
        "code_generation": 0.88, "code_review": 0.84, "creative_writing": 0.86,
        "analysis": 0.92, "reasoning": 0.92, "function_calling": 0.86,
        "vision": 0.86, "long_document": 0.92, "unknown": 0.88, "general": 0.90,
    },
    "o3": {
        # MMLU 93, reasoning-tuned
        "simple_qa": 0.91, "conversation": 0.86, "translation": 0.86,
        "classification": 0.90, "extraction": 0.88, "summarization": 0.86,
        "code_generation": 0.90, "code_review": 0.86, "creative_writing": 0.82,
        "analysis": 0.92, "reasoning": 0.94, "function_calling": 0.86,
        "vision": 0.84, "long_document": 0.84, "unknown": 0.88, "general": 0.90,
    },
    "o4-mini": {
        # HumanEval 99, reasoning-tuned mini
        "simple_qa": 0.86, "conversation": 0.84, "translation": 0.82,
        "classification": 0.86, "extraction": 0.84, "summarization": 0.82,
        "code_generation": 0.94, "code_review": 0.86, "creative_writing": 0.78,
        "analysis": 0.88, "reasoning": 0.90, "function_calling": 0.84,
        "vision": 0.80, "long_document": 0.80, "unknown": 0.84, "general": 0.86,
    },
    "claude-opus-4-20250514-extended": {
        # Same model, extended thinking
        "simple_qa": 0.90, "conversation": 0.91, "translation": 0.88,
        "classification": 0.90, "extraction": 0.90, "summarization": 0.92,
        "code_generation": 0.93, "code_review": 0.92, "creative_writing": 0.92,
        "analysis": 0.92, "reasoning": 0.94, "function_calling": 0.88,
        "vision": 0.86, "long_document": 0.90, "unknown": 0.90, "general": 0.92,
    },
}


def main():
    path = Path(__file__).parent.parent / "router" / "models.json"
    data = json.loads(path.read_text())

    # Sanity: every model in registry must have ratings here
    registry_ids = {m["model_id"] for m in data["models"]}
    missing = registry_ids - RATINGS.keys()
    extra = RATINGS.keys() - registry_ids
    assert not missing, f"Missing ratings for: {missing}"
    assert not extra, f"Extra ratings (no such model): {extra}"

    # Validate vision==0 if model lacks vision capability
    for m in data["models"]:
        ratings = dict(RATINGS[m["model_id"]])
        if "vision" not in m["capabilities"]:
            ratings["vision"] = 0.00
        m["quality_ratings"] = ratings

    data["last_updated"] = "2026-05-10"
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Updated {len(data['models'])} models in {path}")


if __name__ == "__main__":
    main()
