"""
Post-response quality estimation — heuristic only, no LLM calls.
Target: < 2 ms.

After the model returns a response, this module scores it on several
dimensions and feeds the result back to AdaptiveWeights so future routing
decisions for the same (model, task_type) pair can be informed by observed
quality.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from .schemas import ModelOption, QualityScore, RoutingRequest, TaskAnalysis

if TYPE_CHECKING:
    from .adaptive_weights import AdaptiveWeights

log = structlog.get_logger(__name__)

# ── Quality dimension weights (must sum to 1.0) ────────────────────────────
_WEIGHTS = {
    "length":      0.20,
    "format":      0.20,
    "refusal":     0.15,
    "hallucination": 0.15,
    "repetition":  0.10,
    "cutoff":      0.10,
    "latency":     0.10,
}

# Refusal phrases that indicate the model under-served the request
_REFUSAL_PHRASES = re.compile(
    r"\b(i can'?t|i cannot|i am unable|i'm unable|as an ai|"
    r"i don'?t have access|i do not have access|i'?m not able to|"
    r"i must decline|i'?m afraid i can'?t)\b",
    re.IGNORECASE,
)

# Hallucination risk signals
_URL_RE       = re.compile(r"https?://\S+")
_CITATION_RE  = re.compile(r"\[\d+\]|\((?:Smith|Jones|Wang|et al)[^)]*\d{4}\)")
_OVERCONFIDENT_RE = re.compile(
    r"\b(definitely|absolutely|certainly|guaranteed|100%|without a doubt)\b",
    re.IGNORECASE,
)

# Cut-off signals
_OPEN_FENCE_RE = re.compile(r"```[^\n]*\n(?:(?!```).)*$", re.DOTALL)  # unclosed code fence


# Expected output length ranges (in words) per task type.
# Used for length_appropriateness scoring.
_EXPECTED_WORDS: dict[str, tuple[int, int]] = {
    "simple_qa":        (10,  200),
    "conversation":     (5,   200),
    "translation":      (1,   9999),   # proportional — checked separately
    "classification":   (1,    50),
    "extraction":       (20,  500),
    "summarization":    (50,  600),
    "code_generation":  (50, 2000),
    "code_review":      (30,  800),
    "creative_writing": (100, 3000),
    "analysis":         (100, 1500),
    "reasoning":        (100, 2000),
    "function_calling": (20,  500),
    "vision":           (20,  600),
    "long_document":    (100, 3000),
    "unknown":          (10, 1000),
}


class QualityScorer:
    """
    Scores a model response across seven heuristic dimensions and records the
    result in AdaptiveWeights so the router learns over time.
    """

    def __init__(self, adaptive_weights: "AdaptiveWeights") -> None:
        self._adaptive = adaptive_weights

    def score_response(
        self,
        request: RoutingRequest,
        analysis: TaskAnalysis,
        response_text: str,
        model_used: ModelOption,
        latency_ms: int,
    ) -> QualityScore:
        """
        Produce a QualityScore and update adaptive weights.
        All checks are regex / length-based — no external calls.
        """
        words = response_text.split()
        word_count = len(words)
        task_type  = analysis.task_type

        length_score     = self._score_length(task_type, word_count, analysis.estimated_output_tokens)
        format_score     = self._score_format(request, response_text, task_type)
        refusal_detected = bool(_REFUSAL_PHRASES.search(response_text))
        refusal_score    = 0.0 if refusal_detected else 1.0
        hallucination    = self._score_hallucination(response_text)
        repetition_det   = self._detect_repetition(response_text)
        repetition_score = 0.2 if repetition_det else 1.0
        cutoff_detected  = self._detect_cutoff(response_text)
        cutoff_score     = 0.2 if cutoff_detected else 1.0
        latency_score    = self._score_latency(latency_ms)

        overall = (
            length_score          * _WEIGHTS["length"]
            + format_score        * _WEIGHTS["format"]
            + refusal_score       * _WEIGHTS["refusal"]
            + (1.0 - hallucination) * _WEIGHTS["hallucination"]
            + repetition_score    * _WEIGHTS["repetition"]
            + cutoff_score        * _WEIGHTS["cutoff"]
            + latency_score       * _WEIGHTS["latency"]
        )
        overall = round(max(0.0, min(1.0, overall)), 4)

        # Feed result back into the learning loop
        base = model_used.quality_ratings.get(task_type, 0.5)
        self._adaptive.record(model_used.model_id, task_type, overall, base)

        log.debug(
            "quality_scored",
            model=model_used.model_id,
            task_type=task_type,
            overall=overall,
            refusal=refusal_detected,
            cutoff=cutoff_detected,
        )

        return QualityScore(
            overall                 = overall,
            length_appropriateness  = length_score,
            format_compliance       = format_score,
            refusal_detected        = refusal_detected,
            hallucination_risk      = hallucination,
            repetition_detected     = repetition_det,
            latency_rating          = latency_score,
            cut_off_detected        = cutoff_detected,
        )

    # ── Dimension scorers ───────────────────────────────────────────────────

    @staticmethod
    def _score_length(task_type: str, word_count: int, expected_output_tokens: int) -> float:
        """
        Compare actual length to the expected range for this task type.
        Very short responses on complex tasks and very long ones on simple tasks
        both score poorly.
        """
        low, high = _EXPECTED_WORDS.get(task_type, (10, 1000))

        # For translation we compare to estimated output tokens (proportional to input)
        if task_type == "translation":
            expected_words = max(10, int(expected_output_tokens / 1.3))
            low  = max(1, int(expected_words * 0.5))
            high = int(expected_words * 1.8)

        if word_count < low * 0.20:
            return 0.2   # critically short
        if word_count > high * 3.0:
            return 0.5   # excessively long
        if low <= word_count <= high:
            return 1.0
        # Partial credit for being outside range but not extreme
        if word_count < low:
            return 0.6
        return 0.75

    # Matches explicit JSON output instructions, not just any mention of "json".
    _JSON_OUTPUT_RE = re.compile(
        r"\b(output|respond|return|format|give me|provide).{0,30}json\b"
        r"|\bjson\s+(format|output|response|schema)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _score_format(request: RoutingRequest, text: str, task_type: str) -> float:
        """Check that structured-output requests have valid structure."""
        prompt = request.raw_prompt
        if QualityScorer._JSON_OUTPUT_RE.search(prompt):
            try:
                # Extract first JSON-like block
                start = text.find("{")
                end   = text.rfind("}") + 1
                if start != -1 and end > start:
                    json.loads(text[start:end])
                    return 1.0
            except (json.JSONDecodeError, ValueError):
                pass
            return 0.4   # JSON was requested but response is not valid JSON
        if task_type in ("code_generation", "code_review") and "```" in text:
            return 1.0
        if task_type in ("code_generation",) and "```" not in text:
            return 0.6   # code requested but no fence
        return 1.0

    @staticmethod
    def _score_hallucination(text: str) -> float:
        """
        Rough hallucination risk heuristic (higher = riskier).
        Fabricated URLs, suspicious citations, and overconfident language all
        contribute to the risk score.
        """
        risk = 0.0
        urls = _URL_RE.findall(text)
        # Penalise clearly made-up URLs (not wikipedia, github, etc.)
        for url in urls:
            if not any(d in url for d in ("wikipedia.org", "github.com", "arxiv.org", "openai.com", "anthropic.com")):
                risk += 0.05
        risk += min(0.15, len(_CITATION_RE.findall(text)) * 0.05)
        risk += min(0.10, len(_OVERCONFIDENT_RE.findall(text)) * 0.03)
        return round(min(1.0, risk), 4)

    @staticmethod
    def _detect_repetition(text: str) -> bool:
        """
        Detect repeated sentences or paragraphs — a common sign of a model
        looping / running out of useful content.
        """
        sentences = [s.strip().lower() for s in re.split(r"[.!?]", text) if len(s.strip()) > 20]
        if len(sentences) < 4:
            return False
        seen: set[str] = set()
        for s in sentences:
            if s in seen:
                return True
            seen.add(s)
        return False

    @staticmethod
    def _detect_cutoff(text: str) -> bool:
        """
        Detect responses that were cut off mid-generation:
        trailing ellipsis, ends mid-sentence, or an unclosed code block.
        """
        stripped = text.rstrip()
        if not stripped:
            return True
        if stripped.endswith("...") or stripped.endswith("…"):
            return True
        if _OPEN_FENCE_RE.search(text):
            return True
        # Ends on a word without sentence-ending punctuation in a substantial response
        if len(stripped.split()) > 20 and stripped[-1] not in ".!?\"'`)]":
            return True
        return False

    @staticmethod
    def _score_latency(latency_ms: int) -> float:
        """Convert raw latency to a 0–1 quality signal."""
        if latency_ms < 2_000:
            return 1.0
        if latency_ms < 5_000:
            return 0.8
        if latency_ms < 15_000:
            return 0.5
        return 0.2
