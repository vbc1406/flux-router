"""
Request complexity classifier — pure heuristic, zero external ML.
Target: < 5 ms per call, including fingerprinting.

The classifier's job is to turn a raw RoutingRequest into a TaskAnalysis that
the routing engine can act on without ever touching the actual model APIs.
"""

from __future__ import annotations

import math
import re

from .cache import ResponseCache, fingerprint, is_cache_eligible
from .config import (
    COMPLEXITY_BASE_SCORES,
    COMPLEXITY_MODIFIERS,
    TEMPLATE_LINE_VARIATION_THRESHOLD,
    TEMPLATE_SIMILARITY_RATIO,
)
from .schemas import RoutingRequest, TaskAnalysis

# ── Compiled regexes (module-level, paid once) ──────────────────────────────

_LANG_RE = re.compile(
    r"\b(spanish|french|german|italian|portuguese|japanese|chinese|korean|"
    r"arabic|russian|hindi|dutch|swedish|polish|turkish|vietnamese)\b",
    re.IGNORECASE,
)
_TRANSLATE_RE    = re.compile(r"\b(?:translat\w*|in\s+(?:spanish|french|german|japanese|chinese|korean|arabic))\b", re.IGNORECASE)
_SUMMARIZE_RE    = re.compile(r"\b(?:summar\w*|tldr|tl;dr|key\s+points?|main\s+ideas?|brief\s+(?:me|summary))\b", re.IGNORECASE)
_CLASSIFY_RE     = re.compile(r"\b(?:classif\w*|categoriz\w*|label|sentiment\s+(?:of|analysis)|is\s+this\s+(?:positive|negative))\b", re.IGNORECASE)
_EXTRACT_RE      = re.compile(r"\b(extract|parse\s+(?:all|the)|find\s+all|list\s+(?:all|every)|pull\s+out)\b", re.IGNORECASE)
_CODE_GEN_RE     = re.compile(
    r"\b(?:write\s+(?:a\s+)?(?:function|class|script|program|code|module)"
    r"|implement"
    r"|build\s+(?:a\s+)?(?:full\s+|complete\s+)?(?:function|class|api|service|tool|application|app|component|server|client|system)"
    r"|create\s+a\s+(?:function|class|program))\b",
    re.IGNORECASE,
)
_LANG_NAME_RE    = re.compile(
    r"\b(?:python|javascript|typescript|java|c\+\+|c#|go|rust|ruby|swift|kotlin|php|scala"
    r"|matlab|sql|bash|shell"
    r"|react|vue|angular|fastapi|django|flask|node(?:\.js)?|express|next(?:\.js)?|nuxt"
    r"|svelte|laravel|rails|spring)\b",
    re.IGNORECASE,
)
_CODE_REVIEW_RE  = re.compile(r"\b(review\s+(?:this\s+)?(?:code|function|class)|debug|fix\s+(?:this|the|my)|what(?:'s|\s+is)\s+wrong|find\s+(?:the\s+)?bug)\b", re.IGNORECASE)
_CREATIVE_RE     = re.compile(
    r"\b(?:write\s+a(?:\s+[\w-]+){0,4}\s+(?:story|poem|haiku|essay|blog|article|letter|song|script|novel)"
    r"|draft\s+a|compose\s+a)\b",
    re.IGNORECASE,
)
_ANALYSIS_RE     = re.compile(
    r"\b(?:analy[sz]\w*|compar\w*|evaluat\w*|pros\s+and\s+cons|trade.?offs?|assess|critique|explai\w+|risks?\b)\b",
    re.IGNORECASE,
)
_REASONING_RE    = re.compile(r"\b(why\s+(?:does|is|do|did)|explain\s+why|prove\s+(?:that)?|solve|step\s+by\s+step|deriv\w*|theorem|proof)\b", re.IGNORECASE)
_SIMPLE_QA_RE    = re.compile(r"^(what\s+is|what's|who\s+is|who's|define|when\s+did|where\s+is|how\s+many|which\s+is)", re.IGNORECASE)
_STEP_BY_STEP_RE = re.compile(r"\bstep.by.step\b|\bstep\s+\d\b", re.IGNORECASE)
_STRUCT_OUT_RE   = re.compile(r"\b(json|xml|yaml|schema|structured\s+output|in\s+(?:json|yaml|xml)\s+format)\b", re.IGNORECASE)
_CODE_FENCE_RE   = re.compile(r"```")
_MATH_SYM_RE     = re.compile(r"[∑∫≥≤∀∃≠²³√∞±∇∂∈∉⊆⊇∩∪]|(?<!\w)=(?!\w)|(?<!\w)[+\-*/^](?!\w)")
_BULLET_RE       = re.compile(r"^(\s*[-*•]\s|\s*\d+\.\s)", re.MULTILINE)
_QUESTION_RE     = re.compile(r"\?")
_DOMAIN_SPECIFIC_RE = re.compile(
    r"\b(kkt\s+conditions?|lagrangian|hamiltonian|riemann|zeta\s+function|"
    r"gdpr|ecj|indemnif|tort\s+law|habeas\s+corpus|promissory|"
    r"myocardial|pharmacokinetics|ld50|in\s+vitro|homeomorphism|"
    r"eigenvector|eigenvalue|fourier\s+transform|laplace\s+transform|"
    r"byzantine\s+fault|paxos|raft\s+consensus|merkle\s+tree|"
    r"amortized\s+(?:complexity|analysis)|ramanujan|p\s*[≠!=]\s*np)\b",
    re.IGNORECASE,
)
_SENSITIVE_RESTRICTED_RE = re.compile(
    r"\b(ssn|social\s+security\s+number|bank\s+account|routing\s+number|"
    r"private\s+key|secret\s+key|password|passwd|classified|top\s+secret)\b",
    re.IGNORECASE,
)
_SENSITIVE_CONFIDENTIAL_RE = re.compile(
    r"\b(confidential|proprietary|internal\s+only|nda|non.disclosure|"
    r"trade\s+secret|under\s+embargo)\b",
    re.IGNORECASE,
)
_SENSITIVE_INTERNAL_RE = re.compile(
    r"\b(internal\s+use|team\s+only|company\s+(?:internal|policy|data)|"
    r"employee\s+(?:data|record))\b",
    re.IGNORECASE,
)


class RequestClassifier:
    """
    Multi-signal classifier that converts a RoutingRequest into a TaskAnalysis.

    All classification is purely textual / structural — no model calls, no
    external libraries.  Keeping it fast means the routing hot-path stays
    comfortably under 5 ms.
    """

    def __init__(self, cache: ResponseCache) -> None:
        self._cache = cache

    # ── Public ──────────────────────────────────────────────────────────────

    def analyze(self, request: RoutingRequest) -> TaskAnalysis:
        """
        Main entry point.  Returns a fully populated TaskAnalysis.
        Call this BEFORE any cache or budget check — both depend on what
        the classifier produces.
        """
        prompt      = request.raw_prompt
        history     = request.message_history
        sys_prompt  = request.system_prompt or ""
        priority    = request.priority

        # 1. Token economics
        input_tokens   = self._count_tokens(prompt + " " + sys_prompt + " " + self._history_text(history))
        task_type      = self._detect_task_type(request)
        output_tokens  = self._estimate_output_tokens(task_type, input_tokens)
        history_tokens = self._count_tokens(self._history_text(history))
        sys_tokens     = self._count_tokens(sys_prompt)
        total_context  = input_tokens + output_tokens + sys_tokens  # history already in input_tokens

        # 2. Boolean signal extraction
        signals = self._extract_signals(request, input_tokens, history_tokens, sys_tokens)

        # 3. Complexity score
        complexity = _compute_complexity(task_type, signals, priority)

        # 4. Derived flags
        requires_reasoning = (
            task_type in ("reasoning",)
            or signals.get("requires_reasoning", False)
            or signals.get("math_symbols_present", False)
        )
        requires_creativity = task_type == "creative_writing"
        requires_precision  = task_type in ("code_generation", "code_review", "extraction") or signals.get("requires_structured_output", False)

        # 5. Sensitivity
        sensitivity = self._detect_sensitivity(request)

        # 6. Cache eligibility + fingerprint
        cache_eligible = is_cache_eligible(task_type, request.temperature)
        fp = fingerprint(prompt, request.system_prompt, history, request.temperature)

        return TaskAnalysis(
            complexity_score       = complexity,
            estimated_input_tokens = input_tokens,
            estimated_output_tokens= output_tokens,
            total_context_needed   = total_context,
            task_type              = task_type,
            requires_reasoning     = requires_reasoning,
            requires_creativity    = requires_creativity,
            requires_precision     = requires_precision,
            requires_multilingual  = signals.get("multilingual", False),
            requires_streaming     = request.prefer_streaming,
            is_multi_turn          = len(history) > 0,
            question_count         = len(_QUESTION_RE.findall(prompt)),
            has_code_fences        = bool(_CODE_FENCE_RE.search(prompt)),
            has_math_symbols       = signals.get("math_symbols_present", False),
            sensitivity_level      = sensitivity,
            cache_eligible         = cache_eligible,
            prompt_fingerprint     = fp,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Fast token estimate: word count × 1.3 (GPT tokeniser approximation)."""
        return max(1, int(len(text.split()) * 1.3))

    @staticmethod
    def _history_text(history: list[dict]) -> str:
        """Flatten message history to plain text for token counting."""
        parts: list[str] = []
        for msg in history:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return " ".join(parts)

    @staticmethod
    def _estimate_output_tokens(task_type: str, input_tokens: int) -> int:
        """
        Rough output-size estimate by task type.
        Deliberately conservative — actual model output may differ.
        """
        _output_map = {
            "simple_qa":        150,
            "conversation":     150,
            "classification":   50,
            "code_generation":  1000,
            "code_review":      600,
            "creative_writing": 1500,
            "analysis":         800,
            "reasoning":        1200,
            "function_calling": 400,
            "vision":           400,
        }
        if task_type in _output_map:
            return _output_map[task_type]
        if task_type == "translation":
            return max(50, int(input_tokens * 0.9))
        if task_type == "extraction":
            return min(500, max(100, int(input_tokens * 0.5)))
        if task_type == "summarization":
            return max(100, int(input_tokens * 0.2))
        if task_type == "long_document":
            return max(400, int(input_tokens * 0.15))
        return 400

    def _detect_task_type(self, request: RoutingRequest) -> str:
        """
        Rule-based task classification using regex + keyword patterns.
        Ordering matters: more specific patterns come first.
        """
        prompt  = request.raw_prompt
        history = request.message_history
        meta    = request.metadata

        # Vision: images in message content
        if "vision" in request.required_capabilities:
            return "vision"
        for msg in history + [{"role": "user", "content": prompt}]:
            content = msg.get("content", "")
            if isinstance(content, list):
                if any(isinstance(b, dict) and b.get("type") in ("image_url", "image") for b in content):
                    return "vision"

        # Function calling: explicit in metadata or capabilities
        if "function_calling" in request.required_capabilities or meta.get("tools"):
            return "function_calling"

        # Long document: large input (>~2700 tokens already warrants long-doc routing)
        word_count = len(prompt.split())
        if word_count > 2000:
            return "long_document"

        # Conversation: trivially short, casual, greeting-like
        stripped = prompt.strip()
        if len(stripped) < 50 and not _QUESTION_RE.search(stripped):
            casual_words = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
                            "sure", "yes", "no", "great", "cool", "bye", "goodbye", "lol"}
            if any(w in stripped.lower().split() for w in casual_words) or len(stripped.split()) <= 3:
                return "conversation"

        # Unit / currency conversion → route as simple_qa (fast, deterministic answer)
        if re.search(r"\bconvert\b", prompt, re.IGNORECASE) and re.search(r"\bto\b", prompt, re.IGNORECASE):
            return "simple_qa"

        # Architecture / system design
        if re.search(r"\bdesign\s+(?:a|an|the)\b", prompt, re.IGNORECASE):
            return "analysis"

        # Code review: code fences + review/debug keywords
        has_fences = bool(_CODE_FENCE_RE.search(prompt))
        if has_fences and _CODE_REVIEW_RE.search(prompt):
            return "code_review"

        # Code generation: explicit "write code" / language names
        if _CODE_GEN_RE.search(prompt) or (
            _LANG_NAME_RE.search(prompt) and re.search(r"\b(write|implement|build|create|generate)\b", prompt, re.IGNORECASE)
        ):
            return "code_generation"

        # Reasoning: proof / step-by-step / heavy math
        if _REASONING_RE.search(prompt) or _STEP_BY_STEP_RE.search(prompt):
            if _MATH_SYM_RE.search(prompt) or re.search(r"\b(proof|theorem|derive|solve)\b", prompt, re.IGNORECASE):
                return "reasoning"

        # Translation
        if _TRANSLATE_RE.search(prompt) and _LANG_RE.search(prompt):
            return "translation"

        # Summarisation
        if _SUMMARIZE_RE.search(prompt):
            return "summarization"

        # Classification
        if _CLASSIFY_RE.search(prompt):
            return "classification"

        # Extraction
        if _EXTRACT_RE.search(prompt):
            return "extraction"

        # Creative writing
        if _CREATIVE_RE.search(prompt):
            return "creative_writing"

        # Analysis / comparison
        if _ANALYSIS_RE.search(prompt):
            return "analysis"

        # Reasoning (broader: why / explain) — only for substantive prompts
        # Short "why is X?" questions are simple_qa, not reasoning tasks
        if _REASONING_RE.search(prompt) and word_count > 8:
            return "reasoning"

        # Simple Q&A: short, starts with question word
        if _SIMPLE_QA_RE.match(prompt.strip()) or (len(prompt.strip()) < 120 and prompt.strip().endswith("?")):
            return "simple_qa"

        # Code review without explicit review keyword (just code fences)
        if has_fences:
            return "code_review"

        return "unknown"

    def _extract_signals(
        self,
        request: RoutingRequest,
        input_tokens: int,
        history_tokens: int,
        sys_tokens: int,
    ) -> dict[str, bool]:
        """Extract the boolean modifier signals defined in config.COMPLEXITY_MODIFIERS."""
        prompt = request.raw_prompt
        turns  = len(request.message_history)
        fences = len(_CODE_FENCE_RE.findall(prompt))
        lines  = prompt.splitlines() or [""]

        bullet_lines = sum(1 for ln in lines if _BULLET_RE.match(ln))
        bullet_density = bullet_lines / max(len(lines), 1)

        # Structured output detection in prompt OR system prompt
        sys = request.system_prompt or ""
        structured = bool(_STRUCT_OUT_RE.search(prompt) or _STRUCT_OUT_RE.search(sys))

        question_count = len(_QUESTION_RE.findall(prompt))

        # Domain-specific heuristic
        domain_specific = bool(
            _DOMAIN_SPECIFIC_RE.search(prompt)
            or _DOMAIN_SPECIFIC_RE.search(sys)
            or (re.search(r"\b\w{12,}\b", prompt) and self._avg_word_length(prompt) > 7.5)
        )

        # Complex system prompt: long or contains many instructions
        complex_sys = sys_tokens > 200 or sys.count("\n") > 5

        return {
            "input_tokens_gt_2000":        input_tokens > 2000,
            "input_tokens_gt_8000":        input_tokens > 8000,
            "conversation_turns_gt_5":     turns > 5,
            "requires_structured_output":  structured,
            "requires_reasoning":          bool(_REASONING_RE.search(prompt) or _STEP_BY_STEP_RE.search(prompt)),
            "multilingual":                bool(_LANG_RE.search(prompt)),
            "domain_specific":             domain_specific,
            "complex_system_prompt":       complex_sys,
            "multiple_questions_detected": question_count > 1,
            "math_symbols_present":        bool(_MATH_SYM_RE.search(prompt)),
            "high_code_fence_density":     fences >= 2,
            "high_bullet_density":         bullet_density > 0.25,
            "repetitive_templated":        self._is_templated(prompt),
            "expected_short_response":     len(prompt.split()) < 15 and question_count <= 1,
        }

    @staticmethod
    def _avg_word_length(text: str) -> float:
        words = text.split()
        return sum(len(w) for w in words) / max(len(words), 1)

    @staticmethod
    def _is_templated(text: str) -> bool:
        """Detect fill-in-the-blank / repetitive templates (low complexity)."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 3:
            return False
        lengths = [len(ln) for ln in lines]
        avg = sum(lengths) / len(lengths)
        similar = sum(
            1 for ln in lengths
            if abs(ln - avg) / max(avg, 1) < TEMPLATE_LINE_VARIATION_THRESHOLD
        )
        return similar / len(lines) > TEMPLATE_SIMILARITY_RATIO

    def _detect_sensitivity(self, request: RoutingRequest) -> str:
        """
        Determine sensitivity level.  Explicit metadata override takes priority;
        otherwise we scan prompt + system prompt for keywords.
        """
        # Explicit override from caller
        if override := request.metadata.get("sensitivity_level"):
            if override in ("public", "internal", "confidential", "restricted"):
                return override  # type: ignore[return-value]

        combined = (request.raw_prompt + " " + (request.system_prompt or "")).lower()
        if _SENSITIVE_RESTRICTED_RE.search(combined):
            return "restricted"
        if _SENSITIVE_CONFIDENTIAL_RE.search(combined):
            return "confidential"
        if _SENSITIVE_INTERNAL_RE.search(combined):
            return "internal"
        return "public"


def _compute_complexity(task_type: str, signals: dict[str, bool], priority: str) -> float:
    """
    Sigmoid-normalised complexity score in [0, 1].

    The sigmoid keeps the score smooth and bounded, avoiding hard cliff-edges
    between tiers.  Priority clamps are applied AFTER the sigmoid so they can
    override the natural score for time-critical or intentionally cheap requests.
    """
    base = COMPLEXITY_BASE_SCORES.get(task_type, COMPLEXITY_BASE_SCORES["unknown"])
    modifier_sum = sum(
        COMPLEXITY_MODIFIERS[key]
        for key, applies in signals.items()
        if applies and key in COMPLEXITY_MODIFIERS
    )
    raw = base + modifier_sum

    # Sigmoid centred at 0.5 with steepness 5; clamp input to avoid overflow
    clamped = max(-2.0, min(2.0, raw - 0.5))
    score = 1.0 / (1.0 + math.exp(-5.0 * clamped))

    # Priority hard clamps
    if priority == "critical":
        score = max(score, 0.70)
    if priority == "low":
        score = min(score, 0.40)

    return round(score, 4)
