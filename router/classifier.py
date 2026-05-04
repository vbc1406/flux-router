"""
File: router/classifier.py

Purpose:
Convert a raw RoutingRequest into a TaskAnalysis (task_type + complexity_score +
token estimates + signals).  Pure heuristic — no ML models, no API calls.
Target latency: < 5 ms per call.

Main Classes:
  RequestClassifier   — entry point: call analyze(request) to get a TaskAnalysis

Config Dependencies (all in config.py):
  COMPLEXITY_BASE_SCORES      — base complexity score per task type
  COMPLEXITY_MODIFIERS        — per-signal additive adjustments
  TEMPLATE_LINE_VARIATION_THRESHOLD, TEMPLATE_SIMILARITY_RATIO — template detection

Key Methods to Modify:
  _detect_task_type()   — add new task types here (regex patterns, ordered most-specific first)
  _extract_signals()    — add new boolean complexity signals here
  _estimate_output_tokens() — add output token estimates for new task types here

🔧 EXTENSION POINT: to add a new task type, follow these steps:
  1. Add its base score to COMPLEXITY_BASE_SCORES in config.py
  2. Add a regex pattern in _detect_task_type() — MORE SPECIFIC patterns must come FIRST
  3. Add an output token estimate in _estimate_output_tokens()
  4. Add a test in test_classifier.py

Things NOT to change without discussion:
  - The ordering of patterns in _detect_task_type() (more specific must precede more general)
  - The sigmoid normalisation in _compute_complexity() (changing it re-scales all scores)
  - The priority hard-clamps in _compute_complexity() ("critical" ≥ 0.70, "low" ≤ 0.40)
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
_TRANSLATE_RE = re.compile(
    r"\b(?:translat\w*|in\s+(?:spanish|french|german|japanese|chinese|korean|arabic))\b",
    re.IGNORECASE,
)
_SUMMARIZE_RE = re.compile(
    r"\b(?:summar\w*|tldr|tl;dr|key\s+points?|main\s+ideas?|brief\s+(?:me|summary))\b",
    re.IGNORECASE,
)
_CLASSIFY_RE = re.compile(
    r"\b(?:classif\w*|categoriz\w*|label|sentiment\s+(?:of|analysis)|is\s+this\s+(?:positive|negative))\b",
    re.IGNORECASE,
)
_EXTRACT_RE = re.compile(
    r"\b(extract|parse\s+(?:all|the)|find\s+all|list\s+(?:all|every)|pull\s+out)\b", re.IGNORECASE
)
_CODE_GEN_RE = re.compile(
    r"\b(?:write\s+(?:a\s+)?(?:function|class|script|program|code|module)"
    r"|implement"
    r"|build\s+(?:a\s+)?(?:full\s+|complete\s+)?(?:function|class|api|service|tool|application|app|component|server|client|system)"
    r"|create\s+(?:a\s+)?(?:function|class|program|app|tool|script|api|service|bot|website)"
    r"|make\s+(?:a\s+)?(?:tool|app|script|program|function|api|service|bot|website)"
    r"|set\s+up(?:\s+a(?:n)?)?\s+(?:server|api|service|database|pipeline|workflow)"
    r")\b",
    re.IGNORECASE,
)
_REASONING_EXTENDED_RE = re.compile(
    r"\b(?:think\s+(?:about|through)|figure\s+out|help\s+me\s+understand|help\s+me\s+(?:figure|think))\b",
    re.IGNORECASE,
)
_CREATIVE_EXTENDED_RE = re.compile(
    r"\b(?:come\s+up\s+with|brainstorm|imagine\s+(?:a|if|that)|think\s+of\s+ideas?)\b",
    re.IGNORECASE,
)
_LANG_NAME_RE = re.compile(
    r"\b(?:python|javascript|typescript|java|c\+\+|c#|go|rust|ruby|swift|kotlin|php|scala"
    r"|matlab|sql|bash|shell"
    r"|react|vue|angular|fastapi|django|flask|node(?:\.js)?|express|next(?:\.js)?|nuxt"
    r"|svelte|laravel|rails|spring)\b",
    re.IGNORECASE,
)
_CODE_REVIEW_RE = re.compile(
    r"\b(review\s+(?:this\s+)?(?:code|function|class)|debug|fix\s+(?:this|the|my)|what(?:'s|\s+is)\s+wrong|find\s+(?:the\s+)?bug)\b",
    re.IGNORECASE,
)
_CREATIVE_RE = re.compile(
    r"\b(?:write\s+a(?:\s+[\w-]+){0,4}\s+(?:story|poem|haiku|essay|blog|article|letter|song|script|novel)"
    r"|draft\s+a|compose\s+a)\b",
    re.IGNORECASE,
)
_ANALYSIS_RE = re.compile(
    r"\b(?:analy[sz]\w*|compar\w*|evaluat\w*|pros\s+and\s+cons|trade.?offs?|assess|critique|explai\w+|risks?\b)\b",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\b(why\s+(?:does|is|do|did)|explain\s+why|prove\s+(?:that)?|solve|step\s+by\s+step|deriv\w*|theorem|proof)\b",
    re.IGNORECASE,
)
_SIMPLE_QA_RE = re.compile(
    r"^(what\s+is|what's|who\s+is|who's|define|when\s+did|where\s+is|how\s+many|which\s+is)",
    re.IGNORECASE,
)
_STEP_BY_STEP_RE = re.compile(r"\bstep.by.step\b|\bstep\s+\d\b", re.IGNORECASE)
_STRUCT_OUT_RE = re.compile(
    r"\b(json|xml|yaml|schema|structured\s+output|in\s+(?:json|yaml|xml)\s+format)\b", re.IGNORECASE
)
_CODE_FENCE_RE = re.compile(r"```")
_MATH_SYM_RE = re.compile(r"[∑∫≥≤∀∃≠²³√∞±∇∂∈∉⊆⊇∩∪]|(?<!\w)=(?!\w)|(?<!\w)[+\-*/^](?!\w)")
_BULLET_RE = re.compile(r"^(\s*[-*•]\s|\s*\d+\.\s)", re.MULTILINE)
_QUESTION_RE = re.compile(r"\?")
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


# ── Keyword groups for secondary scoring pass ────────────────────────────────

_KW_CODE = frozenset(
    {
        "python",
        "javascript",
        "api",
        "database",
        "function",
        "class",
        "deploy",
        "server",
        "frontend",
        "backend",
        "typescript",
        "sql",
        "react",
        "node",
        "flask",
        "django",
        "fastapi",
        "docker",
    }
)
_KW_REASONING = frozenset(
    {
        "why",
        "how",
        "compare",
        "difference",
        "explain",
        "analyze",
        "analyse",
        "pros",
        "cons",
        "tradeoff",
        "tradeoffs",
    }
)
_KW_CREATIVE = frozenset(
    {
        "story",
        "poem",
        "creative",
        "fiction",
        "blog",
        "article",
        "brainstorm",
        "imagine",
        "narrative",
        "character",
        "plot",
    }
)


def estimate_tokens(text: str) -> int:
    """
    Better token estimation without external tokenizers.

    Heuristics:
    - English prose: ~4 chars per token
    - Code: ~3 chars per token
    - CJK: ~1.5 chars per token
    """
    if not text:
        return 0

    total_chars = len(text)

    # Detect code blocks
    code_indicators = text.count("```") + text.count("def ") + text.count("function ")
    symbol_ratio = sum(1 for c in text if c in "{}[]()=<>|&;:") / max(total_chars, 1)

    is_code_heavy = code_indicators > 0 or symbol_ratio > 0.05

    # Detect CJK
    cjk_count = sum(1 for c in text if ("一" <= c <= "鿿" or "぀" <= c <= "ヿ" or "가" <= c <= "힯"))

    cjk_ratio = cjk_count / max(total_chars, 1)

    if cjk_ratio > 0.3:
        tokens = int(total_chars / 1.5)
    elif is_code_heavy:
        tokens = int(total_chars / 3.0)
    else:
        tokens = int(total_chars / 4.0)

    word_count = len(text.split())

    return max(tokens, word_count)


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
        prompt = request.raw_prompt
        history = request.message_history
        sys_prompt = request.system_prompt or ""
        priority = request.priority

        # 1. Token economics
        input_tokens = self._count_tokens(
            prompt + " " + sys_prompt + " " + self._history_text(history)
        )
        task_type, _ = self._detect_task_type(request)
        output_tokens = self._estimate_output_tokens(task_type, input_tokens)
        history_tokens = self._count_tokens(self._history_text(history))
        sys_tokens = self._count_tokens(sys_prompt)
        total_context = input_tokens + output_tokens + sys_tokens  # history already in input_tokens

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
        requires_precision = task_type in (
            "code_generation",
            "code_review",
            "extraction",
        ) or signals.get("requires_structured_output", False)

        # 5. Sensitivity
        sensitivity = self._detect_sensitivity(request)

        # 6. Cache eligibility + fingerprint
        cache_eligible = is_cache_eligible(task_type, request.temperature)
        fp = fingerprint(prompt, request.system_prompt, history, request.temperature)

        return TaskAnalysis(
            complexity_score=complexity,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            total_context_needed=total_context,
            task_type=task_type,
            requires_reasoning=requires_reasoning,
            requires_creativity=requires_creativity,
            requires_precision=requires_precision,
            requires_multilingual=signals.get("multilingual", False),
            requires_streaming=request.prefer_streaming,
            is_multi_turn=len(history) > 0,
            question_count=len(_QUESTION_RE.findall(prompt)),
            has_code_fences=bool(_CODE_FENCE_RE.search(prompt)),
            has_math_symbols=signals.get("math_symbols_present", False),
            sensitivity_level=sensitivity,
            cache_eligible=cache_eligible,
            prompt_fingerprint=fp,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        Heuristic token estimator with CJK and code awareness.
        ~4 chars/token for prose, ~3 for code, ~1.5 for CJK.
        Falls back to word-count lower bound.
        """
        return estimate_tokens(text)

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
            "simple_qa": 150,
            "conversation": 150,
            "classification": 50,
            "code_generation": 1000,
            "code_review": 600,
            "creative_writing": 1500,
            "analysis": 800,
            "reasoning": 1200,
            "function_calling": 400,
            "vision": 400,
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

    def _detect_task_type(self, request: RoutingRequest) -> tuple[str, float]:
        """
        Rule-based task classification using regex + keyword patterns.
        Returns (task_type, confidence). Ordering matters: more specific patterns first.
        Falls back to keyword scoring, then "general" when confidence < 0.5.
        """
        prompt = request.raw_prompt
        history = request.message_history
        meta = request.metadata

        # Vision: images in message content
        if "vision" in request.required_capabilities:
            return "vision", 0.9
        for msg in history + [{"role": "user", "content": prompt}]:
            content = msg.get("content", "")
            if isinstance(content, list):
                if any(
                    isinstance(b, dict) and b.get("type") in ("image_url", "image") for b in content
                ):
                    return "vision", 0.9

        # Function calling: explicit in metadata or capabilities
        if "function_calling" in request.required_capabilities or meta.get("tools"):
            return "function_calling", 0.9

        # Long document: large input (>~2700 tokens already warrants long-doc routing)
        word_count = len(prompt.split())
        if word_count > 2000:
            return "long_document", 0.9

        # Conversation: trivially short, casual, greeting-like
        stripped = prompt.strip()
        if len(stripped) < 50 and not _QUESTION_RE.search(stripped):
            casual_words = {
                "hi",
                "hello",
                "hey",
                "thanks",
                "thank you",
                "ok",
                "okay",
                "sure",
                "yes",
                "no",
                "great",
                "cool",
                "bye",
                "goodbye",
                "lol",
            }
            if (
                any(w in stripped.lower().split() for w in casual_words)
                or len(stripped.split()) <= 3
            ):
                return "conversation", 0.6

        # Unit / currency conversion → route as simple_qa (fast, deterministic answer)
        if re.search(r"\bconvert\b", prompt, re.IGNORECASE) and re.search(
            r"\bto\b", prompt, re.IGNORECASE
        ):
            return "simple_qa", 0.7

        # Architecture / system design
        if re.search(r"\bdesign\s+(?:a|an|the)\b", prompt, re.IGNORECASE):
            return "analysis", 0.7

        # Code review: code fences + review/debug keywords
        has_fences = bool(_CODE_FENCE_RE.search(prompt))
        if has_fences and _CODE_REVIEW_RE.search(prompt):
            return "code_review", 0.8

        # Code generation: explicit patterns or language name + action verb
        if _CODE_GEN_RE.search(prompt) or (
            _LANG_NAME_RE.search(prompt)
            and re.search(r"\b(write|implement|build|create|generate)\b", prompt, re.IGNORECASE)
        ):
            return "code_generation", 0.8

        # Reasoning: proof / step-by-step / heavy math
        if _REASONING_RE.search(prompt) or _STEP_BY_STEP_RE.search(prompt):
            if _MATH_SYM_RE.search(prompt) or re.search(
                r"\b(proof|theorem|derive|solve)\b", prompt, re.IGNORECASE
            ):
                return "reasoning", 0.8

        # Extended reasoning patterns
        if _REASONING_EXTENDED_RE.search(prompt):
            return "reasoning", 0.7

        # Translation
        if _TRANSLATE_RE.search(prompt) and _LANG_RE.search(prompt):
            return "translation", 0.8

        # Summarisation
        if _SUMMARIZE_RE.search(prompt):
            return "summarization", 0.8

        # Classification
        if _CLASSIFY_RE.search(prompt):
            return "classification", 0.8

        # Extraction
        if _EXTRACT_RE.search(prompt):
            return "extraction", 0.8

        # Creative writing (primary patterns)
        if _CREATIVE_RE.search(prompt):
            return "creative_writing", 0.8

        # Creative writing (extended patterns)
        if _CREATIVE_EXTENDED_RE.search(prompt):
            return "creative_writing", 0.7

        # Analysis / comparison
        if _ANALYSIS_RE.search(prompt):
            return "analysis", 0.7

        # Reasoning (broader: why / explain) — only for substantive prompts
        if _REASONING_RE.search(prompt) and word_count > 8:
            return "reasoning", 0.7

        # Simple Q&A: short, starts with question word
        if _SIMPLE_QA_RE.match(prompt.strip()) or (
            len(prompt.strip()) < 120 and prompt.strip().endswith("?")
        ):
            return "simple_qa", 0.6

        # Code review without explicit review keyword (just code fences)
        if has_fences:
            return "code_review", 0.7

        # ── Secondary keyword scoring pass (tiebreaker / fallback) ────────────
        words = set(prompt.lower().split())
        # Multi-word phrase detection
        prompt_lower = prompt.lower()
        code_hits = len(words & _KW_CODE)
        reasoning_hits = len(words & _KW_REASONING)
        if "pros and cons" in prompt_lower:
            reasoning_hits += 1
        creative_hits = len(words & _KW_CREATIVE)
        if "write me" in prompt_lower:
            creative_hits += 1

        best_hits = max(code_hits, reasoning_hits, creative_hits)
        # Need at least 2 keyword hits for confidence to reach 0.5
        if best_hits >= 2:
            kw_confidence = min(0.5 + (best_hits - 2) * 0.05, 0.70)
            if code_hits >= reasoning_hits and code_hits >= creative_hits:
                return "code_generation", kw_confidence
            if reasoning_hits >= creative_hits:
                return "reasoning", kw_confidence
            return "creative_writing", kw_confidence

        return "general", 0.4

    def _extract_signals(
        self,
        request: RoutingRequest,
        input_tokens: int,
        history_tokens: int,
        sys_tokens: int,
    ) -> dict[str, bool]:
        """Extract the boolean modifier signals defined in config.COMPLEXITY_MODIFIERS."""
        prompt = request.raw_prompt
        turns = len(request.message_history)
        fences = len(_CODE_FENCE_RE.findall(prompt))
        lines = prompt.splitlines() or [""]

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
            "input_tokens_gt_2000": input_tokens > 2000,
            "input_tokens_gt_8000": input_tokens > 8000,
            "conversation_turns_gt_5": turns > 5,
            "requires_structured_output": structured,
            "requires_reasoning": bool(
                _REASONING_RE.search(prompt) or _STEP_BY_STEP_RE.search(prompt)
            ),
            "multilingual": bool(_LANG_RE.search(prompt)),
            "domain_specific": domain_specific,
            "complex_system_prompt": complex_sys,
            "multiple_questions_detected": question_count > 1,
            "math_symbols_present": bool(_MATH_SYM_RE.search(prompt)),
            "high_code_fence_density": fences >= 2,
            "high_bullet_density": bullet_density > 0.25,
            "repetitive_templated": self._is_templated(prompt),
            "expected_short_response": len(prompt.split()) < 15 and question_count <= 1,
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
            1 for ln in lengths if abs(ln - avg) / max(avg, 1) < TEMPLATE_LINE_VARIATION_THRESHOLD
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
