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
# Structured-output extraction: "return/output/give JSON ... from <text>",
# "as JSON: ...", or explicit field-list requests.
_EXTRACT_STRUCTURED_RE = re.compile(
    r"\b(?:return|output|produce|give\s+me|respond\s+with)\b[^\n]{0,120}?"
    r"\b(?:json|xml|yaml|fields?|schema|object)\b[^\n]{0,200}?\bfrom\b",
    re.IGNORECASE,
)
# Code-defect inspection: "find the race condition", "deadlock", etc. — strong
# signal for code_review even without code fences.
_CODE_DEFECT_RE = re.compile(
    r"\b(race\s+condition|deadlock|memory\s+leak|null\s+pointer|"
    r"off.by.one|infinite\s+loop|undefined\s+behavio[u]?r|"
    r"data\s+race|use.after.free|double\s+free|buffer\s+overflow)\b",
    re.IGNORECASE,
)
# Long-form creative writing: "write a <N>-word scene/chapter/prologue ..."
# requires both a "write a" lead AND a narrative-form noun, AND a fiction cue
# (novel/story/screenplay/etc.) to avoid matching "accident scene" prose.
_CREATIVE_LONGFORM_RE = re.compile(
    r"\bwrite\s+a\b[^\n]{0,80}?"
    r"\b(?:scene|chapter|prologue|epilogue|monologue|vignette|novella)\b"
    r"[^\n]{0,120}?"
    r"\b(?:novel|story|screenplay|fiction|mystery|thriller|fantasy|"
    r"sci.?fi|cyberpunk|romance|noir|book)\b",
    re.IGNORECASE,
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
# Fenced code blocks, stripped before math-symbol detection: `total += x` and
# `total / len(xs)` are ordinary code, not mathematical notation, and counting
# them made a five-line off-by-one question score *higher* than a
# cross-goroutine deadlock question. Math symbols in the prose around the code
# still count.
_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
# "Do X without Y", "keep Z while W" — each marker is a constraint the answer
# has to satisfy simultaneously. Two or more means the solution space is
# genuinely narrow, which is a within-task difficulty signal.
_CONSTRAINT_RE = re.compile(
    r"\b(without|instead\s+of|rather\s+than|while\s+(?:still\s+)?(?:keeping|maintaining|"
    r"preserving|the\b)|but\s+(?:still\s+)?(?:keep|maintain|preserve|avoid)|"
    r"must\s+not|cannot\s+use|can't\s+use|no\s+(?:global|shared|additional)\s+\w+|"
    r"ensur\w+|guarante\w+|handle\s+(?:malformed|invalid|edge))\b",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^(\s*[-*•]\s|\s*\d+\.\s)", re.MULTILINE)
# ── Explicit output-length cues (see _explicit_output_tokens) ────────────────
_WORD_COUNT_RE = re.compile(r"\b(\d{2,5})[\s-]*word\b", re.IGNORECASE)
_SENTENCE_COUNT_RE = re.compile(
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)[\s-]*sentences?\b",
    re.IGNORECASE,
)
_PARAGRAPH_COUNT_RE = re.compile(
    r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)[\s-]*paragraphs?\b",
    re.IGNORECASE,
)
_NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# Forms whose length is fixed by the form itself, regardless of task type.
_FIXED_FORM_TOKENS: dict[re.Pattern[str], int] = {
    re.compile(r"\bhaiku\b", re.IGNORECASE): 40,
    re.compile(r"\b(limerick|couplet)\b", re.IGNORECASE): 80,
    re.compile(r"\bsonnet\b", re.IGNORECASE): 250,
    re.compile(r"\b(tweet|headline|tagline|subject\s+line)\b", re.IGNORECASE): 60,
    re.compile(r"\bin\s+a\s+(single|one)\s+(word|line)\b", re.IGNORECASE): 30,
}
# Floor for any explicit-length estimate — even "in one word" needs room for a
# short preamble, and a near-zero estimate would make cost scoring meaningless.
_MIN_OUTPUT_TOKENS = 30
_QUESTION_RE = re.compile(r"\?")
# Agent planning language: "plan the steps", "how should I approach this",
# "what should I do next", "break this down into steps". Only consulted for
# requests carrying a run_id — see _infer_step_type — so ordinary chat like
# "plan a trip to Kyoto" is untouched.
_PLAN_RE = re.compile(
    r"\b(plan\s+(?:your|the|out|this|a\s+few)|make\s+a\s+plan|come\s+up\s+with\s+a\s+plan|"
    r"break\s+(?:this|it|the\s+task)\s+(?:down|into)|"
    r"how\s+should\s+(?:I|we|you)\s+(?:approach|tackle|start)|"
    r"what\s+should\s+(?:I|we|you)\s+do\s+(?:next|first)|"
    r"decide\s+(?:the|your)\s+next\s+(?:step|action)|"
    r"outline\s+(?:the|your)\s+(?:steps|approach)|"
    r"first,?\s+plan|start\s+by\s+planning)\b"
    r"|^\s*plan[.!]?\s*$",
    re.IGNORECASE,
)
# step_type inference — "reflect" language (self-critique / plan revision).
_REFLECT_RE = re.compile(
    r"\b(reflect\s+on|self.critique|critique\s+(your|the)|review\s+your\s+(own\s+)?"
    r"(answer|work|response|plan)|did\s+I\s+miss|what\s+went\s+wrong|revise\s+the\s+plan)\b",
    re.IGNORECASE,
)
# bugfix: explicit "final answer" language in the prompt — highest-
# confidence signal, checked before anything else so it can't be shadowed by
# a stale tool-result message sitting earlier in history.
_FINAL_ANSWER_RE = re.compile(
    r"\b(final\s+answer|final\s+response|here(?:'s|\s+is)\s+(?:my|the)\s+(?:final\s+)?answer|"
    r"provide\s+(?:your|the)\s+(?:final\s+)?answer|respond\s+to\s+the\s+user|"
    r"give\s+(?:your|the)\s+final\s+(?:answer|response))\b",
    re.IGNORECASE,
)
_DOMAIN_SPECIFIC_RE = re.compile(
    r"\b(kkt\s+conditions?|lagrangian|hamiltonian|riemann|zeta\s+function|"
    r"gdpr|ecj|indemnif|tort\s+law|habeas\s+corpus|promissory|"
    r"myocardial|pharmacokinetics|ld50|in\s+vitro|homeomorphism|"
    r"eigenvector|eigenvalue|fourier\s+transform|laplace\s+transform|"
    r"byzantine\s+fault|paxos|raft\s+consensus|merkle\s+tree|"
    r"amortized\s+(?:complexity|analysis)|ramanujan|p\s*[≠!=]\s*np)\b",
    re.IGNORECASE,
)
# Substantive high-stakes MEDICAL judgment/advice detection (see task_type
# "medical" in COMPLEXITY_BASE_SCORES / DOMAIN_TIER_FLOORS). Deliberately
# requires a judgment-seeking VERB PHRASE (diagnose, "what medication should
# I", "is it safe to take", "am I having a heart attack", drug interactions,
# side effects, prescribing, ER/emergency triage) rather than bare topic
# words — "doctor", "medicine", "hospital" alone must NOT match, so a request
# like "write a story about a doctor" or "summarize this medical report"
# stays on its normal (cheap) task_type. Checked before extraction/
# summarization/translation in _detect_task_type so a request that ALSO asks
# Flux to render a substantive judgment on supplied text is still floored,
# while a pure transformation of supplied text is not.
_MEDICAL_SUBSTANTIVE_RE = re.compile(
    r"\b("
    r"diagnos(?:e|is|ing|ed)\s+(?:me\b|this\b|my\b|the\s+patient\b|and\s+treat\b)|"
    r"what(?:'s| is)\s+wrong\s+with\s+me|"
    r"do\s+i\s+have\b.{0,40}\b(?:cancer|diabetes|infection|disease|condition)|"
    r"treat(?:ment|ing)?\s+(?:for|of)\s+(?:my|this)|"
    r"how\s+(?:should|do)\s+i\s+treat\s+(?:my|this)|"
    r"what\s+(?:medication|drug|dose|dosage)\s+(?:should|can)\s+i\s+(?:take|use)|"
    r"is\s+it\s+safe\s+(?:for\s+me\s+)?to\s+take|"
    r"is\s+it\s+safe\s+(?:for\s+me\s+)?to\s+(?:combine|mix)\b.{0,80}\b(?:and|with)\b|"
    r"should\s+i\s+(?:stop|start|continue|skip|change)\s+taking\b|"
    r"(?:stop|start|continue|skip|change)\s+(?:my\s+)?(?:medication|medicine|dose|dosage)\b|"
    r"drug\s+interaction(?:s)?\s+(?:with|between)|"
    r"side\s+effects?\s+of\s+(?:taking\s+)?\w+|"
    r"symptoms?\s+of\b.{0,40}\b(?:mean|indicate|suggest|could\s+be)|"
    r"(?:do|could|might)\s+(?:these|my|the)\s+symptoms?\s+(?:mean|indicate|suggest|be)\b|"
    r"am\s+i\s+having\s+a\s+(?:heart\s+attack|stroke|seizure)|"
    r"should\s+i\s+go\s+to\s+the\s+(?:er|emergency\s+room|hospital)|"
    r"(?:i|my\s+(?:baby|child|toddler)|the\s+(?:baby|child|toddler))\b.{0,50}\b"
    r"(?:swallowed|ingested|drank|ate)\b.{0,60}\b(?:poison|cleaning|chemical|medication|pills?|fluid)|"
    r"(?:swallowed|ingested|drank)\b.{0,60}\b(?:poison|clean(?:er|ing\s+fluid)|chemical|medication|pills?)|"
    r"(?:poison(?:ed|ing)?|overdos(?:e|ed))\b.{0,80}\b(?:what\s+should|what\s+do|help|emergency)|"
    r"is\s+this\s+(?:rash|lump|mole|pain)\b.{0,40}\bserious|"
    r"prescri(?:be|ption)\s+(?:me|for\s+me)|"
    r"contraindicat\w*|"
    r"how\s+much\s+\w+\s+(?:is|would\s+be)\s+(?:an\s+)?overdose"
    r")\b",
    re.IGNORECASE,
)

# Substantive high-stakes LEGAL judgment/advice detection (see task_type
# "legal" above). Same design constraint as medical: requires an explicit
# judgment/liability/compliance verb phrase — bare mentions of "contract",
# "court", "law" do not match, so "extract the parties from this contract"
# or "summarize this court ruling" keeps its normal (cheap) task_type.
_LEGAL_SUBSTANTIVE_RE = re.compile(
    r"\b("
    r"is\s+(?:this|it)\s+legal(?:ly)?\s+(?:binding|enforceable|allowed|permitted)|"
    r"is\s+this\s+contract\s+enforceable|"
    r"am\s+i\s+(?:legally\s+)?liable|"
    r"can\s+i\s+(?:be\s+)?sue|"
    r"can\s+i\s+be\s+sued|"
    r"what\s+are\s+my\s+(?:legal\s+)?rights|"
    r"(?:can|should)\s+i\s+sue\b|"
    r"(?:landlord|employer|tenant|police|debt\s+collector)\b.{0,100}\b"
    r"(?:security\s+deposit|evict(?:ed|ion)?|fired|wages?|custody|arrest(?:ed)?|"
    r"withheld|kept|rights?)\b.{0,80}\b(?:what\s+can\s+i\s+do|what\s+are\s+my\s+rights|legal)|"
    r"(?:what\s+can\s+i\s+do|what\s+are\s+my\s+rights)\b.{0,100}\b"
    r"(?:landlord|employer|tenant|police|debt\s+collector|security\s+deposit|eviction|wages?|custody)|"
    r"debt\s+collector\b.{0,100}\b(?:calling|contacting|harass(?:ing|ment)?)\b.{0,80}\b"
    r"(?:what\s+can\s+i\s+do|what\s+are\s+my\s+rights|legal)|"
    r"breach\s+of\s+contract|"
    r"liability\s+(?:exposure|risk|clause)|"
    r"indemnif(?:y|ication|ies|ied)|"
    r"regulatory\s+compliance|"
    r"comply\s+with\s+(?:gdpr|hipaa|sox|ccpa|the\s+regulations?)|"
    r"does\s+(?:this|releasing\s+this|doing\s+this)\b.{0,60}\bviolate|"
    r"is\s+this\s+a\s+violation\s+of|"
    r"(?:need|want|give\s+me)\s+legal\s+advice|"
    r"should\s+i\s+sign\s+this|"
    r"non.compete\s+(?:clause\s+)?enforceable|"
    r"statute\s+of\s+limitations|"
    r"file\s+a\s+lawsuit"
    r")\b",
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


def _cache_scope_key(request: RoutingRequest, sensitivity: str) -> str:
    """
    Isolation boundary for the shared, process-wide ResponseCache (see
    cache.py::fingerprint()'s scope_key parameter). Without this, two
    different callers issuing byte-identical prompts (a very real scenario
    for shared system prompts / templated agent steps) would be served each
    other's cached response — a cross-tenant/cross-user information
    disclosure if the cache is ever enabled in a multi-tenant deployment.

    Built from every dimension a cache hit must not cross: tenant (if set),
    user, plan (a cached response bought on one budget must not be handed
    out "for free" under another), and sensitivity level (an internal/
    confidential answer must never surface from a public-labelled cache
    lookup or vice versa).
    """
    return "|".join(
        [
            f"tenant:{request.tenant_id or ''}",
            f"user:{request.user_id}",
            f"plan:{request.plan}",
            f"sensitivity:{sensitivity}",
        ]
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
        prompt = request.raw_prompt
        history = request.message_history
        sys_prompt = request.system_prompt or ""
        priority = request.priority

        # 1. Token economics
        input_tokens = self._count_tokens(
            prompt + " " + sys_prompt + " " + self._history_text(history)
        )
        task_type, _ = self._detect_task_type(request)
        # An explicit length instruction in the prompt beats the flat
        # per-task-type table — see _explicit_output_tokens.
        output_tokens = self._explicit_output_tokens(prompt) or self._estimate_output_tokens(
            task_type, input_tokens
        )
        # Billing-relevant figure: never let the task-type heuristic UNDER-
        # estimate what the caller could actually be billed for. Per-model
        # capping still happens downstream in routing_engine._estimate_cost.
        #
        # When max_tokens_requested is set it is a hard ceiling the provider
        # cannot exceed, so it IS the worst-case bill — using it directly keeps
        # the anti-gaming property (asking for a huge completion still routes
        # and budgets as a huge completion) while no longer over-charging a
        # deliberately capped short one against a large task-type default.
        billing_output_tokens = request.max_tokens_requested or output_tokens
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
        fp = fingerprint(
            prompt,
            request.system_prompt,
            history,
            request.temperature,
            scope_key=_cache_scope_key(request, sensitivity),
        )

        # 7. step_type (orthogonal to task_type)
        step_type = request.step_type or self._infer_step_type(request)

        return TaskAnalysis(
            complexity_score=complexity,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            billing_output_tokens=billing_output_tokens,
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
            step_type=step_type,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _infer_step_type(request: RoutingRequest) -> str:
        """
        infer step_type when the caller didn't set it explicitly.

        Signal priority (most specific / highest-stakes first):
          1. explicit "final answer" language in raw_prompt -> final_answer
             (the most confident signal available; must win even over a
             tool-result message sitting in history, see bugfix note below)
          2. the MOST RECENT history message is a tool result -> tool_result_summarize
             (the model is being asked to digest a tool's output right now,
             not choose one). Bugfix: this used to scan the ENTIRE history for
             any tool-role message, so a tool call from several turns back
             would misclassify the actual final-answer turn as
             tool_result_summarize, stripping its quality floor. Only the
             latest message is evidence of "a tool result just came back."
          3. tools offered -> tool_select
          4. structured response_format requested -> extract (schema-fill is
             low-stakes to verify downstream, same floor as extraction)
          5. "reflect"/"critique"/"review" language in the prompt -> reflect
          6. in-run planning language ("plan the steps", "how should I
             approach", "what should I do next") -> plan. STEP_TYPE_FLOORS
             defines a "plan" floor but nothing used to produce that step_type,
             so a planning step fell to "unknown" and got NO floor — the one
             step where a bad model wastes every step that follows. Measured:
             "What should I do next?" routed to a free-tier model, while the
             same prompt with an explicit step_type="plan" was floored to mid.

             NOTE this is language-gated, not position-gated. Inferring "plan"
             from "first step of a run" is tempting and wrong: server.py
             auto-generates a run_id for every proxy request that arrives
             without X-Flux-Run-Id, so "has a run_id and no history yet"
             describes ordinary single-shot chat through the proxy just as well
             as an agent's opening step, and flooring on it drags all plain
             proxy traffic up to mid. A trajectory whose opening step carries
             no planning language therefore still gets no floor; closing that
             needs the caller to send step_type="plan" (or a real
             client-supplied-run_id signal reaching the classifier).
          7. fail-safe: mid-run (run_id set) with history already underway and
             no tools offered this step -> final_answer. An agent step that
             can't be confidently classified any other way but is clearly
             part of a tracked trajectory defaults to the highest defined
             floor rather than "unknown" (no floor at all) — better to
             occasionally over-protect a cheap step than silently leave a
             real final-answer step unprotected. Single-shot, non-run
             requests (the common case for plain chat use of Flux) are
             unaffected and still resolve to "unknown".
          8. no signal and not part of a run -> "unknown" (no floor applied)

        The plan rule is gated on run_id for the same reason rule 7 is: outside
        a tracked trajectory, "plan a trip to Kyoto" is ordinary chat and must
        keep its cheap routing rather than being floored to mid.
        """
        if _FINAL_ANSWER_RE.search(request.raw_prompt):
            return "final_answer"
        history = request.message_history
        if history and history[-1].get("role") == "tool":
            return "tool_result_summarize"
        if request.tools:
            return "tool_select"
        if request.response_format:
            return "extract"
        if _REFLECT_RE.search(request.raw_prompt):
            return "reflect"
        if request.run_id and _PLAN_RE.search(request.raw_prompt):
            return "plan"
        if request.run_id and history and not request.tools:
            return "final_answer"
        return "unknown"

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
    def _explicit_output_tokens(prompt: str) -> int | None:
        """
        Read an explicit length instruction out of the prompt, in tokens.

        The task-type table below is flat: every creative_writing request is
        assumed to produce 1500 output tokens, so a haiku and a 500-word story
        are costed identically (and both ~10x over for the haiku). Since the
        cost estimate feeds model scoring, that flat figure biases short
        creative requests toward cheaper models for no reason. When the user
        states a length, believe them.

        Returns None when the prompt carries no length instruction.
        """
        # "write a 500-word story", "in about 250 words"
        m = _WORD_COUNT_RE.search(prompt)
        if m:
            # ~1.4 tokens per English word, +25% headroom for overshoot.
            return max(_MIN_OUTPUT_TOKENS, int(int(m.group(1)) * 1.4 * 1.25))

        m = _SENTENCE_COUNT_RE.search(prompt)
        if m:
            count = _NUMBER_WORDS.get(m.group(1).lower()) or int(m.group(1))
            return max(_MIN_OUTPUT_TOKENS, count * 30)

        m = _PARAGRAPH_COUNT_RE.search(prompt)
        if m:
            count = _NUMBER_WORDS.get(m.group(1).lower()) or int(m.group(1))
            return max(_MIN_OUTPUT_TOKENS, count * 120)

        # Fixed-length forms: a haiku is 17 syllables no matter the task type.
        for pattern, tokens in _FIXED_FORM_TOKENS.items():
            if pattern.search(prompt):
                return tokens

        return None

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
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") in ("image_url", "image") for b in content
            ):
                return "vision", 0.9

        # Function calling: explicit in metadata or capabilities
        if "function_calling" in request.required_capabilities or meta.get("tools"):
            return "function_calling", 0.9

        # Substantive legal/medical judgment calls: checked before
        # long_document/summarization/translation/extraction so a request
        # asking Flux to render a high-stakes judgment on supplied text is
        # still floored to DOMAIN_TIER_FLOORS, while a PURE transformation
        # (summarize/translate/extract/format supplied text) — which does not
        # match these judgment-seeking patterns — keeps its normal, cheaper
        # task_type. Checked against prompt + system prompt.
        sys_prompt_text = request.system_prompt or ""
        if _MEDICAL_SUBSTANTIVE_RE.search(prompt) or _MEDICAL_SUBSTANTIVE_RE.search(
            sys_prompt_text
        ):
            return "medical", 0.85
        if _LEGAL_SUBSTANTIVE_RE.search(prompt) or _LEGAL_SUBSTANTIVE_RE.search(sys_prompt_text):
            return "legal", 0.85

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

        # Code review: code fences + review/debug keywords, OR a known
        # defect-class term (race condition, deadlock, memory leak, …).
        has_fences = bool(_CODE_FENCE_RE.search(prompt))
        if has_fences and _CODE_REVIEW_RE.search(prompt):
            return "code_review", 0.8
        if _CODE_DEFECT_RE.search(prompt):
            return "code_review", 0.8

        # Structured-output extraction: "return JSON with X from <text>"
        if _EXTRACT_STRUCTURED_RE.search(prompt):
            return "extraction", 0.85

        # Long-form creative writing: "write a 900-word scene for a novel"
        if _CREATIVE_LONGFORM_RE.search(prompt):
            return "creative_writing", 0.85

        # Code generation: explicit patterns or language name + action verb
        if _CODE_GEN_RE.search(prompt) or (
            _LANG_NAME_RE.search(prompt)
            and re.search(r"\b(write|implement|build|create|generate)\b", prompt, re.IGNORECASE)
        ):
            return "code_generation", 0.8

        # Reasoning: proof / step-by-step / heavy math. "prove" as bare verb
        # is just as strong a signal as the noun "proof".
        if (_REASONING_RE.search(prompt) or _STEP_BY_STEP_RE.search(prompt)) and (
            _MATH_SYM_RE.search(prompt)
            or re.search(
                r"\b(prove|proof|theorem|derive|solve|irrational|rational|"
                r"converges?|diverges?)\b",
                prompt,
                re.IGNORECASE,
            )
        ):
            return "reasoning", 0.8

        # Analysis / comparison — checked before the extended reasoning
        # patterns below: "help me understand the tradeoffs between X and Y"
        # matches _REASONING_EXTENDED_RE's "help me understand", but its real
        # signal is the comparison, not open-ended reasoning. Comparison
        # prompts belong in the cheaper analysis bucket, not the premium
        # reasoning one — checking analysis first only redirects prompts that
        # ALSO carry an explicit analysis keyword, so plain reasoning prompts
        # ("help me figure out this proof") are unaffected.
        if _ANALYSIS_RE.search(prompt):
            return "analysis", 0.7

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
        word_count = len(prompt.split())
        # Math symbols are judged on the prose only — see _FENCED_BLOCK_RE.
        unfenced = _FENCED_BLOCK_RE.sub(" ", prompt)

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
            "math_symbols_present": bool(_MATH_SYM_RE.search(unfenced)),
            "high_code_fence_density": fences >= 2,
            "high_bullet_density": bullet_density > 0.25,
            "repetitive_templated": self._is_templated(prompt),
            "expected_short_response": word_count < 15 and question_count <= 1,
            # Within-task magnitude signals (see COMPLEXITY_MODIFIERS).
            "prompt_words_gt_60": word_count > 60,
            "prompt_words_gt_150": word_count > 150,
            "code_defect_class": bool(_CODE_DEFECT_RE.search(prompt)),
            "multi_constraint": len(_CONSTRAINT_RE.findall(prompt)) >= 2,
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
        override = request.metadata.get("sensitivity_level")
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
