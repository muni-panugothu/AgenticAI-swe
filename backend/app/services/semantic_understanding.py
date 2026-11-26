"""Phase B2 — deterministic semantic intent + entity understanding.

No provider calls. Builds on B1 normalization. Produces a validated
``InterpretedRequest`` that advises routing; ``INTENT_TOOL_ALLOWLIST`` remains
authoritative for tool eligibility. Never invents tool arguments or entities.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.agent_orchestration import (
    INTENT_CALC,
    INTENT_CODEBASE,
    INTENT_CURRENT,
    INTENT_DEBUG,
    INTENT_GENERAL,
    INTENT_MEMORY,
    INTENT_TECHNICAL,
    classify_intent,
)
from app.services.input_normalization import (
    CITY_ALIASES,
    PLACE_ALIASES,
    normalize_user_input,
)
from app.validation.semantic_models import (
    ExtractedEntity,
    InterpretedRequest,
    SemanticIntent,
)

# ---------------------------------------------------------------------------
# Lexicons (extensible cues — not fixed question dictionaries)
# ---------------------------------------------------------------------------

PROGRAMMING_LANGUAGES = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "golang",
        "rust",
        "c++",
        "cpp",
        "c#",
        "csharp",
        "ruby",
        "php",
        "kotlin",
        "swift",
        "scala",
        "r",
        "sql",
        "bash",
        "shell",
    }
)

FRAMEWORKS = frozenset(
    {
        "react",
        "vue",
        "angular",
        "next.js",
        "nextjs",
        "fastapi",
        "django",
        "flask",
        "express",
        "nestjs",
        "spring",
        "rails",
        "laravel",
        "tailwind",
        "pytorch",
        "tensorflow",
        "langchain",
        "langgraph",
    }
)

TECHNOLOGIES = frozenset(
    {
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "redis",
        "mongodb",
        "postgres",
        "postgresql",
        "mysql",
        "graphql",
        "rest",
        "jwt",
        "oauth",
        "oauth2",
        "rag",
        "llm",
        "embedding",
        "vector",
        "kafka",
        "nginx",
        "linux",
        "git",
        "ci",
        "cd",
    }
)

# Cue weights: (pattern, weight) per semantic intent.
# Patterns are searched case-insensitively against normalized text.
_INTENT_CUES: dict[str, list[tuple[str, float]]] = {
    "current_information": [
        (r"\b(current|currently|right now|as of today|latest|live|today'?s)\b", 1.4),
        (r"\b(weather|forecast|exchange rate|stock price|news|headlines)\b", 2.0),
        (
            r"\b(usd|inr|eur|gbp|bitcoin|btc).{0,24}(price|rate|exchange)\b",
            2.2,
        ),
        (r"\b(chief minister|prime minister|president|governor|cm of)\b", 2.2),
        (r"\b(who is the current|who(?:'s| is) currently)\b", 2.0),
        (r"\b(search the web|look up online)\b", 1.8),
    ],
    "mathematics": [
        (r"\b(calculate|compute|calculator|average|mean|sum of|total of)\b", 2.0),
        (r"\d+\s*[\+\-\*/x×÷]\s*\d+", 2.4),
        (r"\b(percentage|percent|ratio|equation)\b", 1.2),
    ],
    "debugging": [
        (r"\b(debug|traceback|stack trace|exception|typeerror|assertionerror)\b", 2.4),
        (r"\b(bug|failing|broken|crash(?:es|ed|ing)?|error in)\b", 1.8),
        (r"\b(why (is|does|did) .{0,40} (fail|error|crash))\b", 2.0),
        (r"\b(fix|diagnose)\b.{0,30}\b(bug|error|issue|failure)\b", 2.0),
    ],
    "code_review": [
        (r"\b(review|code review|race condition|security review)\b", 2.0),
        (r"\b(critique|feedback on).{0,20}\b(code|pr|pull request|logic)\b", 1.8),
    ],
    "code_generation": [
        (r"\b(create|build|generate|write|implement|scaffold|make)\b."
         r"{0,40}\b(function|class|component|endpoint|api|page|script|module)\b", 2.2),
        (r"\b(create|build|generate|write|implement)\b."
         r"{0,40}\b(react|vue|angular|fastapi|django|flask|python|typescript)\b", 2.4),
        (r"\b(unit tests?|test cases?)\b", 1.2),
        (r"\busing (react|vue|fastapi|django|tailwind)\b", 1.6),
    ],
    "writing_documentation": [
        (r"\b(draft|write|document|documentation|readme|release notes|blog|api docs)\b", 2.0),
        (r"\b(in markdown|changelog|user guide|tutorial)\b", 1.6),
    ],
    "summarization": [
        (r"\b(summarize|summary|tl;dr|tldr|in \d+ bullets?|condense)\b", 2.4),
        (r"\b(key points|brief overview)\b", 1.4),
    ],
    "data_analysis": [
        (r"\b(analyze|analysis|dataset|csv|dataframe|correlation|insight)\b", 1.8),
        (r"\b(average|median|percentile|distribution)\b", 1.4),
        (r"\b(plot|chart|trend)\b", 1.2),
    ],
    "rag_document": [
        (r"\b(according to|based on)\b", 1.8),
        (r"\b(in (the|my) (document|doc|handbook|policy|pdf)|leave policy|handbook)\b", 2.2),
        (r"\b(uploaded|my documents?|knowledge base|from the file)\b", 2.0),
        (r"\b(what does (the|my) .{0,20}(say|state|mention))\b", 1.6),
    ],
    "personal_memory": [
        (
            r"\b("
            r"remember that|forget that|"
            r"search my memor(?:y|ies)|"
            r"what do you remember|show my preferences|"
            r"my memor(?:y|ies)"
            r")\b",
            2.6,
        ),
        (r"\b(my preferred|do i prefer|saved (fact|preference|memory))\b", 2.2),
        (r"\b(from memor(?:y|ies)|in my memor(?:y|ies))\b", 2.0),
    ],
    "technical_question": [
        (r"\b(what is|what's|explain|how does|how do|difference between|why does)\b", 1.6),
        (r"\b(concept|architecture|design pattern|algorithm|protocol)\b", 1.4),
        (r"\b(oauth|jwt|rag|acl|dependency injection|rest api)\b", 1.2),
    ],
    "general_knowledge": [
        (r"\b(who invented|when was|capital of|population of)\b", 1.8),
        (r"\b(history of|famous for)\b", 1.2),
    ],
    "creative": [
        (r"\b(poem|story|joke|song|brainstorm|creative|imagine)\b", 2.2),
        (r"\b(rewrite in (a )?(fun|witty|poetic) (way|style))\b", 1.6),
    ],
    "follow_up": [
        (r"^(and|also|what about|how about|the second|that one|continue)\b", 1.8),
        (r"\b(as (i|we) (said|mentioned)|based on (that|your) (answer|reply))\b", 1.6),
    ],
    "tool_execution": [
        (r"\b(use (the )?(tool|calculator|web_search|search_code|read_file))\b", 2.4),
        (r"\b(call|run|execute)\b.{0,20}\b(tool|function|calculator)\b", 1.8),
    ],
    "multi_step": [
        (r"\b(and then|then also|after that|as well as)\b", 1.4),
        (r"\b(both|first .{0,20} then)\b", 1.2),
    ],
    "ambiguous": [
        (r"^(help|do something|fix it|handle this)\s*[.?!]*$", 2.0),
        (r"\b(not sure|maybe|whatever|something)\b", 1.2),
        (r"^(hi|hello|hey)\s*[.?!]*$", 0.5),
    ],
    "unsupported": [
        (r"\b(hack into|bypass (auth|security)|steal|malware)\b", 2.5),
        (
            r"\b("
            r"ignore (all )?(prior|previous|safety) (rules|instructions|labels)|"
            r"memory dump|dump all memor(?:y|ies)|"
            r"reveal .{0,20}(secrets?|jwt|password)"
            r")\b",
            2.2,
        ),
    ],
}

_SEMANTIC_TO_EXECUTION: dict[str, str] = {
    "general_knowledge": INTENT_TECHNICAL,
    "current_information": INTENT_CURRENT,
    "technical_question": INTENT_TECHNICAL,
    "code_generation": INTENT_TECHNICAL,
    "debugging": INTENT_DEBUG,
    "code_review": INTENT_TECHNICAL,
    "mathematics": INTENT_CALC,
    "writing_documentation": INTENT_TECHNICAL,
    "summarization": INTENT_TECHNICAL,
    "data_analysis": INTENT_TECHNICAL,
    "rag_document": INTENT_TECHNICAL,
    "personal_memory": INTENT_MEMORY,
    "tool_execution": INTENT_TECHNICAL,
    "multi_step": INTENT_TECHNICAL,
    "creative": INTENT_GENERAL,
    "follow_up": INTENT_GENERAL,
    "ambiguous": INTENT_GENERAL,
    "unsupported": INTENT_GENERAL,
}

_TOOL_EXPECTATIONS: dict[str, list[str]] = {
    "current_information": ["web_search", "get_current_time"],
    "mathematics": ["calculator"],
    "debugging": ["search_code", "read_file"],
    "code_generation": [],
    "code_review": [],
    "technical_question": [],
    "writing_documentation": [],
    "summarization": [],
    "personal_memory": ["search_memory"],
    "rag_document": [],
}


def _score_intents(text: str) -> dict[str, float]:
    scores: dict[str, float] = {k: 0.0 for k in _INTENT_CUES}
    for intent, cues in _INTENT_CUES.items():
        for pattern, weight in cues:
            if re.search(pattern, text, flags=re.IGNORECASE):
                scores[intent] += weight
    # Multi-step boost when multiple tool families appear.
    tool_family = 0
    if re.search(r"\b(search|find|locate|read_file|app/)\b", text, re.I):
        tool_family += 1
    if re.search(r"\b(calculate|compute|\d\s*[\+\-\*/])\b", text, re.I):
        tool_family += 1
    if re.search(r"\b(weather|web|news|current)\b", text, re.I):
        tool_family += 1
    if tool_family >= 2:
        scores["multi_step"] += 2.0
    return scores


def _rank_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract entities that are literally present — never invent values."""
    if not text:
        return []
    entities: list[ExtractedEntity] = []
    seen: set[tuple[str, str]] = set()

    def add(etype: str, value: str, span: str, start: int | None, end: int | None) -> None:
        key = (etype, value.casefold())
        if key in seen or not value.strip():
            return
        seen.add(key)
        entities.append(
            ExtractedEntity(
                type=etype,  # type: ignore[arg-type]
                value=value,
                source_span=span,
                start=start,
                end=end,
            )
        )

    # Files / paths
    for m in re.finditer(
        r"\b(?:app|backend|frontend|src|tests?)/[\w./\\-]+\.\w+\b"
        r"|\b[\w.-]+\.(?:py|ts|tsx|js|jsx|md|json|ya?ml|toml|css|html|pdf|docx)\b",
        text,
        flags=re.IGNORECASE,
    ):
        add("file", m.group(0), m.group(0), m.start(), m.end())

    # Numbers / quantities
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        add("number", m.group(0), m.group(0), m.start(), m.end())

    # Dates / times (conservative)
    for m in re.finditer(
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
        r"today|tomorrow|yesterday|current time)\b",
        text,
        flags=re.IGNORECASE,
    ):
        add("date_time", m.group(0), m.group(0), m.start(), m.end())

    # Places from known aliases appearing as whole words / phrases
    lowered = text.casefold()
    for alias, canon in sorted(PLACE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in {"up", "uk", "in", "or", "me"}:
            # Only when longer political context present.
            if not re.search(r"\b(chief minister|cm|state|capital)\b", lowered):
                continue
        pattern = re.compile(rf"(?i)(?<!\w){re.escape(alias)}(?!\w)")
        m = pattern.search(text)
        if m:
            add("place", canon, m.group(0), m.start(), m.end())
    for alias, canon in CITY_ALIASES.items():
        m = re.search(rf"(?i)(?<!\w){re.escape(alias)}(?!\w)", text)
        if m:
            add("place", canon, m.group(0), m.start(), m.end())
        m2 = re.search(rf"(?i)(?<!\w){re.escape(canon)}(?!\w)", text)
        if m2:
            add("place", canon, m2.group(0), m2.start(), m2.end())

    # Technologies / languages / frameworks (literal tokens only)
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9.+#\-]*", text))
    token_cf = {t.casefold(): t for t in tokens}
    for lang in PROGRAMMING_LANGUAGES:
        if lang in token_cf:
            raw = token_cf[lang]
            idx = text.casefold().find(lang)
            add("programming_language", lang, raw, idx if idx >= 0 else None, None)
    for fw in FRAMEWORKS:
        # allow next.js style
        compact = fw.replace(".", "")
        if fw in token_cf or compact in token_cf:
            raw = token_cf.get(fw) or token_cf.get(compact) or fw
            idx = text.casefold().find(fw.casefold())
            add("framework", fw, raw, idx if idx >= 0 else None, None)
    for tech in TECHNOLOGIES:
        if tech in token_cf:
            raw = token_cf[tech]
            idx = text.casefold().find(tech)
            add("technology", tech, raw, idx if idx >= 0 else None, None)

    # Project references
    for m in re.finditer(r"(?i)\b(?:project|repo|repository)\s+([A-Za-z0-9_\-]+)\b", text):
        add("project", m.group(1), m.group(0), m.start(), m.end())

    # Organizations (light): known uppercase tokens 2+ letters if ALL CAPS uncommon — skip inventing.
    # People: "Chief Minister of X" does not invent the person name.

    # Documents
    for m in re.finditer(
        r"(?i)\b(handbook|policy|leave policy|readme|document|pdf)\b",
        text,
    ):
        add("document", m.group(0), m.group(0), m.start(), m.end())

    return entities


def _confidence_from_scores(
    ranked: list[tuple[str, float]],
) -> tuple[str, float, str | None]:
    if not ranked:
        return "low", 0.0, "no_signals"
    top_name, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top - second
    # Normalize score to 0..1 for reporting (cap at 6.0 cues).
    score01 = max(0.0, min(1.0, top / 6.0))
    if top < 1.0:
        return "low", score01, "weak_signals"
    if margin < 0.8 and second >= 1.0:
        return "low", score01, f"competing:{ranked[0][0]}|{ranked[1][0]}"
    if top >= 3.0 and margin >= 1.2:
        return "high", score01, None
    if top >= 2.0 and margin >= 0.8:
        return "high", score01, None
    if top >= 1.5:
        return "medium", score01, None
    return "low", score01, "insufficient_margin"


def _requested_output(text: str, intent: str) -> str:
    if intent == "summarization":
        return "summary"
    if intent == "code_generation":
        return "code"
    if intent == "mathematics":
        return "numeric_result"
    if intent == "writing_documentation":
        return "document"
    if intent in {"debugging", "code_review"}:
        return "analysis"
    if intent == "current_information":
        return "fresh_answer"
    if re.search(r"(?i)\b(yes or no|true or false)\b", text):
        return "boolean"
    if re.search(r"(?i)\bjson\b", text):
        return "json"
    return "prose"


def _user_goal(text: str, intent: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) > 160:
        cleaned = cleaned[:157] + "..."
    return cleaned or intent


def _clarification_prompt(
    *,
    intent: str,
    ambiguity_reason: str | None,
    text: str,
) -> str | None:
    if intent == "unsupported":
        return (
            "I can’t help with that request as phrased. "
            "Please restate a legitimate goal (for example: explain a concept, "
            "debug code, or look up current public information)."
        )
    if intent == "ambiguous" or (ambiguity_reason and ambiguity_reason.startswith("competing")):
        return (
            "I want to make sure I help correctly. "
            "Could you clarify whether you want an explanation, code, "
            "a calculation, a document lookup, or current/live information?"
        )
    if ambiguity_reason == "weak_signals" and len(text.split()) <= 4:
        return (
            "Could you add a bit more detail about what you need "
            "(goal, topic, or desired output)?"
        )
    return None


def map_semantic_to_execution(
    semantic_intent: str,
    *,
    legacy_intent: str,
    confidence: str,
    requires_fresh_info: bool,
) -> str:
    """Merge semantic label with legacy classify_intent for safe execution.

    Safety rules:
    - Never downgrade CURRENT or MEMORY when legacy detected them.
    - High-confidence semantic labels may refine GENERAL → TECHNICAL/DEBUG/CALC.
    - Tool allowlists still come from execution intent only.
    """
    # Preserve authoritative safety-sensitive legacy routes.
    if legacy_intent == INTENT_CURRENT or requires_fresh_info:
        return INTENT_CURRENT
    if legacy_intent == INTENT_MEMORY:
        return INTENT_MEMORY
    if legacy_intent == INTENT_CODEBASE:
        return INTENT_CODEBASE
    if legacy_intent == INTENT_DEBUG:
        return INTENT_DEBUG
    if legacy_intent == INTENT_CALC:
        return INTENT_CALC

    mapped = _SEMANTIC_TO_EXECUTION.get(semantic_intent, INTENT_GENERAL)

    # Memory via semantic alone: only when high confidence (patterns are strong).
    if mapped == INTENT_MEMORY and confidence != "high":
        return legacy_intent or INTENT_GENERAL

    if confidence == "high":
        return mapped
    if confidence == "medium" and mapped in {
        INTENT_TECHNICAL,
        INTENT_CALC,
        INTENT_DEBUG,
        INTENT_GENERAL,
    }:
        # Prefer semantic refinement over bare GENERAL.
        if legacy_intent == INTENT_GENERAL and mapped != INTENT_GENERAL:
            return mapped
        return legacy_intent or mapped
    # Low confidence → legacy remains authoritative.
    return legacy_intent or INTENT_GENERAL


def interpret_request(
    message: str | None,
    *,
    normalized_message: str | None = None,
    rag_context: Any = None,
) -> InterpretedRequest:
    """Interpret one user message into a validated semantic structure."""
    original = "" if message is None else str(message)
    if normalized_message is None:
        norm = normalize_user_input(original)
        normalized = norm.normalized
    else:
        normalized = str(normalized_message)

    if rag_context:
        return InterpretedRequest(
            original_message=original,
            normalized_message=normalized,
            semantic_intent="rag_document",
            user_goal=_user_goal(normalized, "rag_document"),
            entities=extract_entities(normalized),
            requested_output="grounded_answer",
            context_dependent=True,
            requires_fresh_info=False,
            expected_tools=[],
            execution_intent=INTENT_TECHNICAL,
            confidence="high",
            confidence_score=1.0,
            signals=["rag_context_present"],
        )

    if not original.strip():
        return InterpretedRequest(
            original_message=original,
            normalized_message=normalized,
            semantic_intent="ambiguous",
            user_goal="",
            confidence="low",
            confidence_score=0.0,
            ambiguity_reason="empty_input",
            needs_clarification=True,
            clarification_prompt="Please enter a message describing what you need.",
            execution_intent=INTENT_GENERAL,
            signals=["empty"],
        )

    text = normalized or original
    scores = _score_intents(text)
    ranked = _rank_scores(scores)
    top_intent, top_score = ranked[0]
    secondary = [name for name, sc in ranked[1:4] if sc >= 1.0 and name != top_intent]

    # If top is zero, fall through via legacy-shaped defaults (no invented cues).
    if top_score <= 0:
        if re.match(
            r"(?i)^\s*((hi|hello|hey)([,!]|\s)+)?"
            r"(how are you( doing)?|good (morning|afternoon|evening)|"
            r"thanks|thank you|bye|goodbye|hi|hello|hey)"
            r"[\s.!?]*$",
            text,
        ):
            top_intent = "creative"  # treated as casual/general via mapping
        elif re.search(r"(?i)\b(what|how|why|explain|difference)\b", text):
            top_intent = "technical_question"
        else:
            top_intent = "ambiguous"

    confidence, score01, ambiguity_reason = _confidence_from_scores(ranked)
    if top_score <= 0:
        confidence, score01, ambiguity_reason = "low", 0.0, "no_cue_match"

    # Prefer document-grounded intent when policy/handbook cues compete with "what is".
    if (
        scores.get("rag_document", 0) >= 2.0
        and scores.get("rag_document", 0) + 0.5 >= top_score
    ):
        top_intent = "rag_document"
        top_score = scores["rag_document"]
        secondary = [n for n, s in ranked if n != "rag_document" and s >= 1.0][:3]

    # Multi-step as secondary shouldn't win over a strong primary unless dominant.
    semantic_intent: SemanticIntent = top_intent  # type: ignore[assignment]
    if top_intent == "multi_step" and secondary:
        semantic_intent = secondary[0]  # type: ignore[assignment]
        secondary = ["multi_step", *secondary[1:]]

    requires_fresh = semantic_intent == "current_information" or bool(
        re.search(
            r"(?i)\b(current|latest|right now|live|weather|exchange rate)\b",
            text,
        )
    )

    legacy = classify_intent(text)
    execution = map_semantic_to_execution(
        semantic_intent,
        legacy_intent=legacy,
        confidence=confidence,
        requires_fresh_info=requires_fresh and semantic_intent == "current_information",
    )

    # Agreement boosts confidence.
    mapped_for_legacy = _SEMANTIC_TO_EXECUTION.get(semantic_intent)
    signals = [f"semantic:{semantic_intent}", f"legacy:{legacy}"]
    if mapped_for_legacy == legacy and confidence != "low":
        confidence = "high"
        score01 = max(score01, 0.75)
        signals.append("semantic_legacy_agree")

    entities = extract_entities(text)
    constraints: list[str] = []
    if re.search(r"(?i)\b(simple words|eli5|briefly|concise|short)\b", text):
        constraints.append("simple_language")
    if re.search(r"(?i)\b(json only|yes or no|exactly \d+)\b", text):
        constraints.append("strict_format")

    needs_clarification = False
    unsupported = semantic_intent == "unsupported" and top_score >= 2.0
    if unsupported:
        needs_clarification = True
        confidence = "high"
    elif semantic_intent == "ambiguous":
        needs_clarification = True
    elif confidence == "low" and semantic_intent in {"follow_up"}:
        needs_clarification = True
    elif ambiguity_reason and ambiguity_reason.startswith("competing") and top_score < 3.0:
        needs_clarification = True
        semantic_intent = "ambiguous"  # type: ignore[assignment]

    clarification = _clarification_prompt(
        intent="unsupported" if unsupported else semantic_intent,
        ambiguity_reason=ambiguity_reason,
        text=text,
    )
    if not needs_clarification:
        clarification = None

    context_dependent = semantic_intent == "follow_up" or bool(
        re.search(r"(?i)\b(that|this|previous|above|earlier)\b", text)
    ) and len(text.split()) <= 8

    expected_tools = list(_TOOL_EXPECTATIONS.get(semantic_intent, []))
    if "multi_step" in secondary or semantic_intent == "multi_step":
        # Expected tools are informational only — allowlist still decides.
        expected_tools = list(
            dict.fromkeys(expected_tools + ["search_code", "calculator", "web_search"])
        )

    breakdown = {k: round(v, 3) for k, v in ranked if v > 0}

    return InterpretedRequest(
        original_message=original,
        normalized_message=text,
        semantic_intent=semantic_intent,
        secondary_intents=secondary[:3],  # type: ignore[arg-type]
        user_goal=_user_goal(text, semantic_intent),
        entities=entities,
        constraints=constraints,
        requested_output=_requested_output(text, semantic_intent),
        context_dependent=context_dependent,
        requires_fresh_info=requires_fresh,
        expected_tools=expected_tools,
        execution_intent=execution,
        confidence=confidence,  # type: ignore[arg-type]
        confidence_score=round(score01, 4),
        ambiguity_reason=ambiguity_reason,
        needs_clarification=needs_clarification,
        clarification_prompt=clarification,
        unsupported=unsupported,
        score_breakdown=breakdown,
        signals=signals,
    )
