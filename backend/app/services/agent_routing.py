"""Typed request-level agent routing (Topic 36).

Wraps existing intent classification and tool allowlisting without changing
behavior. Per-step ``resolve_tool_choice`` and memory short-circuits remain
in ``chat_with_ai`` — they are not request-level route selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.services.agent_orchestration import (
    INTENT_CALC,
    INTENT_CODEBASE,
    INTENT_CURRENT,
    INTENT_DEBUG,
    INTENT_GENERAL,
    INTENT_MEMORY,
    INTENT_TECHNICAL,
    INTENT_TOOL_ALLOWLIST,
    _targets_user_repo,
    classify_intent,
    intent_allows_tools,
    tools_for_intent,
)
from app.services.contextual_resolution import resolve_contextual_request
from app.services.input_normalization import normalize_user_input
from app.services.semantic_understanding import interpret_request
from app.services.task_decomposition import decompose_request
from app.validation.task_models import TaskDecomposition


# Deterministic route kinds (not specialist agents).
ROUTE_DIRECT = "direct"
ROUTE_INVESTIGATION = "investigation"
ROUTE_MEMORY = "memory"
ROUTE_RAG = "rag"
ROUTE_MULTI_TOOL = "multi_tool"
ROUTE_CALCULATION = "calculation"
ROUTE_CURRENT = "current"


@dataclass
class RouteDecision:
    """Result of request-level intent and tool eligibility routing."""

    intent: str
    tools_enabled: bool
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    route_kind: str = ROUTE_DIRECT
    # Deterministic confidence metadata (B2 semantic + legacy merge).
    confidence: str = "high"
    reason: str = ""
    fallback_applied: bool = False
    multi_tool: bool = False
    # Phase B1 — routing saw normalized text; original preserved for callers.
    original_message: str = ""
    normalized_message: str = ""
    normalization_transforms: list[str] = field(default_factory=list)
    language_hints: list[str] = field(default_factory=list)
    # Phase B2 — semantic interpretation (advisory; allowlists stay authoritative).
    semantic_intent: str = ""
    semantic_confidence: str = ""
    semantic_confidence_score: float = 0.0
    entities: list[dict[str, Any]] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    interpretation: dict[str, Any] = field(default_factory=dict)
    # Phase B3 — contextual follow-up resolution (advisory; separate confidence).
    is_follow_up: bool = False
    follow_up_type: str = "none"
    resolved_query: str = ""
    contextual_confidence: str = ""
    contextual_confidence_score: float = 0.0
    contextual_references: list[dict[str, Any]] = field(default_factory=list)
    contextual_resolution: dict[str, Any] = field(default_factory=dict)
    # Phase B4 — multi-intent task decomposition (advisory; does not authorize).
    is_multi_intent: bool = False
    task_count: int = 0
    task_decomposition: dict[str, Any] = field(default_factory=dict)
    decomposed_task_ids: list[str] = field(default_factory=list)
    decomposed_execution_order: list[str] = field(default_factory=list)


def _requires_calculator(text: str) -> bool:
    """Local mirror of ``chatbot.requires_calculator`` (avoids circular import)."""
    lowered = (text or "").lower()
    if re.search(r"\b(calculate|compute|calculator)\b", lowered):
        return True
    return bool(
        re.search(
            r"\d+\s*[\+\-\*\/x×]\s*\d+",
            lowered,
        )
    )


_EXTERNAL_LOOKUP_CUES = re.compile(
    r"(?i)\b("
    r"current|latest|recent|today|right now|as of|"
    r"chief minister|prime minister|president|governor|cm\b|"
    r"weather|forecast|news|headlines|"
    r"version of|(latest|current)\s+[\w.+#-]+\s+version|"
    r"[\w.+#-]+\s+version\b"
    r")\b"
)

_DOC_LOOKUP_CUES = re.compile(
    r"(?i)\b(documentation|docs|handbook|manual|readme|api reference)\b"
)

_CODE_DEBUG_CUES = re.compile(
    r"(?i)\b("
    r"bug|error|stack trace|traceback|exception|"
    r"in my (code|codebase|repo|project|file|backend|frontend)|"
    r"app/[\w./\\-]+|search_code|read_file"
    r")\b"
)


def _task_categories(decomposition: TaskDecomposition) -> set[str]:
    return {str(t.category) for t in decomposition.tasks}


def reconcile_execution_intent_with_tasks(
    *,
    text: str,
    intent: str,
    decomposition: TaskDecomposition,
) -> tuple[str, str | None]:
    """Adjust Find… false codebase routing using B4 categories as evidence.

    Does not grant tools by itself — caller still applies INTENT_TOOL_ALLOWLIST.
    Never overrides clear repo/debug targets.
    """
    cats = _task_categories(decomposition)
    lowered = text or ""

    # Repo / debug evidence wins — keep codebase/debug contracts.
    if _targets_user_repo(lowered) or _CODE_DEBUG_CUES.search(lowered):
        if intent in {INTENT_CODEBASE, INTENT_DEBUG}:
            return intent, None
        if "debugging" in cats or re.search(
            r"(?i)\bbug\b.{0,40}\b(code|repo|project|file)\b",
            lowered,
        ):
            return INTENT_CODEBASE, "b4_repo_debug_evidence"

    # Sentence-initial Find… often becomes CODEBASE; prefer CURRENT when B4
    # and wording indicate live/external facts (not a repo lookup).
    if intent == INTENT_CODEBASE and "research_current_information" in cats:
        if _EXTERNAL_LOOKUP_CUES.search(lowered) and not _targets_user_repo(
            lowered
        ):
            return INTENT_CURRENT, "b4_current_over_find_codebase"

    # Docs/product documentation lookup without repo targets → technical
    # (tool-free allowlist), not codebase investigation.
    if intent == INTENT_CODEBASE and not _targets_user_repo(lowered):
        doc_cats = cats & {"documentation", "rag_document", "explanation"}
        if (
            (_DOC_LOOKUP_CUES.search(lowered) or doc_cats)
            and "debugging" not in cats
            and not _CODE_DEBUG_CUES.search(lowered)
            and "research_current_information" not in cats
        ):
            return INTENT_TECHNICAL, "b4_docs_over_find_codebase"

    return intent, None


def _default_is_multi_tool_request(text: str) -> bool:
    """Local mirror of ``chatbot.is_multi_tool_request`` (avoids circular import).

    Production ``chat_with_ai`` passes ``multi_tool_fn=is_multi_tool_request``
    so the live chatbot predicate is used. This fallback keeps unit tests and
    non-chatbot callers aligned with the same rules.
    """
    lowered = (text or "").lower()
    signals = 0
    if re.search(
        r"\b(search|find|look\s+up|locate)\b",
        lowered,
    ) or "search_code" in lowered:
        signals += 1
    if (
        re.search(r"\bread_file\b", lowered)
        or re.search(
            r"\b(read|open|show|inspect)\b.{0,40}\b[\w./\\-]+\.(py|ts|js|tsx|jsx|md)\b",
            lowered,
        )
        or re.search(r"\bapp/[\w./\\-]+", lowered)
    ):
        signals += 1
    if _requires_calculator(text):
        signals += 1
    if re.search(r"\b(current\s+time|what\s+time|today'?s\s+date)\b", lowered):
        signals += 1
    return signals >= 2


def _route_kind_for_intent(intent: str, *, multi_tool: bool, rag: bool) -> str:
    if rag:
        return ROUTE_RAG
    if multi_tool:
        return ROUTE_MULTI_TOOL
    if intent == INTENT_MEMORY:
        return ROUTE_MEMORY
    if intent in {INTENT_CODEBASE, INTENT_DEBUG}:
        return ROUTE_INVESTIGATION
    if intent == INTENT_CALC:
        return ROUTE_CALCULATION
    if intent == INTENT_CURRENT:
        return ROUTE_CURRENT
    return ROUTE_DIRECT


def route_request(
    *,
    user_message: str | None,
    rag_context: Any = None,
    all_tool_definitions: list[dict[str, Any]] | None = None,
    multi_tool_fn: Callable[[str], bool] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> RouteDecision:
    """Select intent and eligible tools for one ``chat_with_ai`` request.

    Pipeline:
    1. B1 ``normalize_user_input``
    2. B3 ``resolve_contextual_request`` (follow-ups / references)
    3. B4 ``decompose_request`` (multi-intent understanding; advisory)
    4. B2 ``interpret_request`` on the trusted routing text
    5. Legacy ``classify_intent`` merged via ``map_semantic_to_execution``
    6. ``INTENT_TOOL_ALLOWLIST`` remains authoritative for tools

    ``rag_context`` still forces ``INTENT_TECHNICAL`` with tools disabled.
    Multi-tool requests keep the full tool catalog (same as chatbot today).

    Contextual/semantic/decomposition layers never bypass JWT/ACL/memory
    permissions and never invent tool arguments. History must already be
    user-scoped by the caller. B4 does not replace Topic 35 AgentPlanning.
    """
    catalog = list(all_tool_definitions or [])
    original = "" if user_message is None else str(user_message)
    # Phase B1: normalize before intent / multi-tool signals. Original is kept
    # on the decision for transcript/debug; normalization never authorizes tools.
    norm = normalize_user_input(original)
    text = norm.normalized
    degenerate = not original.strip()
    multi_fn = multi_tool_fn or _default_is_multi_tool_request

    # Phase B3: resolve follow-ups against caller-supplied history only.
    contextual = resolve_contextual_request(
        original,
        conversation_history=conversation_history,
        normalized_message=text,
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
    )
    # Intent/tool gating uses trusted routing_query only — never raw historical
    # assistant/code/document payloads (those stay in resolved_query boundaries).
    routing_text = text
    if (
        contextual.is_follow_up
        and not (
            contextual.needs_clarification
            and contextual.contextual_confidence == "low"
        )
    ):
        routing_text = (
            contextual.routing_query
            or contextual.current_user_instruction
            or contextual.resolved_query
            or text
        )

    # Phase B4: decompose trusted current-user instruction only.
    trusted_for_tasks = (
        contextual.current_user_instruction
        or text
        or original
    )
    decomposition = decompose_request(
        original,
        normalized_message=text,
        trusted_instruction=trusted_for_tasks,
        contextual_resolution=contextual.model_dump(),
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
    )
    decomp_dict = decomposition.model_dump()

    interpretation = interpret_request(
        original,
        normalized_message=routing_text,
        rag_context=rag_context,
    )
    interp_dict = interpretation.model_dump()
    # Attach primary intent from B2 when available (advisory label only).
    if interpretation.semantic_intent:
        decomposition.primary_intent = str(interpretation.semantic_intent)
        decomp_dict = decomposition.model_dump()

    # Merge clarification: contextual low-confidence references win when set.
    # When B3 confidently resolves a follow-up, do not let B2 ambiguity on the
    # short instruction force a clarification that undoes resolution.
    needs_clarification = bool(interpretation.needs_clarification)
    clarification_prompt = interpretation.clarification_prompt
    if contextual.needs_clarification and contextual.clarification_prompt:
        needs_clarification = True
        clarification_prompt = contextual.clarification_prompt
    elif (
        contextual.is_follow_up
        and not contextual.needs_clarification
        and contextual.contextual_confidence in {"high", "medium"}
    ):
        needs_clarification = False
        clarification_prompt = None
    # B4: surface missing multi-task details without wiping understood tasks.
    # Do not override a B3 low-confidence reference clarification.
    if (
        decomposition.needs_clarification
        and decomposition.clarification_prompts
        and not (
            contextual.needs_clarification
            and contextual.contextual_confidence == "low"
        )
    ):
        needs_clarification = True
        clarification_prompt = decomposition.clarification_prompts[0]

    semantic_meta = {
        "original_message": original,
        "normalized_message": text,
        "normalization_transforms": list(norm.transforms),
        "language_hints": list(norm.language_hints),
        "semantic_intent": interpretation.semantic_intent,
        "semantic_confidence": interpretation.confidence,
        "semantic_confidence_score": float(interpretation.confidence_score),
        "entities": [e.model_dump() for e in interpretation.entities],
        "needs_clarification": needs_clarification,
        "clarification_prompt": clarification_prompt,
        "interpretation": interp_dict,
        "is_follow_up": bool(contextual.is_follow_up),
        "follow_up_type": str(contextual.follow_up_type),
        "resolved_query": str(contextual.resolved_query or routing_text),
        "contextual_confidence": str(contextual.contextual_confidence),
        "contextual_confidence_score": float(contextual.confidence_score),
        "contextual_references": [r.model_dump() for r in contextual.references],
        "contextual_resolution": contextual.model_dump(),
        "is_multi_intent": bool(decomposition.is_multi_intent),
        "task_count": len(decomposition.tasks),
        "task_decomposition": decomp_dict,
        "decomposed_task_ids": [t.id for t in decomposition.tasks],
        "decomposed_execution_order": list(decomposition.execution_order),
    }

    # RAG evidence answers: document-only; tools and personal memory off.
    if rag_context:
        return RouteDecision(
            intent=INTENT_TECHNICAL,
            tools_enabled=False,
            tool_definitions=[],
            route_kind=ROUTE_RAG,
            confidence="high",
            reason="rag_context_forces_technical_no_tools",
            fallback_applied=False,
            multi_tool=False,
            **semantic_meta,
        )

    legacy_intent = classify_intent(routing_text)
    intent = interpretation.execution_intent or legacy_intent
    # B4 categories as supporting evidence for Find… / multi-intent misroutes.
    # Still does not authorize tools — allowlist applied below.
    intent, b4_intent_note = reconcile_execution_intent_with_tasks(
        text=routing_text,
        intent=intent,
        decomposition=decomposition,
    )
    # Empty/whitespace never qualifies as multi-tool (matches chatbot signals).
    multi = bool(multi_fn(routing_text)) and not degenerate
    # Semantic multi-step is advisory; multi_tool_fn remains the gate.
    if (
        not multi
        and interpretation.semantic_intent == "multi_step"
        and interpretation.confidence == "high"
    ):
        # Do not force full catalog unless the deterministic multi-tool
        # predicate agrees — preserves Topic 36 behavior.
        pass

    if multi:
        tool_defs = list(catalog)
        tools_on = True
        reason = "multi_tool_keeps_full_catalog"
    else:
        tool_defs = tools_for_intent(intent, catalog)
        tools_on = intent_allows_tools(intent) and bool(tool_defs)
        reason = f"intent_allowlist:{intent}"
    reason = (
        f"{reason}|semantic:{interpretation.semantic_intent}"
        f"|conf:{interpretation.confidence}"
    )
    if contextual.is_follow_up:
        reason = (
            f"{reason}|follow_up:{contextual.follow_up_type}"
            f"|ctx_conf:{contextual.contextual_confidence}"
        )
    if decomposition.is_multi_intent:
        reason = f"{reason}|b4_tasks:{len(decomposition.tasks)}"
    if b4_intent_note:
        reason = f"{reason}|{b4_intent_note}"
    if norm.transforms:
        reason = f"{reason}|normalized"
    if intent != legacy_intent:
        reason = f"{reason}|legacy:{legacy_intent}"

    # Topic 36 compatibility: route ``confidence`` stays high for resolved
    # non-empty requests. B2 uncertainty lives in ``semantic_confidence``.
    route_confidence = "low" if degenerate else "high"

    if degenerate:
        return RouteDecision(
            intent=intent or INTENT_GENERAL,
            tools_enabled=tools_on,
            tool_definitions=tool_defs,
            route_kind=ROUTE_DIRECT,
            confidence=route_confidence,
            reason="empty_or_whitespace_input",
            fallback_applied=True,
            multi_tool=False,
            **semantic_meta,
        )

    return RouteDecision(
        intent=intent,
        tools_enabled=tools_on,
        tool_definitions=tool_defs,
        route_kind=_route_kind_for_intent(
            intent, multi_tool=multi, rag=False
        ),
        confidence=route_confidence,
        reason=reason,
        fallback_applied=False,
        multi_tool=multi,
        **semantic_meta,
    )


def memory_tools_in_definitions(
    tool_definitions: list[dict[str, Any]] | None,
) -> set[str]:
    """Return memory tool names present in a tool-definition list."""
    memory_allow = INTENT_TOOL_ALLOWLIST.get(INTENT_MEMORY) or frozenset()
    names: set[str] = set()
    for item in tool_definitions or []:
        name = (item.get("function") or {}).get("name")
        if name in memory_allow:
            names.add(str(name))
    return names
