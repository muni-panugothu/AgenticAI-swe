"""Topic 41a — build trusted AgentTurnResult from existing agent outcomes.

The builder never authorizes tools, mutates B5/B6 state, or trusts LLM JSON
for citations or permissions.

Topic 42a SSE mapping (see ``app.services.sse_protocol``):
- meta ← response_type, completion_status, flags + REST citations
- token ← reply chunked from the completed turn (not live Groq tokens)
- done ← terminal success with completion_status
- error ← response_type=error / emit failures (never followed by done)
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from app.validation.agent_output_models import (
    AgentTurnResult,
    CitationRef,
    CompletionStatus,
    AgentResponseType,
)

# Exact abstention string used by POST /api/chat when RAG finds no candidates.
RAG_ABSTAIN_REPLY = (
    "I couldn't find authorized document evidence that answers that question."
)

_INCOMPLETE_MARKERS = (
    "not fully finished",
    "treat the answer above as partial",
)

_BUDGET_MARKERS = (
    "temporarily rate-limited",
    "agent_limit",
    "maximum agent steps",
    "tool execution budget",
)

_ABORT_MARKERS = (
    "generation stopped",
    "request aborted",
)


def citations_from_rag(
    raw_citations: Iterable[Mapping[str, Any]] | None,
) -> list[CitationRef]:
    """Convert server RAG citation dicts into trusted CitationRef models.

    Ignores malformed entries. Never invents citations from assistant text.
    """
    out: list[CitationRef] = []
    if not raw_citations:
        return out
    for item in raw_citations:
        if not isinstance(item, Mapping):
            continue
        doc_id = item.get("documentId") or item.get("document_id") or item.get("id")
        if not doc_id:
            continue
        try:
            out.append(
                CitationRef(
                    index=int(item.get("index", len(out))),
                    document_id=str(doc_id),
                    source=(
                        str(item["source"])
                        if item.get("source") is not None
                        else None
                    ),
                    section=(
                        str(item["section"])
                        if item.get("section") is not None
                        else None
                    ),
                    chunk_index=int(item.get("chunkIndex", item.get("chunk_index", 0))),
                    excerpt=str(item.get("excerpt") or "")[:4000],
                    score=float(item.get("score") or 0.0),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def build_answer(
    reply: str,
    *,
    citations: Iterable[Mapping[str, Any]] | None = None,
    uncertain: bool = False,
) -> AgentTurnResult:
    return AgentTurnResult(
        reply=reply if reply is not None else "",
        response_type="answer",
        completion_status="complete",
        uncertain=uncertain,
        clarification_required=False,
        partial=False,
        citations=citations_from_rag(citations),
    )


def build_clarification(reply: str) -> AgentTurnResult:
    text = reply if reply is not None else ""
    return AgentTurnResult(
        reply=text,
        response_type="clarification",
        completion_status="clarification_needed",
        uncertain=True,
        clarification_required=True,
        partial=False,
        citations=[],
    )


def build_abstain(
    reply: str | None = None,
    *,
    citations: Iterable[Mapping[str, Any]] | None = None,
) -> AgentTurnResult:
    text = RAG_ABSTAIN_REPLY if reply is None else reply
    return AgentTurnResult(
        reply=text,
        response_type="abstain",
        completion_status="complete",
        uncertain=False,
        clarification_required=False,
        partial=False,
        citations=citations_from_rag(citations),
    )


def build_partial(
    reply: str,
    *,
    citations: Iterable[Mapping[str, Any]] | None = None,
    uncertain: bool = True,
    completion_status: CompletionStatus = "partial",
) -> AgentTurnResult:
    return AgentTurnResult(
        reply=reply if reply is not None else "",
        response_type="partial",
        completion_status=completion_status,
        uncertain=uncertain,
        clarification_required=False,
        partial=True,
        citations=citations_from_rag(citations),
    )


def build_aborted(
    reply: str,
    *,
    error_code: str = "ABORTED",
) -> AgentTurnResult:
    return AgentTurnResult(
        reply=reply if reply is not None else "",
        response_type="partial",
        completion_status="aborted",
        uncertain=True,
        clarification_required=False,
        partial=True,
        citations=[],
        error_code=error_code,
        error_message_safe="The request was aborted before completion.",
    )


def build_error(
    reply: str,
    *,
    error_code: str,
    error_message_safe: str,
) -> AgentTurnResult:
    return AgentTurnResult(
        reply=reply if reply is not None else "",
        response_type="error",
        completion_status="failed",
        uncertain=False,
        clarification_required=False,
        partial=False,
        citations=[],
        error_code=error_code,
        error_message_safe=error_message_safe,
    )


def build_from_agent_reply(
    reply: str | None,
    *,
    citations: Iterable[Mapping[str, Any]] | None = None,
    clarification: bool = False,
    rag_abstain: bool = False,
    aborted: bool = False,
    budget_exhausted: bool = False,
    outcome_hint: str | None = None,
) -> AgentTurnResult:
    """Map an existing agent string reply into a trusted AgentTurnResult.

    Classification uses explicit caller hints first, then conservative
    reply-text markers (e.g. B6 incomplete-workflow note). Reply text is
    preserved exactly (empty string if None).
    """
    text = "" if reply is None else str(reply)
    cites = citations_from_rag(citations)
    hint = (outcome_hint or "").strip().lower()

    if clarification or hint == "clarification":
        return build_clarification(text)

    if rag_abstain or hint == "abstain" or text.strip() == RAG_ABSTAIN_REPLY:
        return build_abstain(text, citations=citations)

    if aborted or hint == "aborted":
        return build_aborted(text)

    if budget_exhausted or hint in {"budget", "budget_exhausted", "agent_limit"}:
        return build_partial(text, citations=citations, completion_status="partial")

    lowered = text.casefold()
    if any(m in lowered for m in _INCOMPLETE_MARKERS):
        return build_partial(text, citations=citations)

    if any(m in lowered for m in _ABORT_MARKERS):
        return build_aborted(text)

    if any(m in lowered for m in _BUDGET_MARKERS):
        return build_partial(text, citations=citations)

    if hint in {"partial", "safe_partial", "force_final"}:
        return build_partial(text, citations=citations)

    if hint in {"error", "failed"}:
        return build_error(
            text,
            error_code="AGENT_ERROR",
            error_message_safe="The agent could not complete this request.",
        )

    return build_answer(text, citations=citations)


def turn_result_to_chat_fields(result: AgentTurnResult) -> dict[str, Any]:
    """Map AgentTurnResult to existing ChatResponseBody field values.

    Does not add new public keys. Citations are serialized in the API's
    existing camelCase shape.
    """
    citations: list[dict[str, Any]] = [
        {
            "index": c.index,
            "documentId": c.document_id,
            "source": c.source,
            "section": c.section,
            "chunkIndex": c.chunk_index,
            "excerpt": c.excerpt,
            "score": c.score,
        }
        for c in result.citations
    ]
    return {
        "reply": result.reply,
        "citations": citations,
    }


def reject_untrusted_model_payload(payload: Mapping[str, Any] | None) -> None:
    """Security guard: untrusted LLM JSON must not drive authorization.

    Topic 41a does not parse model JSON into AgentTurnResult. This helper
    documents and enforces that callers never pass model-controlled identity
    or permission fields into the trusted builder.
    """
    if not payload:
        return
    forbidden = {
        "user_id",
        "tenant_id",
        "project_id",
        "tool_allowlist",
        "intent_tool_allowlist",
        "task_status",
        "b5_status",
        "b6_complete",
        "active_task_id",
        "permissions",
        "acl",
    }
    hits = forbidden.intersection(set(payload.keys()))
    if hits:
        raise ValueError(
            "untrusted model payload must not set authorization fields: "
            + ", ".join(sorted(hits))
        )
