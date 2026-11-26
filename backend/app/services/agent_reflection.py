"""Deterministic agent reflection (Topic 37).

Names accept / gather / revise / partial-finalize decisions using existing
evidence helpers. Never calls an LLM. AgentPlanning remains the step
orchestrator — reflection only advises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.agent_orchestration import (
    BLOCKED_TOOL_REASONS,
    is_premature_or_generic_answer,
    reconcile_answer_with_evidence_ledger,
    uncovered_required_targets,
    validate_final_answer_text,
)


# --- actions ---
REFLECT_ACCEPT = "ACCEPT"
REFLECT_GATHER_MORE = "GATHER_MORE"
REFLECT_REVISE = "REVISE"
REFLECT_FINALIZE_PARTIAL = "FINALIZE_PARTIAL"

REFLECT_ACTIONS = frozenset({
    REFLECT_ACCEPT,
    REFLECT_GATHER_MORE,
    REFLECT_REVISE,
    REFLECT_FINALIZE_PARTIAL,
})

# --- discrete tool-quality labels (not scores) ---
QUALITY_OK = "ok"
QUALITY_BLOCKED = "blocked_non_failure"
QUALITY_EMPTY = "empty"
QUALITY_SPAN_LIMITED = "span_limited"
QUALITY_NESTED_FAILURE = "nested_failure"

TOOL_QUALITY_LABELS = frozenset({
    QUALITY_OK,
    QUALITY_BLOCKED,
    QUALITY_EMPTY,
    QUALITY_SPAN_LIMITED,
    QUALITY_NESTED_FAILURE,
})


@dataclass
class ReflectionResult:
    """Typed reflection outcome for one answer or tool-result inspection."""

    action: str
    reason: str = ""
    premature: bool = False
    missing_targets: list[str] = field(default_factory=list)
    revised_answer: str | None = None
    tool_quality: str | None = None


def _has_ledger_targets(requirements: dict[str, Any] | None) -> bool:
    return bool((requirements or {}).get("required_targets"))


def _deterministic_revise(
    draft: str,
    *,
    requirements: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    pending_continuation: dict[str, Any] | None,
) -> tuple[str, bool]:
    """Validate (+ optional ledger reconcile). Returns (text, changed)."""
    cleaned = validate_final_answer_text(draft or "")
    if _has_ledger_targets(requirements) and evidence is not None:
        revised = reconcile_answer_with_evidence_ledger(
            cleaned,
            requirements=requirements,
            evidence=evidence,
            pending_continuation=pending_continuation,
        )
    else:
        revised = cleaned
    changed = revised != (draft or "").strip()
    return revised, changed


def reflect_on_answer(
    *,
    draft: str | None,
    requirements: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    evidence_snippets: list[str] | None,
    tools_enabled: bool,
    requirements_met: bool,
    pending_continuation: bool = False,
    can_continue_investigation: bool = False,
    successful_searches: int = 0,
    successful_reads: int = 0,
    pending_continuation_meta: dict[str, Any] | None = None,
) -> ReflectionResult:
    """Advise accept / gather / revise / partial-finalize for a model draft.

    Reuses orchestration helpers only. Does not call Groq and does not
    execute tools. Callers still use ``plan_next_action`` to orchestrate.
    """
    req = requirements or {}
    ev = evidence or {}
    snippets = list(evidence_snippets or [])
    text = (draft or "").strip()
    missing = list(uncovered_required_targets(req, ev)) if tools_enabled else []

    # Greetings / direct / RAG (tools off): preserve current non-investigation path.
    if not tools_enabled:
        revised, changed = _deterministic_revise(
            text,
            requirements=None,
            evidence=None,
            pending_continuation=None,
        )
        if changed:
            return ReflectionResult(
                action=REFLECT_REVISE,
                reason="direct_answer_validate_only",
                premature=False,
                missing_targets=[],
                revised_answer=revised,
            )
        return ReflectionResult(
            action=REFLECT_ACCEPT,
            reason="direct_or_rag_tools_disabled",
            premature=False,
            missing_targets=[],
            revised_answer=revised or None,
        )

    premature = is_premature_or_generic_answer(
        text,
        requirements=req,
        evidence_snippets=snippets,
    )

    needs_more_source = bool(missing) and (
        successful_searches + successful_reads >= 1
    )
    gather_signal = bool(
        pending_continuation
        or needs_more_source
        or (premature and snippets and not requirements_met)
    )

    if can_continue_investigation and gather_signal:
        return ReflectionResult(
            action=REFLECT_GATHER_MORE,
            reason=(
                "pending_continuation"
                if pending_continuation
                else (
                    "uncovered_required_targets"
                    if needs_more_source
                    else "premature_with_evidence"
                )
            ),
            premature=premature,
            missing_targets=missing,
        )

    # Match plan_next_action: only force partial finalization when evidence
    # exists and requirements remain incomplete / answer is premature.
    # Without snippets, preserve legacy answer_ready acceptance of the draft.
    if (not requirements_met or premature) and snippets:
        return ReflectionResult(
            action=REFLECT_FINALIZE_PARTIAL,
            reason="requirements_incomplete",
            premature=premature,
            missing_targets=missing,
        )

    # Requirements met (or no evidence to force offline) — cleanup / ledger.
    revised, changed = _deterministic_revise(
        text,
        requirements=req,
        evidence=ev,
        pending_continuation=pending_continuation_meta,
    )
    if changed or (
        _has_ledger_targets(req) and revised and revised != text
    ):
        return ReflectionResult(
            action=REFLECT_REVISE,
            reason="deterministic_ledger_or_validate",
            premature=False,
            missing_targets=missing,
            revised_answer=revised,
        )
    return ReflectionResult(
        action=REFLECT_ACCEPT,
        reason="answer_ready",
        premature=False,
        missing_targets=missing,
        revised_answer=revised or None,
    )


def reflect_on_tool_result(
    *,
    tool_name: str,
    tool_result: dict[str, Any] | None = None,
    blocked: bool = False,
    error_code: str | None = None,
    outer_ok: bool = True,
    nested_ok: bool = True,
    inner_payload: dict[str, Any] | None = None,
) -> ReflectionResult:
    """Classify tool-result quality. Does not alter execution or retries.

    Blocked / duplicate skips are ``blocked_non_failure`` — never failures.
    """
    del tool_name  # reserved for future per-tool labels
    payload = inner_payload if isinstance(inner_payload, dict) else {}
    result = tool_result if isinstance(tool_result, dict) else {}

    if (
        blocked
        or bool(result.get("blocked"))
        or bool(payload.get("blocked"))
        or (error_code in BLOCKED_TOOL_REASONS if error_code else False)
    ):
        return ReflectionResult(
            action=REFLECT_ACCEPT,
            reason="tool_quality_blocked_non_failure",
            tool_quality=QUALITY_BLOCKED,
        )

    if (not nested_ok) or (
        isinstance(payload, dict)
        and "success" in payload
        and not bool(payload.get("success"))
    ):
        return ReflectionResult(
            action=REFLECT_ACCEPT,
            reason="tool_quality_nested_failure",
            tool_quality=QUALITY_NESTED_FAILURE,
        )

    if not outer_ok:
        return ReflectionResult(
            action=REFLECT_ACCEPT,
            reason="tool_quality_nested_failure",
            tool_quality=QUALITY_NESTED_FAILURE,
        )

    if bool(payload.get("span_limited")):
        return ReflectionResult(
            action=REFLECT_ACCEPT,
            reason="tool_quality_span_limited",
            tool_quality=QUALITY_SPAN_LIMITED,
        )

    raw = result.get("result")
    empty = False
    if raw is None:
        empty = True
    elif isinstance(raw, str) and not raw.strip():
        empty = True
    elif isinstance(raw, dict) and not raw:
        empty = True
    if empty:
        return ReflectionResult(
            action=REFLECT_ACCEPT,
            reason="tool_quality_empty",
            tool_quality=QUALITY_EMPTY,
        )

    return ReflectionResult(
        action=REFLECT_ACCEPT,
        reason="tool_quality_ok",
        tool_quality=QUALITY_OK,
    )
