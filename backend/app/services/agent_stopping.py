"""Typed agent-loop stopping façade (Topic 40).

Names existing outcomes without replacing Planning. ``plan_next_action``
remains the sole orchestrator; this module only maps authoritative
``PlanDecision`` values (and a few non-plan terminal outcomes already
enforced in ``chat_with_ai``) into a ``StopDecision``.

Does not introduce new stopping predicates, budgets, or LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.agent_planning import (
    ACTION_CALL_LLM,
    ACTION_CONTINUE_WITH_NUDGE,
    ACTION_FINALIZE,
    ACTION_FORCE_CONTINUATION,
    ACTION_FORCE_TARGET_READ,
    PlanDecision,
)


# --- stop actions ---
STOP_CONTINUE_LLM = "CONTINUE_LLM"
STOP_FORCE_CONTINUATION = "FORCE_CONTINUATION"
STOP_FORCE_TARGET_READ = "FORCE_TARGET_READ"
STOP_CONTINUE_NUDGE = "CONTINUE_NUDGE"
STOP_FINALIZE_COMPLETE = "FINALIZE_COMPLETE"
STOP_FINALIZE_PARTIAL = "FINALIZE_PARTIAL"
STOP_REPEATED_FAILURE = "STOP_REPEATED_FAILURE"
STOP_ABORT = "ABORT"

STOP_ACTIONS = frozenset({
    STOP_CONTINUE_LLM,
    STOP_FORCE_CONTINUATION,
    STOP_FORCE_TARGET_READ,
    STOP_CONTINUE_NUDGE,
    STOP_FINALIZE_COMPLETE,
    STOP_FINALIZE_PARTIAL,
    STOP_REPEATED_FAILURE,
    STOP_ABORT,
})

# Non-plan terminal tags already enforced in chatbot (not new predicates).
OUTCOME_REPEATED_FAILURE = "repeated_tool_failure"
OUTCOME_FINALIZE_NONE = "finalize_none"
OUTCOME_PROVIDER_DEGRADE = "provider_degrade"
OUTCOME_PROVIDER_ABORT = "provider_abort"

KNOWN_OUTCOMES = frozenset({
    OUTCOME_REPEATED_FAILURE,
    OUTCOME_FINALIZE_NONE,
    OUTCOME_PROVIDER_DEGRADE,
    OUTCOME_PROVIDER_ABORT,
})


@dataclass
class StopDecision:
    """Named loop outcome for telemetry / tests; not an alternate orchestrator."""

    action: str
    reason: str = ""
    plan_action: str | None = None


def _finalize_kind(plan_reason: str) -> str:
    """Map existing finalize reasons to complete vs partial stop names."""
    reason = (plan_reason or "").strip().lower()
    if reason in {
        "requirements_incomplete",
        "tool_choice_error",
        "token_limit",
        "tpm",
        "tpd",
        "agent_limit",
    }:
        return STOP_FINALIZE_PARTIAL
    # force_final, answer_ready, resolve_none, etc.
    return STOP_FINALIZE_COMPLETE


def map_plan_decision(
    plan: PlanDecision,
    *,
    tools_enabled: bool = True,
) -> StopDecision:
    """Map an authoritative ``PlanDecision`` to a ``StopDecision``."""
    action = plan.action
    reason = plan.reason or ""

    if action == ACTION_FORCE_CONTINUATION:
        return StopDecision(
            action=STOP_FORCE_CONTINUATION,
            reason=reason or "plan_force_continuation",
            plan_action=action,
        )
    if action == ACTION_FORCE_TARGET_READ:
        return StopDecision(
            action=STOP_FORCE_TARGET_READ,
            reason=reason or "plan_force_target_read",
            plan_action=action,
        )
    if action == ACTION_CONTINUE_WITH_NUDGE:
        return StopDecision(
            action=STOP_CONTINUE_NUDGE,
            reason=reason or "plan_continue_nudge",
            plan_action=action,
        )
    if action == ACTION_CALL_LLM:
        return StopDecision(
            action=STOP_CONTINUE_LLM,
            reason=(
                "tools_disabled_direct"
                if not tools_enabled
                else (reason or "plan_call_llm")
            ),
            plan_action=action,
        )
    if action == ACTION_FINALIZE:
        stop_action = _finalize_kind(reason)
        return StopDecision(
            action=stop_action,
            reason=reason or "plan_finalize",
            plan_action=action,
        )

    # Unknown plan action — do not invent policy; surface as abort label only.
    return StopDecision(
        action=STOP_ABORT,
        reason=f"unmapped_plan_action:{action}",
        plan_action=action,
    )


def evaluate_loop_stop(
    *,
    plan: PlanDecision | None = None,
    outcome: str | None = None,
    tools_enabled: bool = True,
) -> StopDecision:
    """Façade: name the loop outcome already chosen by Planning / chatbot.

    Priority:
    1. Explicit non-plan ``outcome`` tags (repeated failure, provider abort, …)
    2. Non-empty unrecognized ``outcome`` → ABORT (never fall through to plan)
    3. ``plan`` mapping via ``map_plan_decision``
    4. Otherwise CONTINUE_LLM when tools are disabled (direct/RAG), else ABORT

    Callers must still branch on ``plan`` / existing chatbot conditions — never
    on this return value alone.
    """
    tag = (outcome or "").strip().lower()
    if tag:
        if tag not in KNOWN_OUTCOMES:
            return StopDecision(
                action=STOP_ABORT,
                reason=f"unknown_outcome:{tag}",
                plan_action=plan.action if plan is not None else None,
            )
        if tag == OUTCOME_REPEATED_FAILURE:
            return StopDecision(
                action=STOP_REPEATED_FAILURE,
                reason="max_same_tool_failures",
                plan_action=plan.action if plan is not None else None,
            )
        if tag == OUTCOME_FINALIZE_NONE:
            return StopDecision(
                action=STOP_FINALIZE_COMPLETE,
                reason="resolve_none",
                plan_action=plan.action if plan is not None else None,
            )
        if tag == OUTCOME_PROVIDER_DEGRADE:
            return StopDecision(
                action=STOP_FINALIZE_PARTIAL,
                reason="provider_degrade_offline",
                plan_action=plan.action if plan is not None else None,
            )
        # OUTCOME_PROVIDER_ABORT
        return StopDecision(
            action=STOP_ABORT,
            reason="provider_abort",
            plan_action=plan.action if plan is not None else None,
        )

    if plan is not None:
        return map_plan_decision(plan, tools_enabled=tools_enabled)

    if not tools_enabled:
        return StopDecision(
            action=STOP_CONTINUE_LLM,
            reason="tools_disabled_no_plan",
            plan_action=None,
        )

    return StopDecision(
        action=STOP_ABORT,
        reason="missing_plan_and_outcome",
        plan_action=None,
    )
