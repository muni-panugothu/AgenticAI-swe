"""Deterministic agent planning decisions (Topic 35).

Wraps existing investigation / finalization predicates into explicit
``PlanDecision`` values. Does not call an LLM planner and does not replace
Groq tool-calling — it only names the action the loop already takes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.agent_orchestration import FINALIZATION_STEP_RESERVE


# Explicit plan actions (Topic 35).
ACTION_FORCE_CONTINUATION = "FORCE_CONTINUATION"
ACTION_FORCE_TARGET_READ = "FORCE_TARGET_READ"
ACTION_CALL_LLM = "CALL_LLM"
ACTION_FINALIZE = "FINALIZE"
ACTION_CONTINUE_WITH_NUDGE = "CONTINUE_WITH_NUDGE"

PLAN_ACTIONS = frozenset({
    ACTION_FORCE_CONTINUATION,
    ACTION_FORCE_TARGET_READ,
    ACTION_CALL_LLM,
    ACTION_FINALIZE,
    ACTION_CONTINUE_WITH_NUDGE,
})


@dataclass
class PlanDecision:
    """One deterministic next-step decision for the agent loop."""

    action: str
    reason: str = ""
    tool_choice: str | dict[str, Any] | None = None
    nudge: str | None = None
    max_drain: int = 0


@dataclass
class AgentPlan:
    """Static plan snapshot derived from investigation requirements.

    Optional ``task_decomposition`` is Phase B4 advisory metadata only.
    It never changes ``plan_next_action`` predicates or tool allowlists.
    """

    requirements: dict[str, Any] = field(default_factory=dict)
    target_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    needs_complete_function: bool = False
    needs_calculator: bool = False
    needs_search: bool = False
    memory_architecture: bool = False
    # Phase B4 — advisory task understanding (does not execute or re-plan).
    task_decomposition: dict[str, Any] | None = None
    decomposed_task_ids: list[str] = field(default_factory=list)
    decomposed_execution_order: list[str] = field(default_factory=list)


def build_agent_plan(requirements: dict[str, Any] | None) -> AgentPlan:
    """Build an AgentPlan view over existing investigation requirements."""
    req = requirements or {}
    target_ids = [
        str(t.get("id"))
        for t in (req.get("required_targets") or [])
        if t.get("id")
    ]
    topics = [str(t) for t in (req.get("topics") or [])]
    return AgentPlan(
        requirements=req,
        target_ids=target_ids,
        topics=topics,
        needs_complete_function=bool(req.get("needs_complete_function")),
        needs_calculator=bool(req.get("needs_calculator")),
        needs_search=bool(req.get("needs_search")),
        memory_architecture="memory_architecture" in topics,
    )


def attach_task_decomposition(
    plan: AgentPlan,
    decomposition: dict[str, Any] | None,
) -> AgentPlan:
    """Attach B4 decomposition as advisory metadata on an existing AgentPlan.

    Does not invoke an LLM, does not mutate investigation requirements, and
    does not alter ``plan_next_action`` behavior.
    """
    if not decomposition:
        return plan
    hints = decomposition.get("planning_hints") or {}
    plan.task_decomposition = dict(decomposition)
    plan.decomposed_task_ids = [
        str(x)
        for x in (
            hints.get("task_ids")
            or [t.get("id") for t in (decomposition.get("tasks") or []) if isinstance(t, dict)]
        )
        if x
    ]
    plan.decomposed_execution_order = [
        str(x) for x in (decomposition.get("execution_order") or []) if x
    ]
    return plan


def build_initial_investigation_plan_message(
    requirements: dict[str, Any] | None,
) -> str:
    """Step-0 investigation plan system message (text parity with Topic 19 loop)."""
    req = requirements or {}
    target_ids = [
        str(t.get("id"))
        for t in (req.get("required_targets") or [])
        if t.get("id")
    ]
    target_note = (
        f" Required source targets: {', '.join(target_ids)}."
        if target_ids
        else ""
    )
    return (
        "Investigation plan: (1) search_code to discover exact "
        "paths/lines for each required symbol, (2) read_file "
        "only the missing relevant windows using those line "
        "numbers (~70 lines max), (3) if needs_continuation/"
        "next_line is present, continue from next_line until "
        "the function ends, (4) explain only what read_file "
        "verified — search hits are discovery, not proof. "
        "Never finalize while required implementations are "
        "unread. Never re-read overlapping ranges or invent "
        "unread bodies."
        f"{target_note}"
    )


def plan_next_action(
    *,
    phase: str,
    tools_enabled: bool,
    force_final: bool = False,
    step: int = 0,
    max_steps: int = 0,
    pending_continuation: bool = False,
    total_tool_executions: int = 0,
    tool_budget_total: int = 0,
    uncovered_count: int = 0,
    investigation_started: bool = False,
    blocked_in_a_row: int = 0,
    successful_searches: int = 0,
    successful_reads: int = 0,
    memory_architecture: bool = False,
    requirements_met: bool = True,
    premature_answer: bool = False,
    has_evidence_snippets: bool = False,
    finalization_reserve: int | None = None,
) -> PlanDecision:
    """Deterministic next action using existing loop predicates.

    ``phase``:
      - ``pre_llm``: before the main Groq planning call
      - ``post_llm_stop``: model returned without tool_calls
    """
    reserve = max(
        1,
        int(
            FINALIZATION_STEP_RESERVE
            if finalization_reserve is None
            else finalization_reserve
        ),
    )
    steps_left = max_steps - 1 - step
    budget_ok = total_tool_executions < tool_budget_total

    if phase == "pre_llm":
        if not tools_enabled:
            return PlanDecision(
                action=ACTION_CALL_LLM,
                reason="tools_disabled",
            )

        # Mirror force_final deferral: continuation / unread before finalize.
        if force_final:
            if step < max_steps - 1 and budget_ok:
                if pending_continuation:
                    return PlanDecision(
                        action=ACTION_FORCE_CONTINUATION,
                        reason="defer_force_final_continuation",
                    )
                if uncovered_count > 0:
                    return PlanDecision(
                        action=ACTION_FORCE_TARGET_READ,
                        reason="defer_force_final_unread_symbol",
                        max_drain=1,
                    )
            return PlanDecision(
                action=ACTION_FINALIZE,
                reason="force_final",
            )

        # Prefer exact next_line, then unread drains, over another LLM round.
        if steps_left > reserve and budget_ok:
            if pending_continuation:
                return PlanDecision(
                    action=ACTION_FORCE_CONTINUATION,
                    reason="pre_llm_continuation",
                )
            drain_ready = investigation_started and (
                blocked_in_a_row >= 1
                or successful_reads >= 1
                or (memory_architecture and successful_searches >= 1)
            )
            if drain_ready and uncovered_count > 0:
                max_pre_drain = min(
                    4,
                    max(0, tool_budget_total - total_tool_executions),
                )
                return PlanDecision(
                    action=ACTION_FORCE_TARGET_READ,
                    reason="pre_llm_unread_required",
                    max_drain=max_pre_drain,
                )

        return PlanDecision(
            action=ACTION_CALL_LLM,
            reason="pre_llm_call",
        )

    if phase == "post_llm_stop":
        if (
            tools_enabled
            and pending_continuation
            and step < max_steps - 1
            and budget_ok
        ):
            return PlanDecision(
                action=ACTION_FORCE_CONTINUATION,
                reason="post_llm_pending_continuation",
            )

        needs_more_source = bool(uncovered_count > 0) and (
            successful_searches + successful_reads >= 1
        )
        can_keep_investigating = steps_left > reserve
        if (
            tools_enabled
            and can_keep_investigating
            and budget_ok
            and (
                pending_continuation
                or needs_more_source
                or (
                    premature_answer
                    and has_evidence_snippets
                    and not requirements_met
                )
            )
        ):
            if pending_continuation:
                return PlanDecision(
                    action=ACTION_FORCE_CONTINUATION,
                    reason="post_llm_keep_continuation",
                )
            if needs_more_source:
                return PlanDecision(
                    action=ACTION_FORCE_TARGET_READ,
                    reason="post_llm_keep_unread",
                    max_drain=1,
                )
            return PlanDecision(
                action=ACTION_CONTINUE_WITH_NUDGE,
                reason="post_llm_missing_evidence_nudge",
            )

        if tools_enabled and has_evidence_snippets and (
            premature_answer or not requirements_met
        ):
            return PlanDecision(
                action=ACTION_FINALIZE,
                reason="requirements_incomplete",
            )

        return PlanDecision(
            action=ACTION_FINALIZE,
            reason="answer_ready",
        )

    return PlanDecision(
        action=ACTION_CALL_LLM,
        reason=f"unknown_phase:{phase}",
    )
