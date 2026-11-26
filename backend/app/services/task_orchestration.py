"""Phase B6 — task-aware execution orchestration (advisory coordinator).

Coordinates B4 decomposition + B5 progress with Topic 35 planning inside the
existing ``chat_with_ai`` loop. Does not plan, execute tools, retry, or grant
permissions.

Lifecycle (Stage 4):
- Decisions / active-task hints are advisory before the LLM call.
- ``start_task`` runs only at the authorized tool-execution boundary.
- Completion uses B5 ``complete_task`` after observable evidence validation.
- Stop paths interrupt running tasks via B5 ``interrupt_task`` (no new transitions).
"""

from __future__ import annotations

from typing import Any, Iterable

from app.services.agent_planning import (
    ACTION_CALL_LLM,
    ACTION_CONTINUE_WITH_NUDGE,
    ACTION_FINALIZE,
    ACTION_FORCE_CONTINUATION,
    ACTION_FORCE_TARGET_READ,
    PlanDecision,
)
from app.services.task_progress import (
    TaskProgressError,
    complete_task,
    eligible_task_ids,
    interrupt_task,
    start_task,
)
from app.validation.task_models import TaskDecomposition
from app.validation.task_orchestration_models import OrchestrationDecision
from app.validation.task_progress_models import TaskProgressBoard

# No write_file / test runner — these must not be started for tool completion.
_UNSUPPORTED_START_CATEGORIES = frozenset(
    {
        "code_generation",
        "code_modification",
        "api_development",
        "database_integration",
        "authentication",
        "authorization",
        "project_work",
        "documentation",
        "testing",
    }
)

_TOOL_BACKED_CATEGORIES = frozenset(
    {
        "calculation",
        "research_current_information",
        "debugging",
        "code_review",
        "personal_memory",
    }
)

# Tool → evidence kinds accepted for start/attribution matching.
_TOOL_EVIDENCE_KINDS: dict[str, frozenset[str]] = {
    "calculator": frozenset({"calculator_result"}),
    "web_search": frozenset({"web_search_result"}),
    "get_current_time": frozenset({"current_time_result"}),
    "search_code": frozenset({"code_search_result"}),
    "read_file": frozenset({"file_read_result"}),
    "search_memory": frozenset({"memory_tool_result"}),
    "save_memory": frozenset({"memory_tool_result"}),
    "forget_memory": frozenset({"memory_tool_result"}),
    "detect_memory_conflict": frozenset({"memory_tool_result"}),
    "request_memory_update": frozenset({"memory_tool_result"}),
    "confirm_memory_update": frozenset({"memory_tool_result"}),
    "reinforce_memory": frozenset({"memory_tool_result"}),
    "inspect_memory_metadata": frozenset({"memory_tool_result"}),
    "summarize_memories": frozenset({"memory_tool_result"}),
    "search_memory_summaries": frozenset({"memory_tool_result"}),
    "expand_memory_summary": frozenset({"memory_tool_result"}),
}

_CATEGORY_EVIDENCE: dict[str, frozenset[str]] = {
    "calculation": frozenset({"calculator_result"}),
    "research_current_information": frozenset(
        {"web_search_result", "current_time_result"}
    ),
    "debugging": frozenset(
        {
            "code_search_result",
            "file_read_result",
            "investigation_verified_target",
        }
    ),
    "code_review": frozenset({"code_search_result", "file_read_result"}),
    "personal_memory": frozenset({"memory_tool_result"}),
    "explanation": frozenset({"final_explanation_output"}),
    "comparison": frozenset({"final_explanation_output", "web_search_result"}),
    "rag_document": frozenset({"rag_retrieval_result", "final_explanation_output"}),
    "general": frozenset({"final_explanation_output", "other_observable"}),
    "other": frozenset({"other_observable", "final_explanation_output"}),
}

_FORCE_ACTIONS = frozenset(
    {
        ACTION_FORCE_CONTINUATION,
        ACTION_FORCE_TARGET_READ,
    }
)

_HINT_ACTIONS = frozenset(
    {
        ACTION_CALL_LLM,
        ACTION_CONTINUE_WITH_NUDGE,
    }
)

_CLEAR_HINT_STATUSES = frozenset(
    {
        "plan_blocks",
        "partial_finalize",
        "budget_exhausted",
        "abort",
        "clarification_required",
        "unsupported_path",
        "no_eligible",
        "no_board",
    }
)


def _as_decomposition(
    snapshot: TaskDecomposition | dict[str, Any] | None,
) -> TaskDecomposition | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, TaskDecomposition):
        return snapshot
    if isinstance(snapshot, dict) and snapshot.get("tasks") is not None:
        return TaskDecomposition.model_validate(snapshot)
    return None


def _incomplete_ids(board: TaskProgressBoard) -> list[str]:
    out: list[str] = []
    for tid in board.execution_order or list(board.tasks.keys()):
        rec = board.tasks.get(tid)
        if rec and rec.status != "completed":
            out.append(tid)
    return out


def _clarification_for_task(
    decomp: TaskDecomposition | None,
    task_id: str,
) -> str | None:
    if decomp is None:
        return None
    for t in decomp.tasks:
        if t.id == task_id and t.needs_clarification:
            return t.clarification_prompt or "Task requires user clarification"
    return None


def _board_needs_clarification(decomp: TaskDecomposition | None) -> bool:
    if decomp is None:
        return False
    if decomp.needs_clarification:
        return True
    return any(t.needs_clarification for t in decomp.tasks)


def tool_compatible_with_category(tool_name: str, category: str) -> bool:
    """True when the tool's evidence kind is accepted by the task category."""
    kinds = _TOOL_EVIDENCE_KINDS.get(tool_name or "")
    if not kinds:
        return False
    allowed = _CATEGORY_EVIDENCE.get(category, frozenset())
    return bool(kinds & allowed)


def select_candidate_task_id(
    board: TaskProgressBoard | None,
    *,
    decomposition: TaskDecomposition | dict[str, Any] | None = None,
) -> tuple[str | None, list[str], str]:
    """Pick at most one eligible task preserving B4 ``execution_order``.

    Does not start the task. Reuses B5 ``eligible_task_ids``.
    """
    if board is None:
        return None, [], "no_progress_board"

    eligible = list(eligible_task_ids(board))
    if not eligible:
        return None, [], "no_eligible_tasks"

    decomp = _as_decomposition(decomposition)
    order = list(board.execution_order)
    if decomp is not None and decomp.execution_order:
        order = [tid for tid in decomp.execution_order if tid in board.tasks]
        b4_ids = {t.id for t in decomp.tasks}
        eligible = [tid for tid in eligible if tid in b4_ids]
        if not eligible:
            return None, [], "eligible_not_in_b4_snapshot"

    eligible_set = set(eligible)
    for tid in order:
        if tid in eligible_set:
            return tid, eligible, f"first_eligible_in_execution_order:{tid}"

    return eligible[0], eligible, f"first_eligible:{eligible[0]}"


def decide_task_orchestration(
    *,
    board: TaskProgressBoard | None,
    plan_decision: PlanDecision | None,
    decomposition: TaskDecomposition | dict[str, Any] | None = None,
    tools_enabled: bool = False,
    authorized_tool_names: Iterable[str] | None = None,
    step: int = 0,
    max_steps: int = 0,
    total_tool_executions: int = 0,
    tool_budget_total: int = 0,
    stop_action: str | None = None,
    outcome: str | None = None,
) -> OrchestrationDecision:
    """Advisory orchestration decision. Never mutates the board or starts tasks.

    ``execution_may_proceed`` means a later authorized tool boundary *may*
    start the hinted task — not that the task is running now.
    """
    auth_names = {
        str(n) for n in (authorized_tool_names or []) if n
    }
    decomp = _as_decomposition(decomposition)
    plan_action = getattr(plan_decision, "action", None) if plan_decision else None

    if board is None:
        return OrchestrationDecision(
            status="no_board",
            selection_reason="no_progress_board",
            plan_action=plan_action,
            plan_compatible=True,
            block_reason="no_progress_board",
        )

    incomplete = _incomplete_ids(board)
    candidate, eligible, sel_reason = select_candidate_task_id(
        board, decomposition=decomp
    )

    if stop_action == "ABORT" or (
        outcome and str(outcome).startswith("provider_abort")
    ):
        return OrchestrationDecision(
            status="abort",
            selected_task_id=None,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            block_reason="topic40_abort",
            incomplete_task_ids=incomplete,
        )

    budget_exhausted = False
    if max_steps > 0 and step >= max_steps - 1:
        budget_exhausted = True
    if tool_budget_total > 0 and total_tool_executions >= tool_budget_total:
        budget_exhausted = True
    if budget_exhausted:
        return OrchestrationDecision(
            status="budget_exhausted",
            selected_task_id=None,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            partial_finalize_recommended=bool(incomplete),
            block_reason="agent_or_tool_budget_exhausted",
            incomplete_task_ids=incomplete,
        )

    if plan_action == ACTION_FINALIZE:
        return OrchestrationDecision(
            status="partial_finalize" if incomplete else "plan_blocks",
            selected_task_id=None,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            partial_finalize_recommended=bool(incomplete),
            block_reason="plan_finalize_blocks_task_advancement",
            incomplete_task_ids=incomplete,
        )

    if plan_action in _FORCE_ACTIONS:
        return OrchestrationDecision(
            status="plan_blocks",
            selected_task_id=None,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            block_reason=f"forced_plan_action:{plan_action}",
            incomplete_task_ids=incomplete,
        )

    if plan_action not in _HINT_ACTIONS:
        return OrchestrationDecision(
            status="plan_blocks",
            selected_task_id=None,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            block_reason=f"unsupported_or_missing_plan_action:{plan_action!r}",
            incomplete_task_ids=incomplete,
        )

    if _board_needs_clarification(decomp) and decomp is not None and (
        decomp.needs_clarification
        or (candidate and _clarification_for_task(decomp, candidate))
    ):
        prompt = None
        if candidate:
            prompt = _clarification_for_task(decomp, candidate)
        if decomp.needs_clarification:
            prompt = prompt or (
                decomp.clarification_prompts[0]
                if decomp.clarification_prompts
                else "Clarification required before task execution"
            )
        return OrchestrationDecision(
            status="clarification_required",
            selected_task_id=candidate,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            clarification_required=True,
            block_reason=prompt or "clarification_required",
            incomplete_task_ids=incomplete,
        )

    if candidate is None:
        return OrchestrationDecision(
            status="no_eligible",
            selected_task_id=None,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=True,
            execution_may_proceed=False,
            block_reason="no_eligible_task",
            incomplete_task_ids=incomplete,
        )

    rec = board.tasks.get(candidate)
    category = rec.category if rec else "general"

    if category in _UNSUPPORTED_START_CATEGORIES:
        return OrchestrationDecision(
            status="unsupported_path",
            selected_task_id=candidate,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            block_reason="no_reliable_tool_completion_pathway",
            unsupported_category=category,
            incomplete_task_ids=incomplete,
            partial_finalize_recommended=True,
        )

    if category in _TOOL_BACKED_CATEGORIES and not tools_enabled:
        return OrchestrationDecision(
            status="unsupported_path",
            selected_task_id=candidate,
            eligible_task_ids=eligible,
            selection_reason=sel_reason,
            plan_action=plan_action,
            plan_compatible=False,
            execution_may_proceed=False,
            block_reason="tools_disabled_for_tool_backed_category",
            unsupported_category=category,
            incomplete_task_ids=incomplete,
        )

    # If an allowlist is provided and none of its tools can serve this category,
    # treat as unsupported (does not widen permissions).
    if auth_names and category in _TOOL_BACKED_CATEGORIES:
        if not any(
            tool_compatible_with_category(name, category) for name in auth_names
        ):
            return OrchestrationDecision(
                status="unsupported_path",
                selected_task_id=candidate,
                eligible_task_ids=eligible,
                selection_reason=sel_reason,
                plan_action=plan_action,
                plan_compatible=False,
                execution_may_proceed=False,
                block_reason="authorized_catalog_cannot_serve_category",
                unsupported_category=category,
                incomplete_task_ids=incomplete,
            )

    return OrchestrationDecision(
        status="candidate_selected",
        selected_task_id=candidate,
        eligible_task_ids=eligible,
        selection_reason=sel_reason,
        plan_action=plan_action,
        plan_compatible=True,
        # Deferred start: may proceed only at authorized tool boundary.
        execution_may_proceed=True,
        active_task_hint=candidate,
        incomplete_task_ids=incomplete,
    )


def bind_active_task_hint(agent_state: Any, decision: OrchestrationDecision) -> None:
    """Bind or clear request-local ``active_task_id`` from a decision.

    Does not start the task. Clears stale hints on blocking statuses.
    """
    if agent_state is None:
        return
    if decision.execution_may_proceed and decision.active_task_hint:
        # Advisory only — task is not running until tool-boundary start.
        agent_state.active_task_id = decision.active_task_hint
        return
    if decision.status in _CLEAR_HINT_STATUSES:
        agent_state.active_task_id = None


def clear_active_task_id(agent_state: Any) -> None:
    if agent_state is not None:
        agent_state.active_task_id = None


def ensure_task_started_for_tool(
    board: TaskProgressBoard | None,
    decision: OrchestrationDecision | None,
    agent_state: Any,
    *,
    tool_name: str,
    reason: str = "b6_authorized_tool_boundary",
) -> bool:
    """Start the hinted task only when an authorized tool is about to run.

    Requires ``execution_may_proceed`` and tool/category compatibility.
    Idempotent when already running.
    """
    if board is None or decision is None or not decision.execution_may_proceed:
        return False
    tid = decision.active_task_hint or decision.selected_task_id
    if not tid:
        return False
    rec = board.tasks.get(tid)
    if rec is None:
        return False
    if not tool_compatible_with_category(tool_name, rec.category):
        return False
    if rec.status == "running":
        if agent_state is not None:
            agent_state.active_task_id = tid
        return True
    if rec.status != "pending":
        return False
    start_task(board, tid, reason=reason)
    if agent_state is not None:
        agent_state.active_task_id = tid
    return True


def try_complete_active_task(
    board: TaskProgressBoard | None,
    task_id: str | None,
    agent_state: Any = None,
) -> bool:
    """Ask B5 to complete when evidence is sufficient. Never fabricates success."""
    if board is None or not task_id:
        return False
    rec = board.tasks.get(task_id)
    if rec is None or rec.status != "running":
        return False
    try:
        complete_task(board, task_id, reason="b6_evidence_satisfied")
    except TaskProgressError:
        return False
    if agent_state is not None and getattr(agent_state, "active_task_id", None) == task_id:
        agent_state.active_task_id = None
    return True


def interrupt_running_tasks(
    board: TaskProgressBoard | None,
    *,
    reason: str,
    agent_state: Any = None,
) -> list[str]:
    """Interrupt all *running* tasks via B5. Clears active_task_id when provided.

    Never mutates completed, failed, pending, or blocked tasks.
    """
    interrupted: list[str] = []
    if board is None:
        clear_active_task_id(agent_state)
        return interrupted
    for tid, rec in list(board.tasks.items()):
        # Explicit guard: completed (and non-running) must not be touched.
        if rec.status != "running":
            continue
        try:
            interrupt_task(board, tid, reason=reason)
            interrupted.append(tid)
        except TaskProgressError:
            continue
    clear_active_task_id(agent_state)
    return interrupted


def safe_cleanup_request_tasks(
    agent_state: Any,
    *,
    reason: str = "unexpected_exit",
) -> list[str]:
    """Best-effort B6 cleanup for request termination / unexpected exits.

    Interrupts only running tasks and clears ``active_task_id``. Never raises:
    cleanup failures are logged so they cannot mask the original exception.
    Idempotent when no running tasks remain.
    """
    interrupted: list[str] = []
    if agent_state is None:
        return interrupted
    try:
        board = getattr(agent_state, "task_progress", None)
        interrupted = interrupt_running_tasks(
            board,
            reason=f"b6_request_cleanup:{reason}",
            agent_state=agent_state,
        )
    except Exception as cleanup_exc:  # noqa: BLE001 — must not mask callers
        try:
            print(
                "[B6 cleanup] failed during request termination: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc!r}"
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            clear_active_task_id(agent_state)
        except Exception:  # noqa: BLE001
            pass
    return interrupted


def apply_orchestration_outcome(
    *,
    board: TaskProgressBoard | None,
    agent_state: Any = None,
    decision: OrchestrationDecision | None = None,
    outcome: str,
    tool_success: bool | None = None,
    active_task_id: str | None = None,
) -> dict[str, Any]:
    """Apply stop/failure/success coordination using existing B5 transitions.

    Outcomes:
    - ``tool_success``: try evidence-gated complete (caller must have recorded evidence)
    - ``tool_error`` / ``validation_error``: interrupt if running and no evidence yet
    - ``finalize`` / ``abort`` / ``budget`` / ``clarification`` / ``llm_text_stop``:
      interrupt all running
    - ``unknown``: interrupt running (never treat as success)
    """
    del decision  # advisory context for callers; board/state drive mutations
    tid = active_task_id or (
        getattr(agent_state, "active_task_id", None) if agent_state else None
    )
    tag = (outcome or "").strip().lower()

    if tag in {"finalize", "abort", "budget", "clarification", "llm_text_stop", "unsupported"}:
        interrupted = interrupt_running_tasks(
            board, reason=f"b6_stop:{tag}", agent_state=agent_state
        )
        return {
            "complete_allowed": False,
            "treat_as_success": False,
            "reason": f"stop:{tag}",
            "interrupted": interrupted,
        }

    if tag == "unknown":
        interrupted = interrupt_running_tasks(
            board, reason="b6_unknown_outcome", agent_state=agent_state
        )
        return {
            "complete_allowed": False,
            "treat_as_success": False,
            "reason": "unknown_outcome_not_success",
            "interrupted": interrupted,
        }

    if tag in {"tool_error", "validation_error", "retry_exhausted"}:
        if board is not None and tid:
            rec = board.tasks.get(tid)
            if rec is not None and rec.status == "running":
                # No evidence yet → interrupt orphan start; else keep for more tools.
                if not rec.evidence:
                    try:
                        interrupt_task(
                            board, tid, reason=f"b6_{tag}_no_evidence"
                        )
                    except TaskProgressError:
                        pass
                    clear_active_task_id(agent_state)
                    return {
                        "complete_allowed": False,
                        "treat_as_success": False,
                        "reason": f"{tag}_interrupted_empty",
                        "interrupted": [tid],
                    }
        return {
            "complete_allowed": False,
            "treat_as_success": False,
            "reason": tag,
            "interrupted": [],
        }

    if tag == "tool_success" or tool_success is True:
        completed = try_complete_active_task(board, tid, agent_state)
        return {
            "complete_allowed": completed,
            "treat_as_success": False,
            "reason": (
                "completed_via_b5_evidence"
                if completed
                else "evidence_recorded_incomplete"
            ),
            "interrupted": [],
            "completed": completed,
        }

    return {
        "complete_allowed": False,
        "treat_as_success": False,
        "reason": "no_outcome",
        "interrupted": [],
    }


# Backward-compatible alias used by Stage 2 tests / callers.
def observe_orchestration_outcome(
    *,
    decision: OrchestrationDecision,
    tool_success: bool | None = None,
    unknown_outcome: bool = False,
    retry_exhausted: bool = False,
) -> dict[str, Any]:
    """Pure guidance helper (no board mutation). Prefer ``apply_orchestration_outcome``."""
    if unknown_outcome:
        return {
            "complete_allowed": False,
            "treat_as_success": False,
            "reason": "unknown_outcome_not_success",
            "selected_task_id": decision.selected_task_id,
        }
    if retry_exhausted:
        return {
            "complete_allowed": False,
            "treat_as_success": False,
            "reason": "retry_exhausted",
            "selected_task_id": decision.selected_task_id,
        }
    if tool_success is False:
        return {
            "complete_allowed": False,
            "treat_as_success": False,
            "reason": "tool_error",
            "selected_task_id": decision.selected_task_id,
        }
    if tool_success is True:
        return {
            "complete_allowed": False,
            "treat_as_success": False,
            "reason": "tool_success_evidence_only",
            "selected_task_id": decision.selected_task_id,
        }
    return {
        "complete_allowed": False,
        "treat_as_success": False,
        "reason": "no_outcome",
        "selected_task_id": decision.selected_task_id,
    }


def format_incomplete_workflow_note(board: TaskProgressBoard | None) -> str:
    """Plain-language suffix for partial workflows. No internal IDs/schemas.

    Boards that were never entered (all tasks still at attempt_count=0 /
    init-only pending) must not emit a partial-workflow note. That covers
    plain greetings and other single-turn answers where B4 attached an
    advisory task that orchestration never started.
    """
    if board is None or not board.tasks:
        return ""
    pending = running = blocked = failed = completed = 0
    # Dependency ``block`` at init does not mean orchestration entered a task.
    # Only start / interrupt / complete / fail (or attempt_count) count.
    workflow_entered = False
    for rec in board.tasks.values():
        if int(getattr(rec, "attempt_count", 0) or 0) > 0:
            workflow_entered = True
        if rec.status in {"running", "failed", "completed"}:
            workflow_entered = True
        transitions = getattr(rec, "transitions", None) or []
        if any(
            getattr(t, "action", None)
            in {"start", "interrupt", "complete", "fail", "retry"}
            for t in transitions
        ):
            workflow_entered = True
        if rec.status == "pending":
            pending += 1
        elif rec.status == "running":
            running += 1
        elif rec.status == "blocked":
            blocked += 1
        elif rec.status == "failed":
            failed += 1
        elif rec.status == "completed":
            completed += 1
    incomplete = pending + running + blocked + failed
    if incomplete == 0:
        return ""
    # Advisory-only board: decomposition existed, but no task was ever started.
    if not workflow_entered:
        return ""
    parts: list[str] = []
    if completed:
        parts.append(
            f"{completed} completed" if completed != 1 else "1 completed"
        )
    if pending:
        parts.append(f"{pending} waiting" if pending != 1 else "1 waiting")
    if running:
        parts.append(
            f"{running} still in progress"
            if running != 1
            else "1 still in progress"
        )
    if blocked:
        parts.append(f"{blocked} blocked" if blocked != 1 else "1 blocked")
    if failed:
        parts.append(f"{failed} failed" if failed != 1 else "1 failed")
    # Keep a stable phrase for dedupe across finalize helpers.
    parts.append(
        f"{incomplete} not fully finished"
        if incomplete != 1
        else "1 not fully finished"
    )
    summary = ", ".join(parts)
    return (
        "\n\nNote: This request involved multiple steps "
        f"({summary}). Treat the answer above as partial where work "
        "could not be fully verified."
    )


def append_incomplete_workflow_note(
    answer: str,
    board: TaskProgressBoard | None,
) -> str:
    note = format_incomplete_workflow_note(board)
    if not note:
        return answer
    text = (answer or "").rstrip()
    if "not fully finished" in text.casefold():
        return text
    return text + note
