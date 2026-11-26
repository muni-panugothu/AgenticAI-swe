"""Phase B5 — execution-aware task progress and dependency tracking.

Request-scoped only. Does not execute tasks, grant tools, or replace
Topic 35 planning / Topic 38 retries / investigation checklist semantics.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.validation.task_models import TaskDecomposition
from app.validation.task_progress_models import (
    DependencyEdge,
    EvidenceKind,
    EvidenceRecord,
    TaskProgressBoard,
    TaskProgressRecord,
    TaskStatus,
    TransitionRecord,
    UncertainDependencyPolicy,
)

# Category → evidence kinds that may satisfy completion (conservative).
_COMPLETION_EVIDENCE: dict[str, frozenset[str]] = {
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
    ),  # completion still requires search+read or verified target (see below)
    "code_review": frozenset({"code_search_result", "file_read_result"}),
    "testing": frozenset(
        {
            # Generated tests are artifacts only — not proof of execution.
            "generated_tests_artifact",
            # Executed tests require a runner this repo does not provide.
            "executed_tests_result",
        }
    ),
    "code_generation": frozenset({"generated_code_artifact"}),
    "code_modification": frozenset(
        {"generated_code_artifact", "file_read_result"}
    ),
    "api_development": frozenset({"generated_code_artifact"}),
    "database_integration": frozenset({"generated_code_artifact"}),
    "authentication": frozenset({"generated_code_artifact"}),
    "authorization": frozenset({"generated_code_artifact"}),
    "project_work": frozenset({"generated_code_artifact", "file_read_result"}),
    "documentation": frozenset(
        {"generated_document_artifact", "final_explanation_output"}
    ),
    "explanation": frozenset({"final_explanation_output"}),
    "comparison": frozenset({"final_explanation_output", "web_search_result"}),
    "rag_document": frozenset({"rag_retrieval_result", "final_explanation_output"}),
    "data_analysis": frozenset(
        {"rag_retrieval_result", "final_explanation_output", "calculator_result"}
    ),
    "personal_memory": frozenset({"memory_tool_result"}),
    "general": frozenset({"final_explanation_output", "other_observable"}),
    "other": frozenset({"other_observable", "final_explanation_output"}),
}

# Categories that need *executed* tests to fully complete when criteria say so.
_TESTING_NEEDS_EXECUTION = re.compile(
    r"(?i)\b(pass|passed|execute|run (the )?tests|test results?)\b"
)

_TOOL_TO_EVIDENCE: dict[str, EvidenceKind] = {
    "calculator": "calculator_result",
    "web_search": "web_search_result",
    "get_current_time": "current_time_result",
    "search_code": "code_search_result",
    "read_file": "file_read_result",
    "search_memory": "memory_tool_result",
    "save_memory": "memory_tool_result",
    "forget_memory": "memory_tool_result",
    "update_memory": "memory_tool_result",
}

_UNSUPPORTED_COMPLETION_KINDS = frozenset({"llm_claim"})


class TaskProgressError(ValueError):
    """Invalid progress transition or evidence policy violation."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fingerprint_decomposition(decomp: TaskDecomposition | dict[str, Any]) -> str:
    if isinstance(decomp, TaskDecomposition):
        payload = decomp.model_dump()
    else:
        payload = dict(decomp)
    tasks = payload.get("tasks") or []
    parts = [
        str(payload.get("trusted_instruction") or ""),
        "|".join(
            f"{t.get('id')}:{t.get('category')}:{t.get('description')}"
            for t in tasks
            if isinstance(t, dict)
        ),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


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


def build_progress_board_from_decomposition(
    snapshot: TaskDecomposition | dict[str, Any] | None,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    request_id: str | None = None,
    uncertain_dependency_policy: UncertainDependencyPolicy = "block",
) -> TaskProgressBoard | None:
    """Initialize a board from an immutable B4 snapshot (copy, do not mutate)."""
    decomp = _as_decomposition(snapshot)
    if decomp is None or not decomp.tasks:
        return None

    tasks: dict[str, TaskProgressRecord] = {}
    for t in decomp.tasks:
        edges = [
            DependencyEdge(predecessor_id=dep, kind=t.dependency_kind)
            for dep in t.depends_on
        ]
        fp = hashlib.sha256(
            f"{t.id}|{t.category}|{t.description}".encode("utf-8")
        ).hexdigest()[:16]
        rec = TaskProgressRecord(
            task_id=t.id,
            description=t.description,
            category=t.category,
            status="pending",
            depends_on=edges,
            expected_outputs=list(t.expected_outputs),
            acceptance_criteria=list(t.acceptance_criteria),
            decomposition_task_fingerprint=fp,
            transitions=[
                TransitionRecord(
                    action="init",
                    from_status=None,
                    to_status="pending",
                    reason="initialized_from_b4_snapshot",
                )
            ],
        )
        tasks[t.id] = rec

    board = TaskProgressBoard(
        board_id=str(uuid.uuid4()),
        request_id=request_id,
        user_id=user_id,
        project_id=project_id,
        tenant_id=tenant_id,
        decomposition_fingerprint=_fingerprint_decomposition(decomp),
        trusted_instruction=decomp.trusted_instruction,
        uncertain_dependency_policy=uncertain_dependency_policy,
        tasks=tasks,
        execution_order=list(decomp.execution_order)
        or [t.id for t in decomp.tasks],
    )
    # Initial block pass for tasks with unsatisfied required deps.
    for tid in list(board.tasks):
        _recompute_blocked(board, tid, reason="init_dependency_scan")
    return board


def set_uncertain_dependency_policy(
    board: TaskProgressBoard,
    policy: UncertainDependencyPolicy,
    *,
    reason: str,
) -> TaskProgressBoard:
    """Explicit override — uncertain edges are never silently optional."""
    if not reason.strip():
        raise TaskProgressError("uncertain policy override requires a reason")
    board.uncertain_dependency_policy = policy
    board.board_version += 1
    board.updated_at = _now()
    for rec in board.tasks.values():
        rec.transitions.append(
            TransitionRecord(
                action="policy_override",
                from_status=rec.status,
                to_status=rec.status,
                reason=reason,
                meta={"uncertain_dependency_policy": policy},
            )
        )
        _recompute_blocked(board, rec.task_id, reason="policy_override_recompute")
    return board


def _edge_blocks_start(
    board: TaskProgressBoard,
    edge: DependencyEdge,
) -> bool:
    pred = board.tasks.get(edge.predecessor_id)
    if pred is None:
        # Integrity: missing record should have failed board validation;
        # treat as blocking if encountered dynamically.
        return True
    if edge.kind == "optional":
        return False
    if edge.kind == "uncertain":
        if board.uncertain_dependency_policy == "treat_as_optional":
            return False
        # Default "block": uncertain predecessor must be completed explicitly.
        return pred.status != "completed"
    # required
    return pred.status != "completed"


def _blocking_predecessors(board: TaskProgressBoard, tid: str) -> list[str]:
    rec = board.tasks[tid]
    return [
        e.predecessor_id
        for e in rec.depends_on
        if _edge_blocks_start(board, e)
    ]


def eligible_task_ids(board: TaskProgressBoard) -> list[str]:
    """Tasks that *could* start — does not start them."""
    out: list[str] = []
    for tid in board.execution_order or board.tasks.keys():
        rec = board.tasks.get(tid)
        if not rec:
            continue
        if rec.status not in {"pending", "blocked"}:
            continue
        if not _blocking_predecessors(board, tid):
            out.append(tid)
    return out


def _recompute_blocked(
    board: TaskProgressBoard,
    tid: str,
    *,
    reason: str,
) -> None:
    rec = board.tasks[tid]
    if rec.status in {"completed", "failed", "running"}:
        return
    blockers = _blocking_predecessors(board, tid)
    if blockers and rec.status == "pending":
        _set_status(
            board,
            rec,
            "blocked",
            action="block",
            reason=reason,
            meta={"blocked_by": blockers},
        )
        rec.blocked_by = blockers
    elif not blockers and rec.status == "blocked":
        _set_status(
            board,
            rec,
            "pending",
            action="unblock",
            reason=reason,
            meta={"cleared_blockers": True},
        )
        rec.blocked_by = []
    elif blockers and rec.status == "blocked":
        rec.blocked_by = blockers
        rec.updated_at = _now()


def _set_status(
    board: TaskProgressBoard,
    rec: TaskProgressRecord,
    new_status: TaskStatus,
    *,
    action: str,
    reason: str,
    meta: dict[str, Any] | None = None,
) -> None:
    old = rec.status
    if old == "completed" and new_status != "completed":
        raise TaskProgressError("completed tasks are terminal")
    rec.transitions.append(
        TransitionRecord(
            action=action,  # type: ignore[arg-type]
            from_status=old,
            to_status=new_status,
            reason=reason,
            meta=meta or {},
        )
    )
    rec.status = new_status
    rec.updated_at = _now()
    board.updated_at = rec.updated_at
    board.board_version += 1


def start_task(board: TaskProgressBoard, task_id: str, *, reason: str = "start") -> None:
    rec = _require(board, task_id)
    if rec.status == "blocked":
        raise TaskProgressError("cannot start blocked task directly; unblock first")
    if rec.status != "pending":
        raise TaskProgressError(f"cannot start task from status {rec.status!r}")
    blockers = _blocking_predecessors(board, task_id)
    if blockers:
        raise TaskProgressError(
            f"required/uncertain dependencies not completed: {blockers}"
        )
    # Attempt accounting: new starts increment; resume-after-interrupt keeps the
    # same attempt; retry already reserved the next attempt number.
    last_action = rec.transitions[-1].action if rec.transitions else None
    if last_action == "interrupt":
        pass
    elif last_action == "retry":
        pass
    else:
        rec.attempt_count += 1
    if rec.attempt_count < 1:
        rec.attempt_count = 1
    rec.started_at = _now()
    rec.blocked_by = []
    rec.failure_code = None
    rec.failure_detail = None
    _set_status(board, rec, "running", action="start", reason=reason)


def fail_task(
    board: TaskProgressBoard,
    task_id: str,
    *,
    failure_code: str,
    detail: str,
) -> None:
    rec = _require(board, task_id)
    if rec.status != "running":
        raise TaskProgressError(f"cannot fail task from status {rec.status!r}")
    if not failure_code.strip():
        raise TaskProgressError("failure_code required")
    rec.failure_code = failure_code
    rec.failure_detail = detail
    _set_status(
        board,
        rec,
        "failed",
        action="fail",
        reason=failure_code,
        meta={"detail": detail},
    )
    # Dependents with required edges become blocked.
    for other_id, other in board.tasks.items():
        if other_id == task_id:
            continue
        _recompute_blocked(
            board,
            other_id,
            reason=f"predecessor_failed:{task_id}",
        )


def interrupt_task(
    board: TaskProgressBoard,
    task_id: str,
    *,
    reason: str,
) -> None:
    rec = _require(board, task_id)
    if rec.status != "running":
        raise TaskProgressError(f"cannot interrupt task from status {rec.status!r}")
    if not reason.strip():
        raise TaskProgressError("interrupt requires an explicit reason")
    _set_status(board, rec, "pending", action="interrupt", reason=reason)
    _recompute_blocked(board, task_id, reason="post_interrupt")


def retry_task(
    board: TaskProgressBoard,
    task_id: str,
    *,
    reason: str,
) -> None:
    """Explicit failed → pending only (no automatic retry)."""
    rec = _require(board, task_id)
    if rec.status != "failed":
        raise TaskProgressError(f"cannot retry task from status {rec.status!r}")
    if not reason.strip():
        raise TaskProgressError("retry/resume requires an explicit reason")
    rec.failure_code = None
    # Keep failure_detail in transition history; clear active detail.
    prior = rec.failure_detail
    rec.failure_detail = None
    # Reserve the next attempt number now so prior-attempt evidence cannot
    # silently satisfy completion after resume (history list is preserved).
    rec.attempt_count = max(1, int(rec.attempt_count)) + 1
    _set_status(
        board,
        rec,
        "pending",
        action="retry",
        reason=reason,
        meta={
            "prior_failure_detail": prior,
            "attempt_count": rec.attempt_count,
            "prior_evidence_superseded_for_completion": True,
        },
    )
    _recompute_blocked(board, task_id, reason="post_retry")


def record_evidence(
    board: TaskProgressBoard,
    task_id: str | None,
    evidence: EvidenceRecord,
) -> None:
    """Attach evidence to a task or the unassigned pool. Does not complete."""
    if evidence.kind in _UNSUPPORTED_COMPLETION_KINDS and evidence.verified:
        raise TaskProgressError("llm_claim cannot be verified evidence")
    if task_id is None:
        board.unassigned_evidence.append(evidence)
        board.board_version += 1
        board.updated_at = _now()
        return
    rec = _require(board, task_id)
    if rec.status == "completed":
        # Idempotent ignore of additional evidence on completed tasks.
        return
    stamped = dict(evidence.details or {})
    stamped.setdefault("attempt", rec.attempt_count)
    ev = evidence.model_copy(
        update={"task_id": task_id, "details": stamped}
    )
    rec.evidence.append(ev)
    rec.transitions.append(
        TransitionRecord(
            action="record_evidence",
            from_status=rec.status,
            to_status=rec.status,
            reason=f"evidence:{ev.kind}",
            meta={"tool_name": ev.tool_name, "fingerprint": ev.fingerprint},
        )
    )
    rec.updated_at = _now()
    board.updated_at = rec.updated_at
    board.board_version += 1


def note_observable_tool_result(
    board: TaskProgressBoard | None,
    *,
    tool_name: str,
    success: bool,
    fingerprint: str | None = None,
    result_excerpt: str = "",
    preferred_task_id: str | None = None,
) -> None:
    """Loop hook: store observable tool evidence without starting/completing.

    Optional ``preferred_task_id`` (B6 active-task hint) is used only when that
    task is *running* and its category accepts this evidence kind. Never
    attaches evidence to a different or non-running task.
    """
    if board is None or not success:
        return
    kind = _TOOL_TO_EVIDENCE.get(tool_name)
    if kind is None:
        return
    summary = (result_excerpt or f"{tool_name} succeeded").strip()[:500]
    evidence = EvidenceRecord(
        kind=kind,
        source="tool_result",
        summary=summary or f"{tool_name} ok",
        tool_name=tool_name,
        fingerprint=fingerprint,
        verified=True,
    )
    # Strict ownership when B6 supplies preferred_task_id: attach only to that
    # running task when its category accepts the kind; otherwise unassigned.
    # Never broadcast to a different running task.
    if preferred_task_id is not None:
        pref = board.tasks.get(preferred_task_id)
        if pref is not None and pref.status == "running":
            allowed = _COMPLETION_EVIDENCE.get(pref.category, frozenset())
            if kind in allowed:
                record_evidence(board, preferred_task_id, evidence)
                return
        record_evidence(board, None, evidence)
        return
    # No preferred owner: first running task whose category accepts the kind
    # (legacy B5 observe path). Callers with an active task should pass preferred.
    matched = False
    for tid, rec in board.tasks.items():
        if rec.status != "running":
            continue
        allowed = _COMPLETION_EVIDENCE.get(rec.category, frozenset())
        if kind in allowed:
            record_evidence(board, tid, evidence)
            matched = True
            break
    if not matched:
        record_evidence(board, None, evidence)


def _acceptable_evidence(rec: TaskProgressRecord) -> list[EvidenceRecord]:
    """Evidence that may count toward completion for the *current* attempt.

    Prior-attempt evidence remains on the record for history but does not
    satisfy completion after an explicit failed→pending retry.
    """
    allowed = _COMPLETION_EVIDENCE.get(rec.category, frozenset())
    current = int(rec.attempt_count)
    out: list[EvidenceRecord] = []
    for e in rec.evidence:
        if not e.verified:
            continue
        if e.kind not in allowed or e.kind in _UNSUPPORTED_COMPLETION_KINDS:
            continue
        attempt = e.details.get("attempt", current)
        try:
            attempt_i = int(attempt)
        except (TypeError, ValueError):
            attempt_i = current
        if attempt_i != current:
            continue
        out.append(e)
    return out


def _evidence_satisfies(rec: TaskProgressRecord) -> tuple[bool, str]:
    accepted = _acceptable_evidence(rec)
    if not accepted:
        return False, "no_acceptable_evidence"

    kinds = {e.kind for e in accepted}

    # Debugging: read_file alone is insufficient discovery — require a verified
    # investigation target or both search and read evidence.
    if rec.category == "debugging":
        if "investigation_verified_target" in kinds:
            pass
        elif "code_search_result" in kinds and "file_read_result" in kinds:
            pass
        else:
            return False, "debugging_needs_search_and_read_or_verified_target"

    # Testing: generated ≠ executed.
    if rec.category == "testing":
        criteria = " ".join(rec.acceptance_criteria + rec.expected_outputs)
        needs_exec = bool(_TESTING_NEEDS_EXECUTION.search(criteria)) or any(
            "executed" in (o or "").casefold() for o in rec.expected_outputs
        )
        if needs_exec and "executed_tests_result" not in kinds:
            return False, "executed_tests_required_but_unavailable"
        if not needs_exec and not (
            kinds & {"generated_tests_artifact", "executed_tests_result"}
        ):
            return False, "testing_artifact_required"
        # Explicit: generated_tests alone never proves execution.
        if kinds == {"generated_tests_artifact"} and needs_exec:
            return False, "generated_tests_are_not_executed_tests"

    # Multiple acceptance criteria → require multiple acceptable evidence items.
    if len(rec.acceptance_criteria) >= 2 and len(accepted) < 2:
        return False, "multiple_acceptance_criteria_need_multiple_evidence"

    if rec.category == "personal_memory" and not any(
        e.kind == "memory_tool_result" for e in accepted
    ):
        return False, "memory_tool_result_required"

    if rec.category == "rag_document" and not any(
        e.kind == "rag_retrieval_result" for e in accepted
    ):
        # Allow final explanation only when RAG evidence also present in pool
        # attached to task — conservative.
        return False, "rag_retrieval_result_required"

    # Implementation categories must not complete on explanation-only evidence.
    if rec.category in {
        "code_generation",
        "api_development",
        "authentication",
        "authorization",
        "database_integration",
        "code_modification",
    } and kinds <= {"final_explanation_output"}:
        return False, "implementation_requires_code_artifact"

    return True, "ok"


def complete_task(
    board: TaskProgressBoard,
    task_id: str,
    *,
    reason: str = "evidence_satisfied",
) -> None:
    rec = _require(board, task_id)
    if rec.status == "completed":
        # Idempotent duplicate completion — no mutation.
        return
    if rec.status != "running":
        raise TaskProgressError(
            f"cannot complete task from status {rec.status!r}"
        )
    ok, detail = _evidence_satisfies(rec)
    if not ok:
        raise TaskProgressError(f"completion rejected: {detail}")
    rec.completed_at = _now()
    rec.blocked_by = []
    _set_status(board, rec, "completed", action="complete", reason=reason)
    for other_id in board.tasks:
        if other_id == task_id:
            continue
        _recompute_blocked(
            board,
            other_id,
            reason=f"predecessor_completed:{task_id}",
        )


def _require(board: TaskProgressBoard, task_id: str) -> TaskProgressRecord:
    rec = board.tasks.get(task_id)
    if rec is None:
        raise TaskProgressError(f"unknown task id: {task_id!r}")
    return rec


def board_snapshot(board: TaskProgressBoard | None) -> dict[str, Any] | None:
    """JSON-friendly snapshot (persistence-ready; not persisted in B5)."""
    if board is None:
        return None
    return board.model_dump(mode="json")
