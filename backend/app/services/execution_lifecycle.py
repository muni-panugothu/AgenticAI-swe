"""Server-owned execution lifecycle for late-worker commit safety.

Synchronous Python threads are never forcibly killed. Instead, protected
side-effect tools must call ``try_begin_commit`` immediately before an
irreversible write. If the caller has timed out or cancelled, the commit
is rejected even if the worker is still running.

LLM/client code cannot forge execution IDs or flip commit authorization.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from app.services.agent_security_audit import record_security_event
from app.services.request_cancellation import CancellationToken

# Tools that implement (or inherit via memory_service) a commit boundary.
COMMIT_GUARDED_TOOLS: frozenset[str] = frozenset({
    "save_memory",
    "forget_memory",
    "reinforce_memory",
    "request_memory_update",
    "confirm_memory_update",
    "summarize_memories",
})

# High-risk registered side-effect tools that lack a safe commit boundary.
# Keep empty unless a tool is registered without a guard — fail closed.
COMMIT_UNSUPPORTED_PROTECTED_TOOLS: frozenset[str] = frozenset()


class CommitRejected(Exception):
    """Raised when a side effect must not commit after invalidation."""

    def __init__(
        self,
        reason: str = "execution_invalidated",
        *,
        error_code: str = "COMMIT_REJECTED",
    ) -> None:
        self.reason = reason
        self.error_code = error_code
        super().__init__(reason)


@dataclass
class ExecutionRecord:
    execution_id: str
    tool_name: str
    approval_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None
    # reserved → active → committing → committed
    # reserved|active → invalidated
    state: str = "reserved"
    invalidate_reason: str | None = None
    commit_count: int = 0
    created_at: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


_REGISTRY: dict[str, ExecutionRecord] = {}
_BY_APPROVAL: dict[str, str] = {}
_REG_LOCK = threading.RLock()

_current_execution: ContextVar[ExecutionRecord | None] = ContextVar(
    "execution_lifecycle_current",
    default=None,
)
_current_cancellation: ContextVar[CancellationToken | None] = ContextVar(
    "execution_lifecycle_cancel",
    default=None,
)


def reset_execution_lifecycle_for_tests() -> None:
    with _REG_LOCK:
        _REGISTRY.clear()
        _BY_APPROVAL.clear()


def requires_commit_guard(tool_name: str) -> bool:
    return str(tool_name or "") in COMMIT_GUARDED_TOOLS


def is_commit_unsupported_protected(tool_name: str) -> bool:
    return str(tool_name or "") in COMMIT_UNSUPPORTED_PROTECTED_TOOLS


def new_execution_id() -> str:
    return str(uuid.uuid4())


def reserve_execution(
    *,
    tool_name: str,
    execution_id: str | None = None,
    approval_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Create a reserved lease before the worker starts (caller-owned id)."""
    eid = execution_id or new_execution_id()
    rec = ExecutionRecord(
        execution_id=eid,
        tool_name=str(tool_name or ""),
        approval_id=str(approval_id) if approval_id else None,
        user_id=str(user_id) if user_id else None,
        request_id=str(request_id) if request_id else None,
        state="reserved",
    )
    with _REG_LOCK:
        _REGISTRY[eid] = rec
        if rec.approval_id:
            _BY_APPROVAL[rec.approval_id] = eid
    return eid


def activate_execution(
    execution_id: str,
    *,
    approval_id: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    tool_name: str | None = None,
) -> bool:
    """Mark reserved→active after authorization/prepare succeeds.

    Fails closed for unknown execution IDs (no silent create — prevents forgery).
    """
    with _REG_LOCK:
        rec = _REGISTRY.get(execution_id)
        if rec is None:
            return False
        with rec.lock:
            if rec.state == "invalidated":
                return False
            if rec.state == "reserved":
                rec.state = "active"
            if approval_id and not rec.approval_id:
                rec.approval_id = str(approval_id)
                _BY_APPROVAL[rec.approval_id] = execution_id
            if user_id and not rec.user_id:
                rec.user_id = str(user_id)
            if request_id and not rec.request_id:
                rec.request_id = str(request_id)
            if tool_name and not rec.tool_name:
                rec.tool_name = str(tool_name)
            return rec.state == "active"


def bind_execution(
    execution_id: str | None,
    *,
    cancellation: CancellationToken | None = None,
) -> tuple[Token, Token]:
    """Bind execution (+ optional cancel token) to this worker context."""
    rec = None
    if execution_id:
        with _REG_LOCK:
            rec = _REGISTRY.get(execution_id)
    t1 = _current_execution.set(rec)
    t2 = _current_cancellation.set(cancellation)
    return t1, t2


def unbind_execution(tokens: tuple[Token, Token]) -> None:
    _current_execution.reset(tokens[0])
    _current_cancellation.reset(tokens[1])


def get_execution(execution_id: str | None) -> ExecutionRecord | None:
    if not execution_id:
        return None
    with _REG_LOCK:
        return _REGISTRY.get(execution_id)


def current_execution() -> ExecutionRecord | None:
    return _current_execution.get()


def invalidate_execution(
    execution_id: str | None,
    reason: str = "timeout",
) -> bool:
    """Invalidate if commit has not started. Returns True if invalidated."""
    if not execution_id:
        return False
    with _REG_LOCK:
        rec = _REGISTRY.get(execution_id)
        if rec is None:
            return False
    with rec.lock:
        if rec.state in {"committed", "committing"}:
            # Commit already won or in progress — do not rewrite history.
            return False
        if rec.state == "invalidated":
            return True
        rec.state = "invalidated"
        rec.invalidate_reason = str(reason or "timeout")
    record_security_event(
        "execution_failed",
        tool_name=rec.tool_name,
        request_id=rec.request_id,
        approval_id=rec.approval_id,
        detail={"reason": "invalidated", "cause": rec.invalidate_reason},
    )
    return True


def invalidate_by_approval(
    approval_id: str | None,
    reason: str = "timeout",
) -> bool:
    if not approval_id:
        return False
    with _REG_LOCK:
        eid = _BY_APPROVAL.get(str(approval_id))
    return invalidate_execution(eid, reason=reason)


def try_begin_commit() -> bool:
    """Acquire the commit boundary. False means the side effect must not occur.

    No-op allow when no execution is bound (legacy direct service unit tests).
    """
    rec = _current_execution.get()
    if rec is None:
        return True
    cancel = _current_cancellation.get()
    with rec.lock:
        if cancel is not None and cancel.is_cancelled():
            if rec.state not in {"committed", "committing"}:
                rec.state = "invalidated"
                rec.invalidate_reason = f"cancelled:{cancel.reason}"
            return False
        if rec.state == "committed":
            # Idempotent re-entry after successful commit.
            return True
        if rec.state != "active":
            return False
        rec.state = "committing"
        return True


def finish_commit_success() -> None:
    rec = _current_execution.get()
    if rec is None:
        return
    with rec.lock:
        if rec.state in {"committing", "active", "committed"}:
            rec.state = "committed"
            rec.commit_count += 1


def finish_commit_abort() -> None:
    """Abort a try_begin_commit that did not write (rollback to active/invalidated)."""
    rec = _current_execution.get()
    if rec is None:
        return
    cancel = _current_cancellation.get()
    with rec.lock:
        if rec.state != "committing":
            return
        if cancel is not None and cancel.is_cancelled():
            rec.state = "invalidated"
            rec.invalidate_reason = f"cancelled:{cancel.reason}"
        elif rec.invalidate_reason:
            rec.state = "invalidated"
        else:
            rec.state = "active"


def assert_commit_allowed() -> None:
    if not try_begin_commit():
        rec = _current_execution.get()
        reason = (rec.invalidate_reason if rec else None) or "execution_invalidated"
        raise CommitRejected(reason)


def commit_rejected_result(exc: CommitRejected | None = None) -> dict[str, Any]:
    reason = exc.reason if exc else "execution_invalidated"
    code = exc.error_code if exc else "COMMIT_REJECTED"
    return {
        "success": False,
        "error": "Protected side effect rejected after timeout or cancellation",
        "error_code": code,
        "reason": reason,
        "mutation_committed": False,
        "execution_completed": False,
        "commit_rejected": True,
    }


def execution_public_state(execution_id: str | None) -> dict[str, Any] | None:
    rec = get_execution(execution_id)
    if rec is None:
        return None
    with rec.lock:
        return {
            "execution_id": rec.execution_id,
            "tool_name": rec.tool_name,
            "approval_id": rec.approval_id,
            "state": rec.state,
            "invalidate_reason": rec.invalidate_reason,
            "commit_count": rec.commit_count,
        }
