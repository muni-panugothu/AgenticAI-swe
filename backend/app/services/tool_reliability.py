"""Topics 45–47 — tool timeouts, tool-level retries, and policy fallbacks.

Separate from Topic 38 (``decide_llm_retry`` / provider retries). This module
only wraps authorized tool execution via ``execute_tool`` / prepare boundary.

Sync timeouts: stop *waiting* for a worker; do **not** kill the OS thread.
Global concurrency slots stay held until the worker actually finishes.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import random
import threading
import time
from concurrent.futures import Future, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Collection, Sequence

from app.services.request_cancellation import CancellationToken, RequestCancelled


def _call_execute_fn(
    execute_fn: Callable[..., dict],
    tool_name: str,
    tool_arguments: dict[str, Any] | None,
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    approval_id: str | None = None,
    execution_id: str | None = None,
    cancellation: CancellationToken | None = None,
) -> dict[str, Any]:
    """Invoke execute_fn, passing trusted lifecycle kwargs when accepted."""
    kwargs: dict[str, Any] = {
        "user_id": user_id,
        "request_id": request_id,
        "project_id": project_id,
        "tenant_id": tenant_id,
    }
    try:
        sig = inspect.signature(execute_fn)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        def _maybe(name: str, value: Any) -> None:
            if value is None:
                return
            if accepts_var_kw or name in params:
                kwargs[name] = value

        _maybe("approval_id", approval_id)
        _maybe("execution_id", execution_id)
        _maybe("cancellation", cancellation)
    except (TypeError, ValueError):
        if approval_id is not None:
            kwargs["approval_id"] = approval_id
        if execution_id is not None:
            kwargs["execution_id"] = execution_id
        if cancellation is not None:
            kwargs["cancellation"] = cancellation
    return execute_fn(tool_name, tool_arguments, **kwargs)

# ---------------------------------------------------------------------------
# Configuration (validated env with safe defaults)
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


DEFAULT_TOOL_TIMEOUT_SECONDS = _env_float(
    "AGENT_TOOL_TIMEOUT_SECONDS", 30.0, lo=0.5, hi=300.0
)
MAX_TOOL_TIMEOUT_SECONDS = _env_float(
    "AGENT_TOOL_TIMEOUT_MAX_SECONDS", 120.0, lo=1.0, hi=600.0
)
DEFAULT_TOOL_MAX_RETRIES = _env_int("AGENT_TOOL_MAX_RETRIES", 2, lo=0, hi=5)
DEFAULT_TOOL_RETRY_BUDGET_SECONDS = _env_float(
    "AGENT_TOOL_RETRY_BUDGET_SECONDS", 45.0, lo=1.0, hi=300.0
)
DEFAULT_TOOL_RETRY_BASE_SECONDS = _env_float(
    "AGENT_TOOL_RETRY_BASE_SECONDS", 0.25, lo=0.05, hi=10.0
)
DEFAULT_TOOL_RETRY_MAX_BACKOFF = _env_float(
    "AGENT_TOOL_RETRY_MAX_BACKOFF_SECONDS", 8.0, lo=0.1, hi=60.0
)
DEFAULT_TOOL_FALLBACK_DEPTH = _env_int(
    "AGENT_TOOL_FALLBACK_MAX_DEPTH", 1, lo=0, hi=3
)
DEFAULT_MAX_OUTSTANDING_TIMED_OUT = _env_int(
    "AGENT_TOOL_MAX_OUTSTANDING_TIMED_OUT", 16, lo=1, hi=64
)


class ToolFailureCategory(str, Enum):
    SUCCESS = "success"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    VALIDATION = "validation"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Metadata (extends Topics 43–44 concepts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolReliabilityMeta:
    name: str
    timeout_seconds: float | None = None
    retry_safe: bool = False
    idempotent: bool = False
    max_retries: int | None = None
    retryable_categories: frozenset[ToolFailureCategory] = frozenset(
        {
            ToolFailureCategory.TRANSIENT,
            ToolFailureCategory.RATE_LIMIT,
            ToolFailureCategory.TIMEOUT,
        }
    )
    fallbacks: tuple[str, ...] = ()


_RELIABILITY_META: dict[str, ToolReliabilityMeta] = {
    "calculator": ToolReliabilityMeta(
        "calculator",
        timeout_seconds=5.0,
        retry_safe=True,
        idempotent=True,
        max_retries=2,
        fallbacks=(),
    ),
    "get_current_time": ToolReliabilityMeta(
        "get_current_time",
        timeout_seconds=5.0,
        retry_safe=True,
        idempotent=True,
        max_retries=2,
    ),
    "search_code": ToolReliabilityMeta(
        "search_code",
        timeout_seconds=20.0,
        retry_safe=True,
        idempotent=True,
        max_retries=2,
        fallbacks=("read_file",),
    ),
    "read_file": ToolReliabilityMeta(
        "read_file",
        timeout_seconds=15.0,
        retry_safe=True,
        idempotent=True,
        max_retries=2,
    ),
    "web_search": ToolReliabilityMeta(
        "web_search",
        timeout_seconds=25.0,
        retry_safe=True,
        idempotent=False,
        max_retries=1,
        retryable_categories=frozenset(
            {ToolFailureCategory.TRANSIENT, ToolFailureCategory.RATE_LIMIT}
        ),
        fallbacks=("get_current_time",),
    ),
    "search_memory": ToolReliabilityMeta(
        "search_memory",
        timeout_seconds=20.0,
        retry_safe=True,
        idempotent=True,
        max_retries=1,
        fallbacks=("search_memory_summaries",),
    ),
    "search_memory_summaries": ToolReliabilityMeta(
        "search_memory_summaries",
        timeout_seconds=20.0,
        retry_safe=True,
        idempotent=True,
        max_retries=1,
    ),
    "save_memory": ToolReliabilityMeta(
        "save_memory",
        timeout_seconds=20.0,
        retry_safe=False,
        idempotent=False,
        max_retries=0,
        retryable_categories=frozenset(),
    ),
    "forget_memory": ToolReliabilityMeta(
        "forget_memory",
        timeout_seconds=20.0,
        retry_safe=False,
        idempotent=False,
        max_retries=0,
        retryable_categories=frozenset(),
    ),
    "reinforce_memory": ToolReliabilityMeta(
        "reinforce_memory",
        timeout_seconds=20.0,
        retry_safe=False,
        idempotent=False,
        max_retries=0,
        retryable_categories=frozenset(),
    ),
    "confirm_memory_update": ToolReliabilityMeta(
        "confirm_memory_update",
        timeout_seconds=20.0,
        retry_safe=False,
        idempotent=False,
        max_retries=0,
        retryable_categories=frozenset(),
    ),
    "request_memory_update": ToolReliabilityMeta(
        "request_memory_update",
        timeout_seconds=20.0,
        retry_safe=False,
        idempotent=False,
        max_retries=0,
        retryable_categories=frozenset(),
    ),
}


def get_reliability_meta(name: str) -> ToolReliabilityMeta:
    meta = _RELIABILITY_META.get(name)
    if meta is not None:
        return meta
    # Unknown tools: conservative — no retry, no fallback, default timeout.
    return ToolReliabilityMeta(
        name=name,
        timeout_seconds=DEFAULT_TOOL_TIMEOUT_SECONDS,
        retry_safe=False,
        idempotent=False,
        max_retries=0,
        retryable_categories=frozenset(),
        fallbacks=(),
    )


def resolve_tool_timeout(name: str) -> float:
    meta = get_reliability_meta(name)
    raw = (
        meta.timeout_seconds
        if meta.timeout_seconds is not None
        else DEFAULT_TOOL_TIMEOUT_SECONDS
    )
    return max(0.5, min(float(raw), MAX_TOOL_TIMEOUT_SECONDS))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_tool_failure(
    result: dict[str, Any] | None = None,
    *,
    exc: BaseException | None = None,
) -> ToolFailureCategory:
    if isinstance(exc, RequestCancelled):
        return ToolFailureCategory.CANCELLATION
    if result is not None and result.get("success"):
        return ToolFailureCategory.SUCCESS

    if result is not None:
        code = str(result.get("error_code") or result.get("error") or "").lower()
        details = str(result.get("details") or "").lower()
        blob = f"{code} {details}"
        if result.get("timed_out") or "tool_timeout" in blob or code == "tool_timeout":
            return ToolFailureCategory.TIMEOUT
        if result.get("worker_may_still_run"):
            return ToolFailureCategory.AMBIGUOUS
        if "invalid tool arguments" in blob or "validation" in blob:
            return ToolFailureCategory.VALIDATION
        if result.get("approval_required") or result.get("pending_human_review"):
            return ToolFailureCategory.AUTHORIZATION
        if (
            "authenticated user_id" in blob
            or "authorization" in blob
            or "permission" in blob
            or "forbidden" in blob
            or "approval_required" in blob
            or "approval_" in blob
            or "sandbox_" in blob
            or "untrusted_security" in blob
            or "tool_not_permitted" in blob
            # Sandbox / path denials are policy failures — never transient.
            or "path_denied" in blob
            or "path_traversal" in blob
            or code.startswith("path_")
            or code.startswith("sandbox_")
        ):
            return ToolFailureCategory.AUTHORIZATION
        if "tool not found" in blob or "no validation model" in blob:
            return ToolFailureCategory.PERMANENT
        if "rate" in blob and "limit" in blob:
            return ToolFailureCategory.RATE_LIMIT
        if any(
            token in blob
            for token in (
                "timeout",
                "temporar",
                "unavailable",
                "connection",
                "network",
                "503",
                "502",
                "429",
            )
        ):
            return ToolFailureCategory.TRANSIENT
        if result.get("mutation_committed") is False and result.get("success") is False:
            # Failed before commit — may be transient only if classified above.
            return ToolFailureCategory.UNKNOWN
        return ToolFailureCategory.UNKNOWN

    if exc is not None:
        text = str(exc).lower()
        if "timeout" in text:
            return ToolFailureCategory.TIMEOUT
        if any(t in text for t in ("temporar", "unavailable", "connection", "network")):
            return ToolFailureCategory.TRANSIENT
        return ToolFailureCategory.UNKNOWN

    return ToolFailureCategory.UNKNOWN


def parse_retry_after_seconds(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    raw = result.get("retry_after_seconds")
    if raw is None and isinstance(result.get("details"), dict):
        raw = result["details"].get("retry_after_seconds")
    try:
        if raw is None:
            return None
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


# ---------------------------------------------------------------------------
# Outstanding timed-out sync workers (honest capacity accounting)
# ---------------------------------------------------------------------------

_outstanding_lock = threading.Lock()
_outstanding_timed_out: dict[str, Future] = {}
_outstanding_count = 0


def outstanding_timed_out_workers() -> int:
    with _outstanding_lock:
        return len(_outstanding_timed_out)


def reset_reliability_state_for_tests() -> None:
    global _outstanding_count
    with _outstanding_lock:
        _outstanding_timed_out.clear()
        _outstanding_count = 0


def _track_outstanding(key: str, fut: Future) -> None:
    global _outstanding_count
    with _outstanding_lock:
        _outstanding_timed_out[key] = fut
        _outstanding_count = len(_outstanding_timed_out)


def _untrack_outstanding(key: str) -> None:
    global _outstanding_count
    with _outstanding_lock:
        _outstanding_timed_out.pop(key, None)
        _outstanding_count = len(_outstanding_timed_out)


def _has_outstanding(key: str) -> bool:
    with _outstanding_lock:
        fut = _outstanding_timed_out.get(key)
        if fut is None:
            return False
        return not fut.done()


# ---------------------------------------------------------------------------
# Timeout / retry helpers
# ---------------------------------------------------------------------------


@dataclass
class ReliabilityClock:
    """Injectable clock + sleeper for deterministic tests."""

    monotonic_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def monotonic(self) -> float:
        return float(self.monotonic_fn())

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.sleep_fn(seconds)

    def jittered_backoff(
        self,
        attempt: int,
        *,
        base: float = DEFAULT_TOOL_RETRY_BASE_SECONDS,
        cap: float = DEFAULT_TOOL_RETRY_MAX_BACKOFF,
        jitter_ratio: float = 0.2,
    ) -> float:
        exp = min(cap, base * (2 ** max(0, attempt - 1)))
        jitter = exp * jitter_ratio * self.rng.random()
        return min(cap, exp + jitter)


def _timeout_result(
    tool_name: str,
    *,
    worker_may_still_run: bool,
    attempts: int,
) -> dict[str, Any]:
    return {
        "success": False,
        "error": "Tool execution timed out",
        "error_code": "TOOL_TIMEOUT",
        "timed_out": True,
        "worker_may_still_run": bool(worker_may_still_run),
        "mutation_committed": False,
        "reliability": {
            "attempts": attempts,
            "failure_category": ToolFailureCategory.TIMEOUT.value,
            "timed_out": True,
            "worker_may_still_run": bool(worker_may_still_run),
            "original_tool": tool_name,
            "executed_tool": tool_name,
            "fallback_used": False,
        },
    }


def _annotate_reliability(
    result: dict[str, Any],
    *,
    original_tool: str,
    executed_tool: str,
    attempts: int,
    category: ToolFailureCategory,
    fallback_used: bool = False,
    fallback_denied_reason: str | None = None,
    worker_may_still_run: bool = False,
) -> dict[str, Any]:
    out = dict(result)
    rel = dict(out.get("reliability") or {})
    rel.update(
        {
            "attempts": attempts,
            "failure_category": category.value,
            "timed_out": bool(out.get("timed_out") or category == ToolFailureCategory.TIMEOUT),
            "worker_may_still_run": bool(
                worker_may_still_run or out.get("worker_may_still_run")
            ),
            "original_tool": original_tool,
            "executed_tool": executed_tool,
            "fallback_used": fallback_used,
            "fallback_denied_reason": fallback_denied_reason,
        }
    )
    out["reliability"] = rel
    return out


def cancel_aware_sleep(
    seconds: float,
    cancellation: CancellationToken | None,
    clock: ReliabilityClock,
) -> None:
    """Sleep in small slices; raise if cancelled."""
    deadline = clock.monotonic() + max(0.0, seconds)
    while True:
        if cancellation is not None:
            cancellation.check()
        remaining = deadline - clock.monotonic()
        if remaining <= 0:
            return
        clock.sleep(min(0.05, remaining))


def should_retry_tool(
    *,
    tool_name: str,
    category: ToolFailureCategory,
    attempt: int,
    elapsed: float,
    retry_after: float | None,
    worker_may_still_run: bool,
) -> tuple[bool, float, str]:
    """Return (retry?, wait_seconds, reason)."""
    meta = get_reliability_meta(tool_name)
    max_retries = (
        DEFAULT_TOOL_MAX_RETRIES if meta.max_retries is None else int(meta.max_retries)
    )
    if category == ToolFailureCategory.CANCELLATION:
        return False, 0.0, "cancelled"
    if category in {
        ToolFailureCategory.VALIDATION,
        ToolFailureCategory.AUTHORIZATION,
        ToolFailureCategory.PERMANENT,
        ToolFailureCategory.SUCCESS,
    }:
        return False, 0.0, f"non_retryable:{category.value}"
    if category == ToolFailureCategory.AMBIGUOUS:
        return False, 0.0, "ambiguous_outcome"
    if worker_may_still_run and not (meta.retry_safe and meta.idempotent):
        return False, 0.0, "worker_may_still_run"
    if not meta.retry_safe:
        return False, 0.0, "not_retry_safe"
    if category not in meta.retryable_categories:
        return False, 0.0, f"category_not_retryable:{category.value}"
    if attempt > max_retries:
        return False, 0.0, "max_attempts"
    remaining_budget = DEFAULT_TOOL_RETRY_BUDGET_SECONDS - elapsed
    if remaining_budget <= 0:
        return False, 0.0, "retry_budget_exhausted"
    wait = 0.0
    if retry_after is not None:
        wait = float(retry_after)
        if wait > remaining_budget:
            return False, 0.0, "retry_after_exceeds_budget"
    return True, wait, "retry_ok"


def select_fallback_tool(
    tool_name: str,
    category: ToolFailureCategory,
    *,
    permitted_tools: Collection[str] | None,
    registered_tools: Collection[str] | None,
    visited: Collection[str],
    depth: int,
) -> tuple[str | None, str | None]:
    """Pick next fallback or (None, denial_reason)."""
    if depth >= DEFAULT_TOOL_FALLBACK_DEPTH:
        return None, "fallback_depth_exceeded"
    if category in {
        ToolFailureCategory.CANCELLATION,
        ToolFailureCategory.AUTHORIZATION,
        ToolFailureCategory.VALIDATION,
        ToolFailureCategory.AMBIGUOUS,
        ToolFailureCategory.SUCCESS,
    }:
        return None, f"fallback_denied:{category.value}"
    meta = get_reliability_meta(tool_name)
    for candidate in meta.fallbacks:
        if candidate in visited:
            continue
        if registered_tools is not None and candidate not in registered_tools:
            return None, "fallback_unregistered"
        if permitted_tools is not None and candidate not in permitted_tools:
            return None, "fallback_not_permitted"
        # Same-name cycles already skipped via visited.
        return candidate, None
    return None, "no_eligible_fallback"


# ---------------------------------------------------------------------------
# Single-attempt invocation with timeout
# ---------------------------------------------------------------------------


def invoke_sync_once_with_timeout(
    tool_name: str,
    tool_arguments: dict[str, Any] | None,
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    cancellation: CancellationToken | None = None,
    execute_fn: Callable[..., dict] | None = None,
    timeout_seconds: float | None = None,
    call_key: str | None = None,
    clock: ReliabilityClock | None = None,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Run one sync attempt with caller timeout (cooperative slot accounting)."""
    from app.services.tool_execution import (
        _global_slots,
        get_global_tool_pool,
    )

    clock = clock or ReliabilityClock()
    if timeout_seconds is not None:
        # Explicit caller timeout may be sub-second (tests / tight budgets).
        timeout = max(0.01, min(float(timeout_seconds), MAX_TOOL_TIMEOUT_SECONDS))
    else:
        timeout = max(0.5, min(resolve_tool_timeout(tool_name), MAX_TOOL_TIMEOUT_SECONDS))
    key = call_key or f"{tool_name}:{id(tool_arguments)}"

    if cancellation is not None:
        cancellation.check()
    if execute_fn is None:
        from app.chatbot import execute_tool as execute_fn  # type: ignore[assignment]

    if outstanding_timed_out_workers() >= DEFAULT_MAX_OUTSTANDING_TIMED_OUT:
        return {
            "success": False,
            "error": "Too many outstanding timed-out tool workers",
            "error_code": "TOOL_WORKER_CAPACITY",
            "mutation_committed": False,
        }

    acquired = _global_slots.acquire(blocking=True, timeout=min(120.0, timeout + 30.0))
    if not acquired:
        return {
            "success": False,
            "error": "global_tool_concurrency_timeout",
            "error_code": "TOOL_CONCURRENCY",
            "mutation_committed": False,
        }

    # True while *this* caller still owns the slot release duty.
    slot_owned = True

    # Pre-reserve a commit lease for protected mutating tools so the caller
    # can invalidate on timeout/cancel even before the worker finishes prepare.
    from app.services.execution_lifecycle import (
        invalidate_by_approval,
        invalidate_execution,
        requires_commit_guard,
        reserve_execution,
    )

    execution_id: str | None = None
    if requires_commit_guard(tool_name):
        execution_id = reserve_execution(
            tool_name=tool_name,
            approval_id=approval_id,
            user_id=user_id,
            request_id=request_id,
        )

    def _invalidate_late_worker(reason: str) -> None:
        # Does not kill the OS thread. Blocks late commits via commit guard.
        invalidate_execution(execution_id, reason=reason)
        if approval_id:
            try:
                from app.services.approval_service import mark_execution_failed

                invalidate_by_approval(str(approval_id), reason=reason)
                mark_execution_failed(str(approval_id))
            except Exception:
                pass

    def _run() -> dict[str, Any]:
        if cancellation is not None:
            cancellation.check()
        # approval_id / execution_id are trusted server kwargs — never from LLM.
        return _call_execute_fn(
            execute_fn,
            tool_name,
            tool_arguments,
            user_id=user_id,
            request_id=request_id,
            project_id=project_id,
            tenant_id=tenant_id,
            approval_id=approval_id,
            execution_id=execution_id,
            cancellation=cancellation,
        )

    pool = get_global_tool_pool()
    fut: Future = pool.submit(_run)

    def _release_when_done(_f: Future) -> None:
        try:
            if _f.done():
                try:
                    _f.result()
                except Exception:
                    pass
        finally:
            _global_slots.release()
            _untrack_outstanding(key)

    try:
        deadline = clock.monotonic() + timeout
        while True:
            if cancellation is not None and cancellation.is_cancelled():
                # Do not kill the worker; invalidate commit lease and keep slot.
                _invalidate_late_worker(f"cancelled:{cancellation.reason}")
                _track_outstanding(key, fut)
                fut.add_done_callback(_release_when_done)
                slot_owned = False
                raise RequestCancelled(cancellation.reason)
            remaining = deadline - clock.monotonic()
            if remaining <= 0:
                break
            try:
                return fut.result(timeout=min(0.05, remaining))
            except FuturesTimeout:
                continue
        # Caller timeout — worker may still run; slot released only when done.
        # Invalidate the commit lease so protected tools cannot late-commit.
        _track_outstanding(key, fut)
        fut.add_done_callback(_release_when_done)
        slot_owned = False
        _invalidate_late_worker("timeout")
        return _timeout_result(tool_name, worker_may_still_run=True, attempts=1)
    except RequestCancelled:
        raise
    except Exception as exc:
        return {
            "success": False,
            "error": "Tool execution exception",
            "details": str(exc)[:300],
            "mutation_committed": False,
        }
    finally:
        if slot_owned:
            _global_slots.release()
            slot_owned = False


async def invoke_async_once_with_timeout(
    tool_name: str,
    tool_arguments: dict[str, Any] | None,
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    cancellation: CancellationToken | None = None,
    execute_fn: Callable[..., Any] | None = None,
    timeout_seconds: float | None = None,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Async attempt with ``asyncio.timeout``; cancels the awaited coroutine."""
    from app.services.tool_execution import (
        _acquire_global_slot_async,
        _global_slots,
        is_native_async_callable,
    )

    if timeout_seconds is not None:
        timeout = max(0.01, min(float(timeout_seconds), MAX_TOOL_TIMEOUT_SECONDS))
    else:
        timeout = max(0.5, min(resolve_tool_timeout(tool_name), MAX_TOOL_TIMEOUT_SECONDS))
    if cancellation is not None:
        cancellation.check()

    async def _call() -> dict[str, Any]:
        if execute_fn is not None and is_native_async_callable(execute_fn):
            await _acquire_global_slot_async(cancellation)
            try:
                result = await execute_fn(
                    tool_name,
                    tool_arguments,
                    user_id=user_id,
                    request_id=request_id,
                    project_id=project_id,
                    tenant_id=tenant_id,
                )
                return (
                    result
                    if isinstance(result, dict)
                    else {"success": True, "result": result}
                )
            finally:
                _global_slots.release()

        if execute_fn is None:
            from app.tool_registry import tools as registry

            tool_fn = registry.get(tool_name)
            if tool_fn is not None and inspect.iscoroutinefunction(tool_fn):
                from app.chatbot import (
                    finalize_tool_error,
                    finalize_tool_success,
                    prepare_tool_invocation,
                )

                await _acquire_global_slot_async(cancellation)
                raw_result = None
                try:
                    if cancellation is not None:
                        cancellation.check()
                    err, tool_function, call_kwargs = prepare_tool_invocation(
                        tool_name,
                        tool_arguments,
                        user_id=user_id,
                        request_id=request_id,
                        project_id=project_id,
                        tenant_id=tenant_id,
                        approval_id=approval_id,
                    )
                    if err is not None:
                        return err
                    assert tool_function is not None and call_kwargs is not None
                    raw_result = await tool_function(**call_kwargs)
                    return finalize_tool_success(tool_name, raw_result)
                except RequestCancelled:
                    raise
                except Exception as exc:
                    return finalize_tool_error(tool_name, raw_result, exc)
                finally:
                    _global_slots.release()

        # Sync path via thread — still honor timeout on the await side.
        return await asyncio.to_thread(
            invoke_sync_once_with_timeout,
            tool_name,
            tool_arguments,
            user_id=user_id,
            request_id=request_id,
            project_id=project_id,
            tenant_id=tenant_id,
            cancellation=cancellation,
            execute_fn=execute_fn,
            timeout_seconds=timeout,
        )

    try:
        async with asyncio.timeout(timeout):
            # Also poll cancellation while awaiting.
            task = asyncio.create_task(_call())
            while True:
                if cancellation is not None and cancellation.is_cancelled():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, RequestCancelled):
                        pass
                    raise RequestCancelled(cancellation.reason)
                done, _ = await asyncio.wait({task}, timeout=0.05)
                if task in done:
                    return task.result()
    except TimeoutError:
        return _timeout_result(tool_name, worker_may_still_run=False, attempts=1)


# ---------------------------------------------------------------------------
# Full reliability pipeline (timeout + retry + fallback)
# ---------------------------------------------------------------------------


def run_tool_with_reliability(
    tool_name: str,
    tool_arguments: dict[str, Any] | None,
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    cancellation: CancellationToken | None = None,
    execute_fn: Callable[..., dict] | None = None,
    permitted_tools: Collection[str] | None = None,
    registered_tools: Collection[str] | None = None,
    clock: ReliabilityClock | None = None,
    fallback_depth: int = 0,
    visited: Sequence[str] | None = None,
    tool_call_id: str | None = None,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Execute one authorized tool with timeout, bounded retries, and fallbacks."""
    clock = clock or ReliabilityClock()
    started = clock.monotonic()
    visited_list = list(visited or ())
    visited_list.append(tool_name)
    original_tool = visited_list[0]
    call_key = f"{tool_call_id or tool_name}:{request_id or ''}:{tool_name}"
    # Approval binds to one tool+args; never reuse on fallback tools.
    active_approval_id = approval_id if fallback_depth == 0 else None

    # Intent / turn allowlist is an execution gate when provided (not fallback-only).
    if permitted_tools is not None and tool_name not in permitted_tools:
        denied = {
            "success": False,
            "error": "Tool not permitted for this turn",
            "error_code": "TOOL_NOT_PERMITTED",
            "mutation_committed": False,
            "execution_completed": False,
        }
        return _annotate_reliability(
            denied,
            original_tool=original_tool,
            executed_tool=tool_name,
            attempts=0,
            category=ToolFailureCategory.AUTHORIZATION,
            fallback_used=False,
            fallback_denied_reason="tool_not_permitted",
        )

    meta = get_reliability_meta(tool_name)
    max_retries = (
        DEFAULT_TOOL_MAX_RETRIES if meta.max_retries is None else int(meta.max_retries)
    )
    attempt = 0
    last_result: dict[str, Any] = {
        "success": False,
        "error": "Tool execution error",
        "mutation_committed": False,
    }
    last_category = ToolFailureCategory.UNKNOWN

    while True:
        attempt += 1
        if cancellation is not None:
            cancellation.check()

        if _has_outstanding(call_key) and not (meta.retry_safe and meta.idempotent):
            last_result = {
                "success": False,
                "error": "Previous timed-out attempt may still be running",
                "error_code": "TOOL_AMBIGUOUS_IN_FLIGHT",
                "worker_may_still_run": True,
                "mutation_committed": False,
            }
            last_category = ToolFailureCategory.AMBIGUOUS
            break

        try:
            last_result = invoke_sync_once_with_timeout(
                tool_name,
                tool_arguments,
                user_id=user_id,
                request_id=request_id,
                project_id=project_id,
                tenant_id=tenant_id,
                cancellation=cancellation,
                execute_fn=execute_fn,
                timeout_seconds=resolve_tool_timeout(tool_name),
                call_key=call_key,
                clock=clock,
                approval_id=active_approval_id,
            )
        except RequestCancelled:
            raise

        last_category = classify_tool_failure(last_result)
        if last_category == ToolFailureCategory.SUCCESS:
            return _annotate_reliability(
                last_result,
                original_tool=original_tool,
                executed_tool=tool_name,
                attempts=attempt,
                category=last_category,
                fallback_used=(tool_name != original_tool),
            )

        elapsed = clock.monotonic() - started
        retry_after = parse_retry_after_seconds(last_result)
        worker_may = bool(last_result.get("worker_may_still_run"))
        ok, wait_hint, reason = should_retry_tool(
            tool_name=tool_name,
            category=last_category,
            attempt=attempt,
            elapsed=elapsed,
            retry_after=retry_after,
            worker_may_still_run=worker_may,
        )
        if not ok or attempt > max_retries:
            break

        wait = wait_hint
        if wait <= 0:
            wait = clock.jittered_backoff(attempt)
        remaining = DEFAULT_TOOL_RETRY_BUDGET_SECONDS - (clock.monotonic() - started)
        if wait > remaining:
            break
        try:
            cancel_aware_sleep(wait, cancellation, clock)
        except RequestCancelled:
            raise

    # Fallback pass (explicit, allowlist-bound).
    if registered_tools is None:
        try:
            from app.tool_registry import tools as registry

            registered_tools = set(registry.keys())
        except Exception:
            registered_tools = set()

    fallback, deny = select_fallback_tool(
        tool_name,
        last_category,
        permitted_tools=permitted_tools,
        registered_tools=registered_tools,
        visited=visited_list,
        depth=fallback_depth,
    )
    if fallback is None:
        return _annotate_reliability(
            last_result,
            original_tool=original_tool,
            executed_tool=tool_name,
            attempts=attempt,
            category=last_category,
            fallback_used=False,
            fallback_denied_reason=deny,
            worker_may_still_run=bool(last_result.get("worker_may_still_run")),
        )

    if cancellation is not None:
        cancellation.check()

    fb_result = run_tool_with_reliability(
        fallback,
        tool_arguments,
        user_id=user_id,
        request_id=request_id,
        project_id=project_id,
        tenant_id=tenant_id,
        cancellation=cancellation,
        execute_fn=execute_fn,
        permitted_tools=permitted_tools,
        registered_tools=registered_tools,
        clock=clock,
        fallback_depth=fallback_depth + 1,
        visited=visited_list,
        tool_call_id=tool_call_id,
        approval_id=None,
    )
    fb_cat = classify_tool_failure(fb_result)
    annotated = _annotate_reliability(
        fb_result,
        original_tool=original_tool,
        executed_tool=str(
            (fb_result.get("reliability") or {}).get("executed_tool") or fallback
        ),
        attempts=attempt + int((fb_result.get("reliability") or {}).get("attempts") or 1),
        category=fb_cat if fb_cat == ToolFailureCategory.SUCCESS else last_category,
        fallback_used=True,
        worker_may_still_run=bool(fb_result.get("worker_may_still_run")),
    )
    # Never claim the original tool succeeded if only the fallback did.
    if annotated.get("success") and original_tool != annotated["reliability"]["executed_tool"]:
        annotated["reliability"]["original_tool_succeeded"] = False
        annotated["reliability"]["fallback_succeeded"] = True
    return annotated
