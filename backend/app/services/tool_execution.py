"""Async + parallel tool execution (Topics 43–44) — agent-loop dispatcher.

``chatbot.execute_tool`` remains the authoritative validation / allowlist /
scope-injection boundary. This module only schedules that function (or a
test double) and never bypasses it.

- Native async tool callables are awaited directly (not wrapped in to_thread).
- Sync tools use a process-wide bounded ThreadPoolExecutor.
- A process-wide semaphore caps concurrent tool work across requests.
- Parallelism is opt-in via explicit metadata; unknown/mutating tools stay sequential.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait as futures_wait,
)
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.services.request_cancellation import CancellationToken, RequestCancelled

DEFAULT_MAX_PARALLEL_TOOLS = max(
    1, int(os.getenv("AGENT_MAX_PARALLEL_TOOLS", "3"))
)
DEFAULT_TOOL_POOL_WORKERS = max(
    1, int(os.getenv("AGENT_TOOL_POOL_WORKERS", "4"))
)
# Global cap across all requests (in addition to per-batch limits).
DEFAULT_GLOBAL_TOOL_CONCURRENCY = max(
    1, int(os.getenv("AGENT_GLOBAL_TOOL_CONCURRENCY", "8"))
)

_pool_lock = threading.Lock()
_global_pool: ThreadPoolExecutor | None = None
_global_slots = threading.BoundedSemaphore(DEFAULT_GLOBAL_TOOL_CONCURRENCY)


def get_global_tool_pool() -> ThreadPoolExecutor:
    """Process-wide bounded pool for synchronous tool workers."""
    global _global_pool
    with _pool_lock:
        if _global_pool is None:
            _global_pool = ThreadPoolExecutor(
                max_workers=DEFAULT_TOOL_POOL_WORKERS,
                thread_name_prefix="agent-tool",
            )
        return _global_pool


def reset_global_tool_pool_for_tests() -> None:
    """Test helper — recreate pool/semaphore with current env defaults."""
    global _global_pool, _global_slots
    with _pool_lock:
        if _global_pool is not None:
            _global_pool.shutdown(wait=False, cancel_futures=True)
            _global_pool = None
        _global_slots = threading.BoundedSemaphore(
            max(1, int(os.getenv("AGENT_GLOBAL_TOOL_CONCURRENCY", "8")))
        )
    try:
        from app.services.tool_reliability import reset_reliability_state_for_tests

        reset_reliability_state_for_tests()
    except Exception:
        pass


@dataclass(frozen=True)
class ToolExecutionMeta:
    name: str
    supports_async: bool = False
    parallel_safe: bool = False
    read_only: bool = False
    idempotent: bool = False
    max_concurrency: int = 1
    # Topics 45–47 — declared here for discovery; authoritative policy lives in
    # ``tool_reliability.get_reliability_meta``.
    timeout_seconds: float | None = None
    retry_safe: bool = False
    max_retries: int | None = None
    fallbacks: tuple[str, ...] = ()


_TOOL_META: dict[str, ToolExecutionMeta] = {
    "calculator": ToolExecutionMeta(
        "calculator",
        parallel_safe=True,
        read_only=True,
        idempotent=True,
        max_concurrency=4,
    ),
    "get_current_time": ToolExecutionMeta(
        "get_current_time",
        parallel_safe=True,
        read_only=True,
        idempotent=True,
        max_concurrency=4,
    ),
    "search_code": ToolExecutionMeta(
        "search_code",
        parallel_safe=True,
        read_only=True,
        idempotent=True,
        max_concurrency=3,
    ),
    "read_file": ToolExecutionMeta(
        "read_file",
        parallel_safe=True,
        read_only=True,
        idempotent=True,
        max_concurrency=3,
    ),
    "web_search": ToolExecutionMeta(
        "web_search",
        parallel_safe=True,
        read_only=True,
        idempotent=False,
        max_concurrency=2,
    ),
    "search_memory": ToolExecutionMeta(
        "search_memory", read_only=True, parallel_safe=False, max_concurrency=1
    ),
    "search_memory_summaries": ToolExecutionMeta(
        "search_memory_summaries",
        read_only=True,
        parallel_safe=False,
        max_concurrency=1,
    ),
    "inspect_memory_metadata": ToolExecutionMeta(
        "inspect_memory_metadata",
        read_only=True,
        parallel_safe=False,
        max_concurrency=1,
    ),
    "expand_memory_summary": ToolExecutionMeta(
        "expand_memory_summary",
        read_only=True,
        parallel_safe=False,
        max_concurrency=1,
    ),
    "detect_memory_conflict": ToolExecutionMeta(
        "detect_memory_conflict",
        read_only=True,
        parallel_safe=False,
        max_concurrency=1,
    ),
    "save_memory": ToolExecutionMeta(
        "save_memory", parallel_safe=False, max_concurrency=1
    ),
    "reinforce_memory": ToolExecutionMeta(
        "reinforce_memory", parallel_safe=False, max_concurrency=1
    ),
    "forget_memory": ToolExecutionMeta(
        "forget_memory", parallel_safe=False, max_concurrency=1
    ),
    "request_memory_update": ToolExecutionMeta(
        "request_memory_update", parallel_safe=False, max_concurrency=1
    ),
    "confirm_memory_update": ToolExecutionMeta(
        "confirm_memory_update", parallel_safe=False, max_concurrency=1
    ),
    "summarize_memories": ToolExecutionMeta(
        "summarize_memories", parallel_safe=False, max_concurrency=1
    ),
}


def get_tool_meta(name: str) -> ToolExecutionMeta:
    meta = _TOOL_META.get(name)
    if meta is not None:
        return meta
    return ToolExecutionMeta(name=name, parallel_safe=False, max_concurrency=1)


def is_parallel_safe(name: str) -> bool:
    meta = get_tool_meta(name)
    return bool(meta.parallel_safe and meta.read_only)


def is_native_async_callable(fn: Any) -> bool:
    if fn is None:
        return False
    if inspect.iscoroutinefunction(fn):
        return True
    # Partial / wrappers
    return bool(getattr(fn, "_agent_native_async", False))


@dataclass
class PreparedToolCall:
    index: int
    tool_call_id: str
    tool_name: str
    tool_arguments: dict[str, Any]
    context: dict[str, Any] | None = None


def execute_tool_sync(
    tool_name: str,
    tool_arguments: dict[str, Any] | None,
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    cancellation: CancellationToken | None = None,
    execute_fn: Callable[..., dict] | None = None,
    permitted_tools: Sequence[str] | None = None,
    tool_call_id: str | None = None,
    enable_reliability: bool = True,
    approval_id: str | None = None,
) -> dict[str, Any]:
    """Run authoritative ``execute_tool`` with Topics 45–47 reliability.

    When ``enable_reliability`` is True (default), applies per-tool timeout,
    bounded tool retries, and policy fallbacks. Topic 38 provider retries are
    never invoked here.
    """
    if cancellation is not None:
        cancellation.check()
    if enable_reliability:
        from app.services.tool_reliability import run_tool_with_reliability

        return run_tool_with_reliability(
            tool_name,
            tool_arguments,
            user_id=user_id,
            request_id=request_id,
            project_id=project_id,
            tenant_id=tenant_id,
            cancellation=cancellation,
            execute_fn=execute_fn,
            permitted_tools=permitted_tools,
            tool_call_id=tool_call_id,
            approval_id=approval_id,
        )

    if execute_fn is None:
        from app.chatbot import execute_tool as execute_fn  # type: ignore[assignment]

    acquired = _global_slots.acquire(blocking=True, timeout=120)
    if not acquired:
        return {"success": False, "error": "global_tool_concurrency_timeout"}
    try:
        if cancellation is not None:
            cancellation.check()
        from app.services.tool_reliability import _call_execute_fn

        return _call_execute_fn(
            execute_fn,
            tool_name,
            tool_arguments,
            user_id=user_id,
            request_id=request_id,
            project_id=project_id,
            tenant_id=tenant_id,
            approval_id=approval_id,
        )
    finally:
        _global_slots.release()


async def _acquire_global_slot_async(
    cancellation: CancellationToken | None,
) -> None:
    """Wait for a global concurrency slot without blocking the event loop."""
    while True:
        if cancellation is not None:
            cancellation.check()
        if _global_slots.acquire(blocking=False):
            return
        await asyncio.sleep(0.01)


async def execute_tool_async(
    tool_name: str,
    tool_arguments: dict[str, Any] | None,
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    cancellation: CancellationToken | None = None,
    execute_fn: Callable[..., Any] | None = None,
    permitted_tools: Sequence[str] | None = None,
    tool_call_id: str | None = None,
    enable_reliability: bool = True,
) -> dict[str, Any]:
    """Dispatch one tool without blocking the event loop.

    - Native async tools: awaited with ``asyncio.timeout`` (Topic 45).
    - Sync tools: reliability path via worker pool + caller timeout.
    """
    if cancellation is not None:
        cancellation.check()

    if enable_reliability:
        # Prefer async timeout path for native async callables / registry tools.
        if execute_fn is None or is_native_async_callable(execute_fn):
            from app.services.tool_reliability import invoke_async_once_with_timeout

            # Single-attempt async timeout; retries/fallbacks use sync reliability
            # wrapper for deterministic policy (same classifier / budgets).
            if execute_fn is not None and is_native_async_callable(execute_fn):
                return await invoke_async_once_with_timeout(
                    tool_name,
                    tool_arguments,
                    user_id=user_id,
                    request_id=request_id,
                    project_id=project_id,
                    tenant_id=tenant_id,
                    cancellation=cancellation,
                    execute_fn=execute_fn,
                )
            from app.tool_registry import tools as registry

            tool_fn = registry.get(tool_name) if execute_fn is None else None
            if tool_fn is not None and inspect.iscoroutinefunction(tool_fn):
                return await invoke_async_once_with_timeout(
                    tool_name,
                    tool_arguments,
                    user_id=user_id,
                    request_id=request_id,
                    project_id=project_id,
                    tenant_id=tenant_id,
                    cancellation=cancellation,
                    execute_fn=None,
                )

        return await asyncio.to_thread(
            execute_tool_sync,
            tool_name,
            tool_arguments,
            user_id=user_id,
            request_id=request_id,
            project_id=project_id,
            tenant_id=tenant_id,
            cancellation=cancellation,
            execute_fn=execute_fn,
            permitted_tools=permitted_tools,
            tool_call_id=tool_call_id,
            enable_reliability=True,
        )

    # Legacy path without Topics 45–47.
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
                    approval_id=None,
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

    return await asyncio.to_thread(
        execute_tool_sync,
        tool_name,
        tool_arguments,
        user_id=user_id,
        request_id=request_id,
        project_id=project_id,
        tenant_id=tenant_id,
        cancellation=cancellation,
        execute_fn=execute_fn,
        enable_reliability=False,
    )


def _cap_workers(names: Sequence[str], requested: int) -> int:
    if requested < 1:
        requested = 1
    per_tool = min(get_tool_meta(n).max_concurrency for n in names) if names else 1
    return max(
        1,
        min(
            requested,
            per_tool,
            DEFAULT_MAX_PARALLEL_TOOLS,
            DEFAULT_GLOBAL_TOOL_CONCURRENCY,
            len(names) or 1,
        ),
    )


def can_parallelize(calls: Sequence[PreparedToolCall]) -> bool:
    if len(calls) < 2:
        return False
    return all(is_parallel_safe(c.tool_name) for c in calls)


def partition_execution_batches(
    calls: Sequence[PreparedToolCall],
) -> list[list[PreparedToolCall]]:
    """Split into maximal parallel-safe runs; unsafe tools become solo batches."""
    batches: list[list[PreparedToolCall]] = []
    current: list[PreparedToolCall] = []
    for call in calls:
        if is_parallel_safe(call.tool_name):
            current.append(call)
        else:
            if current:
                batches.append(current)
                current = []
            batches.append([call])
    if current:
        batches.append(current)
    return batches


def execute_tools_bounded(
    calls: Sequence[PreparedToolCall],
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    cancellation: CancellationToken | None = None,
    max_concurrency: int | None = None,
    execute_fn: Callable[..., dict] | None = None,
    permitted_tools: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute prepared calls in original order with bounded parallelism.

    Used by the real agent loop. Parallel only when every call in a partition
    is parallel_safe; otherwise sequential. Always goes through ``execute_fn``
    / ``execute_tool`` (via Topics 45–47 reliability wrapper).
    """
    if not calls:
        return []
    if cancellation is not None:
        cancellation.check()

    # Preserve global order: run partitions in sequence, parallel within a safe batch.
    # Always key results by PreparedToolCall.index — never by tool_call_id alone
    # (duplicate ids from a model must not overwrite sibling slots).
    ordered: list[dict[str, Any] | None] = [None] * len(calls)

    for batch in partition_execution_batches(list(calls)):
        if cancellation is not None:
            cancellation.check()

        if len(batch) == 1 or not can_parallelize(batch):
            for call in batch:
                if cancellation is not None:
                    cancellation.check()
                ordered[call.index] = execute_tool_sync(
                    call.tool_name,
                    call.tool_arguments,
                    user_id=user_id,
                    request_id=request_id,
                    project_id=project_id,
                    tenant_id=tenant_id,
                    cancellation=cancellation,
                    execute_fn=execute_fn,
                    permitted_tools=permitted_tools,
                    tool_call_id=call.tool_call_id,
                )
            continue

        workers = _cap_workers(
            [c.tool_name for c in batch],
            max_concurrency or DEFAULT_MAX_PARALLEL_TOOLS,
        )
        pool = get_global_tool_pool()
        futures: dict[Future, PreparedToolCall] = {}

        def _one(call: PreparedToolCall) -> tuple[str, dict[str, Any]]:
            if cancellation is not None and cancellation.is_cancelled():
                raise RequestCancelled(cancellation.reason)
            result = execute_tool_sync(
                call.tool_name,
                call.tool_arguments,
                user_id=user_id,
                request_id=request_id,
                project_id=project_id,
                tenant_id=tenant_id,
                cancellation=cancellation,
                execute_fn=execute_fn,
                permitted_tools=permitted_tools,
                tool_call_id=call.tool_call_id,
            )
            return call.tool_call_id, result

        # Submit at most ``workers`` at a time while respecting cancel.
        pending_calls = list(batch)
        in_flight: dict[Future, PreparedToolCall] = {}

        def _submit_next() -> None:
            while pending_calls and len(in_flight) < workers:
                if cancellation is not None and cancellation.is_cancelled():
                    raise RequestCancelled(cancellation.reason)
                call = pending_calls.pop(0)
                fut = pool.submit(_one, call)
                in_flight[fut] = call
                futures[fut] = call

        try:
            _submit_next()
            while in_flight:
                if cancellation is not None and cancellation.is_cancelled():
                    for fut in list(in_flight):
                        fut.cancel()
                    raise RequestCancelled(cancellation.reason)
                done_set, _ = futures_wait(
                    in_flight.keys(),
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if not done_set:
                    _submit_next()
                    continue
                for fut in done_set:
                    call = in_flight.pop(fut)
                    _call_id, result = fut.result()
                    ordered[call.index] = result
                _submit_next()
        except RequestCancelled:
            for fut in list(in_flight):
                fut.cancel()
            raise

    return [
        r if r is not None else {"success": False, "error": "missing_result"}
        for r in ordered
    ]


async def execute_tools_bounded_async(
    calls: Sequence[PreparedToolCall],
    *,
    user_id: str | None = None,
    request_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
    cancellation: CancellationToken | None = None,
    max_concurrency: int | None = None,
    execute_fn: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Async batch runner (native async tools awaited; sync via to_thread)."""
    if cancellation is not None:
        cancellation.check()
    if not calls:
        return []

    ordered: list[dict[str, Any] | None] = [None] * len(calls)

    for batch in partition_execution_batches(list(calls)):
        if cancellation is not None:
            cancellation.check()
        if len(batch) == 1 or not can_parallelize(batch):
            for call in batch:
                ordered[call.index] = await execute_tool_async(
                    call.tool_name,
                    call.tool_arguments,
                    user_id=user_id,
                    request_id=request_id,
                    project_id=project_id,
                    tenant_id=tenant_id,
                    cancellation=cancellation,
                    execute_fn=execute_fn,
                )
            continue

        workers = _cap_workers(
            [c.tool_name for c in batch],
            max_concurrency or DEFAULT_MAX_PARALLEL_TOOLS,
        )
        sem = asyncio.Semaphore(workers)

        async def _one(call: PreparedToolCall) -> None:
            async with sem:
                if cancellation is not None:
                    cancellation.check()
                ordered[call.index] = await execute_tool_async(
                    call.tool_name,
                    call.tool_arguments,
                    user_id=user_id,
                    request_id=request_id,
                    project_id=project_id,
                    tenant_id=tenant_id,
                    cancellation=cancellation,
                    execute_fn=execute_fn,
                )

        await asyncio.gather(*[_one(c) for c in batch])

    return [
        r if r is not None else {"success": False, "error": "missing_result"}
        for r in ordered
    ]
