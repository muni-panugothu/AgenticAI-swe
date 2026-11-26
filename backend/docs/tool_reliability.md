# Topics 45–47 — Tool Timeouts, Retries, and Fallbacks

This layer wraps authorized tool execution (`execute_tool` / `prepare_tool_invocation`).
It does **not** replace Topic 38 provider/LLM retries (`decide_llm_retry`).

## Defaults (env)

| Variable | Default | Role |
|---|---|---|
| `AGENT_TOOL_TIMEOUT_SECONDS` | 30 | Default per-tool timeout |
| `AGENT_TOOL_TIMEOUT_MAX_SECONDS` | 120 | Hard cap |
| `AGENT_TOOL_MAX_RETRIES` | 2 | Default retries for retry-safe tools |
| `AGENT_TOOL_RETRY_BUDGET_SECONDS` | 45 | Max elapsed retry window |
| `AGENT_TOOL_RETRY_BASE_SECONDS` | 0.25 | Backoff base |
| `AGENT_TOOL_RETRY_MAX_BACKOFF_SECONDS` | 8 | Backoff cap |
| `AGENT_TOOL_FALLBACK_MAX_DEPTH` | 1 | Max fallback hops |
| `AGENT_TOOL_MAX_OUTSTANDING_TIMED_OUT` | 16 | Cap on timed-out sync workers still running |

Per-tool overrides live in `app.services.tool_reliability._RELIABILITY_META`
(timeout, `retry_safe`, `idempotent`, `max_retries`, fallbacks). LLM-supplied
timeouts are ignored.

## Sync vs async timeouts

- **Async:** `asyncio.timeout` cancels the awaited coroutine.
- **Sync:** caller stops waiting; the worker thread is **not** killed. The global
  concurrency slot stays held until the worker finishes (`worker_may_still_run`).

## Topic 38 vs Topic 46

- Topic 38: Groq/provider request retries only.
- Topic 46: tool execution retries only (classified, bounded, cancellation-aware).

## Fallbacks

Explicit allowlisted candidates only. Authorization, validation, cancellation,
and ambiguous side-effect failures never select a fallback.
