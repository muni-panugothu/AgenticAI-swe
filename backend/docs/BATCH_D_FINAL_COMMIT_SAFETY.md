# Batch D-Final — Late Worker Commit Safety

**Date:** 2026-11-26  
**Verdict:** ACCEPTED WITH LIMITATIONS

## 1. Original late-worker vulnerability

1. User approves a protected tool.
2. `prepare_tool_invocation` consumes the approval.
3. The sync tool body starts on a worker thread.
4. Caller times out or cancels.
5. System records timeout / `mark_execution_failed`.
6. **Worker may still commit** a durable side effect afterward.

Python threads are **not** forcibly terminated. That remains true.

## 2. Execution lifecycle (before → after)

**Before:** approval consume → tool body → DB write (no commit gate).

**After:**

1. Caller reserves a server-owned `execution_id` for commit-guarded tools.
2. Worker runs `prepare_tool_invocation` (permission + approval consume).
3. Worker `activate_execution` (fails if already invalidated).
4. Worker binds execution (+ optional `CancellationToken`) via contextvars.
5. Immediately before irreversible write: `try_begin_commit()`.
6. On timeout/cancel: `invalidate_execution` (and approval `mark_execution_failed`).
7. If invalidated before commit → write rejected (`COMMIT_REJECTED`).
8. If commit already acquired → side effect may complete; state stays `committed` (honest).

## 3. Commit-boundary guarantee

> A commit-guarded tool must not perform a protected durable write after its
> execution lease was invalidated by timeout or cancellation, unless
> `try_begin_commit` had already succeeded (commit won the race).

This is **application-level**. It is **not** OS process kill.

## 4. Tool categories

| Category | Tools | Commit guard |
|----------|-------|--------------|
| Read-only | `calculator`, `get_current_time`, `read_file`, `search_code`, `web_search`, memory search/inspect | Not required |
| Side-effect + guarded | `save_memory`, `forget_memory`, `reinforce_memory`, `request_memory_update`, `confirm_memory_update`, `summarize_memories` | Yes (`COMMIT_GUARDED_TOOLS` + memory write paths) |
| Unsupported protected | `COMMIT_UNSUPPORTED_PROTECTED_TOOLS` (empty) | Would be denied with `COMMIT_BOUNDARY_UNSUPPORTED` |
| Unknown | unregistered / UNKNOWN risk | Already fail closed at registry / approval |

## 5. Race tests

| ID | Scenario | Result |
|----|----------|--------|
| BD_CS01 | Invalidate before commit | Rejected, commit_count=0 |
| BD_CS02 | Cancel before commit | Rejected |
| BD_CS03 | Commit then cancel | State=`committed` |
| BD_CS04 | Timeout after commit | Does not claim prevented |
| BD_CS05 | Repeated cancel | Still blocked |
| BD_CS06 | Forged execution id | `activate` fails closed |
| BD_CS07 | Wrong approval reuse | No execution |
| BD_CS08 | E2E timeout + paused worker | commits=0 |
| BD_CS09 | Legitimate approved path | Still works |
| BD_CS10 | Read-only calculator | Unaffected |
| BD_CS11 | Barrier invalidate | Rejected |

## 6. Remaining limitations

- Sync worker threads continue running after timeout/cancel.
- Prep work (e.g. embeddings) may still run after invalidation; only durable commits are gated.
- Tools that bypass `memory_service` / lifecycle helpers without calling `try_begin_commit` are not protected.
- No OS container / Job Object / thread kill.

## 7. Files

**Created:** `app/services/execution_lifecycle.py`, `tests/security/test_batch_d_final_commit_safety.py`, this doc.

**Modified:** `tool_reliability.py`, `chatbot.py` (`execute_tool`), `memory_service.py`, `memory_summary_service.py`, `IMPLEMENTATION_STATUS.md`.
