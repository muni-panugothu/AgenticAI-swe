# Batch D — Security and Adversarial Validation

**Date:** 2026-11-26  
**Verdict:** ACCEPTED WITH LIMITATIONS  
**Scope:** Topics 48–53 and interactions with Topics 38–40, 42–47, RAG ACL, auth/scope, memory permissions.  
**Not claimed:** production readiness; OS-level sandbox isolation; live Atlas/Redis/provider adversarial runs.

## 1. Scope and threat model

Trusted boundaries under test:

1. JWT / server-injected `user_id` / `project_id` / `tenant_id`
2. `prepare_tool_invocation` / `execute_tool` (permission, approval, provenance strip, sandbox)
3. Approval atomic consume + argument/tool/scope binding
4. Tool reliability (no retry/fallback on auth/approval)
5. Parallel batching (`execute_tools_bounded`) order + cancel
6. RAG `authorize_candidates` + `_same_scope`
7. Application-level `path_sandbox` (not OS containers)

Untrusted: user messages, RAG/web/tool text, LLM tool args, document metadata fields resembling identity/approval.

## 2. Files inspected and changed

**Inspected (representative):** `chatbot.py` (`prepare_tool_invocation`), `approval_service.py`, `tool_permission_service.py`, `content_provenance.py`, `sandbox_service.py`, `path_sandbox.py`, `tool_reliability.py`, `tool_execution.py`, `rag_acl.py`, `rag_document_service.py`, existing `tests/test_agent_security_48_53.py` and related suites.

**Created:**

- `tests/security/` adversarial suites (`test_batch_d_*.py`, `_fixtures.py`)
- `docs/BATCH_D_SECURITY_AUDIT.md`

**Modified:**

- `app/services/rag_acl.py` — reject missing/non-ready status
- `app/services/approval_service.py` — atomic consume rebinds project/tenant
- `app/services/sandbox_service.py` — validate `folder_path`; directory expect for sensitive dirs
- `app/services/tool_execution.py` — result slots keyed by call index (duplicate tool_call_id safe)
- `docs/IMPLEMENTATION_STATUS.md` — Batch D section only

## 3. Attack scenarios tested

| ID prefix | Category | Entry point |
|-----------|----------|-------------|
| BD-A | Prompt injection / untrusted content | wrap/strip/reject + execute_tool |
| BD-B | Permission bypass | execute_tool, reliability fallback |
| BD-C | Approval attacks | consume + execute_tool + concurrency barrier |
| BD-D | Cross-user/project isolation | approvals, memory args, RAG ACL |
| BD-E | Retry/fallback/cancel abuse | reliability + bounded parallel |
| BD-F | Sandbox / path attacks | sandbox_service + path_sandbox |
| BD-G | Streaming/parallel security | execute_tools_bounded + partitions |
| BD-R | RAG ACL adversarial | authorize_candidates |

## 4. Test IDs and results

Offline adversarial suite under `tests/security/`: **76 collected** in focused run with related suites totaling **224 passed, 10 skipped** (see §8).

Representative IDs: `BD_A01`–`BD_A09`, `BD_B01`–`BD_B09`, `BD_C01`–`BD_C11`, `BD_D01`–`BD_D08`, `BD_E01`–`BD_E10`, `BD_F01`–`BD_F15`, `BD_G01`–`BD_G07`, `BD_R01`–`BD_R08`.

## 5. Confirmed vulnerabilities

### V1 — RAG ACL missing `status` accepted (Medium)

- **Repro:** `authorize_candidates` with candidate lacking `status` but in `allowed_docs`.
- **Impact:** Fusion/rerank/rewrite paths that omit readiness metadata could surface non-ready chunks.
- **Fix:** Require `status == "ready"` (fail closed). Regression: `BD_R01`.

### V2 — Parallel duplicate `tool_call_id` overwrote sibling results (Medium)

- **Repro:** Two `PreparedToolCall`s with the same `tool_call_id`; index map keyed by id.
- **Impact:** One result slot could remain `missing_result` / wrong association.
- **Fix:** Always assign by `PreparedToolCall.index`. Regression: `BD_G04`.

### V3 — `folder_path` not checked at sandbox gate; `.git` as folder not denied as directory (Medium)

- **Repro:** `enforce_sandbox_for_tool("search_code", {"folder_path": ".git"})` returned `None` before fix (body still used `resolve_under_root`).
- **Impact:** Pre-exec gate missed search roots; sensitive directory basename not denied when treated as a file path.
- **Fix:** Include `folder_path`; `expect_dir=True` for directory-shaped keys. Regression: `BD_F10`.

### V4 — Approval atomic consume omitted project/tenant in filter (Low–Medium)

- **Repro:** Tamper stored `project_id` after non-atomic binding check; consume could still succeed if filter ignored scope.
- **Impact:** Scope binding weaker under store tampering / races.
- **Fix:** Include `project_id` / `tenant_id` in `find_one_and_update` filter. Regression: `BD_C11`.

## 6. Fixes and regression coverage

All four findings fixed with Batch D tests above. Existing Topics 48–53 tests remain green.

## 7. Unresolved findings

| Finding | Severity | Notes |
|---------|----------|-------|
| Windows junction / reparse TOCTOU | Medium | No `O_NOFOLLOW`; residual race documented in path_sandbox |
| Sync worker cancel/timeout does not kill OS thread | Medium | Cooperative only; late workers may still finish after `TOOL_TIMEOUT`. Timeout calls `mark_execution_failed` when possible (`BD_FUP07`) but cannot guarantee non-commit |
| No OS container / Job Object / network namespace | High (env) | Application-level only; honest capabilities API |
| Live Atlas/Redis adversarial isolation | — | NOT RUN (Batch C gates; out of Batch D offline scope) |
| LLM-level injection compliance | Medium | Tool/RAG outputs are provenance-wrapped; model obedience not proven without live LLM |

### Follow-up hardenings (post-explore)

- `PATH_*` / sandbox denials → `AUTHORIZATION` (no retry) — `BD_FUP01`/`BD_FUP02`
- Intent `permitted_tools` is a primary execution gate (not fallback-only) — `BD_FUP03`/`BD_FUP04`
- Agent tool messages use `tool_result_for_model` provenance wrap — `BD_FUP05`

## 8. Skipped / NOT RUN

- Symlink escape (`BD_F14`): may skip when Windows denies symlink creation without privilege.
- Path security symlink/junction cases in `test_tool_path_security.py` when unsupported.
- Live Mongo/Redis/Gemini/Groq/Tavily adversarial: **NOT RUN** (no paid/live calls; Batch C opt-in only).
- Production DB/Redis: never used.

## 9. Windows-specific sandbox limitations

- Path confinement via resolve + `relative_to` + sensitive denylist.
- UNC, drive letters, NT device prefixes rejected.
- Symlink leaf refusal + re-resolve before open; junctions remain residual risk.
- No claim of escape-proof OS isolation.

## 10. Application-level vs OS-level gaps

| Control | Level |
|---------|-------|
| Path confinement, sensitive deny, process tool refuse | Application |
| Permission / approval / provenance | Application |
| OS container, Job Object, network namespace | **Not implemented** |
| Killing timed-out sync threads | **Not implemented** |

## 11. Mocks / fakes used

- `InMemoryCollection` for approvals/permissions (same as Topics 48–53).
- Patched tool bodies to assert non-execution on deny.
- Custom `execute_fn` for reliability/parallel scheduling tests.
- **Not** mocked: `prepare_tool_invocation`, permission checks, approval consume, `authorize_candidates`, `path_sandbox` resolve.

## 12. Explicitly NOT tested

- Live provider prompt-injection with Groq/Gemini
- OS-level sandbox escape with admin junction creation beyond available privileges
- Browser/XSS/CSRF frontend attacks
- Network exfiltration beyond app-level web_search allowlisting
- Full red-team against production credentials or real user data
