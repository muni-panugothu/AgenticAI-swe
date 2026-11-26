# Implementation Status

> **Current verification date: November 26, 2026** (Batch E live canaries + Batch D adversarial + Batch A/B/C).
> Historical suite snapshots from **November 10 / 23, 2026** remain below for archive only.

## Batch E — Live integration / security canaries (November 26, 2026)

| Area | Status | Evidence |
|---|---|---|
| Safety gates + budgets | **DONE** | `tests/integration/batch_e_safety.py`; offline gate tests |
| Disposable Atlas Mongo / approvals / commit | **GATED** | `test_mongo_batch_e.py` (`BATCH_E_MONGO_INTEGRATION`) |
| Disposable Redis TTL / isolation / fail-closed | **GATED** | `test_redis_batch_e.py` (+ offline fail-closed contract) |
| Live RAG ingest / ACL | **GATED** | `test_rag_batch_e.py` (`BATCH_E_RAG_INTEGRATION`) |
| Gemini / Groq / Tavily canaries | **OPT-IN NOT_RUN by default** | `BATCH_E_LIVE_*` + request budgets |
| CI live job | **WIRED / NOT RUN** until secrets + `BATCH_E_LIVE_ENABLED` | `.github/workflows/backend-ci.yml` |
| Production readiness | **NOT CLAIMED** | Skips ≠ passes; no OS isolation |

**Report:** [BATCH_E_LIVE_INTEGRATION_REPORT.md](BATCH_E_LIVE_INTEGRATION_REPORT.md).  
**Runbook:** [INTEGRATION_TESTING.md](INTEGRATION_TESTING.md).

## Batch D — Security / adversarial validation (November 26, 2026)

| Area | Status | Evidence |
|---|---|---|
| Threat matrix + adversarial suites | **DONE (offline)** | `tests/security/test_batch_d_*.py`; `docs/BATCH_D_SECURITY_AUDIT.md` |
| Topics 48–53 regression | **PASS** | `tests/test_agent_security_48_53.py` + Batch D |
| RAG ACL missing-status fail-closed | **FIXED** | `rag_acl.py`; `BD_R01` |
| Duplicate tool_call_id parallel slotting | **FIXED** | `tool_execution.py`; `BD_G04` |
| Sandbox `folder_path` / sensitive dirs | **FIXED** | `sandbox_service.py`; `BD_F10` |
| Approval atomic project/tenant bind | **HARDENED** | `approval_service.py`; `BD_C11` |
| Late-worker commit safety | **DONE (app-level)** | `execution_lifecycle.py`; `BD_CS*`; [BATCH_D_FINAL_COMMIT_SAFETY.md](BATCH_D_FINAL_COMMIT_SAFETY.md) |
| Live Atlas/Redis/provider adversarial | **NOT RUN (default)** | Offline Batch D; live paths via Batch E gates |
| Production readiness | **NOT CLAIMED** | Application-level controls; no OS isolation |

**Audit:** [BATCH_D_SECURITY_AUDIT.md](BATCH_D_SECURITY_AUDIT.md).  
**Commit safety:** [BATCH_D_FINAL_COMMIT_SAFETY.md](BATCH_D_FINAL_COMMIT_SAFETY.md).

## Batch C — Live persistence / RAG / providers (November 26, 2026)

| Area | Status | Evidence |
|---|---|---|
| Safe integration gates | **DONE** | `tests/integration/safety.py`; disposable DB name checks; paid flags off by default |
| Mongo auth/memory/approvals | **GATED** | `tests/integration/test_mongo_batch_c.py` (`BATCH_C_MONGO_INTEGRATION`) |
| Redis rate-limit / task context | **GATED** | `tests/integration/test_redis_batch_c.py` (`BATCH_C_REDIS_INTEGRATION`) |
| RAG loaders/chunk/fake ingest | **DONE (offline)** | `tests/integration/test_rag_pipeline_batch_c.py` |
| RAG live Atlas ingest/retrieve | **GATED** | same file + existing `test_rag_acl_integration.py` |
| Live Gemini / Groq / Tavily | **OPT-IN NOT_RUN by default** | `BATCH_C_LIVE_*` flags; see `docs/INTEGRATION_TESTING.md` |
| Parallel overlap proof | **DONE** | barrier-based `test_parallel_overlap_batch_c.py` |
| Web vs investigation finalization | **FIXED + regression** | `scope_investigation_requirements_for_intent`; `tests/e2e/test_batch_c_routing_finalization.py` |

**Runbook:** [INTEGRATION_TESTING.md](INTEGRATION_TESTING.md).

## Agent Topics 33–35 (Batch 1)

| Topic | Status | Evidence |
|---|---|---|
| 33 | Agent State | **DONE** | `app/services/agent_state.py`; integrated in `chatbot.py`; `tests/test_agent_state.py`. Locals remain scalar SoT (documented dual-write debt). |
| 34 | Agent Context | **DONE** | `app/services/agent_context.py`; live `messages` by reference; `tests/test_agent_context.py` |
| 35 | Agent Planning | **DONE** | `app/services/agent_planning.py` (`AgentPlan`, `PlanDecision`, `plan_next_action`); step-0 plan helper; `tests/test_agent_planning.py` |

**Batch 1 regression (November 23, 2026):** `unittest discover -s tests` → **786 passed, 11 skipped**.

**Known debt:** Topic 33 scalar dual-write sync is best-effort (not exhaustive). Do not treat `AgentState` scalars as authoritative until a later topic.

**Not started:** Topics 36–40.

## Completed Stages

| Stage | Topic | Status | Evidence |
|---|---|---|---|
| 16 | Conversation Memory (in-process + Redis) | **DONE** | `chatbot.py` `add_to_conversation_memory`, `temporary_memory_service.py`, `test_chatbot_memory_stage17.py` |
| 16 | Gemini Embeddings | **DONE** | `embedding_service.py` `create_embedding`, `test_embedding.py` |
| 16 | Semantic Vector Search | **DONE** | `memory_service.py` `semantic_search_memories`, Atlas `$vectorSearch` |
| 16 | Tool Registry + Pydantic Validation | **DONE** | `tool_registry.py`, `validation/tool_models.py`, `validation/tool_registry.py` |
| 17 | Fact Extraction + Conflict Detection | **DONE** | `memory_conflict_service.py`, `memory_fact_extraction.py` |
| 17 | Versioned Supersede + Audit Trail | **DONE** | `memory_service.py` `supersede_memory`, `memory_audit_service.py` |
| 17 | Exclusive Fact Keys | **DONE** | `memory_conflict_service.py` `EXCLUSIVE_FACT_KEYS` |
| 18 | Freshness Scoring + Lifecycle | **DONE** | `memory_lifecycle_service.py`, `test_memory_lifecycle_service.py` |
| 18 | Reinforcement Tracking | **DONE** | `memory_service.py` `reinforce_memory` |
| 19 | Memory Summarization | **DONE** | `memory_summary_service.py`, `test_memory_summarization.py` |
| 19 | Deduplication Intelligence | **DONE** | `memory_conflict_service.py` `classify_duplicate` |
| 19 | Audit Events | **DONE** | `memory_audit_service.py`, `test_stage19_security.py` |
| 19 | Multi-statement Fact Extraction | **DONE** | `memory_fact_extraction.py` `extract_facts_from_message` |
| 19 | Agent Orchestration + Investigation | **DONE** | `agent_orchestration.py`, `investigation_state.py`, `test_investigation_architecture.py` |
| 20 | Memory Scoping | **PASS** | `memory_scope_service.py` (+ `conversation_id` gate); user/tenant/project/session Redis scope; `docs/MEMORY_SECURITY.md`; `tests/test_memory_scoping.py` |
| 21 | Memory Permissions | **PASS** | `memory_permission_service.py`; owner/shared/project/public_read; save ignores forged `owner_id`; delete uses owner+scope filter; `tests/test_memory_security.py` (29) + `tests/test_memory_scope_permission_security.py` |
| — | Production Auth (API + CLI + Expo) | **PASS** (CLI refresh 2026-11-10) | JWT + refresh rotation; CLI auto-refresh; server logout with body refresh; see [AUTHENTICATION.md](AUTHENTICATION.md) |

## RAG Topics 22–32 (document retrieval)

**Source of truth:** [RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md).

[RAG_IMPLEMENTATION_PLAN.md](RAG_IMPLEMENTATION_PLAN.md) is a **historical design plan only** — not active work and must not override the implementation notes.

**Current status:** RAG Topics 22–32 are **implemented and verified** (local mocks + disposable Atlas Baseline A/B + ACL suite). **Not production-ready** — known limitations remain (see Verified results and Remaining features).

| Stage | Topic | Status | Evidence |
|---|---|---|---|
| 22 | RAG Fundamentals | **DONE** | `rag_config.py`, `rag_document_service.py`, `docs/RAG_IMPLEMENTATION.md` |
| 23 | Document Loading | **DONE** (Batch 1 + critical fixes) | Multi-format parsers; HTML/binary hardening; flag enforcement |
| 24 | Chunking | **DONE** (Batch 1 + critical fixes) | Exact source-span offsets; structure pack + char fallback |
| 25 | Embedding Documents | **DONE** (Batch 1 + critical fixes) | Vector-only cache hardening; staged re-embed recovery |
| 26 | Vector Search (RAG) | **DONE** | Tunable `numCandidates`/min-score; fail-closed index checks; diagnostics |
| 27 | Metadata Filtering | **DONE** | `_same_scope` + `authorize_candidates` after fuse/rerank/context |
| 28 | Hybrid Search | **DONE** | Active lexical path: **overlap**. Atlas BM25 code exists but is **unavailable** on current M0 (Search index limit) |
| 29 | Reranking | **DONE** (optional) | Baseline B verified **`heuristic`**. Cross-encoder optional; **live CE verification not performed** |
| 30 | Query Rewriting | **DONE** (conservative defaults) | Decomposition on; multi-query/HyDE/conversation off |
| 31 | Context Compression | **DONE** | Token budgets + optional Groq compression (off) |
| 32 | RAG Evaluation | **DONE** (offline + disposable live capture) | Golden v2; Baseline A/B captures + reports; LLM-judge disabled |

### Verified results (November 23, 2026)

| Check | Result |
|---|---|
| **Latest full regression** | **`unittest discover -s tests`: 786 passed, 11 skipped** (includes Agent Topics 33–35) |
| Live Atlas ACL integration | **6/6 passed** (`RAG_ACL_INTEGRATION` + disposable DB) |
| Baseline A (`reranker=none`, `capture01`) | MRR@5 **0.857142**, nDCG@5 **0.903479**, Precision@5 **0.228571**, Recall@5 **1.0** |
| Baseline B (`reranker=heuristic`, `capture02`) | MRR@5 **1.0**, nDCG@5 **1.0**, Precision@5 **0.228571**, Recall@5 **1.0**; **7** rerank calls; **0** provider failures |
| Capture artifacts (preserved) | `captures/baseline_a_captured_results.json`, `captures/baseline_a_report.json`, `captures/baseline_b_captured_results.json`, `captures/baseline_b_report.json` |
| Eval DB markers | `capture01` and `capture02` rows cleaned from `agentic_ai_memory_rag_acl_test.rag_chunks` |
| Lexical backend | **Overlap active.** Atlas BM25 **unavailable** (M0 three Search-index limit) — distinct from “BM25 running” |
| Recall@5 note | Aggregate/per-case `recall@5` can be **vacuously 1.0** when `relevant_chunk_ids` is empty (no positives to miss). Prefer abstention / Precision@5 for negative cases |
| Cross-encoder | **Not live-verified** (optional dependency; Baseline B used heuristic only) |
| Known gaps | Unsupported-question false positives; LLM judge / answer faithfulness disabled; API text encoding unresolved |

**Remaining feature (separate):** frontend document UI — not implemented (API-only). Agent Topics 33–35 done; 36–40 not started.

Details: [RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md), [RAG_EVAL_CAPTURE.md](RAG_EVAL_CAPTURE.md), [RAG_EVALUATION_GOLDEN_REVIEW.md](RAG_EVALUATION_GOLDEN_REVIEW.md).

## Production hardening — persistence & sessions (historical: November 10, 2026)

### Root causes (CLI live testing)

| Symptom | Cause |
|---|---|
| “Session expired” on save/recall | 15m access JWT; **Continue as …** reused stale token; CLI never called `/auth/refresh` |
| Account B “lost” memories | Failed saves during expired session **or** recall questions that skipped LTM (`intent_needs_long_term_memory`) — not Mongo deletion |
| `Unclosed MongoClient` | Shared client at `memory_service.py:91`; closed on FastAPI shutdown + `atexit` |
| `unclosed ssl.SSLSocket` | **Groq / Gemini / requests HTTPS**, not Mongo; Groq `close()` on shutdown via `resource_lifecycle.py` |

### Fixes

- `cli_auth.py`: proactive refresh, 401 retry, rotated refresh persistence, server logout.
- `test_chatbot.py`: tracks session; refreshes tokens through chat.
- `intent_needs_long_term_memory`: patterns for “did I ask/tell you…”.
- `auth_service.refresh_session`: reuse detection revokes all sessions.
- `app/services/resource_lifecycle.py`: Mongo + Groq shutdown.

### Tests (historical snapshot — November 10, 2026)

These counts are **not** the current regression baseline. Latest full suite: **757 passed, 11 skipped** (November 23, 2026).

| Suite | Result (2026-11-10) |
|---|---|
| `discover -s tests` | **502 OK**, **1 skipped** (`test_memory_persistence_integration` — NOT RUN without `MEMORY_PERSISTENCE_INTEGRATION=1`) |
| Security suites (permissions + scope + security + scoping + auth_api) | **75 OK** |
| New: `test_cli_auth_session`, `test_auth_refresh`, `test_resource_lifecycle` | **OK** |

### Integration test (real Mongo)

Set `MEMORY_PERSISTENCE_INTEGRATION=1` and `MEMORY_TEST_DATABASE` to a **disposable** database name before running `tests.test_memory_persistence_integration`.

## Topics 20–21 Verification (historical: November 10, 2026, security audit pass)

### Status

| Topic | Verdict | Why |
|---|---|---|
| 20 Memory Scoping | **PASS** | User/project/tenant isolation enforced; missing ownership denied; session scope helpers present |
| 21 Memory Permissions | **PASS** | Central permission service; mutations + semantic retrieval filtered; shared read-only; denials audited without content |

### Gap closed in this pass

Previously `_ensure_identity_fields` / vector post-processing could invent `user_id`/`owner_id` from the caller for orphan docs, and `public_read` could succeed with missing tenants. Fixed so:

- Missing ownership always denies (including `public_read`).
- Caller identity is never stamped onto orphan documents.
- `public_read` requires explicit matching non-null tenants.
- Conceptual DIRECT_ANSWER prompt no longer implies missing scope ⇒ public.

### Exact implementation ranges

| File | Symbols / ranges |
|---|---|
| `app/services/memory_scope_service.py` | `build_scope_filter` 117–158; `document_matches_scope` 187–229; active scope 257–321 |
| `app/services/memory_permission_service.py` | `_check_permission` 119–251; `filter_readable_documents` 287–305; grant/revoke 308–396 |
| `app/services/memory_service.py` | `_ensure_identity_fields` 205–226; `save_memory` 264–411; `semantic_search_memories` 2165–2448; `reinforce_memory` 2492–2635; `delete_memory` 3322–3437 |

### Security regression tests (executed)

| Test | Result |
|---|---|
| `test_user_a_cannot_read_user_b_private_memory` | OK |
| `test_user_a_cannot_modify_user_b_memory` | OK |
| `test_cross_tenant_access_is_denied` | OK |
| `test_cross_project_access_is_denied` | OK |
| `test_missing_ownership_does_not_grant_public_access` | OK |
| `test_authorized_shared_memory_reads_work` | OK |
| `test_unauthorized_shared_memory_reads_fail` | OK |
| `test_unauthorized_writes_fail` | OK |
| `test_semantic_retrieval_respects_permissions` | OK |
| `test_existing_memory_operations_remain_functional` | OK |

### Suite results (historical snapshots — November 10, 2026)

These are **not** current. Latest full regression: **757 passed, 11 skipped** (November 23, 2026). An earlier November 10 snapshot also recorded **442** tests all passing before later growth of the suite.

| Command | Result (2026-11-10) |
|---|---|
| `python -m unittest tests.test_memory_scope_permission_security tests.test_memory_permissions tests.test_memory_scoping -v` | **36 OK** |
| `python -m unittest tests.test_memory_security tests.test_memory_scope_permission_security -v` | **39 OK** |
| `python -m unittest discover -s tests` | **492 OK**, 2 unrelated failures (file-reader + investigation merge) — historical |

### Obsolete RAG status in this November 10, 2026 snapshot (superseded)

At the time of this security-audit write-up, **RAG Topics 22–32 were not yet implemented** and document upload was still a stub path (including **503** responses where RAG was not available). That statement is **historical only**.

**Current status (November 23, 2026):** RAG Topics 22–32 are **implemented and verified** (see the RAG section above), with known limitations — **not** production-ready. Do not use this November 10 snapshot as the live RAG status.

### Remaining limitations (memory / auth — still applicable)

- Legacy CLI path (`test_chatbot.py` without `USE_API_AUTH`) still trusts env `user_id`; use `cli_auth.py login` + `USE_API_AUTH=true`.
- Memory tools invoked only via authenticated `/api/chat` pass JWT `user_id`; direct in-process CLI bypasses HTTP auth.
- Atlas `$vectorSearch` pre-filter is equality-based; post-filters enforce legacy null/project/tenant rules.
- Shared access is read-only; no ownership-transfer API.
- Cookie-only browser CSRF hardening not fully exercised on Expo (Bearer + SecureStore).
- API text encoding issue remains unresolved (does not block offline RAG evaluation).

## Test Suite

| Suite | Count | Status |
|---|---|---|
| **Latest full regression** (`discover -s tests`) | **757 passed, 11 skipped** | **Current — November 23, 2026** |
| Live RAG ACL integration | **6/6** | Gated; disposable Atlas only (November 23, 2026) |
| Historical: discover ≈442 all passing | — | Snapshot **November 10, 2026** (superseded) |
| Historical: discover 502 OK, 1 skipped | — | Snapshot **November 10, 2026** (superseded) |
| `backend/` root legacy tests | 17 files | Not in `discover` path; run individually |

## Infrastructure

| Component | Provider | Status |
|---|---|---|
| MongoDB Atlas | Cloud | **Connected** — collections + memory vector index + RAG `vector_index` |
| Gemini Embeddings | Google | **Connected** — 768-dimension RAG vectors |
| Groq LLM | Cloud | **Connected** — `openai/gpt-oss-20b` |
| Redis | Upstash | **Connected** — session / active project scope |
| Tavily | Cloud | **Connected** — web search |
| FastAPI | Local | **Wired** — `uvicorn app.main:app --port 8787`; auth + chat + document RAG API |

## Remaining features (not Topics 22–32 completion blockers)

| Item | Status |
|---|---|
| Frontend document UI | **Not implemented** (API-only today) |
| Agentic AI Topics 33–40 | **Not started** |
| Atlas BM25 live on this M0 cluster | **Unavailable** (index limit); overlap lexical active |
| Cross-encoder live verification | **Not performed** |

## Documentation map

| Doc | Role |
|---|---|
| [RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md) | **Active** verified implementation notes |
| [RAG_EVAL_CAPTURE.md](RAG_EVAL_CAPTURE.md) | Capture tooling + Baseline A/B artifacts |
| [RAG_EVALUATION_GOLDEN_REVIEW.md](RAG_EVALUATION_GOLDEN_REVIEW.md) | Golden review process + verified metrics |
| [RAG_IMPLEMENTATION_PLAN.md](RAG_IMPLEMENTATION_PLAN.md) | **Historical** Topics 20–32 design plan only |
