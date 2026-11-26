# Integration Testing Runbook (Batch C + Batch E)

Default backend tests are **offline-safe**. Live MongoDB, Redis, Gemini, Groq, and Tavily suites are **opt-in** and skip with an explicit reason when prerequisites are missing. A skipped live test is **not** a pass.

Batch E uses **stricter** `BATCH_E_*` flags. Batch C flags alone do **not** enable Batch E live suites. See [BATCH_E_LIVE_INTEGRATION_REPORT.md](BATCH_E_LIVE_INTEGRATION_REPORT.md).

## Python environment

```powershell
cd backend
.\venv\Scripts\python.exe --version
```

## Offline groups (always safe)

```powershell
# Unit + API E2E (includes Batch B / routing regressions)
.\venv\Scripts\python.exe -m pytest tests -q --ignore=tests/integration

# Batch C offline: parallel overlap + RAG loader/chunk fakes + provider mocks
.\venv\Scripts\python.exe -m pytest tests/integration/test_parallel_overlap_batch_c.py tests/integration/test_rag_pipeline_batch_c.py::BatchCRagOfflinePipelineTests tests/integration/test_live_providers_batch_c.py::BatchCProviderOfflineOfflineTests tests/e2e/test_batch_c_routing_finalization.py -q

# Batch B E2E
.\venv\Scripts\python.exe -m pytest tests/e2e -q

# Offline assistant eval (no paid calls)
.\venv\Scripts\python.exe -m scripts.run_assistant_eval --fail-on-fail
```

## MongoDB (disposable DB only)

**Required**

| Variable | Purpose |
|----------|---------|
| `MONGO_URI` | Cluster URI (never commit; never print) |
| `MEMORY_TEST_DATABASE` or `RAG_TEST_DATABASE` | Disposable DB name containing `test` / `integration` / `batch_c` |
| `BATCH_C_MONGO_INTEGRATION=1` | Enable memory/auth/approval Mongo suite |
| `BATCH_C_RAG_INTEGRATION=1` or `RAG_ACL_INTEGRATION=1` | Enable RAG Atlas suite |

**Forbidden targets:** `agentic_ai_memory`, `production`, `prod`, or whatever `MEMORY_DATABASE` points at.

```powershell
$env:BATCH_C_MONGO_INTEGRATION="1"
$env:MEMORY_TEST_DATABASE="agentic_ai_memory_batch_c_test"
.\venv\Scripts\python.exe -m pytest tests/integration/test_mongo_batch_c.py -q

$env:BATCH_C_RAG_INTEGRATION="1"
$env:RAG_TEST_DATABASE="agentic_ai_memory_rag_acl_test"
.\venv\Scripts\python.exe -m pytest tests/integration/test_rag_pipeline_batch_c.py -q
```

Cleanup deletes **only** documents/users created by the test markers/IDs. Do not drop entire databases unless you separately own that disposable DB.

## Redis (isolated namespace)

Uses Upstash REST (`UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`).

| Variable | Purpose |
|----------|---------|
| `BATCH_C_REDIS_INTEGRATION=1` | Enable Redis suite |
| `BATCH_C_REDIS_NAMESPACE` | Must start with `agent:` and include `batchc` or `test` |

```powershell
$env:BATCH_C_REDIS_INTEGRATION="1"
$env:BATCH_C_REDIS_NAMESPACE="agent:batchc:test:"
.\venv\Scripts\python.exe -m pytest tests/integration/test_redis_batch_c.py -q
```

## Live paid providers (explicit opt-in)

| Flag | Service | Cost |
|------|---------|------|
| `BATCH_C_LIVE_EMBEDDINGS=1` | Gemini embeddings | Yes |
| `BATCH_C_LIVE_GROQ=1` | Groq LLM | Yes |
| `BATCH_C_LIVE_TAVILY=1` | Tavily search | Yes |

Also requires the corresponding API key env vars (`GOOGLE_API_KEY` / `GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`).

Default CI / local `pytest tests` must **not** set these flags.

## Batch A client regressions

```powershell
cd web
npm test -- src/contexts/ChatContext.batchA.test.tsx src/services/chat/chatApi.batchA.test.ts
cd ../app
npx tsx src/services/streamChatOnce.test.ts
```

## What Batch B vs Batch C prove

| Layer | Batch B | Batch C |
|-------|---------|---------|
| API E2E + fake LLM | Yes | Regression |
| JWT minting with mocked auth deps | Harness | Clarified |
| Real Mongo memory/auth/approvals | Simulated | Gated live |
| Real Redis rate limit / task context | Mocked | Gated live |
| RAG loaders/chunkers | Partial | Offline + gated live |
| Live Gemini/Groq/Tavily | NOT_RUN | Opt-in NOT_RUN by default |
| Parallel overlap | Noted gap | Barrier-proven |

## Batch E (stricter live canaries)

| Variable | Purpose |
|----------|---------|
| `BATCH_E_MONGO_INTEGRATION=1` | Auth/memory/approval/commit on disposable Atlas |
| `BATCH_E_REDIS_INTEGRATION=1` | Rate-limit / TTL / namespace Redis canaries |
| `BATCH_E_RAG_INTEGRATION=1` | Live RAG ingest/ACL on disposable DB |
| `BATCH_E_LIVE_GEMINI=1` | Real `gemini-embedding-001` (budget default 3) |
| `BATCH_E_LIVE_GROQ=1` | Small Groq completion (budget default 2) |
| `BATCH_E_LIVE_TAVILY=1` | Minimal Tavily search (budget default 2) |
| `MEMORY_TEST_DATABASE` / `RAG_TEST_DATABASE` | Disposable DB (name must include test/integration/batch_e) |
| `BATCH_E_REDIS_NAMESPACE` | e.g. `agent:batche:test:` |

```powershell
# Offline Batch E contracts (always safe)
.\venv\Scripts\python.exe -m pytest tests/integration/test_batch_e_offline_gates.py tests/integration/test_rag_batch_e.py::BatchERagOfflineContracts tests/integration/test_live_providers_batch_e.py::BatchEProviderMockOffline -q

# Live groups — only with disposable infra + explicit flags
$env:BATCH_E_MONGO_INTEGRATION="1"
$env:MEMORY_TEST_DATABASE="agentic_ai_memory_batch_e_test"
.\venv\Scripts\python.exe -m pytest tests/integration/test_mongo_batch_e.py -q

$env:BATCH_E_REDIS_INTEGRATION="1"
$env:BATCH_E_REDIS_NAMESPACE="agent:batche:test:"
.\venv\Scripts\python.exe -m pytest tests/integration/test_redis_batch_e.py -q

$env:BATCH_E_RAG_INTEGRATION="1"
$env:RAG_TEST_DATABASE="agentic_ai_memory_batch_e_test"
.\venv\Scripts\python.exe -m pytest tests/integration/test_rag_batch_e.py -q
```

**CI:** Offline job always runs Batch E gate tests. Live job `batch-e-live` runs only on `main` pushes when `vars.BATCH_E_LIVE_ENABLED=true` and disposable Mongo secrets exist. Never on pull requests.

## Limitations

- Application-level sandbox (Topic 53) is not OS isolation.
- Live Atlas RAG still needs a READY `vector_index` on the disposable DB.
- Without `BATCH_C_LIVE_EMBEDDINGS` / `BATCH_E_LIVE_GEMINI`, Mongo RAG tests may patch embeddings and verify persistence/ACL only.
- Approval **UI** is not covered; API persistence/consume is.
- Batch E live groups are **NOT RUN** unless `BATCH_E_*` flags and disposable targets are set.
