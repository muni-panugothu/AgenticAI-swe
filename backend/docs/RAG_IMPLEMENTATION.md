# RAG Topics 22–32 — Implementation Notes

This document describes the **verified** document-RAG implementation in the
authenticated backend. It is the source of truth for current behavior,
configuration, and tests.

**Current verification date: November 23, 2026.**

`RAG_IMPLEMENTATION_PLAN.md` and materials under `RAG_Project_Documentation/`
are **historical design references only** — not active work. They may describe
intended work that differs from what ships today and must not override this file.

## Current architecture and verified Atlas state

- Document chunks use the existing `agentic_ai_memory.rag_chunks` collection. No separate document collection was introduced; document metadata is repeated on its chunks so the already configured vector index can serve them.
- The verified Atlas Vector Search index is `vector_index`, `READY`, with `embedding` at 768 dimensions and `similarity: cosine`, plus filter fields `user_id`, `project_id`, `tenant_id`, `document_id`, and `status`. Runtime `verify_rag_index()` fail-closes when readiness, dimensions, similarity, or required filters do not match `RAG_EMBEDDING_DIMENSIONS` / `RAG_VECTOR_SIMILARITY`. Update that definition in Atlas (do not recreate the collection) and wait until status is `READY` and `queryable` is true before relying on retrieval. The memory collection has a separate `memory_vector_index` at 3072 dimensions.
- RAG explicitly requests `gemini-embedding-001`, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`, and 768 dimensions. It uses a distinct bounded-timeout Gemini client. Personal-memory embedding code and index dimensions are unchanged.
- The implementation does not create or alter MongoDB collections or indexes. Ingestion and vector retrieval write/query only after checking the existing Atlas index definition at runtime.

## Implemented behavior

1. Authenticated document upload accepts UTF-8 `.txt` / `.md` / `.markdown` plus optional `.pdf`, `.docx`, `.html`/`.htm`, and `.csv` (feature flags below), up to 5 MiB (`RAG_MAX_UPLOAD_BYTES`). Parsers never execute macros, never fetch remote URLs, and reject encrypted/corrupt/empty inputs. OCR for scanned PDFs is **not** implemented.
2. Per-user upload rate limiting is enforced via Redis (`RAG_UPLOAD_RATE_LIMIT_COUNT` uploads per `RAG_UPLOAD_RATE_LIMIT_WINDOW_SECONDS`). Exceeding the limit returns `429` with `UPLOAD_RATE_LIMIT_EXCEEDED`; Redis/rate-limit failures fail closed with `503` / `UPLOAD_RATE_LIMIT_UNAVAILABLE`.
3. Chunking: default `RAG_CHUNKING_STRATEGY=char` keeps the deterministic character window. `markdown` enables structure-aware Markdown packing (headings, lists, fenced code). PDF/DOCX/HTML/CSV use parser structure blocks (pages/sections/rows) with `heading_path` / `page` metadata when available. Parent-child chunk storage is **deferred**.
4. Embedding batches are limited to 32 inputs and 256 chunks per upload. Retry is bounded. Optional Redis **vector-only** cache (`RAG_EMBEDDING_CACHE`) keys by model/dimensions/task/version/content hash and never stores document text; cache failure falls back to Gemini. Chunks store `embedding_version`, `parser_version`, and `chunking_version`. Controlled re-embedding is available via `python -m app.services.rag_reembed` (dry-run by default; requires `RAG_REEMBED_CONFIRM=1` + `--apply`). Partial database writes remain in `ingesting` / `reembedding` and are hidden from retrieval.
5. `user_id`/owner identity comes from the authenticated principal. Documents are private by default. Owners can share read-only with active users in the same explicit tenant; each request checks current share metadata. Revoking a share updates chunk ACL metadata and takes effect on subsequent requests.
6. Vector retrieval uses the verified index with configurable `numCandidates` (`RAG_NUM_CANDIDATES`, auto-capped), `limit`, and `RAG_MIN_VECTOR_SCORE`. Atlas prefilters authorized owners (`user_id` `$in`), `project_id`, authorized `document_id` (`$in`), `status: ready`, and `tenant_id` when non-null. `_same_scope()` and `authorize_candidates()` recheck after vector/lexical fusion, after rerank, and before citations enter the LLM context.
7. Lexical retrieval defaults to bounded token-overlap (`RAG_LEXICAL_BACKEND=overlap`). Optional Atlas Search BM25 (`bm25`) uses a **separate** search index (`RAG_SEARCH_INDEX=rag_text_index`) with readiness validation; missing/misconfigured BM25 falls back to overlap and is never silently labeled as BM25. Hybrid mode fuses vector + lexical rankings with weighted Reciprocal Rank Fusion (`RAG_RRF_K`, `RAG_HYBRID_*_WEIGHT`).
8. Optional reranking (`RAG_RERANKER=none|heuristic|cross_encoder`, default `none`) runs only on authorized candidates (top-N). Cross-encoder is lazy and optional (`sentence-transformers`); failures fall back to heuristic. Query decomposition remains on by default; conversation rewrite, Groq multi-query, and HyDE are off by default. HyDE passages are retrieval-only (never cited/stored as evidence) and use the same ACL filters.
9. `retrieve()` packs context with a **token** budget (`RAG_CONTEXT_TOKEN_BUDGET`, tiktoken `cl100k_base`) including evidence-wrapper overhead, near-duplicate removal, and optional Groq compression (`RAG_ENABLE_SEMANTIC_COMPRESSION=false` by default) that preserves `source_id` mappings. Character-budget `select_context(..., budget_chars=...)` remains for compatibility.
10. The authenticated chat endpoint invokes RAG only when `useRag` is true and the query indicates a document request. A document request with no authorized evidence abstains. Retrieval failures fail closed with a sanitized `503`. RAG answers use no tools or personal-memory context and treat retrieved text as untrusted evidence.
11. Document listing uses cursor pagination (`limit` 1–100, opaque `cursor` / `next_cursor`). Read-only sharing, owner deletion, chat citations, and citations in stream metadata are available through the existing API. No frontend was added.

## Configuration

All application `RAG_*` settings are defined in `app/services/rag_config.py` and listed in `backend/.env.example`. Defaults below are the code defaults.

| Variable | Default | Required? | Purpose |
|---|---|---|---|
| `RAG_CHUNKS_COLLECTION` | `rag_chunks` | Optional | Mongo collection for document chunks |
| `RAG_VECTOR_INDEX` | `vector_index` | Optional | Atlas Vector Search index name |
| `RAG_EMBEDDING_MODEL` | `gemini-embedding-001` | Optional | Embedding model id |
| `RAG_EMBEDDING_DIMENSIONS` | `768` | Optional | Must match Atlas index dimensions |
| `RAG_VECTOR_SIMILARITY` | `cosine` | Optional | Must match Atlas index similarity |
| `RAG_MAX_UPLOAD_BYTES` | `5242880` (5 MiB) | Optional | Max upload body size |
| `RAG_UPLOAD_RATE_LIMIT_COUNT` | `10` | Optional | Max uploads per user per window |
| `RAG_UPLOAD_RATE_LIMIT_WINDOW_SECONDS` | `60` | Optional | Rate-limit window length |
| `RAG_CHUNK_SIZE_CHARS` | `1200` | Optional | Chunk window size |
| `RAG_CHUNK_OVERLAP_CHARS` | `160` | Optional | Chunk overlap |
| `RAG_MAX_CHUNKS_PER_DOCUMENT` | `256` | Optional | Hard cap per upload |
| `RAG_CHUNKING_STRATEGY` | `char` | Optional | `char` or `markdown` |
| `RAG_ENABLE_PDF` | `true` | Optional | Enable PDF parser (`pypdf`) |
| `RAG_ENABLE_DOCX` | `true` | Optional | Enable DOCX parser (`python-docx`) |
| `RAG_ENABLE_HTML` | `true` | Optional | Enable HTML parser (`beautifulsoup4`) |
| `RAG_ENABLE_CSV` | `true` | Optional | Enable CSV parser (stdlib) |
| `RAG_EMBEDDING_CACHE` | `true` | Optional | Redis vector-only embed cache |
| `RAG_EMBEDDING_CACHE_TTL_SECONDS` | `604800` | Optional | Cache TTL (7 days) |
| `RAG_EMBEDDING_VERSION` | `rag-embed-v1` | Optional | Logical embedding version label |
| `RAG_TOP_K` | `6` | Optional | Chunks considered for context packing |
| `RAG_CANDIDATE_LIMIT` | `80` | Optional | Max fused/reranked candidates |
| `RAG_NUM_CANDIDATES` | `0` (auto) | Optional | Atlas `numCandidates`; `0` ⇒ `min(limit*10, 800)` |
| `RAG_LEXICAL_SCAN_LIMIT` | `2000` | Optional | Lexical / readable-doc scan bound |
| `RAG_CONTEXT_BUDGET_CHARS` | `12000` | Optional | Character budget (compat / explicit callers) |
| `RAG_CONTEXT_TOKEN_BUDGET` | `3000` | Optional | Token budget for `retrieve()` packing + wrapper |
| `RAG_CONTEXT_MAX_CHUNKS` | `8` | Optional | Max chunks packed into context |
| `RAG_MIN_VECTOR_SCORE` | `0.25` | Optional | Minimum vector score retained |
| `RAG_LEXICAL_BACKEND` | `overlap` | Optional | `overlap` or `bm25` (requires Atlas Search) |
| `RAG_SEARCH_INDEX` | `rag_text_index` | Optional | Atlas Search index name (BM25) |
| `RAG_HYBRID_VECTOR_WEIGHT` | `1.0` | Optional | RRF weight for vector ranking |
| `RAG_HYBRID_LEXICAL_WEIGHT` | `1.0` | Optional | RRF weight for lexical ranking |
| `RAG_RRF_K` | `60` | Optional | RRF smoothing constant |
| `RAG_RERANKER` | `none` | Optional | `none` / `heuristic` / `cross_encoder` |
| `RAG_ENABLE_RERANKING` | `false` | Optional | Legacy; if true and reranker=`none` ⇒ heuristic |
| `RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional | Cross-encoder model id |
| `RAG_RERANK_TOP_N` | `20` | Optional | Max candidates rescored |
| `RAG_RERANK_TIMEOUT_SECONDS` | `8` | Optional | Soft timeout for cross-encoder |
| `RAG_ENABLE_QUERY_DECOMPOSITION` | `true` | Optional | Multi-part query clause split |
| `RAG_ENABLE_CONVERSATION_REWRITE` | `false` | Optional | Pronoun/ellipsis rewrite |
| `RAG_ENABLE_MULTI_QUERY` | `false` | Optional | Groq multi-query expansion |
| `RAG_MAX_REWRITTEN_QUERIES` | `3` | Optional | Cap on total retrieval queries |
| `RAG_ENABLE_HYDE` | `false` | Optional | Hypothetical passage retrieval (not evidence) |
| `RAG_ENABLE_SEMANTIC_COMPRESSION` | `false` | Optional | Optional Groq evidence compression |
| `RAG_COMPRESSION_MAX_OUTPUT_TOKENS` | `512` | Optional | Compression output cap |
| `RAG_NEAR_DUP_JACCARD` | `0.92` | Optional | Near-duplicate drop threshold |

Test-only gates (not application settings; see Live ACL section):

| Variable | Default | Required for live ACL? | Purpose |
|---|---|---|---|
| `RAG_ACL_INTEGRATION` | unset / off | Yes (`1` / `true` / `yes`) | Enables live Atlas ACL suite |
| `RAG_TEST_DATABASE` | unset | Yes | Disposable Atlas DB name (never production) |

Do not change the embedding model or dimension without verifying a corresponding index and re-embedding plan. Do not use personal-memory indexes for RAG.

## Feature maturity

| Area | Status | Notes |
|---|---|---|
| Text/Markdown ingest, chunking, 768-d embeddings | Implemented | Always on |
| PDF / DOCX / HTML / CSV parsers | Implemented (flagged) | No OCR; no macro/URL execution |
| Structure-aware Markdown + document blocks | Implemented | `RAG_CHUNKING_STRATEGY`; char fallback |
| Parent-child chunk storage | Deferred | ACL-safe expansion reserved for later |
| Embedding cache (Redis vectors only) | Implemented | Fail-open to provider |
| Controlled re-embedding CLI | Implemented | Per-document; dry-run default |
| Atlas vector retrieval + ACL filters | Implemented | Fail-closed; configurable candidates/score |
| Lexical **overlap** (active) | Implemented (default) | Active on current M0; never labeled as BM25 |
| Atlas Search BM25 | Code present; **unavailable on current M0** | Needs separate `rag_text_index`; Free-tier 3-index limit blocks it here; app falls back to overlap |
| Hybrid weighted RRF | Implemented | Vector + lexical; ACL recheck after fuse |
| Heuristic reranking | Optional; **Baseline B verified** | `RAG_RERANKER=heuristic` |
| Cross-encoder reranking | Optional (off) | Needs `sentence-transformers`; **live CE verification not performed** |
| Query decomposition | Implemented (on) | Explicit multi-part clauses only |
| Conversation rewrite | Optional (off) | Conservative pronoun rewrite |
| Multi-query / HyDE | Optional (off) | Groq; HyDE never cited as evidence |
| Token-aware context packing | Implemented | tiktoken; wrapper overhead included |
| LLM context compression | Optional (off) | Preserves source IDs; fails open |
| Offline evaluation harness | Implemented | Golden v2 + legacy fixture |
| Disposable live capture (Baseline A/B) | Verified 2026-11-23 | See Evaluation below; not production quality claims |
| Document list cursor pagination | Implemented | Opaque cursor; ACL-scoped |
| Upload rate limiting | Implemented | Redis per-user window |
| Frontend document UI | **Remaining feature** | Not implemented; API-only (separate from Topics 22–32 backend completion) |
| Agentic AI Topics 33–40 | Not started | Out of scope; not active RAG 22–32 work |

## Evaluation

Offline scorer: `app/services/rag_evaluation.py` (no provider/DB calls). Metrics include Recall@K, Precision@K, MRR, nDCG, answer relevance/keyword match, citation precision/recall, faithfulness heuristic, abstention accuracy, and ACL isolation checks. **LLM-as-judge is disabled** (`llm_judge.enabled=false`). Retrieval-only captures leave `answer=""` so answer keyword match and faithfulness are **not** meaningful quality gates.

Fixtures:
- `tests/fixtures/rag_eval_v1.json` — legacy synthetic regression set
- `tests/fixtures/rag_golden_v2.json` — human-reviewable golden set (multi-format, multi-doc, abstention, metadata filters, ACL). Review process: `docs/RAG_EVALUATION_GOLDEN_REVIEW.md`
- Capture tooling: `docs/RAG_EVAL_CAPTURE.md`

### Verified disposable-Atlas Baseline A/B (2026-11-23)

Same synthetic corpus and golden v2 cases; hybrid + overlap lexical; eval DB `agentic_ai_memory_rag_acl_test`. Markers `capture01` / `capture02` were cleaned after capture. Artifacts retained under `captures/`.

| Baseline | Reranker | MRR@5 | nDCG@5 | Precision@5 | Recall@5 | Notes |
|---|---|---|---|---|---|---|
| A | `none` | 0.857142 | 0.903479 | 0.228571 | 1.0 | `baseline_a_captured_results.json` / `baseline_a_report.json` |
| B | `heuristic` | 1.0 | 1.0 | 0.228571 | 1.0 | 7 rerank calls; 0 provider failures; `baseline_b_*` |

Heuristic rerank improved ranking (MRR/nDCG) without changing Precision@5 on this 9-case set. Unsupported / isolation cases still show abstention false positives (overlap + hybrid over-retrieval); see limitations.

**Recall@5 caveat:** when a golden case has **no** `relevant_chunk_ids`, the offline scorer sets `recall@5 = 1.0` vacuously (there are no relevant IDs to miss), even if retrieval returned false positives. That does **not** mean perfect negative-case behavior — use `abstention_match`, `acl_isolation_accuracy`, and `precision@5` for those cases.

```powershell
cd backend
.\venv\Scripts\python.exe -m app.services.rag_evaluation `
  --dataset tests/fixtures/rag_golden_v2.json `
  --results captures/baseline_a_captured_results.json --k 5 --output captures/baseline_a_report.json
```

Do not export real user documents into evaluation payloads. These Baseline metrics are **not** a production readiness claim.

## Atlas Search index (BM25) — manual setup

Do **not** modify `vector_index`. Create a separate Atlas Search index named `rag_text_index` (or `RAG_SEARCH_INDEX`) on `rag_chunks`:

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "text": { "type": "string", "analyzer": "lucene.standard" },
      "status": { "type": "token" },
      "document_id": { "type": "token" },
      "user_id": { "type": "token" },
      "tenant_id": { "type": "token" },
      "project_id": { "type": "token" }
    }
  }
}
```

Wait until the search index is READY/queryable, then set `RAG_LEXICAL_BACKEND=bm25`. Application code validates readiness and falls back to overlap when BM25 is unavailable. Automatic production index creation from app code is intentionally not performed.

**Current M0 cluster:** Atlas BM25 is **unavailable** (three Search indexes already in use). The **active** lexical path is **overlap**. Do not read “BM25 implemented in code” as “BM25 verified live on this cluster.”

## Test commands

### Unit / mocked RAG tests

```powershell
cd backend
.\venv\Scripts\python.exe -m unittest tests.test_rag_topics_26_32 -v
.\venv\Scripts\python.exe -m unittest tests.test_rag_pipeline -v
.\venv\Scripts\python.exe -m unittest tests.test_rag_batch1_document_intelligence tests.test_rag_batch1_critical_fixes -v
.\venv\Scripts\python.exe -m unittest tests.test_rag_document_listing -v
.\venv\Scripts\python.exe -m unittest tests.test_upload_rate_limit_service tests.test_upload_rate_limit_api tests.test_upload_body_limit -v
```

### Safe re-embedding (operator CLI)

```powershell
cd backend
# Dry-run (default): report how many chunks would be updated — no DB writes
.\venv\Scripts\python.exe -m app.services.rag_reembed `
  --document-id <64-hex> --owner-id <user-id>

# Apply (requires explicit confirmation env + --apply)
$env:RAG_REEMBED_CONFIRM = "1"
.\venv\Scripts\python.exe -m app.services.rag_reembed `
  --document-id <64-hex> --owner-id <user-id> --apply
```

**Recovery design (staged, retryable):**
1. Select owned chunks in `ready` **or** `reembedding` (so stuck jobs are retryable).
2. Generate and validate **all** replacement vectors first. Live `embedding` stays unchanged while status is still `ready`.
3. Stage into `embedding_next` + `reembed_generation` (ignored by retrieval).
4. Briefly mark targets `reembedding`, then promote every staged chunk: copy next→`embedding`, set version metadata, clear staging, set `status=ready`.
5. Gemini failure before staging: no DB mutation of live embeddings; document remains searchable.
6. Incomplete promote: raises; re-run the same CLI command — idempotent `UpdateOne` on stable `chunk_id` (no duplicates). ACL fields are never written.

Retrieval continues to require `status: ready` and never reads `embedding_next`.

Chat RAG routing (mocked retrieve / ACL args) lives in `tests.test_auth_api`:

```powershell
cd backend
.\venv\Scripts\python.exe -m unittest tests.test_auth_api -v
```

### Full backend suite

```powershell
cd backend
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Verified **2026-11-23: 757 passed, 11 skipped**. The RAG unit tests use deterministic mocks and do not require live database writes. The full backend suite contains existing integration tests; review their environment gates before pointing them at any non-disposable database.

### Live Atlas ACL integration

`tests.test_rag_acl_integration` is **skipped** unless both gates are set:

- `RAG_ACL_INTEGRATION=1` (also accepts `true` / `yes`)
- `RAG_TEST_DATABASE=<disposable database name>` that is **not** `MEMORY_DATABASE`, `agentic_ai_memory`, `production`, or `prod`

Prerequisites:

1. Working `MONGO_URI` and Gemini/`GOOGLE_API_KEY` for embeddings.
2. Disposable Atlas database named by `RAG_TEST_DATABASE`.
3. Collection `rag_chunks` in that database.
4. READY Atlas Vector Search index named `vector_index` (or `RAG_VECTOR_INDEX`) on that collection with:
   - path `embedding`, **768** dimensions, **cosine** similarity
   - filter fields: `user_id`, `project_id`, `tenant_id`, `document_id`, `status`
5. Ordinary unit runs must **never** enable this gate against production data.

The suite creates temporary documents/chunks and deletes only the data it created. Re-verified **2026-11-23: six passing tests** against disposable DB `agentic_ai_memory_rag_acl_test`. That verification demonstrates the covered ACL scenarios (owner vs stranger, share/revoke, tenant isolation, project isolation, hybrid/vector/lexical/rerank unauthorized denial, and cleanup). It does **not** prove every possible production security scenario.

Exact PowerShell commands (replace the disposable DB name):

```powershell
cd backend
$env:RAG_ACL_INTEGRATION = "1"
$env:RAG_TEST_DATABASE = "agentic_ai_memory_rag_acl_test"
.\venv\Scripts\python.exe -m unittest tests.test_rag_acl_integration -v
```

Do not run the live suite unless the disposable database and READY index are configured as above.

## Error handling (current API)

| Situation | Typical response |
|---|---|
| Unsupported/oversized/invalid upload | `422` `DOCUMENT_REJECTED` |
| Upload rate limit exceeded | `429` `UPLOAD_RATE_LIMIT_EXCEEDED` |
| Rate-limit store unavailable | `503` `UPLOAD_RATE_LIMIT_UNAVAILABLE` |
| Ingestion failure | `503` `RAG_INGESTION_FAILED` |
| Listing/share/delete store failure | `503` `RAG_STORE_UNAVAILABLE` |
| Invalid list cursor | `400` `INVALID_DOCUMENT_CURSOR` |
| Chat retrieval failure (fail closed) | `503` `RAG_RETRIEVAL_FAILED` |
| Missing document / unauthorized share target | `404` / `400` as coded |

## Current limitations

- **Not production-ready** as an end-to-end product claim; Topics 22–32 are implemented with the verified results above and the gaps below.
- No OCR for scanned/image-only PDFs.
- Parent-child chunk expansion is deferred (structure metadata is stored flat).
- **Atlas BM25 unavailable** on the current M0 cluster (three Search index limit). **Overlap lexical fallback is active** and was used for Baseline A/B. BM25 was not live-verified here.
- Unsupported-question / ACL-isolation cases can still return authorized but irrelevant packed candidates (false positives); abstention is empty-pack only.
- `recall@5` can be vacuously 1.0 for negative cases with empty `relevant_chunk_ids` (scorer convention).
- LLM-as-judge and answer faithfulness evaluation remain disabled / non-meaningful for retrieval-only captures.
- API text encoding issue remains unresolved.
- Cross-encoder rerank requires an optional `sentence-transformers` install; not pinned as a required dependency (Python 3.14 / size constraints). **Live cross-encoder verification was not performed** (Baseline B used heuristic only).
- Multi-query, HyDE, and LLM compression are off by default and depend on Groq availability.
- Weighted vector score blending beyond RRF was not added without further evaluation evidence.
- **Frontend document UI** remains a separate unimplemented feature (API-only).
- Re-embedding is explicit/per-document only — never an automatic full-corpus migration.
- Staged re-embed uses additive `embedding_next` fields (not a separate collection); Mongo multi-document transactions are not required.
- Live ACL coverage is the six gated integration tests above, plus mocked unit/API tests — not an exhaustive production penetration suite.
- Personal-memory vectors/indexes remain separate (3072-d) and must not be reused for RAG.
- Agentic AI Topics 33–40 were not started.

## Batch 1 critical fixes (Topics 23–25)

| Fix | Change |
|---|---|
| Parser security | Stronger HTML sanitization / binary mismatch rejection; disabled flags enforced; parse failures never write `ingesting` rows |
| Chunk offsets | Structure packing stores exact `source[start_char:end_char]` text; empty pack falls back to char chunking |
| Embedding cache | Rejects tainted payloads; stores vector+dimensions only; versioned keys; Redis failure → live embed |
| Re-embed recovery | Generate→stage→promote; retry includes `reembedding`; dry-run is write-free; ACL fields untouched |
