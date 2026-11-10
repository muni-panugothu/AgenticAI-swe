# Memory + RAG Implementation Plan

> Investigation date: November 2026
> Status: **Plan only — awaiting approval before development begins.**

---

## 1. Current Architecture Assessment

### 1.1 Verified Existing Stack

| Component | Technology | Evidence |
|---|---|---|
| LLM provider | **Groq** (`openai/gpt-oss-20b`) | `chatbot.py:175` — `MODEL = "openai/gpt-oss-20b"` |
| Embeddings | **Google Gemini** (`gemini-embedding-001`) | `embedding_service.py:42` — 768-dimension vectors |
| Database | **MongoDB Atlas** (database: `agentic_ai_memory`) | `memory_service.py:44` — three collections verified |
| Vector search | **MongoDB Atlas Vector Search** (`$vectorSearch`) | `memory_service.py:1717` — index `memory_vector_index` on `embedding` path |
| Session cache | **Upstash Redis** | `temporary_memory_service.py:11` — TTL-based context |
| Web search | **Tavily** | `web_search.py:3` — `TavilyClient` |
| Validation | **Pydantic** v2 (`ConfigDict(extra="forbid")`) | `tool_models.py`, `memory_schema.py` |
| Framework | **FastAPI** (in `requirements.txt`, no routes file found) | Not wired yet; `test_chatbot.py` is CLI-only |
| Agent loop | Custom multi-step loop in `chatbot.py` | `chat_with_ai()` at line 2449 — max 8 iterations |

### 1.2 MongoDB Collections (Verified)

| Collection | Purpose | Indexes |
|---|---|---|
| `memories_history` | Long-term memory documents with embeddings | `user_status_updated_idx`, `user_fact_status_idx`, `user_operation_id_idx`, plus Atlas vector index `memory_vector_index` |
| `memory_audit_events` | Immutable audit trail | `audit_user_created_idx`, `audit_operation_id_idx` |
| `memory_summaries` | Conflict-aware memory digests | `user_type_scope_status_idx`, `user_status_updated_idx`, `user_source_hash_idx`, `user_summary_operation_id_idx` |

### 1.3 Existing Memory Features (Verified Working)

| Stage | Feature | Evidence |
|---|---|---|
| 16 | Conversation memory (in-process dict + Redis context) | `chatbot.py:941` `add_to_conversation_memory`, `temporary_memory_service.py` |
| 16 | Gemini embedding generation | `embedding_service.py:49` — tested in `test_embedding.py` |
| 17 | Semantic vector search with freshness ranking | `memory_service.py:1957` `semantic_search_memories` |
| 17 | Fact extraction and conflict detection | `memory_conflict_service.py:578` `detect_memory_conflict` |
| 17 | Versioned supersede with audit trail | `memory_service.py` `supersede_memory` |
| 17 | Exclusive fact keys (role, company, preferred_database, etc.) | `memory_conflict_service.py:152` `EXCLUSIVE_FACT_KEYS` |
| 18 | Freshness scoring and lifecycle metadata | `memory_lifecycle_service.py:32` `calculate_memory_freshness_score` |
| 18 | Reinforcement tracking | `memory_service.py` `reinforce_memory` |
| 19 | Memory summarization with scope isolation | `memory_summary_service.py:216` `memory_scope_keys` |
| 19 | Deduplication intelligence | `memory_conflict_service.py:464` `classify_duplicate` |
| 19 | Audit event system | `memory_audit_service.py:77` `record_audit_event` |
| 19 | Memory metadata inspection | `memory_service.py` `inspect_memory_metadata` |
| 19 | Multi-statement fact extraction | `memory_fact_extraction.py` `extract_facts_from_message` |

### 1.4 Existing Tool Registry (Verified)

12 registered tools in `tool_registry.py`:
`calculator`, `get_current_time`, `read_file`, `search_code`, `web_search`,
`search_memory`, `save_memory`, `detect_memory_conflict`, `request_memory_update`,
`confirm_memory_update`, `reinforce_memory`, `forget_memory`,
`inspect_memory_metadata`, `summarize_memories`, `search_memory_summaries`,
`expand_memory_summary`.

### 1.5 Memory Scoping & Permissions (Topics 20–21 — DONE)

- **User / project / tenant isolation:** `memory_scope_service.build_scope_filter` and `build_vector_search_filter`; post-filters via `document_matches_scope`.
- **Session / active project:** Redis `set_active_project_scope` / `get_active_project_scope` with in-process fallback; chatbot injects authenticated scope into memory tools only.
- **Summary scoping:** Existing `memory_scope_keys` plus project/tenant-aware `_load_raw_memories`.
- **Permissions:** `memory_permission_service` enforces owner / shared_with / permission_level at the data-access layer; audit events `permission_denied|granted|revoked`.
- **Defaults:** Legacy docs without new fields behave as private user-scoped memories.

### 1.6 Test Structure (Verified)

- **Test files** in `backend/tests/` using `unittest` (includes `test_memory_scoping.py`, `test_memory_permissions.py`).
- **17 legacy test files** in `backend/` root (earlier stages).
- **442 tests passing** in the `tests/` suite (verified 2026-11-10).
- Tests use temporary MongoDB collections via `mongomock` or real Atlas with cleanup.
- No pytest; framework is `unittest` throughout.

### 1.7 What Does NOT Exist Yet

| Capability | Status |
|---|---|
| Explicit memory permissions model | **Done (Topic 21)** |
| Permission-aware retrieval | **Done (Topic 21)** |
| Document ingestion pipeline | **Not implemented** |
| Text chunking | **Not implemented** |
| RAG document store | **Not implemented** (only user memory exists) |
| Metadata-filtered search | **Partially** — `$match` after `$vectorSearch` exists for `status`, not for arbitrary metadata |
| Hybrid search (keyword + vector) | **Not implemented** — lexical reranking exists but is post-retrieval only |
| Dedicated reranking model | **Not implemented** — heuristic score adjustment in `semantic_search_memories` |
| Query rewriting | **Not implemented** |
| Context compression | **Not implemented** |
| RAG evaluation metrics | **Not implemented** |
| FastAPI routes | **Not wired** — `requirements.txt` has `fastapi` but no routes directory |
| Frontend | **Out of scope** |

---

## 2. Memory Topics 20–21 — COMPLETED (2026-11-10)

### 2.1 Topic 20 — Memory Scoping — **DONE**

**Objective:** Enforce user, session, project, and tenant boundaries so memories never leak across scopes.

#### Implemented

| Scope | Status | Location |
|---|---|---|
| User isolation | Done | `build_scope_filter` / vector pre-filter `user_id` |
| Session / active project | Done | Redis + in-process active scope helpers |
| Summary scoping | Done | `memory_scope_keys` + scoped `_load_raw_memories` |
| Project scoping | Done | `project_id` on documents + queries |
| Tenant isolation | Done | `tenant_id` on documents + queries |

#### Key files

| File | Role |
|---|---|
| `app/services/memory_scope_service.py` | Scope context, filters, active project scope |
| `app/schemas/memory_schema.py` | `project_id`, `tenant_id`, `scope` |
| `app/services/memory_service.py` | Scope-aware create/retrieve/search/mutate + indexes |
| `app/services/memory_summary_service.py` | Scope-aware summary source loading |
| `app/tools/memory_tools.py` / `app/chatbot.py` | Authenticated scope injection (never from LLM) |
| `tests/test_memory_scoping.py` | Isolation + default/missing-scope coverage |

#### Defaults & security

- Missing `project_id` / `tenant_id` on a query matches only unscoped/legacy docs (never other projects/tenants).
- Missing ownership fields default to private user ownership — never grant cross-user access.
- Production still needs real auth middleware for CLI `user_id` injection.

---

### 2.2 Topic 21 — Memory Permissions — **DONE**

**Objective:** Ownership tracking, read/write permissions, access validation, and unauthorized access prevention.

#### Implemented

- `owner_id`, `shared_with`, `permission_level` on memory documents.
- Central checks: `check_read_permission` / `check_write_permission`.
- Share lifecycle: `grant_access` / `revoke_access` (sharees are read-only).
- Levels: `private` | `project_read` | `project_write` | `public_read`.
- Enforced in get/search/reinforce/supersede/delete paths; denied attempts audit without memory text.

#### Key files

| File | Role |
|---|---|
| `app/services/memory_permission_service.py` | Permission service |
| `app/services/memory_service.py` | Mutation + retrieval enforcement |
| `app/services/memory_audit_service.py` | `permission_denied|granted|revoked` |
| `tests/test_memory_permissions.py` | Owner/shared/unauthorized/injection tests |

#### Remaining limitations

- No ownership-transfer API yet.
- `project_write` / `public_read` require matching project/tenant context.
- Audit logs store ids/metadata only (never memory content).

---

## 3. RAG Topics 22–32

### 3.1 RAG Architecture Overview

Two connected workflows that extend the existing agent:

#### Document Ingestion Pipeline

```
Document sources (files, URLs, text)
    → Document Loader (Topic 23)
    → Text Extraction + Cleaning
    → Chunking (Topic 24)
    → Metadata Attachment (Topic 27)
    → Embedding Generation (Topic 25 — reuses existing Gemini service)
    → Vector Storage (Topic 26 — reuses existing MongoDB Atlas)
```

#### Query + Retrieval Pipeline

```
User query
    → Query Rewriting (Topic 30)
    → Permission Check (Topic 21 — already planned)
    → Hybrid Search: Vector + Keyword (Topic 28)
    → Metadata Filtering (Topic 27)
    → Reranking (Topic 29)
    → Context Compression (Topic 31)
    → LLM Response Generation (existing agent loop)
    → RAG Evaluation (Topic 32)
```

#### Connection to Existing Agent

- RAG retrieval integrates as a new tool: `search_documents`.
- Document ingestion integrates as a new tool: `ingest_document`.
- The existing `search_memory` (personal facts) and `search_documents` (knowledge base) remain separate retrieval paths.
- The agent's intent classifier (`classify_intent`) gains a `KNOWLEDGE_RETRIEVAL` intent for RAG queries.
- RAG results enter the LLM context using the same untrusted-reference pattern as `USER_MEMORY_REFERENCE`.
- Permission enforcement happens before retrieved chunks enter the context window.

### 3.2 Proposed Folder Structure

```
backend/
├── app/
│   ├── schemas/
│   │   ├── memory_schema.py          (existing — add scope/permission fields)
│   │   ├── memory_summary_schema.py  (existing — unchanged)
│   │   └── rag_schema.py             (new — Document, Chunk, RAGResult models)
│   ├── services/
│   │   ├── memory_service.py         (existing — permission integration)
│   │   ├── memory_scope_service.py   (new — scope validation)
│   │   ├── memory_permission_service.py (new — permission checks)
│   │   ├── embedding_service.py      (existing — reused for document embeddings)
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   ├── document_loader.py    (new — Topic 23)
│   │   │   ├── chunking_service.py   (new — Topic 24)
│   │   │   ├── document_store.py     (new — Topics 25–26)
│   │   │   ├── metadata_service.py   (new — Topic 27)
│   │   │   ├── hybrid_search.py      (new — Topic 28)
│   │   │   ├── reranker.py           (new — Topic 29)
│   │   │   ├── query_rewriter.py     (new — Topic 30)
│   │   │   ├── context_compressor.py (new — Topic 31)
│   │   │   └── evaluator.py          (new — Topic 32)
│   │   └── ...
│   ├── tools/
│   │   ├── rag_tools.py              (new — search_documents, ingest_document)
│   │   └── ...
│   └── ...
├── tests/
│   ├── test_memory_scoping.py        (new — Topic 20)
│   ├── test_memory_permissions.py    (new — Topic 21)
│   ├── test_rag_fundamentals.py      (new — Topic 22)
│   ├── test_document_loader.py       (new — Topic 23)
│   ├── test_chunking.py              (new — Topic 24)
│   ├── test_document_embeddings.py   (new — Topic 25)
│   ├── test_vector_search.py         (new — Topic 26)
│   ├── test_metadata_filtering.py    (new — Topic 27)
│   ├── test_hybrid_search.py         (new — Topic 28)
│   ├── test_reranking.py             (new — Topic 29)
│   ├── test_query_rewriting.py       (new — Topic 30)
│   ├── test_context_compression.py   (new — Topic 31)
│   ├── test_rag_evaluation.py        (new — Topic 32)
│   └── ...
└── docs/
    ├── RAG_IMPLEMENTATION_PLAN.md    (this file)
    └── IMPLEMENTATION_STATUS.md      (new)
```

---

### Topic 22 — RAG Fundamentals

**Objective:** Establish the core data model, database collections, and integration pattern for document-based retrieval alongside the existing memory system.

#### Existing code that can be reused

- `embedding_service.py` — Gemini embedding generation (768 dimensions).
- `memory_service.py` — MongoDB Atlas `$vectorSearch` pipeline pattern.
- `memory_schema.py` — Pydantic model pattern with `ConfigDict(extra="forbid")`.
- `memory_audit_service.py` — Audit event recording.

#### Missing functionality

- RAG document and chunk Pydantic models.
- `rag_documents` and `rag_chunks` MongoDB collections.
- Atlas vector search index for chunk embeddings.
- Agent integration via new `search_documents` tool.

#### Proposed files

| File | Purpose |
|---|---|
| `app/schemas/rag_schema.py` | `RagDocument`, `RagChunk`, `RagSearchResult` Pydantic models |
| `app/services/rag/document_store.py` | CRUD for documents and chunks; vector index management |
| `app/tools/rag_tools.py` | `search_documents` tool callable by the agent |

#### Data models

```python
class RagDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    user_id: str = Field(min_length=1)          # Owner
    tenant_id: str | None = None
    project_id: str | None = None
    title: str = Field(min_length=1)
    source_type: Literal["file", "url", "text", "code"]
    source_path: str | None = None              # Original file path or URL
    content_hash: str = Field(min_length=1)      # SHA-256 for deduplication
    chunk_count: int = Field(default=0, ge=0)
    metadata: dict = Field(default_factory=dict)
    status: Literal["processing", "ready", "failed", "archived"] = "processing"
    permission_level: str = "private"
    created_at: datetime
    updated_at: datetime
    operation_id: str | None = None

class RagChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    document_id: str = Field(min_length=1)       # References RagDocument._id
    user_id: str = Field(min_length=1)
    tenant_id: str | None = None
    project_id: str | None = None
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    embedding: list[float] | None = None
    token_count: int = Field(default=0, ge=0)
    metadata: dict = Field(default_factory=dict)  # Section, page, language, etc.
    created_at: datetime
```

#### Database changes

```
New collections:
- rag_documents       (indexed on user_id + project_id + status)
- rag_chunks          (indexed on document_id, user_id + project_id)
- Atlas vector search index: rag_chunk_vector_index on rag_chunks.embedding
```

#### Tests

- Document insert and retrieve round-trip.
- Chunk creation linked to document.
- Vector search returns relevant chunks.
- User isolation on document queries.
- Status lifecycle (processing → ready → archived).

#### Acceptance criteria

- RAG schema models pass Pydantic validation.
- Document and chunk CRUD operations work with MongoDB.
- Existing memory tests continue to pass.

---

### Topic 23 — Document Loading

**Objective:** Load documents from multiple source types (plain text, Markdown, Python files, URLs) into the ingestion pipeline.

#### Existing code that can be reused

- `file_reader.py` — File path resolution and content reading.
- `code_search.py` — File enumeration patterns.
- `web_search.py` — HTTP request patterns (for URL loading).

#### Missing functionality

- Text extraction from different file formats.
- URL content fetching and cleaning.
- Batch document loading.
- Source deduplication via content hash.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/document_loader.py` | `load_from_file(path)`, `load_from_url(url)`, `load_from_text(text, title)`, `load_from_directory(path, extensions)` |

#### Dependencies

- No new dependencies for `.txt`, `.md`, `.py` files (stdlib only).
- `requests` (already in `requirements.txt`) for URL loading.
- Future: `PyPDF2` or `pdfplumber` for PDF support (deferred — justify when needed).

#### Tests

- Load a `.txt` file and verify content extraction.
- Load a `.md` file and verify metadata (title from first heading).
- Load a `.py` file and verify code content.
- Load from URL (mocked HTTP response).
- Content hash deduplication prevents re-ingestion.
- Invalid file path returns structured error.
- Directory loading discovers all matching files.

#### Acceptance criteria

- Three source types (file, text, URL) produce valid `RagDocument` records.
- Content hash prevents duplicate ingestion.
- No external dependencies beyond `requests`.

---

### Topic 24 — Chunking

**Objective:** Split loaded documents into semantically coherent chunks suitable for embedding and retrieval.

#### Existing code that can be reused

- `file_reader.py:79` `ast_enclosing_symbol_span` — AST-aware boundaries for Python code.
- `memory_fact_extraction.py` — Sentence splitting patterns.

#### Missing functionality

- Configurable text chunking strategies (fixed-size, sentence-based, recursive).
- Code-aware chunking for Python files.
- Overlap handling for context preservation.
- Token counting for chunk size management.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/chunking_service.py` | `chunk_text(text, strategy, chunk_size, overlap)`, `chunk_code(text, language)`, `ChunkingConfig` dataclass |

#### Chunking strategies

1. **Fixed-size:** Split by character/token count with overlap. Simple baseline.
2. **Sentence-based:** Split at sentence boundaries. Better semantic coherence.
3. **Recursive:** Split by paragraph → sentence → character with configurable separators.
4. **Code-aware:** Split by function/class boundaries using AST. Leverages existing `ast_enclosing_symbol_span`.

#### Configuration

```python
@dataclass
class ChunkingConfig:
    strategy: Literal["fixed", "sentence", "recursive", "code"] = "recursive"
    chunk_size: int = 512          # Target tokens per chunk
    chunk_overlap: int = 64        # Overlap tokens between adjacent chunks
    min_chunk_size: int = 50       # Minimum viable chunk
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
```

#### Dependencies

- No new dependencies. Token estimation uses character count / 4 (calibrated for Gemini tokenizer).
- Future: `tiktoken` for precise token counting if needed.

#### Tests

- Fixed-size chunking produces expected chunk count and sizes.
- Sentence chunking does not split mid-sentence.
- Recursive chunking respects separator hierarchy.
- Code chunking splits at function boundaries.
- Overlap is correctly applied between adjacent chunks.
- Empty input returns empty chunk list.
- Very small documents produce a single chunk.
- `chunk_index` is sequential and starts at 0.

#### Acceptance criteria

- All four strategies produce valid `RagChunk` records.
- No chunk exceeds `chunk_size` by more than one sentence.
- Overlap preserves context between chunks.
- Code chunking produces function-level chunks for Python.

---

### Topic 25 — Embedding Documents

**Objective:** Generate and store vector embeddings for document chunks using the existing Gemini embedding service.

#### Existing code that can be reused

- `embedding_service.py:49` `create_embedding` — Gemini `gemini-embedding-001` (768 dimensions).
- `memory_service.py:270` — Embedding storage pattern in MongoDB documents.

#### Missing functionality

- Batch embedding generation with rate limiting.
- Embedding storage in `rag_chunks` collection.
- Retry logic for Gemini API failures.
- Progress tracking for large document sets.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/document_store.py` | `embed_and_store_chunks(document_id, chunks)`, batch embedding logic |

#### Implementation notes

- Reuse existing `create_embedding()` — no new embedding provider needed.
- Batch size: 10 chunks per API call (Gemini supports batch embedding via `embed_content` with multiple texts).
- Rate limiting: 1-second delay between batches to respect Gemini free-tier limits.
- Store `embedding` field on each `RagChunk` document.
- Update parent `RagDocument.status` to `"ready"` when all chunks are embedded.

#### Tests

- Single chunk embedding generates 768-dimension vector.
- Batch embedding correctly processes multiple chunks.
- Failed embedding retries and records error.
- Document status transitions: processing → ready.
- Partially embedded document is recoverable.

#### Acceptance criteria

- All chunks in a document receive embeddings.
- Gemini API errors do not leave orphaned documents.
- Existing memory embedding tests continue to pass.

---

### Topic 26 — Vector Search

**Objective:** Query document chunks using MongoDB Atlas Vector Search, analogous to the existing memory vector search.

#### Existing code that can be reused

- `memory_service.py:1707` `_vector_search_memories` — Complete `$vectorSearch` pipeline pattern.
- `memory_service.py:1717–1748` — Pipeline with `$match`, `$project`, `vectorSearchScore`.

#### Missing functionality

- `rag_chunk_vector_index` Atlas Search index on the `rag_chunks` collection.
- `search_chunks(query, user_id, project_id, limit)` function.
- Score thresholding and result formatting for RAG context.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/document_store.py` | `search_chunks(query_embedding, user_id, ...)` using `$vectorSearch` |

#### Database changes

- **Required:** A new Atlas Vector Search index `rag_chunk_vector_index` must be created in the MongoDB Atlas UI (not via PyMongo — Atlas Search indexes are managed through the Atlas dashboard or Atlas Admin API).

```json
{
  "name": "rag_chunk_vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 768,
        "similarity": "cosine"
      },
      {
        "type": "filter",
        "path": "user_id"
      },
      {
        "type": "filter",
        "path": "project_id"
      }
    ]
  }
}
```

#### Tests

- Vector search returns relevant chunks ranked by cosine similarity.
- User isolation: other users' chunks are not returned.
- Project isolation: other projects' chunks are not returned.
- Score threshold filters low-relevance results.
- Empty collection returns empty results.

#### Acceptance criteria

- `search_chunks` returns semantically relevant chunks.
- Permission and scope filters are enforced before results are returned.
- Performance: search completes within 2 seconds for up to 10,000 chunks.

---

### Topic 27 — Metadata Filtering

**Objective:** Attach structured metadata to documents/chunks and filter search results by metadata fields.

#### Existing code that can be reused

- `memory_service.py` `$match` stage after `$vectorSearch` — same pattern for metadata filtering.
- `memory_summary_service.py:130` `classify_summary_type` — Metadata classification pattern.

#### Missing functionality

- Structured metadata extraction from document content.
- Metadata-filtered vector search (pre-filter in `$vectorSearch` or post-filter via `$match`).
- Metadata schema validation.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/metadata_service.py` | `extract_metadata(content, source_type)`, `validate_metadata(metadata)` |

#### Metadata fields

```python
# Standard metadata on RagChunk
metadata = {
    "source_type": "python",           # file type
    "language": "python",              # programming language (for code)
    "section": "Authentication",       # document section heading
    "page": 3,                         # page number (for PDFs)
    "file_path": "app/auth/service.py",
    "created_date": "2026-01-15",
    "tags": ["authentication", "security"],
}
```

#### Database changes

- Add `$match` metadata filters to the vector search pipeline.
- MongoDB Atlas Vector Search supports `filter` fields — register `source_type`, `language`, `tags` as filter paths in the index definition.

#### Tests

- Metadata extracted from Python file includes language and file path.
- Metadata extracted from Markdown includes section headings.
- Filtered search excludes non-matching metadata.
- Empty metadata filter returns all results.
- Invalid metadata is rejected by validation.

#### Acceptance criteria

- Metadata is automatically attached during ingestion.
- Search supports filtering by any registered metadata field.
- Filtering does not degrade search relevance.

---

### Topic 28 — Hybrid Search

**Objective:** Combine vector similarity search with keyword (lexical) search for better retrieval accuracy.

#### Existing code that can be reused

- `memory_service.py:2070` `_lexical_relevance_boost` — Post-retrieval keyword scoring.
- `code_search.py:42` `search_code` — Full-text substring search.

#### Missing functionality

- MongoDB Atlas full-text search index on chunk content.
- Combined scoring: `alpha * vector_score + (1 - alpha) * text_score`.
- Reciprocal Rank Fusion (RRF) for merging ranked lists.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/hybrid_search.py` | `hybrid_search(query, user_id, ...)`, RRF merging, score normalization |

#### Implementation approach

**Option A — MongoDB Atlas Search (recommended):**
Create an Atlas Search index with `text` type on `content` field. Use `$search` for keyword matching alongside `$vectorSearch` for semantic matching. Merge with RRF.

**Option B — Application-level hybrid (fallback):**
Run vector search and substring matching separately. Merge results with normalized scoring. This approach works without additional Atlas configuration.

**Decision needed:** Verify that the current MongoDB Atlas tier supports both `$vectorSearch` and `$search` in the same aggregation pipeline. If not, use Option B.

#### Tests

- Pure keyword query finds exact term matches.
- Pure semantic query finds conceptually related chunks.
- Hybrid search outperforms either alone on mixed queries.
- RRF fusion correctly interleaves ranked lists.
- Alpha parameter controls keyword vs. semantic weighting.

#### Acceptance criteria

- Hybrid search returns results from both vector and keyword sources.
- Results are ranked by fused score.
- Performance: hybrid search completes within 3 seconds.

---

### Topic 29 — Reranking

**Objective:** Reorder initial search results using a more accurate scoring model.

#### Existing code that can be reused

- `memory_service.py:2070–2104` — `_lexical_relevance_boost`, `_project_intent_boost`, and priority scoring.
- `memory_lifecycle_service.py:60` `calculate_memory_priority_score` — Multi-signal ranking formula.

#### Missing functionality

- Cross-encoder or LLM-based reranking for RAG chunks.
- Configurable reranking pipeline.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/reranker.py` | `rerank(query, chunks, strategy)`, pluggable reranking strategies |

#### Reranking strategies

1. **Heuristic (no new deps):** Multi-signal scoring using keyword overlap, metadata relevance, recency, and source authority. Extends existing `calculate_memory_priority_score` pattern.

2. **LLM-based (uses existing Groq):** Send `(query, chunk_text)` pairs to the LLM and ask for relevance scores. Expensive but accurate.

3. **Cross-encoder model (future):** Use a dedicated reranking model (e.g., `sentence-transformers` cross-encoder). Deferred — requires justification for the new dependency.

**Recommendation:** Start with heuristic reranking (strategy 1). Add LLM-based reranking (strategy 2) as an opt-in for high-stakes queries.

#### Tests

- Reranking changes the order of initial results.
- Higher-relevance chunks rank above lower-relevance chunks.
- LLM-based reranking (mocked) produces valid scores.
- Reranking with no results returns empty list.

#### Acceptance criteria

- Heuristic reranking improves retrieval precision over raw vector scores.
- Reranking is optional and configurable per query.
- No additional paid services required for the default strategy.

---

### Topic 30 — Query Rewriting

**Objective:** Transform user queries into more effective search queries for better retrieval.

#### Existing code that can be reused

- `memory_service.py:1957` `infer_query_fact_keys` — Query intent extraction.
- `agent_orchestration.py:275` `classify_intent` — Intent classification patterns.
- `memory_conflict_service.py:282` `extract_exclusive_fact` — Structured information extraction.

#### Missing functionality

- Query expansion (add synonyms and related terms).
- Query decomposition (split complex queries into sub-queries).
- Hypothetical Document Embedding (HyDE) for improved vector search.
- Conversation-aware query rewriting using chat history.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/query_rewriter.py` | `rewrite_query(query, conversation_history, strategy)`, HyDE generation, query decomposition |

#### Rewriting strategies

1. **Keyword extraction:** Extract key terms and expand with synonyms. No LLM needed.
2. **Conversation-aware:** Incorporate recent conversation context to resolve pronouns and implicit references. Uses existing conversation memory.
3. **HyDE:** Generate a hypothetical answer using the LLM, then embed that answer for vector search. Uses existing Groq + Gemini.
4. **Multi-query:** Decompose complex queries into multiple simpler queries. Uses existing Groq.

#### Tests

- Keyword extraction produces relevant search terms.
- Conversation-aware rewriting resolves "it" and "that" references.
- HyDE generates a plausible hypothetical document (mocked LLM).
- Multi-query decomposition produces valid sub-queries.
- Rewriting preserves user intent.

#### Acceptance criteria

- At least two rewriting strategies are implemented.
- Query rewriting measurably improves retrieval relevance.
- No additional dependencies required.

---

### Topic 31 — Context Compression

**Objective:** Reduce the size of retrieved context to fit within the LLM's token budget while preserving the most relevant information.

#### Existing code that can be reused

- `agent_orchestration.py` `compact_memory_context` — Memory context truncation.
- `agent_orchestration.py` `prune_messages_for_budget` — Token budget management.
- `chatbot.py:788` `limit_tool_result` — Result size limiting.

#### Missing functionality

- Extractive compression (select most relevant sentences from chunks).
- LLM-based abstractive compression (summarize chunks into a shorter context).
- Token budget allocation between memory context and RAG context.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/context_compressor.py` | `compress_context(chunks, query, max_tokens)`, extractive and abstractive strategies |

#### Compression strategies

1. **Extractive (no LLM):** Score sentences by keyword overlap with the query. Keep top-K sentences. Fast and deterministic.
2. **Abstractive (uses existing Groq):** Send retrieved chunks to the LLM with a compression prompt. More accurate but uses LLM tokens.
3. **Hybrid:** Extractive first to reduce size, then abstractive for final compression.

#### Token budget integration

```python
TOTAL_CONTEXT_BUDGET = 3500          # chars (from MAX_MEMORY_CONTEXT_CHARS)
MEMORY_CONTEXT_SHARE = 0.4          # 40% for personal memory
RAG_CONTEXT_SHARE = 0.6             # 60% for RAG documents
```

#### Tests

- Extractive compression reduces token count below budget.
- Compressed context retains answer-relevant sentences.
- Abstractive compression (mocked LLM) produces valid summary.
- Budget allocation between memory and RAG contexts is respected.
- Empty input returns empty compressed output.

#### Acceptance criteria

- Compressed context fits within the allocated token budget.
- Compression preserves the information needed to answer the query.
- No additional dependencies required.

---

### Topic 32 — RAG Evaluation

**Objective:** Measure RAG pipeline quality using automated metrics.

#### Existing code that can be reused

- `memory_audit_service.py` — Event recording pattern for evaluation logs.
- Existing `unittest` framework for test-based evaluation.

#### Missing functionality

- Retrieval evaluation metrics (precision, recall, MRR, NDCG).
- Answer quality metrics (faithfulness, relevance, completeness).
- Automated evaluation pipeline.
- Ground-truth test dataset.

#### Proposed files

| File | Purpose |
|---|---|
| `app/services/rag/evaluator.py` | `evaluate_retrieval(query, retrieved, ground_truth)`, `evaluate_answer(query, answer, context, ground_truth)` |
| `tests/test_rag_evaluation.py` | Evaluation metric tests with known ground-truth data |

#### Metrics

| Metric | Type | Description |
|---|---|---|
| **Precision@K** | Retrieval | Fraction of top-K results that are relevant |
| **Recall@K** | Retrieval | Fraction of relevant documents in top-K |
| **MRR** | Retrieval | Mean Reciprocal Rank of first relevant result |
| **Faithfulness** | Answer | Whether the answer is supported by retrieved context |
| **Relevance** | Answer | Whether the answer addresses the original query |
| **Context Utilization** | Answer | How much of the retrieved context was actually used |

#### Implementation approach

- Retrieval metrics: Computed from ground-truth relevance labels (no LLM needed).
- Answer metrics: LLM-as-judge using existing Groq (prompt-based scoring).
- Evaluation dataset: Manually curated set of 20–50 query/document/answer triples.

#### Tests

- Precision@5 calculation with known relevant documents.
- MRR calculation with ranked results.
- Faithfulness scoring (mocked LLM judge).
- Perfect retrieval produces metric values of 1.0.
- Empty retrieval produces metric values of 0.0.

#### Acceptance criteria

- At least 3 retrieval metrics and 2 answer metrics are implemented.
- Evaluation can run headlessly via `python -m unittest`.
- Results are logged to `memory_audit_events` for tracking over time.

---

## 4. Dependency Order

```
Phase 1 — Foundation (Topics 20, 21, 22)
├── Topic 20: Memory Scoping
│   └── No dependencies — extends existing memory model
├── Topic 21: Memory Permissions
│   └── Depends on: Topic 20 (scope fields)
└── Topic 22: RAG Fundamentals
    └── Depends on: Topic 20 (scope fields in RAG schema)

Phase 2 — Ingestion Pipeline (Topics 23, 24, 25)
├── Topic 23: Document Loading
│   └── Depends on: Topic 22 (RagDocument model)
├── Topic 24: Chunking
│   └── Depends on: Topic 23 (loaded documents)
└── Topic 25: Embedding Documents
    └── Depends on: Topic 24 (chunks), existing embedding_service.py

Phase 3 — Retrieval Pipeline (Topics 26, 27, 28)
├── Topic 26: Vector Search
│   └── Depends on: Topic 25 (embedded chunks), Atlas vector index
├── Topic 27: Metadata Filtering
│   └── Depends on: Topic 26 (search pipeline)
└── Topic 28: Hybrid Search
    └── Depends on: Topic 26 (vector search), Atlas text index

Phase 4 — Quality (Topics 29, 30, 31, 32)
├── Topic 29: Reranking
│   └── Depends on: Topic 28 (search results to rerank)
├── Topic 30: Query Rewriting
│   └── Depends on: Topic 26 (queries to improve)
├── Topic 31: Context Compression
│   └── Depends on: Topic 29 (reranked results to compress)
└── Topic 32: RAG Evaluation
    └── Depends on: Topics 26–31 (full pipeline to evaluate)
```

---

## 5. Database and Data-Model Design

### New MongoDB Collections

| Collection | Purpose | Key indexes |
|---|---|---|
| `rag_documents` | Document metadata and status | `user_id + project_id + status`, `content_hash + user_id`, `operation_id` |
| `rag_chunks` | Document chunks with embeddings | `document_id`, `user_id + project_id`, Atlas vector index `rag_chunk_vector_index` |

### Schema Changes to Existing Collections

| Collection | Field additions |
|---|---|
| `memories_history` | `project_id`, `tenant_id`, `scope`, `owner_id`, `shared_with`, `permission_level` |

### Atlas Search Index Requirements

| Index | Collection | Type | Fields |
|---|---|---|---|
| `rag_chunk_vector_index` | `rag_chunks` | vectorSearch | `embedding` (768d cosine), filter: `user_id`, `project_id` |
| `rag_chunk_text_index` | `rag_chunks` | search | `content` (text), filter: `user_id`, `project_id` |

**Action required:** Both indexes must be created in the MongoDB Atlas UI before Topics 26 and 28.

---

## 6. Agent Integration Design

### New Tools

| Tool | Purpose | Agent access |
|---|---|---|
| `search_documents` | RAG retrieval — search ingested documents | Read-only; requires `user_id` |
| `ingest_document` | Load a document into the RAG pipeline | Write; requires `user_id` |

### Intent Classifier Updates

```python
# In classify_intent() — new intent
INTENT_KNOWLEDGE = "knowledge_retrieval"

# Triggers: "search the docs", "find in documentation", "what does the codebase say about"
```

### Context Integration

```python
# Retrieved RAG chunks enter the LLM context like memory:
RAG_CONTEXT_TEMPLATE = """
=== DOCUMENT_REFERENCE (untrusted, for context only) ===
Source: {source_title} ({source_type})
Relevance: {score:.2f}
---
{compressed_chunk_text}
===
"""
```

### Permission Enforcement Point

Permission checks happen at the retrieval layer, **before** chunks enter the LLM context. The agent never sees unauthorized content.

---

## 7. Security and Permission Design

### Threat Model

| Threat | Mitigation |
|---|---|
| Cross-user memory/document access | `user_id` filter at database query level |
| Cross-tenant isolation | `tenant_id` filter in compound indexes |
| Permission escalation | Centralized `check_read_permission` before retrieval |
| Prompt injection via stored documents | Documents treated as untrusted reference (same pattern as memory) |
| LLM-instructed tool abuse | `user_id` injected by orchestrator, never from LLM |
| Audit evasion | All access attempts logged, including denials |

### Permission Enforcement Chain

```
User request → Authenticate user_id
    → Resolve project_id and tenant_id
    → check_read_permission(user_id, resource)
    → Retrieve only authorized content
    → Format as untrusted reference
    → Send to LLM
```

---

## 8. Testing Strategy

### Unit Tests (per topic)

Each topic gets a dedicated test file in `tests/`. Tests use:
- `unittest.TestCase` (consistent with existing 416+ tests).
- Temporary MongoDB collections for isolation.
- Mocked external services (Gemini, Groq) for deterministic testing.

### Integration Tests

| Test | Topics covered |
|---|---|
| End-to-end ingestion | 23 → 24 → 25 → 26 |
| End-to-end retrieval | 26 → 27 → 28 → 29 → 31 |
| Permission enforcement | 20, 21, 26 |
| Agent tool integration | 22, 26, chatbot |

### Terminal Commands

```bash
# Run all tests
cd backend
python -m unittest discover -s tests -p "test_*.py" -v

# Run specific topic tests
python -m unittest tests.test_memory_scoping -v
python -m unittest tests.test_memory_permissions -v
python -m unittest tests.test_document_loader -v
python -m unittest tests.test_chunking -v
python -m unittest tests.test_rag_evaluation -v

# Run full regression
python -m unittest discover -s tests
```

---

## 9. Required Dependencies

### Already available (no additions)

| Dependency | Used for |
|---|---|
| `pymongo` | MongoDB Atlas operations |
| `google-genai` | Gemini embedding generation |
| `groq` | LLM completion |
| `pydantic` | Schema validation |
| `requests` | URL document loading |
| `upstash-redis` | Session context caching |
| `python-dotenv` | Environment configuration |

### Potentially needed (justify before adding)

| Dependency | When needed | Justification |
|---|---|---|
| `tiktoken` | Topic 24 (Chunking) | Precise token counting. Alternative: character-based estimation (divide by 4). **Defer unless precision is required.** |
| `PyPDF2` or `pdfplumber` | Topic 23 (Document Loading) | PDF text extraction. **Defer until PDF support is explicitly requested.** |
| `beautifulsoup4` | Topic 23 (Document Loading) | HTML cleaning for URL sources. **Defer — basic regex cleaning may suffice initially.** |
| `sentence-transformers` | Topic 29 (Reranking) | Cross-encoder reranking model. **Defer — heuristic + LLM-based reranking covers initial needs.** |

**Recommendation:** Start with zero new dependencies. Add them one at a time with explicit justification when a topic requires them.

---

## 10. Risks and Unresolved Decisions

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| MongoDB Atlas free tier limits | Vector search index limits may block multiple indexes | Verify tier limits before creating `rag_chunk_vector_index` |
| Gemini embedding rate limits | Batch ingestion may be throttled | Rate limiting with backoff in batch embedding |
| Groq TPM budget sharing | RAG query rewriting / reranking consumes LLM tokens | Token budget allocation between agent reasoning and RAG operations |
| Atlas Search availability | `$search` for hybrid search may not be available on current tier | Fall back to application-level keyword matching (Option B in Topic 28) |
| Schema migration | Adding fields to `MemoryDocument` may affect existing data | Use optional fields with defaults; no migration needed for existing documents |

### Decisions Requiring Approval

1. **Atlas tier verification:** Before Phase 3, verify that the current MongoDB Atlas tier supports creating a second vector search index (`rag_chunk_vector_index`) alongside the existing `memory_vector_index`.

2. **Tenant isolation scope:** Is multi-tenant isolation (Topic 20) needed now, or can it be deferred? Adding `tenant_id` to every query has a small performance cost.

3. **PDF support:** Should Topic 23 include PDF loading (requires `PyPDF2` dependency), or defer to a future topic?

4. **LLM-based reranking budget:** Should Topic 29 include LLM-based reranking (uses Groq tokens from the same TPM budget), or start with heuristic-only?

5. **Hybrid search approach:** Option A (Atlas Search index) or Option B (application-level keyword matching) for Topic 28? Depends on Atlas tier capabilities.

6. **RAG evaluation dataset:** Should the ground-truth evaluation dataset (Topic 32) be curated from this project's own codebase, or from an external domain?

---

## 11. Phase-by-Phase Acceptance Criteria

### Phase 1 — Foundation (Topics 20, 21, 22)
- [ ] Memory scoping prevents cross-user, cross-project, and cross-tenant leakage.
- [ ] Permission checks block unauthorized read and write operations.
- [ ] RAG document and chunk models validate correctly.
- [ ] RAG collections exist in MongoDB with proper indexes.
- [ ] All existing 416+ tests pass.
- [ ] New tests: at least 15 tests across Topics 20–22.

### Phase 2 — Ingestion Pipeline (Topics 23, 24, 25)
- [ ] Documents can be loaded from text, files, and URLs.
- [ ] Chunks are produced with correct sizes and overlap.
- [ ] All chunks receive Gemini embeddings.
- [ ] Document status lifecycle (processing → ready) works correctly.
- [ ] Ingestion is idempotent (content hash prevents re-processing).
- [ ] New tests: at least 20 tests across Topics 23–25.

### Phase 3 — Retrieval Pipeline (Topics 26, 27, 28)
- [ ] Vector search returns semantically relevant chunks.
- [ ] Metadata filtering narrows results correctly.
- [ ] Hybrid search combines vector and keyword results.
- [ ] All searches enforce user and project isolation.
- [ ] `search_documents` tool is registered and callable by the agent.
- [ ] New tests: at least 15 tests across Topics 26–28.

### Phase 4 — Quality (Topics 29, 30, 31, 32)
- [ ] Reranking improves retrieval precision over raw scores.
- [ ] Query rewriting resolves ambiguous queries.
- [ ] Context compression fits retrieved content within token budget.
- [ ] At least 3 retrieval metrics and 2 answer metrics are implemented.
- [ ] Full RAG pipeline works end-to-end through the agent.
- [ ] New tests: at least 20 tests across Topics 29–32.

---

## 12. Recommended Implementation Order

```
1.  Topic 20 — Memory Scoping           (foundation for all RAG permissions)
2.  Topic 21 — Memory Permissions        (required before RAG retrieval)
3.  Topic 22 — RAG Fundamentals          (data models and collections)
4.  Topic 23 — Document Loading          (first ingestion step)
5.  Topic 24 — Chunking                  (second ingestion step)
6.  Topic 25 — Embedding Documents       (third ingestion step)
7.  Topic 26 — Vector Search             (core retrieval)
8.  Topic 27 — Metadata Filtering        (search refinement)
9.  Topic 28 — Hybrid Search             (search quality)
10. Topic 30 — Query Rewriting           (pre-retrieval quality — before reranking)
11. Topic 29 — Reranking                 (post-retrieval quality)
12. Topic 31 — Context Compression       (LLM integration quality)
13. Topic 32 — RAG Evaluation            (pipeline quality measurement)
```

Note: Topic 30 (Query Rewriting) is recommended before Topic 29 (Reranking) because rewriting improves the input to the entire retrieval pipeline, while reranking only improves the output ordering.
