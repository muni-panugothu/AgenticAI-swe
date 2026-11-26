# Agentic AI Software Engineer --- RAG Architecture and Integration Rules

## 1. Purpose

This document defines architecture constraints for adding RAG to the
existing Agentic AI Software Engineer repository. It is a design
contract for the audit and implementation process. The actual repository
must be inspected before choosing file names, classes, APIs, or schema
details.

## 2. Known project context to verify

The project has previously used or reported the following: - Python
backend and a terminal chatbot (`test_chatbot.py`). - FastAPI
authentication/API service (reported on port 8787). - MongoDB Atlas for
persistent data. - Redis/Upstash for selected state or caching. - Groq
chat model configuration, previously `openai/gpt-oss-20b`. - Gemini
embeddings, previously configured with `gemini-embedding-001`. - A RAG
collection named `rag_chunks` and an Atlas vector index named
`vector_index` have previously been created/reported. - Existing
personal-memory storage, scope, permission, retrieval, and audit
functionality. - Refresh-token sessions stored server-side with hashed
opaque refresh tokens and token-family rotation/reuse handling.

These are historical project details, not a substitute for checking the
current `.env` configuration, source code, MongoDB collections, Atlas
index definitions, or deployed services. Report any differences found.

## 3. Core architecture principles

### 3.1 Keep RAG documents separate from personal memory

RAG document chunks and personal conversational memories have different
lifecycle, provenance, retrieval, and permission semantics.

-   Use the existing RAG-specific collection if it is still the intended
    collection.
-   Do not store document chunks in the personal-memory history
    collection.
-   Do not reuse or alter the existing personal-memory vector index for
    RAG.
-   Do not silently migrate or delete existing memory records.
-   If a new collection or index is required, explain why and obtain
    approval before destructive or incompatible changes.

### 3.2 Reuse authentication and server-side identity

-   Reuse the existing authentication mechanism and authenticated
    principal.
-   Derive owner/user identity from validated server-side authentication
    context.
-   Never trust a request body's `user_id`, owner ID, or claimed role as
    proof of identity.
-   Do not create a parallel login system for RAG.
-   Keep refresh tokens, access tokens, and secrets out of document
    metadata, prompts, logs, and test output.

### 3.3 Reuse permission and scope services

-   Inspect the existing memory scope and permission services before
    designing RAG authorization.
-   Reuse shared policy primitives where their semantics fit.
-   Do not assume memory permissions automatically equal document
    permissions; document the mapping.
-   Enforce permissions in the backend and in retrieval query
    construction.
-   Apply equivalent security constraints to vector search, lexical
    search, hybrid fusion, reranking, query rewriting, and citations.
-   A result must not become visible merely because a later pipeline
    stage receives it.
-   Revoked or removed access must be reflected in subsequent retrieval.

### 3.4 Treat documents and retrieved text as untrusted data

Documents, filenames, metadata, extracted text, and retrieved chunks can
contain prompt-injection instructions.

-   Never execute document content.
-   Never interpret retrieved text as system/developer instructions.
-   Delimit retrieved evidence from instructions in model prompts.
-   Do not allow document text to change tools, permissions, identities,
    or system policies.
-   Test prompt-injection attempts embedded in documents.

### 3.5 Preserve provenance and citations

Every chunk should be traceable to: - Its source document. - Its owner
and access scope. - Its chunk order or stable chunk identifier. - Page,
section, or other location when available. - The
parser/chunking/embedding configuration needed for reproducibility.

The exact schema must be chosen after inspecting current code and data.
Avoid adding speculative fields without documenting their purpose.

## 4. Proposed logical pipeline

The following is a logical model, not a required class or file layout:

### Ingestion path

1.  Authenticate the caller.
2.  Validate upload and document policy.
3.  Create or identify a document record with ownership/access metadata.
4.  Parse supported content safely.
5.  Normalize extracted text while retaining source locations.
6.  Split text into chunks.
7.  Generate embeddings using the verified embedding configuration.
8.  Store chunks and metadata in the RAG-specific store.
9.  Mark ingestion status only when the intended work is complete; make
    retries idempotent or safely recoverable.

### Query path

1.  Authenticate the caller.
2.  Validate the query and any user-selectable filters.
3.  Derive owner, role, and access scope from the authenticated server
    context.
4.  Optionally rewrite the query without broadening permissions.
5.  Retrieve vector and/or lexical candidates using the same
    authorization constraints.
6.  Fuse and optionally rerank authorized candidates.
7.  Select or compress evidence within a defined context budget.
8.  Generate an answer using the existing chat/agent orchestration where
    appropriate.
9.  Return answer, source references, and an appropriate abstention when
    evidence is insufficient.
10. Record privacy-conscious telemetry and evaluation data.

## 5. Embedding and vector-index compatibility

Before implementing embedding writes or vector queries: - Inspect the
actual embedding model name and provider. - Inspect the exact output
dimension produced by the configured model/API. - Inspect the existing
RAG Atlas vector index definition and dimensions. - Verify the stored
vector shape and index dimensions agree. - Verify query embeddings use
the same compatible model/configuration as document embeddings. - Do not
mix vectors from incompatible models or dimensions in one index. - Do
not change the existing memory index or its dimensions. - If the RAG
collection/index is incompatible, stop and report the mismatch with a
safe migration proposal. Do not perform a destructive migration
automatically.

A previously reported setup included `gemini-embedding-001` and a
768-dimensional `vector_index`; verify the actual configured output
dimension and index before proceeding. Do not assume those values are
compatible.

## 6. Storage and lifecycle considerations

Inspect the existing MongoDB conventions before defining schemas. At
minimum, the design must address: - Stable document and chunk
identifiers. - Owner and access scope. - Source filename/type and
provenance. - Chunk order and source location. - Embedding
model/configuration version. - Ingestion status and failure/retry
state. - Created/updated timestamps. - Duplicate upload and re-ingestion
behavior. - Document deletion and corresponding chunk cleanup. - Access
revocation and visibility changes.

Do not store raw secrets or unnecessary sensitive content in metadata.
Define retention/deletion semantics before implementing deletion.

## 7. Reliability, security, and operations

-   Validate file size, type, parser output, and query size.
-   Set timeouts and bounded retries for external model/provider calls.
-   Handle provider rate limits and partial ingestion.
-   Avoid duplicate writes on retries.
-   Close database, HTTP, and model-client resources correctly.
-   Do not log raw tokens, passwords, or full private document contents.
-   Prefer IDs, counts, status, timing, and sanitized error categories
    in logs.
-   Use environment variables/secrets management; never hardcode keys.
-   Keep optional providers configurable and fail safely when
    unavailable.
-   Do not weaken rate limits, authentication, or authorization to make
    tests pass.

## 8. Existing chatbot integration

-   Keep the current terminal chatbot usable throughout implementation.
-   Integrate RAG as a capability in the existing orchestration rather
    than replacing the agent loop.
-   Define when RAG is invoked and how it coexists with tools and
    personal-memory retrieval.
-   Keep personal memory retrieval distinct from document RAG retrieval.
-   Avoid mixing user memory and document evidence without explicit,
    documented policy.
-   Preserve existing CLI login, logout, session restoration, and
    automatic refresh behavior.
-   Frontend integration is explicitly out of scope for Topics 22--32
    unless separately requested.

## 9. Architecture decisions that require repository audit

Claude Code must inspect and report before deciding: - Existing
package/module structure and naming conventions. - Current RAG
collection and index status. - Embedding model, dimensions, task
configuration, and batching. - Available Atlas vector and lexical search
capabilities in the actual deployment. - Existing document parsers and
allowed file types. - Existing auth principal and permission service
interfaces. - Whether Redis is appropriate for any
queue/cache/coordination need. - Existing retry, logging, audit, and
resource-lifecycle patterns. - How the current agent loop should call
RAG. - Whether any schema/index change is necessary.

Do not invent module names, APIs, environment variables, or test
commands. Derive them from the repository and document the evidence.

## 10. Non-goals

Unless separately approved, do not: - Build or connect a frontend. -
Replace the authentication system. - Rewrite personal-memory
functionality. - Change personal-memory schemas or vector dimensions. -
Delete or migrate existing user data. - Introduce a new vector database
without a demonstrated need. - Replace the existing LLM provider/model
without a documented reason. - Implement all RAG topics in one
unreviewed patch.
