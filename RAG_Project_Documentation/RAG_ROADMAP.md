# Agentic AI Software Engineer --- RAG Roadmap

## Purpose

This document defines the implementation roadmap for Topics 22--32 of
the Agentic AI Software Engineer project. It is the feature roadmap, not
permission to rewrite the existing application.

The implementation is divided into two parts so each layer can be built,
tested, and reviewed before the next layer depends on it.

## Implementation strategy

1.  Audit the existing repository and the other project documentation
    before changing code.
2.  Produce a repository-specific implementation plan for all eleven
    topics.
3.  Implement Part 1 (Topics 22--27), then test and review it.
4.  Implement Part 2 (Topics 28--32), then test and review it.
5.  Run the full regression suite and verify integration with the
    existing Python chatbot.
6.  Connect a frontend only in a later, separate integration phase.

Do not implement all topics in one unreviewed change.

------------------------------------------------------------------------

## Part 1 --- Core RAG pipeline (Topics 22--27)

### Topic 22 --- RAG Fundamentals

**Goal:** Establish a clear retrieval-augmented generation pipeline that
retrieves relevant, authorized document content and supplies it to the
existing language-model workflow.

**Scope:** - Define the end-to-end RAG flow: query → retrieval →
permission filtering → context preparation → answer generation → source
attribution. - Define the boundaries between personal memory and
user-provided RAG documents. - Reuse the existing agent/chat
orchestration where appropriate; do not create a competing chatbot loop
without a demonstrated need. - Define failure behavior for no relevant
results, inaccessible documents, and retrieval errors.

**Acceptance criteria:** - A documented pipeline and component
responsibility map exist. - RAG retrieval is not allowed to bypass
authentication or authorization. - Answers can distinguish retrieved
evidence from unsupported information. - Existing non-RAG chatbot
behavior remains intact.

### Topic 23 --- Document Loading

**Goal:** Ingest supported documents safely and consistently.

**Scope:** - Audit existing dependencies and supported file types before
choosing parsers. - Validate file type, size, encoding, and content
before processing. - Extract text and retain useful source metadata
(document identity, owner, filename, page/section where available,
timestamps, and provenance). - Handle empty, malformed, duplicate, and
unsupported documents. - Avoid executing uploaded content or treating
document text as trusted instructions. - Make ingestion observable and
retry-safe.

**Acceptance criteria:** - Valid supported documents can be loaded and
traced to their source. - Invalid or unsupported files fail safely with
useful errors. - Duplicate/retry behavior is defined and tested. -
Ownership and access metadata are attached before chunks become
searchable.

### Topic 24 --- Chunking

**Goal:** Split extracted content into retrieval-friendly chunks without
losing source traceability.

**Scope:** - Choose chunking strategies based on document type and
measured retrieval needs. - Configure chunk size and overlap centrally
rather than scattering constants. - Preserve document ID, chunk order,
page/section references, and relevant metadata. - Avoid empty chunks and
excessive duplication. - Make chunking deterministic where possible.

**Acceptance criteria:** - Tests cover short, long, empty,
multilingual/Unicode, and boundary-case documents. - Chunks can be
traced back to their source and position. - Chunking configuration is
documented and can be tuned without unsafe schema changes.

### Topic 25 --- Embedding Documents

**Goal:** Generate and store embeddings for document chunks using the
project's verified embedding configuration.

**Scope:** - Inspect the currently configured Gemini embedding model,
output dimensions, task settings, and existing Atlas index definition
before implementation. - Verify that stored vectors and the vector index
dimensions match exactly. - Reuse the project's existing embedding
client/configuration where appropriate. - Handle rate limits, transient
failures, retries, batching, and partial ingestion safely. - Record
model/configuration provenance so incompatible embeddings are not
silently mixed.

**Acceptance criteria:** - Embeddings are finite, non-empty vectors of
the expected dimension. - Dimension and model compatibility are checked
before storage/search. - Failed batches can be retried without creating
uncontrolled duplicate chunks. - Existing memory embeddings and indexes
are not modified.

### Topic 26 --- Vector Search

**Goal:** Retrieve semantically relevant RAG chunks from the configured
vector store.

**Scope:** - Inspect the actual MongoDB Atlas collection and vector
index definition before querying. - Use the RAG-specific
collection/index; do not repurpose the personal-memory collection or
index. - Apply authorization constraints as part of retrieval, not as a
UI-only post-filter. - Define top-k, score handling, no-result behavior,
and timeouts. - Return source/chunk identifiers and relevance metadata
needed for citations and evaluation.

**Acceptance criteria:** - Relevant documents are retrievable using
semantic queries. - Queries cannot retrieve another user's private
chunks. - Missing index, dimension mismatch, timeout, and empty-result
behavior are handled clearly. - Retrieval results are traceable to
source documents.

### Topic 27 --- Metadata Filtering

**Goal:** Combine semantic retrieval with validated metadata
constraints.

**Scope:** - Define and validate filterable metadata (for example:
authenticated owner, visibility, document ID, document type, and
permitted scope). - Derive security-sensitive filters from the
authenticated server-side identity and permission services. - Never
trust a caller-supplied `user_id`, owner ID, or visibility value as
authorization proof. - Ensure filters are applied in the database query
where supported and are consistent with the permission model. - Test
private, shared, and inaccessible document cases according to existing
project semantics.

**Acceptance criteria:** - Users retrieve only documents they are
authorized to access. - Metadata filters cannot be manipulated to escape
scope. - Tests cover cross-user access, shared access, revoked access,
and missing metadata. - Existing memory permissions and scopes remain
unchanged.

**Part 1 exit gate:** All Part 1 acceptance criteria pass; targeted
tests and existing memory/authentication regression tests pass;
implementation decisions and limitations are documented. Do not begin
Part 2 until this gate is reviewed.

------------------------------------------------------------------------

## Part 2 --- Advanced retrieval and quality (Topics 28--32)

### Topic 28 --- Hybrid Search

**Goal:** Combine semantic/vector retrieval with lexical search when it
improves recall and precision.

**Scope:** - Inspect Atlas search capabilities and the existing
deployment/index configuration. - Define lexical and vector retrieval
paths and a principled result-fusion method. - Normalize or rank scores
appropriately; do not directly compare incompatible raw score scales. -
Preserve identical authorization and metadata filters across both
retrieval paths. - Measure hybrid search against vector-only and
lexical-only baselines.

**Acceptance criteria:** - Hybrid retrieval improves or justifiably
complements the baseline on a defined evaluation set. - Both retrieval
paths enforce the same permissions. - Fusion behavior is deterministic
and tested.

### Topic 29 --- Reranking

**Goal:** Reorder retrieved candidates using a reranking strategy to
improve the relevance of the final context.

**Scope:** - Evaluate available local or API-based reranking options
against project cost, latency, privacy, and reliability constraints. -
Keep reranking optional/configurable and provide safe behavior when
unavailable. - Limit candidate count and input size. - Preserve source
identity and authorization metadata through reranking.

**Acceptance criteria:** - Reranking is evaluated against a baseline
using relevance metrics or reviewed examples. - It cannot introduce
unauthorized candidates. - Timeout, provider failure, and empty results
are handled.

### Topic 30 --- Query Rewriting

**Goal:** Improve retrieval queries while preserving the user's intent
and authorization scope.

**Scope:** - Support useful transformations such as query clarification,
contextual expansion, and decomposition where justified. - Do not
silently change entities, dates, constraints, or user intent. - Keep the
original query for traceability and evaluation. - Treat conversation
context as untrusted input for retrieval planning and retain
user/document scope. - Avoid unnecessary rewrites for already-clear
queries.

**Acceptance criteria:** - Rewritten queries improve retrieval on a
measured test set. - Original and rewritten queries are observable
without leaking private content. - Rewriting cannot broaden permissions
or document scope.

### Topic 31 --- Context Compression

**Goal:** Fit the most useful authorized evidence into the model context
while preserving factual meaning and source traceability.

**Scope:** - Prefer relevance-based selection and safe deduplication
before lossy summarization. - Preserve citations/source IDs and
important qualifiers, numbers, negations, and uncertainty. - Treat
compression as a transformation of untrusted document content, not as an
instruction source. - Define token/context budgets and fallback
behavior. - Do not compress away evidence needed to answer the query
accurately.

**Acceptance criteria:** - Compressed context remains traceable to
original chunks. - Tests detect loss of key facts, negations, numbers,
and source attribution. - Context limits and failure behavior are
explicit.

### Topic 32 --- RAG Evaluation

**Goal:** Measure retrieval and answer quality, security, latency, and
cost using a repeatable evaluation process.

**Scope:** - Build a small, versioned evaluation dataset with questions,
expected source documents/chunks, and expected answer characteristics. -
Measure retrieval quality (for example: Recall@k, Precision@k, MRR or
nDCG where appropriate). - Assess answer faithfulness, relevance,
citation correctness, and appropriate abstention. - Include adversarial
and permission-isolation cases. - Track latency, embedding/reranking
calls, token usage, and provider failures. - Compare Part 2 techniques
against Part 1 baselines.

**Acceptance criteria:** - Evaluation can be rerun from documented
commands. - Metrics and known limitations are reported honestly. -
Security tests are mandatory gates, not quality trade-offs. - No claim
of production readiness is made solely from unit tests.

**Part 2 exit gate:** Advanced retrieval is measurably evaluated, all
security tests pass, and the full backend regression suite is reviewed.

------------------------------------------------------------------------

## Global definition of done

A topic is complete only when: - Its implementation matches the approved
repository-specific design. - Unit and integration tests cover normal,
boundary, and failure cases. - Authorization is enforced server-side. -
Logs avoid secrets and unnecessary private document content. - Errors,
retries, and resource cleanup are handled. - Documentation and test
commands are updated. - Existing chatbot, memory, authentication, and
tool behavior remain regression-tested.

A passing isolated test does not by itself prove production readiness.
