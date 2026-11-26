# Agentic AI Software Engineer --- RAG Implementation and Testing Guide

## 1. How to use this document

Use this guide together with: - `RAG_ROADMAP.md` - `RAG_ARCHITECTURE.md`

Claude Code must read all three RAG documents and inspect the repository
before editing files. These documents define intended outcomes and
constraints; they do not prove that a particular implementation already
exists.

## 2. Mandatory first step: read-only audit

Before making code changes:

1.  Confirm the repository root and active branch.
2.  Read all three RAG Markdown documents completely.
3.  Inspect the project README and existing architecture/developer
    instructions.
4.  Inspect the backend directory tree and relevant source files.
5.  Inspect authentication, session, memory scope, memory permission,
    retrieval, embedding, MongoDB, Redis, logging, and CLI integration
    code.
6.  Inspect current tests and identify the full regression command.
7.  Verify the actual MongoDB database/collections and Atlas
    vector-index definition without modifying data.
8.  Inspect environment-variable names without printing secret values.
9.  Identify the current working tree state; do not overwrite unrelated
    user changes.

### Audit deliverable

Produce a report containing: - Existing architecture and relevant file
paths. - What can be reused. - What is missing. - Current RAG
collection/index state. - Actual embedding model and output
dimensions. - Authentication and permission enforcement points. -
Existing tests and baseline results. - Risks, assumptions, and
unresolved decisions. - A topic-by-topic implementation plan for Topics
22--32. - Proposed files to add/change, with reasons.

**Stop after the audit and plan. Do not implement anything until the
user approves the plan.**

## 3. Implementation workflow

After approval:

### Part 1 --- Topics 22--27

Implement in dependency order: 1. RAG fundamentals and pipeline
boundaries. 2. Document loading. 3. Chunking. 4. Document embeddings. 5.
Vector search. 6. Metadata filtering and authorization.

For each topic: - Explain the intended change. - Make the smallest
coherent implementation. - Add unit tests and integration tests. - Run
targeted tests. - Inspect the diff for unrelated changes and security
regressions. - Update documentation. - Report exact commands and
results.

At the end of Part 1, stop and provide a review report. Do not begin
Part 2 until the user approves.

### Part 2 --- Topics 28--32

After Part 1 is reviewed and approved: 1. Hybrid search. 2. Reranking.
3. Query rewriting. 4. Context compression. 5. RAG evaluation.

Use measured baselines. Do not add complexity unless it produces a
measurable benefit or meets an explicit requirement.

At the end of Part 2, stop and provide a final review report.

## 4. Required testing layers

### 4.1 Document loading tests

Cover: - Each supported file type. - Unsupported extension and
mismatched content type. - Empty and corrupt files. - Large files and
size boundaries. - Unicode and multilingual text. - Duplicate upload and
retry. - Parser failure and partial failure. - Source metadata and
provenance preservation. - Malicious or instruction-like document text.

### 4.2 Chunking tests

Cover: - Short text. - Long text. - Empty/whitespace-only text. -
Unicode and multilingual content. - Chunk size and overlap boundaries. -
Stable chunk ordering and IDs. - Source page/section traceability. - No
accidental loss or uncontrolled duplication.

### 4.3 Embedding tests

Cover: - Correct configured provider/model. - Expected vector dimension
and finite numeric values. - Empty input and invalid provider
response. - Rate limit, timeout, and retry. - Partial batch failure and
retry safety. - No mixing of incompatible embedding configurations. - No
modification of existing memory vectors/indexes.

### 4.4 Vector and metadata retrieval tests

Cover: - Relevant semantic query retrieves expected chunks. - No-result
and low-relevance behavior. - Top-k limits and score handling. - Correct
document/source metadata. - User A cannot retrieve User B's private
chunks. - Shared documents follow the existing approved permission
semantics. - Revoked access is denied on subsequent retrieval. -
Caller-supplied user/owner IDs cannot override authenticated identity. -
Filters apply consistently before results reach the model. - Missing
index, wrong dimensions, timeout, and provider/database failure.

### 4.5 Advanced retrieval tests

For hybrid search: - Compare vector-only, lexical-only, and hybrid
retrieval. - Use the same permission constraints in all paths. - Test
duplicate candidates and fusion ordering.

For reranking: - Compare ranking before and after reranking. - Test
provider failure, timeout, empty candidates, and bounded input. - Verify
source and permission metadata remain attached.

For query rewriting: - Test clear, ambiguous, multi-part, and
context-dependent questions. - Preserve entities, constraints, dates,
and intent. - Verify rewriting never broadens document permissions.

For context compression: - Preserve key facts, numbers, negations,
qualifiers, and citations. - Trace compressed statements to original
chunks. - Test context budget boundaries and compression failure.

### 4.6 RAG evaluation

Build a versioned, repeatable evaluation set with: - User question. -
Expected relevant source/chunk IDs. - Expected answer properties or
reference answer where suitable. - Expected abstention for unanswerable
questions. - Permission/adversarial test cases.

Report suitable retrieval metrics such as Recall@k, Precision@k, MRR, or
nDCG where the dataset supports them. Assess answer faithfulness,
relevance, citation correctness, and abstention. Track latency, token
usage, provider calls, and failure rates.

Do not fabricate scores or claim improvement without running the
evaluation.

## 5. Security acceptance gates

These are mandatory, not optional quality improvements:

-   Authentication is required for private RAG operations.
-   User identity comes from validated server-side authentication
    context.
-   Authorization is applied during retrieval, not only after retrieval
    or in the frontend.
-   Vector, lexical, hybrid, reranking, and citation stages preserve the
    same access restrictions.
-   User A cannot access User B's private documents/chunks.
-   Revoked access is enforced.
-   Retrieved document content is untrusted data and cannot override
    system instructions or invoke tools by itself.
-   No tokens, passwords, API keys, or unnecessary private document text
    appear in logs.
-   Document upload and parsing cannot execute uploaded content.
-   Existing memory permissions and authentication behavior remain
    intact.

Any critical security test failure blocks completion of the affected
part.

## 6. Regression and environment rules

-   Discover the actual project test commands from repository
    configuration and existing documentation.
-   Run focused tests first, then the full backend suite.
-   Use isolated test databases/collections and synthetic users for
    destructive or lifecycle tests.
-   Never point cleanup tests at production or personal data.
-   Do not silently skip a failing test or exclude it merely to obtain a
    green result.
-   If a test requires network/provider access and the environment
    blocks it, report it as environment-blocked and provide a
    deterministic mock/fixture test separately.
-   Explain skipped, failed, deselected, and newly added tests.
-   Compare results with the current baseline, not an outdated test
    count.
-   Preserve test output summaries and exact commands.

## 7. Git and change-management rules

-   Work on a dedicated feature branch for RAG.
-   Do not reset, force-push, or discard unrelated changes.
-   Keep commits or diffs logically grouped by part/topic where
    practical.
-   Review changed files before reporting completion.
-   Do not modify secrets or commit `.env` files.
-   Do not alter existing database indexes or data without explicit
    approval.

## 8. Final report format

For each part, report:

1.  Topics completed.
2.  Architecture decisions and reasons.
3.  Files added or changed.
4.  Database collections/indexes touched (or explicitly not touched).
5.  Security and permission behavior.
6.  Tests and exact commands.
7.  Passed, failed, skipped, and environment-blocked counts.
8.  Evaluation results, where applicable.
9.  Known limitations and follow-up work.
10. Whether the part meets its acceptance criteria.

Use one of these statuses: - `PART PASSED` -
`PART PASSED WITH LIMITATIONS` - `PART FAILED`

Do not call the system production-ready solely because tests pass. State
what has and has not been verified.

## 9. Strict instruction to Claude Code

Read and understand the repository and all three RAG documents first. Do
not assume that historical project notes exactly match the current code.
Do not invent file paths, APIs, database fields, index dimensions, or
test results.

The first task is read-only audit and a complete plan for Topics 22--32.
Do not modify code or database state during that first task. Wait for
explicit user approval before implementation.
