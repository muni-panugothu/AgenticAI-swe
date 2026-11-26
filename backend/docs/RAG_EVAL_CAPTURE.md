# RAG Golden Capture Tooling (Baseline A/B)

Test-only tooling to seed a **synthetic** corpus into a **disposable** Atlas
database and capture real `retrieve()` outputs for `rag_golden_v2.json`.

This does **not** change chat, `retrieve()` semantics, production ingestion, ACL
rules, indexes, or `.env` defaults. Verified Baseline A/B metrics are **not** a
production readiness claim.

**Current verification date: November 23, 2026.**

## Fixture corpus and ID mapping

- Corpus root: `tests/fixtures/rag_corpus/`
- Authoritative chunk texts/IDs: `manifest.json`
- Human mapping notes: `MAPPING.md`
- **ID strategy: direct insertion** with exact golden IDs (`doc-policy-md:2`, …)
- Normal `ingest_document()` hashing is **not** used for capture seeding

Runtime ACL (`_same_scope`, `authorize_candidates`) still applies on every
`retrieve()` call.

## Database safety gates

Required process environment (session only; do not put these in `.env`):

| Variable | Purpose |
|---|---|
| `RAG_EVAL_CAPTURE=1` | Explicit opt-in |
| `RAG_EVAL_DATABASE=<name>` | Disposable DB name |

Rejected names (exact or substring): `MEMORY_DATABASE` value, `agentic_ai_memory`,
`production`, `prod`.

**Approved M0 evaluation database:** `agentic_ai_memory_rag_acl_test`

This exact name is the only deliberate allowlist exception to the
`agentic_ai_memory` substring ban. It already has a READY
`rag_chunks.vector_index` (768-d cosine + ACL filters), so capture can reuse it
without creating a fourth Atlas Search index (M0 limit is three).

The empty database `rag_eval_capture_disposable_v1` is **not** used on the
current M0 cluster: creating another Vector Search index there would exceed the
three-index limit. Prefer the allowlisted ACL test DB instead.

**Never** use `agentic_ai_memory` or the configured `MEMORY_DATABASE` as
`RAG_EVAL_DATABASE`.

Before live seed/capture, the existing `rag_chunks.vector_index` on the approved
eval DB **must** pass the read-only `verify_vector_index` preflight (READY,
queryable, 768-d cosine embedding path, ACL filter fields). Fail closed on mismatch.

**Do not run ACL integration tests and evaluation capture concurrently** against
`agentic_ai_memory_rag_acl_test`. They share the same disposable DB; concurrent
use can interleave documents and confuse marker/cleanup expectations.

Never silently falls back to the application default database.

Live capture uses `eval_database_redirect` so the approved evaluation database
stays active for **index validation, seeding, and every `retrieve()` call**.
The original `memory_service.database` pointer is restored in an outer
`finally` (including on exceptions). Each `retrieve()` is preceded by an
active-DB identity check and aborts if the pointer drifts.

Injected collections must expose `collection.database.name` matching
`RAG_EVAL_DATABASE`; missing or mismatched identities fail closed.

**One active capture marker per disposable DB:** seeding refuses to overwrite
fixture rows that already belong to a different `eval_capture_run_id`. Clean
up the old marker explicitly before starting a new run.

## Verified Baseline A/B (November 23, 2026)

| Run | Marker | Reranker | Artifacts | Headline metrics |
|---|---|---|---|---|
| A | `capture01` | `none` | `captures/baseline_a_captured_results.json`, `captures/baseline_a_report.json` | MRR@5 0.857142, nDCG@5 0.903479, P@5 0.228571, R@5 1.0 |
| B | `capture02` | `heuristic` | `captures/baseline_b_captured_results.json`, `captures/baseline_b_report.json` | MRR@5 1.0, nDCG@5 1.0, P@5 0.228571, R@5 1.0; **7** rerank calls; **0** provider failures |

Both capture JSON and report files are **preserved**. Temporary `capture01` and
`capture02` rows were **cleaned** from `agentic_ai_memory_rag_acl_test.rag_chunks`
via marker-scoped `cleanup_capture_run`.

**Lexical:** **overlap was active.** Atlas BM25 is **unavailable** on this M0
cluster (Free-tier Search index limit). That is separate from “BM25 code exists.”

**Rerank:** Baseline B verified **heuristic** only. **Cross-encoder live
verification was not performed.**

**Recall@5:** can be vacuously 1.0 for negative golden cases with empty
`relevant_chunk_ids`; interpret with abstention / Precision@5.

## Dry-run (safe default)

Dry-run validates golden↔corpus mapping and safety gates, plans seed IDs, and
writes **placeholder** capture JSON when `--output` is set. It performs **no**
Atlas writes, embeddings, or live retrieval. Omitting `--dry-run` still defaults
to dry-run; live mode requires **`--no-dry-run`**.

```powershell
cd C:\Users\Dell\AICodes\agentic-software-engineer-Pratice\backend
$env:RAG_EVAL_CAPTURE = "1"
$env:RAG_EVAL_DATABASE = "agentic_ai_memory_rag_acl_test"
.\venv\Scripts\python.exe -m tests.rag_eval_capture.capture `
  --dataset tests/fixtures/rag_golden_v2.json `
  --output captures/baseline_a_dry_run.json `
  --reranker none `
  --capture-run-id capture01 `
  --dry-run
```

## Live seed + capture

Prerequisites: approved eval DB + READY `vector_index` verified via
`verify_vector_index` + `GOOGLE_API_KEY` for Gemini embeddings. Then:

```powershell
$env:RAG_EVAL_CAPTURE = "1"
$env:RAG_EVAL_DATABASE = "agentic_ai_memory_rag_acl_test"
.\venv\Scripts\python.exe -m tests.rag_eval_capture.capture `
  --dataset tests/fixtures/rag_golden_v2.json `
  --output captures/baseline_a_captured_results.json `
  --reranker none `
  --capture-run-id capture01 `
  --no-dry-run `
  --seed
```

Baseline B: same command with `--reranker heuristic`,
`--output captures/baseline_b_captured_results.json`, and
`--capture-run-id capture02` (clean prior marker first if IDs collide).

Score offline (no DB/API):

```powershell
.\venv\Scripts\python.exe -m app.services.rag_evaluation `
  --dataset tests/fixtures/rag_golden_v2.json `
  --results captures/baseline_a_captured_results.json `
  --k 5 `
  --output captures/baseline_a_report.json
```

## Cleanup

Deletes **only** rows with `eval_capture_run_id == <capture-run-id>`.

Cleanup is **marker-scoped**. It must **not** delete unmarked ACL integration
test records that may already exist in `agentic_ai_memory_rag_acl_test`.

```powershell
.\venv\Scripts\python.exe -c "from tests.rag_eval_capture.seed import cleanup_capture_run; print(cleanup_capture_run(capture_run_id='capture01', dry_run=False))"
```

Use dry-run first (`dry_run=True`) to inspect the marker query. Cleanup verifies
zero remaining marker rows. Seeding does **not** auto-delete another run’s
marker; cleanup must succeed before a new `capture-run-id` can seed.

## ACL violation diagnostics

Capture records `unauthorized_chunk_ids` / `acl_violation` when returned
candidates fall outside the fixture documents readable under that case’s
resolved ACL scope. Runtime `authorize_candidates` / `_same_scope` are not
weakened; this is an extra diagnostic for capture quality.

## Baseline A/B reproducibility

Keep identical across A and B:

- corpus bytes / chunk IDs / ACL metadata
- questions and case scopes
- hybrid mode, top-k, context budgets, vector index, embedding model

Change **only** in-process `--reranker` (`none` vs `heuristic`). Effective config
is recorded under capture JSON `meta`.

## Metrics validity (retrieval-only captures)

**Valid / meaningful:**

- Recall@K / Precision@K / MRR / nDCG on packed `retrieved_chunk_ids` (see vacuous-recall caveat for empty relevant sets)
- Citation precision/recall on `citation_chunk_ids`
- Abstention accuracy using `abstained := (no packed candidates)`
- ACL isolation when captures show empty retrieval for isolation cases

**Not valid from retrieval-only captures:**

- Answer keyword match / faithfulness (answer is always `""`)
- End-to-end chatbot quality
- LLM-as-judge (disabled)

## Limitations

1. `retrieved_chunk_ids` are **packed** `retrieve()` candidates, not full pre-pack ranking.
2. **Overlap** lexical is active; Atlas BM25 is **unavailable** on current M0 (Free-tier index limit).
3. Unsupported-question false positives remain (authorized but irrelevant packs).
4. `recall@5` may be vacuously 1.0 when relevant IDs are empty.
5. Cross-encoder was not live-verified (Baseline B = heuristic).
6. Golden dataset has no binary PDF; handbook text is seeded with `file_type=pdf` metadata.
7. Live capture requires Gemini embeddings and disposable Atlas; offline unit tests mock both.
8. Frontend document UI is a separate remaining feature (out of scope for capture tooling).

`RAG_IMPLEMENTATION_PLAN.md` is historical design only — not active work.
