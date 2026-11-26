# RAG Golden Dataset Review Process

Offline evaluation cases live under `backend/tests/fixtures/`. They must never
contain real user documents, PII, or production credentials.

**Current verification date: November 23, 2026.**

## Review statuses

| `review.status` | Meaning |
|---|---|
| `draft` | Authoring in progress; metrics may be tracked but not claimed as quality gates |
| `approved` | A human reviewer confirmed questions, relevant chunk IDs, abstention expectations, and ACL scenarios |

Synthetic or LLM-generated labels must remain `draft` until a human approves them.
Do not treat generated labels as human-reviewed.

## Approval checklist

1. Each `case_id` is unique and stable.
2. `relevant_chunk_ids` refer only to fixture IDs (never live Atlas IDs from prod).
3. Unanswerable cases set `expected_abstain: true` and empty relevant IDs.
4. ACL cases set `expect_acl_isolation` / `acl_scenario` as appropriate.
5. Multi-document cases list every required chunk ID.
6. Metadata-filter cases document required tenant/project/document constraints.
7. Update `review.reviewed_by` and notes when approving.

## Running evaluation

```powershell
.\venv\Scripts\python.exe -m app.services.rag_evaluation `
  --dataset tests/fixtures/rag_golden_v2.json `
  --results path/to/captured_results.json `
  --k 5 `
  --output rag_eval_report.json
```

Captured results must be produced from fixture corpora or disposable test
databases. Do not export private user documents into evaluation payloads.
LLM-as-judge scoring is intentionally separate and disabled in the offline
scorer (`llm_judge.enabled=false`). Answer faithfulness / keyword metrics are
not meaningful for retrieval-only captures (`answer` is empty).

**Recall@5 caveat:** if `relevant_chunk_ids` is empty, the scorer reports
`recall@5 = 1.0` vacuously (no relevant IDs to miss), even when retrieval
returned false positives. Prefer abstention accuracy, ACL isolation, and
Precision@5 for negative cases.

To produce captures safely, use the test-only tooling documented in
[RAG_EVAL_CAPTURE.md](RAG_EVAL_CAPTURE.md) (`tests/rag_eval_capture/`, synthetic
corpus under `tests/fixtures/rag_corpus/`).

## Verified Baseline A/B (November 23, 2026)

Disposable eval DB `agentic_ai_memory_rag_acl_test`; golden v2 (9 cases).
**Active lexical path: overlap.** Atlas BM25 is **unavailable** on this M0
cluster (Search index limit) — code supports BM25, but it was not live here.
Baseline B used **heuristic** rerank; **cross-encoder was not live-verified**.
Capture markers cleaned afterward; JSON artifacts retained:

| Artifact | Role |
|---|---|
| `captures/baseline_a_captured_results.json` | A: `reranker=none`, marker `capture01` |
| `captures/baseline_a_report.json` | Offline score for A |
| `captures/baseline_b_captured_results.json` | B: `reranker=heuristic`, marker `capture02` |
| `captures/baseline_b_report.json` | Offline score for B |

| Metric | Baseline A | Baseline B |
|---|---|---|
| MRR@5 | 0.857142 | 1.0 |
| nDCG@5 | 0.903479 | 1.0 |
| Precision@5 | 0.228571 | 0.228571 |
| Recall@5 | 1.0 | 1.0 |
| Rerank calls | 0 | 7 |
| Provider failures | 0 | 0 |

These numbers are retrieval ranking results on a synthetic golden set — **not**
a production readiness claim. Known gaps: unsupported-question false positives;
vacuous recall on empty-relevant negatives; LLM judge / faithfulness disabled;
frontend document UI still a separate remaining feature.
See [RAG_IMPLEMENTATION.md](RAG_IMPLEMENTATION.md) limitations.

`RAG_IMPLEMENTATION_PLAN.md` is historical design only — not active work.
