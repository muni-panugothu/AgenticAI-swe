"""Capture CLI: run real retrieve() for golden cases into evaluator JSON.

Retrieval-only: answer is always empty. abstained := no packed candidates.
Does not modify .env. Reranker is patched in-process for Baseline A/B.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from app.services.rag_evaluation import evaluate, validate_golden_dataset
from tests.rag_eval_capture.case_scope import resolve_all_case_scopes
from tests.rag_eval_capture.corpus import (
    DEFAULT_GOLDEN,
    DEFAULT_MANIFEST,
    load_corpus_manifest,
    load_golden_dataset,
    validate_corpus_against_golden,
)
from tests.rag_eval_capture.safety import (
    EvalCaptureSafetyError,
    assert_safe_eval_database,
    eval_database_redirect,
    require_marker,
)
from tests.rag_eval_capture.seed import plan_seed, seed_corpus

logger = logging.getLogger("agentic.rag.eval_capture")


def fixture_row_same_scope(
    doc: dict[str, Any],
    *,
    user_id: str,
    tenant_id: str | None,
    project_id: str | None,
) -> bool:
    """Mirror runtime ``_same_scope`` against fixture document metadata."""
    owner = str(doc.get("owner_id") or doc.get("user_id") or "")
    if not owner or doc.get("tenant_id") != tenant_id or doc.get("project_id") != project_id:
        return False
    if owner == user_id:
        return True
    shared = list(doc.get("shared_with") or [])
    permission = doc.get("permission_level") or ("shared_read" if shared else "private")
    return user_id in shared and permission == "shared_read" and tenant_id is not None


def allowed_fixture_ids_for_scope(
    scope: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, set[str]]:
    """Return readable fixture document_ids and chunk_ids for a capture case scope."""
    allowed_docs: set[str] = set()
    allowed_chunks: set[str] = set()
    for doc in manifest.get("documents") or []:
        if not fixture_row_same_scope(
            doc,
            user_id=str(scope["user_id"]),
            tenant_id=scope.get("tenant_id"),
            project_id=scope.get("project_id"),
        ):
            continue
        doc_id = str(doc.get("document_id") or "")
        if not doc_id:
            continue
        allowed_docs.add(doc_id)
        for chunk in doc.get("chunks") or []:
            cid = str(chunk.get("chunk_id") or "")
            if cid:
                allowed_chunks.add(cid)
    return {"document_ids": allowed_docs, "chunk_ids": allowed_chunks}


def observation_from_retrieve_result(
    result: dict[str, Any],
    *,
    allowed_chunk_ids: set[str] | None = None,
    allowed_document_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Map retrieve() output to evaluator-compatible observation.

    retrieved_chunk_ids = packed candidates only (not full pre-pack ranking).
    answer is always empty for retrieval-only captures.
    abstained = True when packed candidates are empty.

    Unauthorized IDs are recorded diagnostically; candidates are not stripped.
    """
    candidates = list(result.get("candidates") or [])
    citations = list(result.get("citations") or [])
    retrieved_ids = [str(row.get("chunk_id") or "") for row in candidates if row.get("chunk_id")]
    citation_ids = [str(row.get("chunk_id") or "") for row in citations if row.get("chunk_id")]
    evidence = [str(row.get("text") or "") for row in candidates if str(row.get("text") or "").strip()]

    unauthorized: list[str] = []
    if allowed_chunk_ids is not None or allowed_document_ids is not None:
        for row in candidates:
            cid = str(row.get("chunk_id") or "")
            doc_id = str(row.get("document_id") or "")
            # When allow-lists are provided, require membership when IDs are present.
            if allowed_chunk_ids is not None and cid and cid not in allowed_chunk_ids:
                unauthorized.append(cid)
                continue
            if allowed_document_ids is not None and doc_id and doc_id not in allowed_document_ids:
                unauthorized.append(cid or doc_id)
        # De-dupe while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for item in unauthorized:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        unauthorized = ordered

    diagnostics = dict(result.get("diagnostics") or {})
    diagnostics.update(
        {
            "degraded": bool(result.get("degraded")),
            "degraded_reason": result.get("degraded_reason"),
            "rewritten_queries": list(result.get("rewritten_queries") or []),
            "has_readable_documents": bool(result.get("has_readable_documents")),
            "ranking_semantics": "packed_candidates_only",
            "answer_semantics": "retrieval_only_empty_answer",
            "abstained_rule": "no_packed_candidates",
        }
    )
    return {
        "retrieved_chunk_ids": retrieved_ids,
        "citation_chunk_ids": citation_ids,
        "answer": "",
        "abstained": len(retrieved_ids) == 0,
        "evidence_texts": evidence,
        "unauthorized_chunk_ids": unauthorized,
        "acl_violation": bool(unauthorized),
        "latency_ms": int(result.get("latency_ms") or 0),
        "embedding_calls": int(result.get("embedding_calls") or 0),
        "rerank_calls": int(result.get("rerank_calls") or 0),
        "diagnostics": diagnostics,
    }


def build_capture_payload(
    *,
    golden: dict[str, Any],
    observations: dict[str, dict[str, Any]],
    reranker: str,
    top_k: int | None,
    capture_run_id: str,
    database: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    expected = [str(case["case_id"]) for case in golden["cases"]]
    missing = [cid for cid in expected if cid not in observations]
    if missing:
        raise EvalCaptureSafetyError(
            "Capture missing cases: " + ", ".join(sorted(missing))
        )
    extra = sorted(set(observations) - set(expected))
    if extra:
        raise EvalCaptureSafetyError(
            "Capture contains unexpected case_ids: " + ", ".join(extra)
        )
    from app.services.rag_config import settings

    return {
        "meta": {
            "dataset_version": golden.get("version"),
            "capture_run_id": capture_run_id,
            "database": database,
            "dry_run": dry_run,
            "reranker": reranker,
            "mode": "hybrid",
            "top_k": top_k if top_k is not None else settings.top_k,
            "context_token_budget": settings.context_token_budget,
            "context_max_chunks": settings.context_max_chunks,
            "lexical_backend": settings.lexical_backend,
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "vector_index": settings.vector_index,
            "ranking_semantics": "packed_candidates_only",
            "answer_semantics": "retrieval_only_empty_answer",
        },
        "cases": {cid: observations[cid] for cid in expected},
    }


def patched_rag_settings(*, reranker: str, top_k: int | None = None):
    """Context manager patching rag + rerank settings in-process (no .env writes)."""
    if reranker not in {"none", "heuristic"}:
        raise EvalCaptureSafetyError("reranker must be 'none' or 'heuristic'")
    from app.services import rag_document_service as rag
    from app.services import rag_rerank
    from app.services.rag_config import settings

    kwargs: dict[str, Any] = {
        "reranker": reranker,
        "enable_reranking": reranker != "none",
    }
    if top_k is not None:
        kwargs["top_k"] = int(top_k)
    patched = replace(settings, **kwargs)
    stack = ExitStack()
    stack.enter_context(patch.object(rag, "settings", patched))
    stack.enter_context(patch.object(rag_rerank, "settings", patched))
    return stack


def capture_cases(
    *,
    golden: dict[str, Any],
    manifest: dict[str, Any],
    reranker: str,
    retrieve_fn: Callable[..., dict[str, Any]],
    top_k: int | None = None,
    before_retrieve: Callable[[], None] | None = None,
) -> dict[str, dict[str, Any]]:
    scopes = resolve_all_case_scopes(golden)
    observations: dict[str, dict[str, Any]] = {}
    with patched_rag_settings(reranker=reranker, top_k=top_k):
        for scope in scopes:
            if before_retrieve is not None:
                before_retrieve()
            result = retrieve_fn(
                scope["question"],
                user_id=scope["user_id"],
                tenant_id=scope["tenant_id"],
                project_id=scope["project_id"],
                mode="hybrid",
            )
            if not isinstance(result, dict):
                raise EvalCaptureSafetyError(
                    f"retrieve() returned non-object for {scope['case_id']}"
                )
            allowed = allowed_fixture_ids_for_scope(scope, manifest)
            observations[scope["case_id"]] = observation_from_retrieve_result(
                result,
                allowed_chunk_ids=allowed["chunk_ids"],
                allowed_document_ids=allowed["document_ids"],
            )
    return observations


def run_capture(
    *,
    dataset_path: Path,
    output_path: Path | None,
    reranker: str,
    top_k: int | None,
    dry_run: bool,
    capture_run_id: str,
    seed: bool,
    retrieve_fn: Callable[..., dict[str, Any]] | None = None,
    memory_service: Any | None = None,
    settings: Any | None = None,
    embed_many: Callable[..., list[list[float]]] | None = None,
) -> dict[str, Any]:
    marker = require_marker(capture_run_id)
    approved_db = assert_safe_eval_database()
    golden = load_golden_dataset(dataset_path)
    validate_golden_dataset(golden)
    manifest = load_corpus_manifest(DEFAULT_MANIFEST)
    validate_corpus_against_golden(golden, manifest)
    resolve_all_case_scopes(golden)

    if dry_run:
        seed_report = plan_seed(capture_run_id=marker, golden_path=dataset_path)
        if retrieve_fn is None:
            observations = {
                str(case["case_id"]): {
                    "retrieved_chunk_ids": [],
                    "citation_chunk_ids": [],
                    "answer": "",
                    "abstained": True,
                    "evidence_texts": [],
                    "unauthorized_chunk_ids": [],
                    "acl_violation": False,
                    "latency_ms": 0,
                    "embedding_calls": 0,
                    "rerank_calls": 0,
                    "diagnostics": {
                        "dry_run": True,
                        "ranking_semantics": "packed_candidates_only",
                        "note": "Dry-run placeholder; retrieve() not executed",
                    },
                }
                for case in golden["cases"]
            }
        else:
            observations = capture_cases(
                golden=golden,
                manifest=manifest,
                reranker=reranker,
                retrieve_fn=retrieve_fn,
                top_k=top_k,
            )
    else:
        # Critical: keep eval DB redirect active for seed AND every retrieve().
        with eval_database_redirect(
            database_name=approved_db,
            memory_service=memory_service,
            settings=settings,
        ) as ctx:
            if seed:
                seed_report = seed_corpus(
                    capture_run_id=marker,
                    dry_run=False,
                    golden_path=dataset_path,
                    memory_service=ctx._memory_service,
                    settings=settings or ctx._settings,
                    embed_many=embed_many,
                    collection=ctx.collection,
                    manage_redirect=False,
                )
            else:
                seed_report = {
                    "seeded": False,
                    "capture_run_id": marker,
                    "database": approved_db,
                }
            if retrieve_fn is None:
                from app.services.rag_document_service import retrieve as live_retrieve

                retrieve_fn = live_retrieve

            def before_retrieve() -> None:
                ctx.assert_still_active()

            observations = capture_cases(
                golden=golden,
                manifest=manifest,
                reranker=reranker,
                retrieve_fn=retrieve_fn,
                top_k=top_k,
                before_retrieve=before_retrieve,
            )

    payload = build_capture_payload(
        golden=golden,
        observations=observations,
        reranker=reranker,
        top_k=top_k,
        capture_run_id=marker,
        database=approved_db,
        dry_run=dry_run,
    )
    payload["meta"]["seed"] = seed_report

    # Ensure evaluator accepts the cases object shape.
    evaluate(golden, {"cases": payload["cases"]}, k=5)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RAG golden capture tooling (test-only). "
            "Requires RAG_EVAL_CAPTURE=1 and RAG_EVAL_DATABASE. "
            "Default is dry-run (no Atlas writes)."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="Path to golden dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for captured_results JSON",
    )
    parser.add_argument(
        "--reranker",
        choices=("none", "heuristic"),
        default="none",
        help="In-process reranker mode (Baseline A=none, B=heuristic)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Optional top_k override")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate/plan only (default: true). Use --no-dry-run for live capture.",
    )
    parser.add_argument(
        "--capture-run-id",
        required=True,
        help="Unique marker for seed/cleanup",
    )
    parser.add_argument(
        "--seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed fixture corpus before live capture (ignored for dry-run plan)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_capture(
            dataset_path=args.dataset,
            output_path=args.output,
            reranker=args.reranker,
            top_k=args.top_k,
            dry_run=args.dry_run,
            capture_run_id=args.capture_run_id,
            seed=args.seed,
        )
    except (EvalCaptureSafetyError, ValueError) as exc:
        logger.error("rag_eval_capture_failed error=%s", exc)
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "meta": payload.get("meta"),
                "case_count": len(payload.get("cases") or {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
