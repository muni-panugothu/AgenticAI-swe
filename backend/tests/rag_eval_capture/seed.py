"""Seed synthetic golden corpus into a disposable Atlas evaluation database.

Never run against production. Requires RAG_EVAL_CAPTURE=1 and RAG_EVAL_DATABASE.
This module does not execute seeding unless explicitly invoked by a caller.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable

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
    require_collection_database_name,
    require_marker,
)

logger = logging.getLogger("agentic.rag.eval_capture")

MARKER_FIELD = "eval_capture_run_id"


def _content_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_chunk_documents(
    manifest: dict[str, Any],
    *,
    capture_run_id: str,
    embeddings_by_chunk_id: dict[str, list[float]],
    embedding_model: str,
    embedding_dimensions: int,
    embedding_version: str,
) -> list[dict[str, Any]]:
    """Build ready rag_chunks rows with exact golden IDs."""
    marker = require_marker(capture_run_id)
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for doc in manifest["documents"]:
        doc_id = str(doc["document_id"])
        shared = list(doc.get("shared_with") or [])
        permission = str(doc.get("permission_level") or ("shared_read" if shared else "private"))
        for chunk in doc["chunks"]:
            cid = str(chunk["chunk_id"])
            text = str(chunk["text"])
            vector = embeddings_by_chunk_id.get(cid)
            if vector is None:
                raise EvalCaptureSafetyError(f"Missing embedding for chunk {cid}")
            if len(vector) != embedding_dimensions:
                raise EvalCaptureSafetyError(
                    f"Embedding dimension mismatch for {cid}: "
                    f"{len(vector)} != {embedding_dimensions}"
                )
            rows.append(
                {
                    "_id": cid,
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "user_id": doc.get("owner_id"),
                    "owner_id": doc.get("owner_id"),
                    "tenant_id": doc.get("tenant_id"),
                    "project_id": doc.get("project_id"),
                    "shared_with": shared,
                    "permission_level": permission,
                    "filename": doc.get("filename") or "",
                    "file_type": doc.get("file_type") or "",
                    "content_sha256": _content_sha(text),
                    "text": text,
                    "chunk_index": int(chunk["chunk_index"]),
                    "start_char": 0,
                    "end_char": len(text),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                    "heading_path": list(chunk.get("heading_path") or []),
                    "page": chunk.get("page"),
                    "embedding": [float(x) for x in vector],
                    "embedding_model": embedding_model,
                    "embedding_dimensions": embedding_dimensions,
                    "embedding_task_type": "RETRIEVAL_DOCUMENT",
                    "embedding_version": embedding_version,
                    "parser_version": "eval-fixture-v1",
                    "chunking_version": "eval-explicit-v1",
                    "status": "ready",
                    "created_at": now,
                    "updated_at": now,
                    MARKER_FIELD: marker,
                    "eval_fixture": True,
                    "eval_synthetic": True,
                }
            )
    return rows


def plan_seed(
    *,
    capture_run_id: str,
    manifest_path=None,
    golden_path=None,
) -> dict[str, Any]:
    """Validate corpus/golden mapping and return a dry-run plan (no DB/API)."""
    marker = require_marker(capture_run_id)
    approved_db = assert_safe_eval_database()
    manifest = load_corpus_manifest(manifest_path or DEFAULT_MANIFEST)
    golden = load_golden_dataset(golden_path or DEFAULT_GOLDEN)
    mapping = validate_corpus_against_golden(golden, manifest)
    planned_ids = [str(chunk["chunk_id"]) for doc in manifest["documents"] for chunk in doc["chunks"]]
    return {
        "dry_run": True,
        "database": approved_db,
        "capture_run_id": marker,
        "marker_field": MARKER_FIELD,
        "document_count": mapping["document_count"],
        "chunk_count": mapping["chunk_count"],
        "chunk_ids": planned_ids,
        "writes": 0,
        "note": "Dry-run only; no Atlas writes, embeddings, or cleanup performed",
    }


def verify_vector_index(collection: Any, settings: Any) -> dict[str, Any]:
    """Fail closed unless the disposable DB vector index matches RAG settings."""
    try:
        rows = list(collection.aggregate([{"$listSearchIndexes": {}}]))
    except Exception as exc:  # pragma: no cover - exercised via mocks
        raise EvalCaptureSafetyError(
            f"Atlas index inspection failed ({type(exc).__name__})"
        ) from exc
    index = next(
        (
            row
            for row in rows
            if isinstance(row, dict) and row.get("name") == settings.vector_index
        ),
        None,
    )
    if (
        not index
        or index.get("type") != "vectorSearch"
        or str(index.get("status") or "").upper() != "READY"
        or not index.get("queryable")
    ):
        raise EvalCaptureSafetyError("Configured RAG vector index is missing or not queryable")
    latest = index.get("latestDefinition")
    if not isinstance(latest, dict) or not isinstance(latest.get("fields"), list):
        raise EvalCaptureSafetyError("Atlas vector index definition is missing or malformed")
    vector = None
    filter_paths: set[str] = set()
    for item in latest["fields"]:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "vector" and item.get("path") == "embedding":
            vector = item
        elif item.get("type") == "filter" and isinstance(item.get("path"), str):
            filter_paths.add(item["path"])
    if vector is None:
        raise EvalCaptureSafetyError("Atlas vector index is missing embedding vector field")
    if int(vector.get("numDimensions") or 0) != int(settings.embedding_dimensions):
        raise EvalCaptureSafetyError("RAG embedding dimension does not match the Atlas vector index")
    if str(vector.get("similarity") or "") != str(settings.vector_similarity):
        raise EvalCaptureSafetyError("RAG vector index similarity metric mismatch")
    required = {"user_id", "project_id", "tenant_id", "document_id", "status"}
    if not required.issubset(filter_paths):
        raise EvalCaptureSafetyError("Atlas vector index is missing required ACL filter fields")
    return {"name": settings.vector_index, "status": "READY", "dimensions": settings.embedding_dimensions}


def embed_corpus_texts(
    rows_without_vectors: list[dict[str, Any]],
    *,
    embed_many: Callable[..., list[list[float]]],
    dimensions: int,
) -> dict[str, list[float]]:
    texts = [str(row["text"]) for row in rows_without_vectors]
    ids = [str(row["chunk_id"]) for row in rows_without_vectors]
    vectors = embed_many(texts, "RETRIEVAL_DOCUMENT")
    if len(vectors) != len(ids):
        raise EvalCaptureSafetyError("Embedding provider returned incomplete batch")
    out: dict[str, list[float]] = {}
    for cid, vector in zip(ids, vectors):
        if len(vector) != dimensions:
            raise EvalCaptureSafetyError(f"Bad embedding length for {cid}")
        out[cid] = [float(x) for x in vector]
    return out


def assert_single_active_capture_marker(
    collection: Any,
    *,
    capture_run_id: str,
    fixture_chunk_ids: list[str],
) -> None:
    """Permit only one active capture marker for fixture rows; never overwrite another run."""
    marker = require_marker(capture_run_id)
    ids = [str(cid) for cid in fixture_chunk_ids]
    if not ids:
        return
    conflicts = list(
        collection.find(
            {
                "$or": [{"_id": {"$in": ids}}, {"chunk_id": {"$in": ids}}],
                MARKER_FIELD: {"$exists": True, "$nin": [marker, None, ""]},
            },
            {"_id": 1, MARKER_FIELD: 1},
        )
    )
    if conflicts:
        foreign = sorted(
            {
                str(row.get(MARKER_FIELD))
                for row in conflicts
                if row.get(MARKER_FIELD) is not None
            }
        )
        raise EvalCaptureSafetyError(
            "Refusing to seed: fixture rows already belong to another capture marker "
            f"{foreign}. Run cleanup_capture_run for that marker before starting a new run."
        )


def seed_corpus(
    *,
    capture_run_id: str,
    dry_run: bool = True,
    database_name: str | None = None,
    manifest_path=None,
    golden_path=None,
    embed_many: Callable[..., list[list[float]]] | None = None,
    collection: Any | None = None,
    settings: Any | None = None,
    memory_service: Any | None = None,
    manage_redirect: bool = True,
) -> dict[str, Any]:
    """Seed fixture chunks into the approved eval DB, or return a dry-run plan.

    When ``dry_run=True`` (default), performs validation only and never writes.
    Live seeding requires explicit ``dry_run=False``.

    When ``manage_redirect=False``, the caller must already hold an active
    ``eval_database_redirect`` (used by live capture so retrieve stays on the eval DB).
    """
    marker = require_marker(capture_run_id)
    approved_db = assert_safe_eval_database(database_name)
    manifest = load_corpus_manifest(manifest_path or DEFAULT_MANIFEST)
    golden = load_golden_dataset(golden_path or DEFAULT_GOLDEN)
    mapping = validate_corpus_against_golden(golden, manifest)
    fixture_ids = [str(chunk["chunk_id"]) for doc in manifest["documents"] for chunk in doc["chunks"]]

    if dry_run:
        return plan_seed(
            capture_run_id=marker,
            manifest_path=manifest_path or DEFAULT_MANIFEST,
            golden_path=golden_path or DEFAULT_GOLDEN,
        )

    if settings is None:
        from app.services.rag_config import settings as rag_settings

        settings = rag_settings
    if embed_many is None:
        from app.services.rag_document_service import _embed_many as rag_embed_many

        embed_many = rag_embed_many

    placeholders = [
        {"chunk_id": chunk["chunk_id"], "text": chunk["text"]}
        for doc in manifest["documents"]
        for chunk in doc["chunks"]
    ]

    def _seed_into(active_collection: Any) -> dict[str, Any]:
        require_collection_database_name(active_collection, approved_name=approved_db)
        verify_vector_index(active_collection, settings)
        assert_single_active_capture_marker(
            active_collection,
            capture_run_id=marker,
            fixture_chunk_ids=fixture_ids,
        )
        embeddings = embed_corpus_texts(
            placeholders,
            embed_many=embed_many,
            dimensions=int(settings.embedding_dimensions),
        )
        rows = build_chunk_documents(
            manifest,
            capture_run_id=marker,
            embeddings_by_chunk_id=embeddings,
            embedding_model=str(settings.embedding_model),
            embedding_dimensions=int(settings.embedding_dimensions),
            embedding_version=str(settings.embedding_version),
        )
        from pymongo import ReplaceOne

        ops = [ReplaceOne({"_id": row["_id"]}, row, upsert=True) for row in rows]
        result = active_collection.bulk_write(ops, ordered=True)
        count = active_collection.count_documents({MARKER_FIELD: marker})
        logger.info(
            "rag_eval_seed_complete database=%s marker=%s upserted=%s marker_rows=%s",
            approved_db,
            marker,
            getattr(result, "upserted_count", None),
            count,
        )
        return {
            "dry_run": False,
            "database": approved_db,
            "capture_run_id": marker,
            "marker_field": MARKER_FIELD,
            "chunk_count": len(rows),
            "marker_row_count": count,
            "chunk_ids": [row["chunk_id"] for row in rows],
            "mapping": mapping,
            "writes": len(ops),
        }

    if not manage_redirect:
        if collection is None:
            if memory_service is None:
                from app.services import memory_service as memory_service_mod

                memory_service = memory_service_mod
            active_name = getattr(getattr(memory_service, "database", None), "name", None)
            from tests.rag_eval_capture.safety import assert_database_is_approved_target

            assert_database_is_approved_target(active_name, approved_name=approved_db)
            collection = memory_service.database[settings.collection_name]
        return _seed_into(collection)

    if collection is not None:
        # Standalone seed with injected collection: verify identity, do not redirect.
        return _seed_into(collection)

    with eval_database_redirect(
        database_name=database_name,
        memory_service=memory_service,
        settings=settings,
    ) as ctx:
        return _seed_into(ctx.collection)


def cleanup_capture_run(
    *,
    capture_run_id: str,
    dry_run: bool = True,
    database_name: str | None = None,
    collection: Any | None = None,
    settings: Any | None = None,
    memory_service: Any | None = None,
    manage_redirect: bool = True,
) -> dict[str, Any]:
    """Delete only rows with the exact capture-run marker."""
    marker = require_marker(capture_run_id)
    approved_db = assert_safe_eval_database(database_name)
    query = {MARKER_FIELD: marker}

    if dry_run:
        return {
            "dry_run": True,
            "database": approved_db,
            "capture_run_id": marker,
            "marker_field": MARKER_FIELD,
            "would_delete_query": query,
            "writes": 0,
            "note": "Dry-run cleanup; no deletions performed",
        }

    if settings is None:
        from app.services.rag_config import settings as rag_settings

        settings = rag_settings

    def _cleanup(active_collection: Any) -> dict[str, Any]:
        require_collection_database_name(active_collection, approved_name=approved_db)
        before = active_collection.count_documents(query)
        result = active_collection.delete_many(query)
        remaining = active_collection.count_documents(query)
        if remaining:
            raise EvalCaptureSafetyError(
                f"Cleanup left {remaining} row(s) for marker {marker}; "
                f"deleted={result.deleted_count} before={before}"
            )
        return {
            "dry_run": False,
            "database": approved_db,
            "capture_run_id": marker,
            "deleted_count": int(result.deleted_count),
            "remaining": remaining,
            "writes": int(result.deleted_count),
        }

    if not manage_redirect:
        if collection is None:
            if memory_service is None:
                from app.services import memory_service as memory_service_mod

                memory_service = memory_service_mod
            from tests.rag_eval_capture.safety import assert_database_is_approved_target

            assert_database_is_approved_target(
                getattr(getattr(memory_service, "database", None), "name", None),
                approved_name=approved_db,
            )
            collection = memory_service.database[settings.collection_name]
        return _cleanup(collection)

    if collection is not None:
        return _cleanup(collection)

    with eval_database_redirect(
        database_name=database_name,
        memory_service=memory_service,
        settings=settings,
    ) as ctx:
        return _cleanup(ctx.collection)
