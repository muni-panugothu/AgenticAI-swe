"""Load and validate the synthetic rag_corpus against rag_golden_v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.rag_evaluation import validate_golden_dataset

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "rag_corpus"
DEFAULT_MANIFEST = CORPUS_DIR / "manifest.json"
DEFAULT_GOLDEN = FIXTURES_DIR / "rag_golden_v2.json"


class CorpusMappingError(ValueError):
    """Corpus cannot satisfy golden labels safely."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or DEFAULT_MANIFEST
    data = load_json(manifest_path)
    if not isinstance(data.get("documents"), list) or not data["documents"]:
        raise CorpusMappingError("Corpus manifest must include a non-empty documents list")
    seen_docs: set[str] = set()
    seen_chunks: set[str] = set()
    for doc in data["documents"]:
        if not isinstance(doc, dict):
            raise CorpusMappingError("Corpus documents must be objects")
        doc_id = str(doc.get("document_id") or "")
        if not doc_id:
            raise CorpusMappingError("Corpus document missing document_id")
        if doc_id in seen_docs:
            raise CorpusMappingError(f"Duplicate corpus document_id: {doc_id}")
        seen_docs.add(doc_id)
        chunks = doc.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise CorpusMappingError(f"Document {doc_id} must declare chunks")
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise CorpusMappingError(f"Invalid chunk under {doc_id}")
            cid = str(chunk.get("chunk_id") or "")
            idx = chunk.get("chunk_index")
            text = str(chunk.get("text") or "").strip()
            if not cid or idx is None or not text:
                raise CorpusMappingError(f"Chunk under {doc_id} missing chunk_id/index/text")
            expected = f"{doc_id}:{int(idx)}"
            if cid != expected:
                raise CorpusMappingError(
                    f"chunk_id {cid!r} must equal document_id:chunk_index ({expected!r})"
                )
            if cid in seen_chunks:
                raise CorpusMappingError(f"Duplicate corpus chunk_id: {cid}")
            seen_chunks.add(cid)
            if "SYNTHETIC EVAL FIXTURE" not in text:
                raise CorpusMappingError(
                    f"Chunk {cid} must be marked SYNTHETIC EVAL FIXTURE"
                )
    return data


def load_golden_dataset(path: Path | None = None) -> dict[str, Any]:
    dataset = load_json(path or DEFAULT_GOLDEN)
    validate_golden_dataset(dataset)
    return dataset


def corpus_chunk_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map chunk_id → {document, chunk}."""
    index: dict[str, dict[str, Any]] = {}
    for doc in manifest["documents"]:
        for chunk in doc["chunks"]:
            index[str(chunk["chunk_id"])] = {"document": doc, "chunk": chunk}
    return index


def corpus_document_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(doc["document_id"]): doc for doc in manifest["documents"]}


def validate_corpus_against_golden(
    golden: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Ensure every golden document and relevant_chunk_id exists in the corpus."""
    docs = corpus_document_index(manifest)
    chunks = corpus_chunk_index(manifest)
    missing_docs: list[str] = []
    for gdoc in golden.get("documents") or []:
        doc_id = str(gdoc.get("document_id") or "")
        if doc_id not in docs:
            missing_docs.append(doc_id)
            continue
        corpus_doc = docs[doc_id]
        for field in ("owner_id", "tenant_id", "project_id"):
            if corpus_doc.get(field) != gdoc.get(field):
                raise CorpusMappingError(
                    f"Corpus {doc_id} field {field}={corpus_doc.get(field)!r} "
                    f"does not match golden {gdoc.get(field)!r}"
                )
        golden_shared = list(gdoc.get("shared_with") or [])
        corpus_shared = list(corpus_doc.get("shared_with") or [])
        if sorted(golden_shared) != sorted(corpus_shared):
            raise CorpusMappingError(
                f"Corpus {doc_id} shared_with {corpus_shared!r} does not match golden {golden_shared!r}"
            )

    if missing_docs:
        raise CorpusMappingError(
            "Corpus missing golden documents: " + ", ".join(sorted(missing_docs))
        )

    missing_chunks: list[str] = []
    for case in golden.get("cases") or []:
        for cid in case.get("relevant_chunk_ids") or []:
            if str(cid) not in chunks:
                missing_chunks.append(str(cid))
    if missing_chunks:
        raise CorpusMappingError(
            "Corpus missing golden relevant_chunk_ids: "
            + ", ".join(sorted(set(missing_chunks)))
        )

    return {
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "case_count": len(golden.get("cases") or []),
        "chunk_ids": sorted(chunks),
    }
