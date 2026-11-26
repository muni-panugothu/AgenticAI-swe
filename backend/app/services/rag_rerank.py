"""Optional RAG reranking: none, heuristic overlap, or lazy cross-encoder.

Cross-encoder mode never downloads models at import/startup. Missing optional
dependencies or inference failures fall back to heuristic scoring.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.services.rag_config import settings

logger = logging.getLogger("agentic.rag")
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)

_cross_encoder = None
_cross_encoder_error: str | None = None


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1]


def heuristic_scores(query: str, candidates: list[dict[str, Any]]) -> list[float]:
    terms = set(_tokens(query))
    if not terms:
        return [0.0] * len(candidates)
    scores: list[float] = []
    for row in candidates:
        overlap = len(terms.intersection(_tokens(row.get("text") or "")))
        scores.append(overlap / max(1, len(terms)))
    return scores


def _load_cross_encoder():
    """Lazy load; never runs at application import."""
    global _cross_encoder, _cross_encoder_error
    if _cross_encoder is not None:
        return _cross_encoder
    if _cross_encoder_error is not None:
        raise RuntimeError(_cross_encoder_error)
    try:
        # Prefer sentence-transformers CrossEncoder when available; otherwise
        # onnxruntime stub path is not bundled — fail closed to heuristic.
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as exc:
        _cross_encoder_error = f"cross_encoder_unavailable:{type(exc).__name__}"
        raise RuntimeError(_cross_encoder_error) from exc
    try:
        _cross_encoder = CrossEncoder(settings.rerank_model)
    except Exception as exc:
        _cross_encoder_error = f"cross_encoder_init_failed:{type(exc).__name__}"
        raise RuntimeError(_cross_encoder_error) from exc
    return _cross_encoder


def cross_encoder_scores(query: str, candidates: list[dict[str, Any]]) -> list[float]:
    model = _load_cross_encoder()
    pairs = [(query, str(row.get("text") or "")) for row in candidates]
    started = time.monotonic()
    raw = model.predict(pairs)
    elapsed = time.monotonic() - started
    if elapsed > settings.rerank_timeout_seconds:
        raise TimeoutError("cross_encoder_timeout")
    return [float(value) for value in raw]


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    mode: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank authorized candidates; preserves retrieval_score, adds rerank_score."""
    resolved = (mode or settings.reranker or "none").strip().lower()
    diagnostics: dict[str, Any] = {
        "reranker": resolved,
        "rerank_applied": False,
        "rerank_fallback": None,
        "rerank_top_n": 0,
    }
    if resolved == "none" or len(candidates) < 2:
        return candidates, diagnostics

    top_n = min(settings.rerank_top_n, len(candidates))
    head = candidates[:top_n]
    tail = candidates[top_n:]
    diagnostics["rerank_top_n"] = top_n

    scores: list[float] | None = None
    used = resolved
    if resolved == "cross_encoder":
        try:
            scores = cross_encoder_scores(query, head)
        except Exception as exc:
            logger.warning("rag_rerank_cross_encoder_failed error=%s", type(exc).__name__)
            diagnostics["rerank_fallback"] = "heuristic"
            used = "heuristic"
            scores = heuristic_scores(query, head)
    elif resolved == "heuristic":
        scores = heuristic_scores(query, head)
    else:
        return candidates, diagnostics

    enriched: list[dict[str, Any]] = []
    for row, score in zip(head, scores):
        item = dict(row)
        if "retrieval_score" not in item:
            item["retrieval_score"] = float(item.get("score") or 0.0)
        item["rerank_score"] = float(score)
        item["score"] = float(score)
        enriched.append(item)
    enriched.sort(key=lambda row: (-float(row.get("rerank_score") or 0.0), -float(row.get("retrieval_score") or 0.0), row["chunk_id"]))
    diagnostics["reranker"] = used
    diagnostics["rerank_applied"] = True
    return enriched + tail, diagnostics


def reset_cross_encoder_for_tests() -> None:
    global _cross_encoder, _cross_encoder_error
    _cross_encoder = None
    _cross_encoder_error = None
