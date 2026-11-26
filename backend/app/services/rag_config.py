"""Runtime configuration for document RAG.

RAG deliberately has its own embedding dimensionality and collection/index
settings. Personal-memory vectors continue using their existing configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes"}


def _chunking_strategy(name: str, default: str = "char") -> str:
    value = (os.getenv(name) or default).strip().lower()
    if value not in {"char", "markdown"}:
        return default
    return value


def _lexical_backend(name: str, default: str = "overlap") -> str:
    value = (os.getenv(name) or default).strip().lower()
    if value not in {"overlap", "bm25"}:
        return default
    return value


def _reranker_mode() -> str:
    """Resolve reranker: RAG_RERANKER wins; else legacy RAG_ENABLE_RERANKING."""
    explicit = (os.getenv("RAG_RERANKER") or "").strip().lower()
    if explicit in {"none", "heuristic", "cross_encoder"}:
        return explicit
    if _bool("RAG_ENABLE_RERANKING", False):
        return "heuristic"
    return "none"


@dataclass(frozen=True)
class RagSettings:
    collection_name: str = os.getenv("RAG_CHUNKS_COLLECTION", "rag_chunks")
    vector_index: str = os.getenv("RAG_VECTOR_INDEX", "vector_index")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dimensions: int = int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "768"))
    # Must match the Atlas vectorSearch field similarity (verified at runtime).
    vector_similarity: str = os.getenv("RAG_VECTOR_SIMILARITY", "cosine")
    max_upload_bytes: int = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
    upload_rate_limit_count: int = int(
        os.getenv("RAG_UPLOAD_RATE_LIMIT_COUNT", "10")
    )
    upload_rate_limit_window_seconds: int = int(
        os.getenv("RAG_UPLOAD_RATE_LIMIT_WINDOW_SECONDS", "60")
    )
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE_CHARS", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "160"))
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    candidate_limit: int = int(os.getenv("RAG_CANDIDATE_LIMIT", "80"))
    # Atlas $vectorSearch numCandidates (capped; never exceeds hard ceiling).
    num_candidates: int = int(os.getenv("RAG_NUM_CANDIDATES", "0"))  # 0 => auto
    max_chunks_per_document: int = int(os.getenv("RAG_MAX_CHUNKS_PER_DOCUMENT", "256"))
    lexical_scan_limit: int = int(os.getenv("RAG_LEXICAL_SCAN_LIMIT", "2000"))
    context_budget_chars: int = int(os.getenv("RAG_CONTEXT_BUDGET_CHARS", "12000"))
    context_token_budget: int = int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", "3000"))
    context_max_chunks: int = int(os.getenv("RAG_CONTEXT_MAX_CHUNKS", "8"))
    min_vector_score: float = float(os.getenv("RAG_MIN_VECTOR_SCORE", "0.25"))
    # Legacy boolean kept for older env files; prefer RAG_RERANKER.
    enable_reranking: bool = _bool("RAG_ENABLE_RERANKING", False)
    reranker: str = _reranker_mode()
    rerank_model: str = os.getenv(
        "RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_top_n: int = int(os.getenv("RAG_RERANK_TOP_N", "20"))
    rerank_timeout_seconds: float = float(os.getenv("RAG_RERANK_TIMEOUT_SECONDS", "8"))
    enable_query_decomposition: bool = _bool("RAG_ENABLE_QUERY_DECOMPOSITION", True)
    enable_conversation_rewrite: bool = _bool("RAG_ENABLE_CONVERSATION_REWRITE", False)
    enable_multi_query: bool = _bool("RAG_ENABLE_MULTI_QUERY", False)
    max_rewritten_queries: int = int(os.getenv("RAG_MAX_REWRITTEN_QUERIES", "3"))
    enable_hyde: bool = _bool("RAG_ENABLE_HYDE", False)
    # Lexical / hybrid (Topic 28)
    lexical_backend: str = _lexical_backend("RAG_LEXICAL_BACKEND", "overlap")
    search_index: str = os.getenv("RAG_SEARCH_INDEX", "rag_text_index")
    hybrid_vector_weight: float = float(os.getenv("RAG_HYBRID_VECTOR_WEIGHT", "1.0"))
    hybrid_lexical_weight: float = float(os.getenv("RAG_HYBRID_LEXICAL_WEIGHT", "1.0"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    # Context compression (Topic 31)
    enable_semantic_compression: bool = _bool("RAG_ENABLE_SEMANTIC_COMPRESSION", False)
    compression_max_output_tokens: int = int(
        os.getenv("RAG_COMPRESSION_MAX_OUTPUT_TOKENS", "512")
    )
    near_duplicate_jaccard: float = float(os.getenv("RAG_NEAR_DUP_JACCARD", "0.92"))
    # Batch 1 — document intelligence
    enable_pdf: bool = _bool("RAG_ENABLE_PDF", True)
    enable_docx: bool = _bool("RAG_ENABLE_DOCX", True)
    enable_html: bool = _bool("RAG_ENABLE_HTML", True)
    enable_csv: bool = _bool("RAG_ENABLE_CSV", True)
    chunking_strategy: str = _chunking_strategy("RAG_CHUNKING_STRATEGY", "char")
    embedding_cache: bool = _bool("RAG_EMBEDDING_CACHE", True)
    embedding_cache_ttl_seconds: int = int(
        os.getenv("RAG_EMBEDDING_CACHE_TTL_SECONDS", str(7 * 24 * 3600))
    )
    embedding_version: str = os.getenv("RAG_EMBEDDING_VERSION", "rag-embed-v1")
    chunking_version_char: str = "char-window-v1"
    chunking_version_markdown: str = "markdown-structure-v1"
    parser_version_plain: str = "plain-text-v1"
    parser_version_pdf: str = "pypdf-v1"
    parser_version_docx: str = "python-docx-v1"
    parser_version_html: str = "bs4-html-v1"
    parser_version_csv: str = "csv-stdlib-v1"
    # Hard ceiling for Atlas vector numCandidates (Atlas allows up to ~10000).
    num_candidates_hard_max: int = 800


settings = RagSettings()


def resolved_num_candidates() -> int:
    """Validated Atlas numCandidates: auto = min(candidate_limit*10, hard max)."""
    if settings.num_candidates > 0:
        return min(settings.num_candidates, settings.num_candidates_hard_max)
    return min(max(settings.candidate_limit * 10, settings.candidate_limit), settings.num_candidates_hard_max)


def validate_settings() -> None:
    if settings.embedding_dimensions < 1:
        raise ValueError("RAG_EMBEDDING_DIMENSIONS must be positive")
    if not str(settings.vector_similarity or "").strip():
        raise ValueError("RAG_VECTOR_SIMILARITY must be non-empty")
    if settings.chunk_size < 1 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("RAG chunk size/overlap configuration is invalid")
    if settings.max_upload_bytes < 1:
        raise ValueError("RAG_MAX_UPLOAD_BYTES must be positive")
    if not 1 <= settings.top_k <= settings.candidate_limit <= 500:
        raise ValueError("RAG top-k/candidate limits are invalid")
    if settings.lexical_scan_limit < 1 or settings.context_budget_chars < 1 or settings.max_chunks_per_document < 1:
        raise ValueError("RAG scan/context limits must be positive")
    if not 0 <= settings.min_vector_score <= 1:
        raise ValueError("RAG_MIN_VECTOR_SCORE must be between 0 and 1")
    if settings.num_candidates < 0:
        raise ValueError("RAG_NUM_CANDIDATES must be >= 0")
    if settings.num_candidates > 0 and settings.num_candidates < settings.candidate_limit:
        raise ValueError("RAG_NUM_CANDIDATES must be >= RAG_CANDIDATE_LIMIT when set")
    if settings.num_candidates > settings.num_candidates_hard_max:
        raise ValueError("RAG_NUM_CANDIDATES exceeds hard maximum")
    if settings.lexical_backend not in {"overlap", "bm25"}:
        raise ValueError("RAG_LEXICAL_BACKEND must be 'overlap' or 'bm25'")
    if not str(settings.search_index or "").strip():
        raise ValueError("RAG_SEARCH_INDEX must be non-empty")
    if settings.hybrid_vector_weight < 0 or settings.hybrid_lexical_weight < 0:
        raise ValueError("RAG hybrid weights must be >= 0")
    if settings.hybrid_vector_weight + settings.hybrid_lexical_weight <= 0:
        raise ValueError("At least one RAG hybrid weight must be positive")
    if settings.rrf_k < 1:
        raise ValueError("RAG_RRF_K must be positive")
    if settings.reranker not in {"none", "heuristic", "cross_encoder"}:
        raise ValueError("RAG_RERANKER must be none, heuristic, or cross_encoder")
    if settings.rerank_top_n < 1 or settings.rerank_top_n > settings.candidate_limit:
        raise ValueError("RAG_RERANK_TOP_N must be between 1 and RAG_CANDIDATE_LIMIT")
    if settings.rerank_timeout_seconds <= 0:
        raise ValueError("RAG_RERANK_TIMEOUT_SECONDS must be positive")
    if settings.max_rewritten_queries < 1 or settings.max_rewritten_queries > 8:
        raise ValueError("RAG_MAX_REWRITTEN_QUERIES must be between 1 and 8")
    if settings.context_token_budget < 1:
        raise ValueError("RAG_CONTEXT_TOKEN_BUDGET must be positive")
    if settings.context_max_chunks < 1:
        raise ValueError("RAG_CONTEXT_MAX_CHUNKS must be positive")
    if settings.compression_max_output_tokens < 1:
        raise ValueError("RAG_COMPRESSION_MAX_OUTPUT_TOKENS must be positive")
    if not 0.5 <= settings.near_duplicate_jaccard <= 1.0:
        raise ValueError("RAG_NEAR_DUP_JACCARD must be between 0.5 and 1.0")

    if settings.upload_rate_limit_count < 1:
        raise ValueError("RAG_UPLOAD_RATE_LIMIT_COUNT must be positive")
    if settings.upload_rate_limit_window_seconds < 1:
        raise ValueError("RAG_UPLOAD_RATE_LIMIT_WINDOW_SECONDS must be positive")
    if settings.embedding_cache_ttl_seconds < 1:
        raise ValueError("RAG_EMBEDDING_CACHE_TTL_SECONDS must be positive")
    if settings.chunking_strategy not in {"char", "markdown"}:
        raise ValueError("RAG_CHUNKING_STRATEGY must be 'char' or 'markdown'")
