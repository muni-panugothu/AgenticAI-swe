"""Test-only RAG golden capture tooling (Baseline A/B).

Not part of the production application. Requires explicit opt-in env gates
before any disposable Atlas access. Offline unit tests must mock DB/embeddings.
"""

from __future__ import annotations

__all__ = [
    "CORPUS_DIR",
    "FORBIDDEN_DB_NAMES",
    "assert_eval_capture_enabled",
    "assert_safe_eval_database",
    "eval_database_redirect",
    "load_corpus_manifest",
    "resolve_case_scope",
    "validate_corpus_against_golden",
]

from tests.rag_eval_capture.corpus import CORPUS_DIR, load_corpus_manifest, validate_corpus_against_golden
from tests.rag_eval_capture.case_scope import resolve_case_scope
from tests.rag_eval_capture.safety import (
    FORBIDDEN_DB_NAMES,
    assert_eval_capture_enabled,
    assert_safe_eval_database,
    eval_database_redirect,
)
