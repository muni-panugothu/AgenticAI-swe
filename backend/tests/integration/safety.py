"""Safe gates for Batch C live integration tests.

Never prints URIs, credentials, or API keys. Rejects production database
names. Live paid provider suites require explicit opt-in flags.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


def _truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


_FORBIDDEN_DB_NAMES = frozenset(
    {
        "agentic_ai_memory",
        "production",
        "prod",
        "main",
        "primary",
    }
)

_SAFE_DB_SUFFIX_RE = re.compile(
    r"(?i)(^|_)(test|tests|integration|acl_test|batch_c|ci)(_|$)"
)


@dataclass(frozen=True)
class GateResult:
    enabled: bool
    reason: str
    database: str | None = None
    namespace: str | None = None


def production_memory_database() -> str:
    return (os.getenv("MEMORY_DATABASE") or "agentic_ai_memory").strip()


def is_forbidden_database(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return True
    if n in _FORBIDDEN_DB_NAMES:
        return True
    if n == production_memory_database():
        return True
    return False


def looks_like_disposable_database(name: str) -> bool:
    n = (name or "").strip()
    if is_forbidden_database(n):
        return False
    return bool(_SAFE_DB_SUFFIX_RE.search(n))


def mongo_memory_gate() -> GateResult:
    if not _truthy("BATCH_C_MONGO_INTEGRATION") and not _truthy(
        "MEMORY_PERSISTENCE_INTEGRATION"
    ):
        return GateResult(
            False,
            "NOT RUN: set BATCH_C_MONGO_INTEGRATION=1 (or MEMORY_PERSISTENCE_INTEGRATION=1) "
            "and MEMORY_TEST_DATABASE to a disposable DB",
        )
    db = (os.getenv("MEMORY_TEST_DATABASE") or "").strip()
    if not db:
        return GateResult(False, "NOT RUN: MEMORY_TEST_DATABASE is required")
    if is_forbidden_database(db):
        return GateResult(
            False,
            f"NOT RUN: MEMORY_TEST_DATABASE={db!r} is forbidden / production-like",
        )
    if not looks_like_disposable_database(db):
        return GateResult(
            False,
            f"NOT RUN: MEMORY_TEST_DATABASE={db!r} must look disposable "
            "(name should include test/integration/batch_c)",
        )
    if not (os.getenv("MONGO_URI") or "").strip():
        return GateResult(False, "NOT RUN: MONGO_URI is not configured")
    return GateResult(True, "ok", database=db)


def mongo_rag_gate() -> GateResult:
    if not _truthy("BATCH_C_RAG_INTEGRATION") and not _truthy("RAG_ACL_INTEGRATION"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_C_RAG_INTEGRATION=1 (or RAG_ACL_INTEGRATION=1) "
            "and RAG_TEST_DATABASE to a disposable DB",
        )
    db = (os.getenv("RAG_TEST_DATABASE") or "").strip()
    if not db:
        return GateResult(False, "NOT RUN: RAG_TEST_DATABASE is required")
    if is_forbidden_database(db):
        return GateResult(
            False,
            f"NOT RUN: RAG_TEST_DATABASE={db!r} is forbidden / production-like",
        )
    if not looks_like_disposable_database(db):
        return GateResult(
            False,
            f"NOT RUN: RAG_TEST_DATABASE={db!r} must look disposable",
        )
    if not (os.getenv("MONGO_URI") or "").strip():
        return GateResult(False, "NOT RUN: MONGO_URI is not configured")
    return GateResult(True, "ok", database=db)


def redis_gate() -> GateResult:
    if not _truthy("BATCH_C_REDIS_INTEGRATION"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_C_REDIS_INTEGRATION=1 and BATCH_C_REDIS_NAMESPACE "
            "to an isolated prefix (e.g. agent:batchc:test:)",
        )
    ns = (os.getenv("BATCH_C_REDIS_NAMESPACE") or "").strip()
    if not ns:
        return GateResult(False, "NOT RUN: BATCH_C_REDIS_NAMESPACE is required")
    if not ns.startswith("agent:"):
        return GateResult(
            False,
            "NOT RUN: BATCH_C_REDIS_NAMESPACE must start with 'agent:'",
        )
    if "batchc" not in ns.lower() and "test" not in ns.lower():
        return GateResult(
            False,
            "NOT RUN: BATCH_C_REDIS_NAMESPACE must include 'batchc' or 'test'",
        )
    # Upstash Redis.from_env() needs these; do not print values.
    if not (os.getenv("UPSTASH_REDIS_REST_URL") or "").strip():
        return GateResult(False, "NOT RUN: UPSTASH_REDIS_REST_URL is not configured")
    if not (os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip():
        return GateResult(False, "NOT RUN: UPSTASH_REDIS_REST_TOKEN is not configured")
    return GateResult(True, "ok", namespace=ns)


def live_embedding_gate() -> GateResult:
    if not _truthy("BATCH_C_LIVE_EMBEDDINGS"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_C_LIVE_EMBEDDINGS=1 to allow Gemini embedding calls",
        )
    if not (
        (os.getenv("GOOGLE_API_KEY") or "").strip()
        or (os.getenv("GEMINI_API_KEY") or "").strip()
    ):
        return GateResult(False, "NOT RUN: GOOGLE_API_KEY/GEMINI_API_KEY missing")
    return GateResult(True, "ok")


def live_groq_gate() -> GateResult:
    if not _truthy("BATCH_C_LIVE_GROQ"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_C_LIVE_GROQ=1 to allow paid Groq calls",
        )
    if not (os.getenv("GROQ_API_KEY") or "").strip():
        return GateResult(False, "NOT RUN: GROQ_API_KEY missing")
    return GateResult(True, "ok")


def live_tavily_gate() -> GateResult:
    if not _truthy("BATCH_C_LIVE_TAVILY"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_C_LIVE_TAVILY=1 to allow paid Tavily calls",
        )
    if not (os.getenv("TAVILY_API_KEY") or "").strip():
        return GateResult(False, "NOT RUN: TAVILY_API_KEY missing")
    return GateResult(True, "ok")
