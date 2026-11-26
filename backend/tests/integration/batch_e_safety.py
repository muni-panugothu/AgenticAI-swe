"""Batch E safety gates — disposable Atlas/Redis + budget-capped provider canaries.

Never prints URIs, tokens, or API keys. Live suites require explicit BATCH_E_*
opt-in (Batch C flags alone are not enough for Batch E reports).
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from tests.integration.safety import (
    GateResult,
    is_forbidden_database,
    looks_like_disposable_database as _batch_c_disposable,
    production_memory_database,
)


def _truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


_SAFE_DB_RE = re.compile(
    r"(?i)(^|_)(test|tests|integration|acl_test|batch_c|batch_e|ci)(_|$)"
)


def looks_like_disposable_database(name: str) -> bool:
    n = (name or "").strip()
    if is_forbidden_database(n):
        return False
    if _batch_c_disposable(n):
        return True
    return bool(_SAFE_DB_RE.search(n))


def new_batch_e_run_id() -> str:
    return f"be_{int(time.time())}_{uuid.uuid4().hex[:10]}"


@dataclass
class ProviderBudget:
    """Hard stop for paid provider canaries."""

    name: str
    max_requests: int
    _count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def consume(self, n: int = 1) -> None:
        with self._lock:
            if self._count + n > self.max_requests:
                raise RuntimeError(
                    f"BATCH_E budget exceeded for {self.name}: "
                    f"{self._count + n}/{self.max_requests}"
                )
            self._count += n

    @property
    def used(self) -> int:
        with self._lock:
            return self._count


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def mongo_gate() -> GateResult:
    if not _truthy("BATCH_E_MONGO_INTEGRATION"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_E_MONGO_INTEGRATION=1 and MEMORY_TEST_DATABASE "
            "to a disposable DB (name must include test/integration/batch_e)",
        )
    db = (os.getenv("MEMORY_TEST_DATABASE") or "").strip()
    if not db:
        return GateResult(False, "NOT RUN: MEMORY_TEST_DATABASE is required")
    if is_forbidden_database(db):
        return GateResult(
            False,
            f"NOT RUN: MEMORY_TEST_DATABASE={db!r} is forbidden / production-like "
            f"(production MEMORY_DATABASE={production_memory_database()!r})",
        )
    if not looks_like_disposable_database(db):
        return GateResult(
            False,
            f"NOT RUN: MEMORY_TEST_DATABASE={db!r} must look disposable",
        )
    if not (os.getenv("MONGO_URI") or "").strip():
        return GateResult(False, "NOT RUN: MONGO_URI is not configured")
    return GateResult(True, "ok", database=db)


def rag_gate() -> GateResult:
    if not _truthy("BATCH_E_RAG_INTEGRATION"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_E_RAG_INTEGRATION=1 and RAG_TEST_DATABASE "
            "(or MEMORY_TEST_DATABASE) to a disposable DB",
        )
    db = (
        os.getenv("RAG_TEST_DATABASE")
        or os.getenv("MEMORY_TEST_DATABASE")
        or ""
    ).strip()
    if not db:
        return GateResult(
            False,
            "NOT RUN: RAG_TEST_DATABASE or MEMORY_TEST_DATABASE is required",
        )
    if is_forbidden_database(db):
        return GateResult(
            False,
            f"NOT RUN: RAG test DB {db!r} is forbidden / production-like",
        )
    if not looks_like_disposable_database(db):
        return GateResult(
            False,
            f"NOT RUN: RAG test DB {db!r} must look disposable",
        )
    if not (os.getenv("MONGO_URI") or "").strip():
        return GateResult(False, "NOT RUN: MONGO_URI is not configured")
    return GateResult(True, "ok", database=db)


def redis_gate() -> GateResult:
    if not _truthy("BATCH_E_REDIS_INTEGRATION"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_E_REDIS_INTEGRATION=1 and "
            "BATCH_E_REDIS_NAMESPACE (e.g. agent:batche:test:)",
        )
    ns = (os.getenv("BATCH_E_REDIS_NAMESPACE") or "").strip()
    if not ns:
        return GateResult(False, "NOT RUN: BATCH_E_REDIS_NAMESPACE is required")
    if not ns.startswith("agent:"):
        return GateResult(
            False,
            "NOT RUN: BATCH_E_REDIS_NAMESPACE must start with 'agent:'",
        )
    low = ns.lower()
    if "batche" not in low and "batch_e" not in low and "test" not in low:
        return GateResult(
            False,
            "NOT RUN: BATCH_E_REDIS_NAMESPACE must include batche/batch_e/test",
        )
    if "prod" in low or "production" in low:
        return GateResult(
            False,
            "NOT RUN: BATCH_E_REDIS_NAMESPACE looks production-like",
        )
    if not (os.getenv("UPSTASH_REDIS_REST_URL") or "").strip():
        return GateResult(False, "NOT RUN: UPSTASH_REDIS_REST_URL is not configured")
    if not (os.getenv("UPSTASH_REDIS_REST_TOKEN") or "").strip():
        return GateResult(False, "NOT RUN: UPSTASH_REDIS_REST_TOKEN is not configured")
    return GateResult(True, "ok", namespace=ns)


def live_gemini_gate() -> GateResult:
    if not _truthy("BATCH_E_LIVE_GEMINI"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_E_LIVE_GEMINI=1 to allow Gemini embedding canaries",
        )
    if not (
        (os.getenv("GOOGLE_API_KEY") or "").strip()
        or (os.getenv("GEMINI_API_KEY") or "").strip()
    ):
        return GateResult(False, "NOT RUN: GOOGLE_API_KEY/GEMINI_API_KEY missing")
    return GateResult(True, "ok")


def live_groq_gate() -> GateResult:
    if not _truthy("BATCH_E_LIVE_GROQ"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_E_LIVE_GROQ=1 to allow paid Groq canaries",
        )
    if not (os.getenv("GROQ_API_KEY") or "").strip():
        return GateResult(False, "NOT RUN: GROQ_API_KEY missing")
    return GateResult(True, "ok")


def live_tavily_gate() -> GateResult:
    if not _truthy("BATCH_E_LIVE_TAVILY"):
        return GateResult(
            False,
            "NOT RUN: set BATCH_E_LIVE_TAVILY=1 to allow paid Tavily canaries",
        )
    if not (os.getenv("TAVILY_API_KEY") or "").strip():
        return GateResult(False, "NOT RUN: TAVILY_API_KEY missing")
    return GateResult(True, "ok")


def gemini_budget() -> ProviderBudget:
    return ProviderBudget(
        "gemini",
        max_requests=_env_int("BATCH_E_GEMINI_MAX_REQUESTS", 3),
    )


def groq_budget() -> ProviderBudget:
    return ProviderBudget(
        "groq",
        max_requests=_env_int("BATCH_E_GROQ_MAX_REQUESTS", 2),
    )


def tavily_budget() -> ProviderBudget:
    return ProviderBudget(
        "tavily",
        max_requests=_env_int("BATCH_E_TAVILY_MAX_REQUESTS", 2),
    )
