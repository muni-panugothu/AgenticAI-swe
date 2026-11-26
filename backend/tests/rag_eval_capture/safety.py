"""Disposable-database and opt-in gates for RAG eval capture."""

from __future__ import annotations

import os
from types import ModuleType
from typing import Any


FORBIDDEN_DB_NAMES = frozenset(
    {
        "agentic_ai_memory",
        "production",
        "prod",
    }
)

# Exact disposable names that may contain a forbidden substring (e.g. agentic_ai_memory)
# but are approved for M0 eval capture reuse. Not env-configurable.
ALLOWED_EVAL_DATABASES = frozenset(
    {
        "agentic_ai_memory_rag_acl_test",
    }
)

OPT_IN_ENV = "RAG_EVAL_CAPTURE"
TARGET_DB_ENV = "RAG_EVAL_DATABASE"


class EvalCaptureSafetyError(RuntimeError):
    """Raised when capture/seed safety prerequisites are not met."""


def _normalized(name: str | None) -> str:
    return (name or "").strip()


def forbidden_identifiers() -> set[str]:
    """Production-like names that must never be capture targets."""
    forbidden = set(FORBIDDEN_DB_NAMES)
    memory_db = _normalized(os.getenv("MEMORY_DATABASE"))
    if memory_db:
        forbidden.add(memory_db)
    # Literal token from requirements / ACL suite docs.
    forbidden.add("MEMORY_DATABASE")
    return {item.casefold() for item in forbidden if item}


def is_forbidden_database_name(name: str) -> bool:
    candidate = _normalized(name)
    if not candidate:
        return True
    lowered = candidate.casefold()
    forbidden = forbidden_identifiers()
    # Exact matches always win (includes configured MEMORY_DATABASE and prod tokens).
    if lowered in forbidden:
        return True
    # Deliberate M0 reuse of the ACL disposable DB — exact allowlist only.
    allowed = {item.casefold() for item in ALLOWED_EVAL_DATABASES}
    if lowered in allowed:
        return False
    for token in forbidden:
        if token in lowered:
            return True
    return False


def assert_eval_capture_enabled(*, environ: dict[str, str] | None = None) -> None:
    env = environ if environ is not None else os.environ
    raw = _normalized(env.get(OPT_IN_ENV))
    if raw.lower() not in {"1", "true", "yes"}:
        raise EvalCaptureSafetyError(
            f"{OPT_IN_ENV}=1 (or true/yes) is required before RAG eval capture/seed"
        )


def assert_safe_eval_database(
    database_name: str | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> str:
    """Return the approved evaluation database name or raise."""
    env = environ if environ is not None else os.environ
    assert_eval_capture_enabled(environ=env)
    configured = _normalized(database_name if database_name is not None else env.get(TARGET_DB_ENV))
    if not configured:
        raise EvalCaptureSafetyError(
            f"{TARGET_DB_ENV} must be set to an explicit disposable database name"
        )
    if is_forbidden_database_name(configured):
        raise EvalCaptureSafetyError(
            f"Refusing RAG eval database {configured!r}: matches forbidden production identifier"
        )
    # Refuse silent mismatch if caller passes a name different from env when env is set.
    env_target = _normalized(env.get(TARGET_DB_ENV))
    if env_target and database_name is not None and configured != env_target:
        raise EvalCaptureSafetyError(
            f"Provided database {configured!r} does not match {TARGET_DB_ENV}={env_target!r}"
        )
    return configured


def assert_database_is_approved_target(
    active_database_name: str | None,
    *,
    approved_name: str,
) -> None:
    if active_database_name is None or not _normalized(str(active_database_name)):
        raise EvalCaptureSafetyError(
            "Active database identity cannot be verified; refusing to proceed"
        )
    if _normalized(str(active_database_name)) != _normalized(approved_name):
        raise EvalCaptureSafetyError(
            f"Active database {active_database_name!r} is not the approved eval database "
            f"{approved_name!r}; refusing to proceed"
        )


def require_collection_database_name(collection: Any, *, approved_name: str) -> str:
    """Fail closed unless an injected collection targets the approved eval DB."""
    database = getattr(collection, "database", None)
    db_name = getattr(database, "name", None) if database is not None else None
    if db_name is None or not _normalized(str(db_name)):
        raise EvalCaptureSafetyError(
            "Injected collection does not expose a verifiable database name; "
            "refusing to continue"
        )
    assert_database_is_approved_target(str(db_name), approved_name=approved_name)
    return str(db_name)


def require_marker(capture_run_id: str) -> str:
    marker = _normalized(capture_run_id)
    if not marker or len(marker) < 8:
        raise EvalCaptureSafetyError("capture_run_id must be at least 8 non-empty characters")
    if any(ch.isspace() for ch in marker):
        raise EvalCaptureSafetyError("capture_run_id must not contain whitespace")
    return marker


class eval_database_redirect:
    """Redirect ``memory_service.database`` to the approved eval DB for a full capture window.

    Restores the original pointer in ``__exit__`` even when operations raise.
    Does not alter normal application database initialization outside this context.
    """

    def __init__(
        self,
        *,
        database_name: str | None = None,
        memory_service: ModuleType | Any | None = None,
        settings: Any | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self._database_name = database_name
        self._memory_service = memory_service
        self._settings = settings
        self._environ = environ
        self.approved_db: str | None = None
        self.database: Any | None = None
        self.collection: Any | None = None
        self._original_database: Any | None = None
        self._entered = False

    def __enter__(self) -> "eval_database_redirect":
        self.approved_db = assert_safe_eval_database(
            self._database_name,
            environ=self._environ,
        )
        if self._settings is None:
            from app.services.rag_config import settings as rag_settings

            self._settings = rag_settings
        if self._memory_service is None:
            from app.services import memory_service as memory_service_mod

            self._memory_service = memory_service_mod

        self._original_database = self._memory_service.database
        test_db = self._memory_service.client[self.approved_db]
        self._memory_service.database = test_db
        self._entered = True
        assert_database_is_approved_target(getattr(test_db, "name", None), approved_name=self.approved_db)
        # Re-check the active module pointer, not only the local handle.
        self.assert_still_active()
        self.database = test_db
        self.collection = test_db[self._settings.collection_name]
        require_collection_database_name(self.collection, approved_name=self.approved_db)
        return self

    def assert_still_active(self) -> None:
        if not self._entered or self.approved_db is None or self._memory_service is None:
            raise EvalCaptureSafetyError("Evaluation database redirect is not active")
        active = getattr(self._memory_service.database, "name", None)
        assert_database_is_approved_target(active, approved_name=self.approved_db)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._entered and self._memory_service is not None:
            self._memory_service.database = self._original_database
        self._entered = False
        return None
