"""Shared backend test process cleanup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_llm_provider_cache():
    """Topic 42b — avoid cross-test provider singleton pollution."""
    try:
        from app.services.llm_provider.factory import reset_llm_provider_cache

        reset_llm_provider_cache()
    except Exception:
        pass
    yield
    try:
        from app.services.llm_provider.factory import reset_llm_provider_cache

        reset_llm_provider_cache()
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    """Close process-wide clients before interpreter teardown begins."""
    del session, exitstatus
    from app.services.resource_lifecycle import shutdown_application_resources

    shutdown_application_resources()
