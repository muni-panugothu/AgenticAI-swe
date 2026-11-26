"""Batch B E2E — write coverage report after the package finishes."""

from __future__ import annotations

from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    del exitstatus
    # Only emit when this package participated in the session.
    collected = [
        item
        for item in getattr(session, "items", [])
        if "tests/e2e" in str(getattr(item, "fspath", ""))
        or "tests\\e2e" in str(getattr(item, "fspath", ""))
    ]
    if not collected:
        return
    try:
        from tests.e2e.coverage_matrix import COVERAGE_MATRIX, render_coverage_markdown
        from tests.e2e import test_batch_b_functional_e2e as suite
    except Exception:
        return
    results = dict(getattr(suite, "_RESULTS", {}))
    for row in COVERAGE_MATRIX:
        results.setdefault(
            row["id"],
            "NOT_RUN" if row["id"].startswith("BB-LIVE") else "DEFINED",
        )
    out = Path(__file__).resolve().parents[2] / "reports" / "batch_b_e2e_coverage.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_coverage_markdown(results) + "\n", encoding="utf-8")
