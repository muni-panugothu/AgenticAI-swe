"""Load versioned assistant evaluation datasets from fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.eval.assistant_eval.schema import validate_dataset

DEFAULT_FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "assistant_eval_v1"
)


def default_dataset_path() -> Path:
    return DEFAULT_FIXTURE_DIR / "dataset.json"


def load_dataset(path: str | Path | None = None) -> dict[str, Any]:
    dataset_path = Path(path) if path else default_dataset_path()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Assistant eval dataset not found: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    validate_dataset(raw)
    return raw


def category_counts(dataset: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in dataset.get("cases") or []:
        cat = str(case.get("category") or "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items()))
