"""General-purpose assistant evaluation framework (Phase A).

Fixtures under ``tests/fixtures/assistant_eval_v1`` are evaluation-only.
They must never be wired into production chatbot behavior.
"""

from app.eval.assistant_eval.runner import run_evaluation
from app.eval.assistant_eval.schema import validate_dataset

__all__ = ["run_evaluation", "validate_dataset"]
