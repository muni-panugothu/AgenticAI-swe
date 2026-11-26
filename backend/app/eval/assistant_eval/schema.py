"""Dataset schema validation for assistant_eval_v1."""

from __future__ import annotations

from typing import Any

DATASET_VERSION_PREFIX = "assistant_eval_v1"

REQUIRED_CASE_FIELDS = ("case_id", "category", "input")
ALLOWED_STATUS = frozenset({"approved", "draft"})

KNOWN_CATEGORIES = frozenset(
    {
        "general_knowledge",
        "current_information",
        "abbreviations",
        "typos_incomplete",
        "multilingual",
        "ambiguous",
        "conversation_followups",
        "mathematics",
        "code_generation",
        "code_explanation_debug",
        "writing_documentation",
        "rag_retrieval_abstention",
        "personal_memory_permissions",
        "tool_selection_execution",
        "multi_step_workflows",
        "prompt_injection_adversarial",
        "error_handling_uncertainty",
        "instruction_following",
    }
)


def validate_dataset(dataset: dict[str, Any]) -> None:
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be an object")
    version = dataset.get("version")
    if not isinstance(version, str) or not version.startswith("1."):
        raise ValueError("Dataset version must be a 1.x string")
    review = dataset.get("review")
    if not isinstance(review, dict) or review.get("status") not in ALLOWED_STATUS:
        raise ValueError("review.status must be 'approved' or 'draft'")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must include a non-empty cases list")

    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                raise ValueError(f"Case missing required field: {field}")
        cid = str(case["case_id"]).strip()
        if not cid:
            raise ValueError("case_id must be non-empty")
        if cid in seen:
            raise ValueError(f"Duplicate case_id: {cid}")
        seen.add(cid)

        category = str(case["category"])
        if category not in KNOWN_CATEGORIES:
            raise ValueError(f"Unknown category '{category}' for case {cid}")

        inp = case["input"]
        if not isinstance(inp, dict) or not str(inp.get("message") or "").strip():
            raise ValueError(f"Case {cid} input.message is required")
        if "history" in inp and not isinstance(inp["history"], list):
            raise ValueError(f"Case {cid} input.history must be a list")

        expect = case.get("expect") or {}
        if not isinstance(expect, dict):
            raise ValueError(f"Case {cid} expect must be an object")
        for list_field in (
            "required_tools",
            "forbidden_tools",
            "answer_must_contain",
            "answer_must_not_contain",
            "required_entities",
            "required_extracted_entities",
            "resolved_must_contain",
            "task_descriptions_must_contain",
            "task_categories_must_include",
            "required_dependencies",
            "execution_order_prefix",
            "decomposition_must_not_contain",
        ):
            if list_field in expect and not isinstance(expect[list_field], list):
                raise ValueError(f"Case {cid} expect.{list_field} must be a list")
        retrieval = expect.get("retrieval")
        if retrieval is not None:
            if not isinstance(retrieval, dict):
                raise ValueError(f"Case {cid} expect.retrieval must be an object")
            for key in ("relevant_chunk_ids", "retrieved_chunk_ids"):
                if key in retrieval and not isinstance(retrieval[key], list):
                    raise ValueError(f"Case {cid} expect.retrieval.{key} must be a list")
