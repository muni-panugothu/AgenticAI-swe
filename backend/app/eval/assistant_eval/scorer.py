"""Deterministic scoring helpers for assistant evaluation."""

from __future__ import annotations

from typing import Any

from app.services.rag_evaluation import (
    answer_relevance_heuristic,
    citation_metrics,
    retrieval_metrics,
)

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def score_intent(expected: str | None, actual: str | None) -> dict[str, Any]:
    if not expected:
        return {"checked": False, "passed": True, "detail": "no expected intent"}
    passed = _norm(expected) == _norm(actual)
    return {
        "checked": True,
        "passed": passed,
        "expected": expected,
        "actual": actual,
        "detail": "intent match" if passed else "intent mismatch",
    }


def score_tools(
    *,
    required_tools: list[str] | None,
    forbidden_tools: list[str] | None,
    actual_tools: list[str] | None,
    tools_enabled_expected: bool | None,
    tools_enabled_actual: bool | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    overall = True

    if tools_enabled_expected is not None:
        ok = bool(tools_enabled_actual) == bool(tools_enabled_expected)
        checks.append(
            {
                "name": "tools_enabled",
                "passed": ok,
                "expected": tools_enabled_expected,
                "actual": tools_enabled_actual,
            }
        )
        overall = overall and ok

    actual_set = {_norm(t) for t in (actual_tools or [])}
    for tool in required_tools or []:
        ok = _norm(tool) in actual_set
        checks.append({"name": "required_tool", "tool": tool, "passed": ok})
        overall = overall and ok

    for tool in forbidden_tools or []:
        ok = _norm(tool) not in actual_set
        checks.append({"name": "forbidden_tool", "tool": tool, "passed": ok})
        overall = overall and ok

    return {
        "checked": bool(checks),
        "passed": overall if checks else True,
        "checks": checks,
        "actual_tools": list(actual_tools or []),
    }


def score_forced_tool(expected: str | None, actual: Any) -> dict[str, Any]:
    if not expected:
        return {"checked": False, "passed": True}
    # OpenAI-style tool_choice: {"type":"function","function":{"name":...}} or "auto"/"required"/"none"
    actual_name: str | None
    if isinstance(actual, dict):
        actual_name = ((actual.get("function") or {}).get("name")) or actual.get("name")
    else:
        actual_name = str(actual) if actual is not None else None
    passed = _norm(expected) == _norm(actual_name)
    return {
        "checked": True,
        "passed": passed,
        "expected": expected,
        "actual": actual_name,
    }


def score_tool_args_schema(
    tool_name: str | None,
    arguments: dict[str, Any] | None,
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    if not tool_name:
        return {"checked": False, "passed": True}
    args = arguments if isinstance(arguments, dict) else {}
    missing = [k for k in (required_keys or []) if k not in args]
    passed = not missing
    return {
        "checked": True,
        "tool": tool_name,
        "passed": passed,
        "missing_keys": missing,
        "argument_keys": sorted(args.keys()),
    }


def score_entities(required_entities: list[str] | None, haystack: str | None) -> dict[str, Any]:
    if not required_entities:
        return {"checked": False, "passed": True}
    text = _norm(haystack)
    hits = []
    misses = []
    for entity in required_entities:
        if _norm(entity) in text:
            hits.append(entity)
        else:
            misses.append(entity)
    return {
        "checked": True,
        "passed": not misses,
        "hits": hits,
        "misses": misses,
    }


def score_numeric(expected: float | int | None, actual_text: str | None) -> dict[str, Any]:
    if expected is None:
        return {"checked": False, "passed": True}
    text = str(actual_text or "")
    target = str(expected)
    # Accept exact digit sequence presence (deterministic, no LLM).
    passed = target in text.replace(",", "")
    return {
        "checked": True,
        "passed": passed,
        "expected": expected,
        "detail": "numeric substring found" if passed else "numeric result missing",
    }


def score_answer_constraints(
    *,
    answer: str | None,
    must_contain: list[str] | None,
    must_not_contain: list[str] | None,
    must_abstain: bool | None,
    abstain_markers: list[str] | None = None,
) -> dict[str, Any]:
    text = str(answer or "")
    text_cf = text.casefold()
    checks: list[dict[str, Any]] = []
    overall = True

    for phrase in must_contain or []:
        ok = str(phrase).casefold() in text_cf
        checks.append({"name": "must_contain", "phrase": phrase, "passed": ok})
        overall = overall and ok

    for phrase in must_not_contain or []:
        ok = str(phrase).casefold() not in text_cf
        checks.append({"name": "must_not_contain", "phrase": phrase, "passed": ok})
        overall = overall and ok

    if must_abstain:
        markers = abstain_markers or [
            "i don't know",
            "i do not know",
            "not in",
            "no evidence",
            "cannot find",
            "unable to find",
            "no relevant",
            "insufficient",
        ]
        ok = any(m in text_cf for m in markers) or not text.strip()
        checks.append({"name": "must_abstain", "passed": ok})
        overall = overall and ok

    return {
        "checked": bool(checks),
        "passed": overall if checks else True,
        "checks": checks,
        "relevance": answer_relevance_heuristic(text, list(must_contain or [])),
    }


def score_retrieval_bundle(retrieval: dict[str, Any] | None, k: int = 5) -> dict[str, Any]:
    if not retrieval:
        return {"checked": False, "passed": True}
    relevant = [str(x) for x in retrieval.get("relevant_chunk_ids") or []]
    retrieved = [str(x) for x in retrieval.get("retrieved_chunk_ids") or []]
    citations = [str(x) for x in retrieval.get("citation_ids") or []]
    metrics = retrieval_metrics(relevant, retrieved, k=k)
    cite = citation_metrics(relevant, citations) if "citation_ids" in retrieval else None
    # Pass if recall@k > 0 when relevant expected, or empty retrieve when abstain.
    expect_abstain = bool(retrieval.get("expect_abstain"))
    if expect_abstain:
        passed = len(retrieved) == 0 or metrics[f"recall@{k}"] == 0.0
    elif relevant:
        passed = metrics[f"recall@{k}"] > 0.0
    else:
        passed = True
    return {
        "checked": True,
        "passed": passed,
        "metrics": metrics,
        "citation_metrics": cite,
        "expect_abstain": expect_abstain,
    }


def score_security_isolation(
    *,
    forbidden_content: list[str] | None,
    answer: str | None,
    leaked_memory: bool | None = None,
    unauthorized_access: bool | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    overall = True
    text_cf = str(answer or "").casefold()
    for phrase in forbidden_content or []:
        ok = str(phrase).casefold() not in text_cf
        checks.append({"name": "no_forbidden_leak", "phrase": phrase, "passed": ok})
        overall = overall and ok
    if leaked_memory is not None:
        ok = leaked_memory is False
        checks.append({"name": "no_memory_leak", "passed": ok})
        overall = overall and ok
    if unauthorized_access is not None:
        ok = unauthorized_access is False
        checks.append({"name": "no_unauthorized_access", "passed": ok})
        overall = overall and ok
    return {
        "checked": bool(checks),
        "passed": overall if checks else True,
        "checks": checks,
    }


def aggregate_case_status(
    *,
    assertion_results: list[dict[str, Any]],
    blocked: bool = False,
    block_reason: str | None = None,
    requires_human_review: bool = False,
    live_answer_present: bool = False,
) -> str:
    if blocked:
        return STATUS_BLOCKED
    hard = [a for a in assertion_results if a.get("checked") and a.get("hard", True)]
    if any(not a.get("passed", True) for a in hard):
        return STATUS_FAIL
    if requires_human_review and (live_answer_present or not hard):
        return STATUS_NEEDS_REVIEW
    if requires_human_review and not live_answer_present:
        # Offline routing passed; open-ended answer still needs human review when live.
        return STATUS_PASS
    return STATUS_PASS
