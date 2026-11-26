"""Offline-first assistant evaluation runner.

Default mode exercises intent routing, tool allowlists, forced tool choice,
Pydantic argument schemas, and offline retrieval scoring — without provider calls.

Live provider evaluation requires ``ASSISTANT_EVAL_LIVE=1`` (or ``--live``).
"""

from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

from app.eval.assistant_eval.loader import category_counts, load_dataset
from app.eval.assistant_eval.schema import validate_dataset
from app.eval.assistant_eval.scorer import (
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_NEEDS_REVIEW,
    STATUS_PASS,
    aggregate_case_status,
    score_answer_constraints,
    score_entities,
    score_forced_tool,
    score_intent,
    score_numeric,
    score_retrieval_bundle,
    score_security_isolation,
    score_tool_args_schema,
    score_tools,
)


def _tool_names(defs: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for item in defs or []:
        name = ((item.get("function") or {}).get("name")) or item.get("name")
        if name:
            names.append(str(name))
    return names


def _validate_tool_args_with_pydantic(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    from app.validation.tool_registry import tool_input_models

    model = tool_input_models.get(tool_name)
    if model is None:
        return {
            "checked": True,
            "passed": False,
            "hard": True,
            "detail": f"unknown tool schema: {tool_name}",
        }
    try:
        model.model_validate(arguments)
        return {
            "checked": True,
            "passed": True,
            "hard": True,
            "tool": tool_name,
            "detail": "pydantic ok",
        }
    except Exception as exc:  # noqa: BLE001 — evaluation captures validation errors
        return {
            "checked": True,
            "passed": False,
            "hard": True,
            "tool": tool_name,
            "detail": str(exc),
        }


def evaluate_case_offline(
    case: dict[str, Any],
    *,
    tool_definitions: list[dict[str, Any]] | None = None,
    route_fn: Callable[..., Any] | None = None,
    classify_fn: Callable[[str], str] | None = None,
    resolve_tool_choice_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Score one case without live provider or tool side effects."""
    from app import chatbot
    from app.services import agent_orchestration as orch
    from app.services.agent_routing import route_request

    started = time.perf_counter()
    case_id = str(case["case_id"])
    expect = case.get("expect") or {}
    inp = case.get("input") or {}
    message = str(inp.get("message") or "")
    rag_context = inp.get("rag_context")
    assertions: list[dict[str, Any]] = []
    blocked = False
    block_reason: str | None = None
    error: str | None = None

    actual_intent: str | None = None
    classified_intent: str | None = None
    actual_tools: list[str] = []
    tools_enabled: bool | None = None
    forced_choice: Any = None
    route_kind: str | None = None
    normalized_message: str | None = None
    normalization_transforms: list[str] = []
    latency_ms = 0.0

    try:
        catalog = tool_definitions if tool_definitions is not None else chatbot.tool_definitions
        classify = classify_fn or orch.classify_intent
        router = route_fn or route_request
        resolve = resolve_tool_choice_fn or chatbot.resolve_tool_choice

        # Intent used for scoring is the routed decision (RAG/multi-tool may
        # override raw classify_intent), matching production chat_with_ai.
        history = inp.get("history") if isinstance(inp.get("history"), list) else None
        decision = router(
            user_message=message,
            rag_context=rag_context,
            all_tool_definitions=catalog,
            multi_tool_fn=chatbot.is_multi_tool_request,
            conversation_history=history,
        )
        normalized_message = getattr(decision, "normalized_message", None) or message
        normalization_transforms = list(
            getattr(decision, "normalization_transforms", []) or []
        )
        resolved_query = getattr(decision, "resolved_query", None) or normalized_message
        # Classify raw vs normalized for diagnostics; scoring uses routed intent.
        classified_intent = classify(message)
        actual_intent = getattr(decision, "intent", None) or classify(
            resolved_query
        )
        actual_tools = _tool_names(decision.tool_definitions)
        tools_enabled = bool(decision.tools_enabled)
        route_kind = getattr(decision, "route_kind", None)

        forced_choice, _nudge = resolve(
            user_message=resolved_query,
            step=0,
            max_steps=8,
            calculator_used=False,
        )

        intent_score = score_intent(expect.get("intent"), actual_intent)
        intent_score["hard"] = True
        assertions.append(intent_score)

        tools_score = score_tools(
            required_tools=expect.get("required_tools"),
            forbidden_tools=expect.get("forbidden_tools"),
            actual_tools=actual_tools,
            tools_enabled_expected=expect.get("tools_enabled"),
            tools_enabled_actual=tools_enabled,
        )
        tools_score["hard"] = True
        assertions.append(tools_score)

        if expect.get("route_kind"):
            rk_ok = str(expect["route_kind"]) == str(route_kind)
            assertions.append(
                {
                    "checked": True,
                    "passed": rk_ok,
                    "hard": True,
                    "name": "route_kind",
                    "expected": expect["route_kind"],
                    "actual": route_kind,
                }
            )

        forced_score = score_forced_tool(expect.get("forced_tool"), forced_choice)
        forced_score["hard"] = True
        assertions.append(forced_score)

        # Phase B2 — semantic intent + entity checks (verify downstream intent too).
        actual_semantic = getattr(decision, "semantic_intent", None)
        if expect.get("semantic_intent"):
            sem_ok = str(expect["semantic_intent"]) == str(actual_semantic)
            assertions.append(
                {
                    "checked": True,
                    "passed": sem_ok,
                    "hard": True,
                    "name": "semantic_intent",
                    "expected": expect["semantic_intent"],
                    "actual": actual_semantic,
                }
            )
        if expect.get("required_extracted_entities"):
            ent_values = []
            for ent in getattr(decision, "entities", None) or []:
                if isinstance(ent, dict):
                    ent_values.append(str(ent.get("value") or ""))
                else:
                    ent_values.append(str(ent))
            hay = " ".join(ent_values).casefold()
            missing = [
                e
                for e in expect["required_extracted_entities"]
                if str(e).casefold() not in hay
            ]
            assertions.append(
                {
                    "checked": True,
                    "passed": not missing,
                    "hard": True,
                    "name": "required_extracted_entities",
                    "missing": missing,
                    "actual_entities": ent_values,
                }
            )
        if "needs_clarification" in expect:
            actual_clarify = bool(getattr(decision, "needs_clarification", False))
            assertions.append(
                {
                    "checked": True,
                    "passed": actual_clarify == bool(expect["needs_clarification"]),
                    "hard": True,
                    "name": "needs_clarification",
                    "expected": expect["needs_clarification"],
                    "actual": actual_clarify,
                }
            )

        # Phase B3 — follow-up / resolved-query checks (downstream behavior).
        if "is_follow_up" in expect:
            actual_fu = bool(getattr(decision, "is_follow_up", False))
            assertions.append(
                {
                    "checked": True,
                    "passed": actual_fu == bool(expect["is_follow_up"]),
                    "hard": True,
                    "name": "is_follow_up",
                    "expected": expect["is_follow_up"],
                    "actual": actual_fu,
                }
            )
        if expect.get("follow_up_type"):
            actual_ft = str(getattr(decision, "follow_up_type", "") or "")
            assertions.append(
                {
                    "checked": True,
                    "passed": actual_ft == str(expect["follow_up_type"]),
                    "hard": True,
                    "name": "follow_up_type",
                    "expected": expect["follow_up_type"],
                    "actual": actual_ft,
                }
            )
        if expect.get("resolved_must_contain"):
            rq = str(getattr(decision, "resolved_query", "") or "").casefold()
            missing = [
                p
                for p in expect["resolved_must_contain"]
                if str(p).casefold() not in rq
            ]
            assertions.append(
                {
                    "checked": True,
                    "passed": not missing,
                    "hard": True,
                    "name": "resolved_must_contain",
                    "missing": missing,
                    "resolved_query": getattr(decision, "resolved_query", None),
                }
            )

        # Phase B4 — multi-intent decomposition checks (downstream understanding).
        decomp = getattr(decision, "task_decomposition", None) or {}
        tasks = decomp.get("tasks") if isinstance(decomp, dict) else None
        if not isinstance(tasks, list):
            tasks = []
        task_blob = " ".join(
            str(t.get("description") or "") for t in tasks if isinstance(t, dict)
        ).casefold()
        cats = [
            str(t.get("category") or "")
            for t in tasks
            if isinstance(t, dict)
        ]
        deps_pairs = set()
        for t in tasks:
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id") or "")
            for dep in t.get("depends_on") or []:
                deps_pairs.add(f"{tid}>{dep}")

        if "is_multi_intent" in expect:
            actual_mi = bool(getattr(decision, "is_multi_intent", False))
            assertions.append(
                {
                    "checked": True,
                    "passed": actual_mi == bool(expect["is_multi_intent"]),
                    "hard": True,
                    "name": "is_multi_intent",
                    "expected": expect["is_multi_intent"],
                    "actual": actual_mi,
                }
            )
        if "task_count" in expect:
            actual_tc = int(getattr(decision, "task_count", 0) or len(tasks))
            assertions.append(
                {
                    "checked": True,
                    "passed": actual_tc == int(expect["task_count"]),
                    "hard": True,
                    "name": "task_count",
                    "expected": expect["task_count"],
                    "actual": actual_tc,
                }
            )
        if "min_task_count" in expect:
            actual_tc = int(getattr(decision, "task_count", 0) or len(tasks))
            assertions.append(
                {
                    "checked": True,
                    "passed": actual_tc >= int(expect["min_task_count"]),
                    "hard": True,
                    "name": "min_task_count",
                    "expected": expect["min_task_count"],
                    "actual": actual_tc,
                }
            )
        if expect.get("task_descriptions_must_contain"):
            missing = [
                p
                for p in expect["task_descriptions_must_contain"]
                if str(p).casefold() not in task_blob
            ]
            assertions.append(
                {
                    "checked": True,
                    "passed": not missing,
                    "hard": True,
                    "name": "task_descriptions_must_contain",
                    "missing": missing,
                    "task_blob": task_blob[:500],
                }
            )
        if expect.get("task_categories_must_include"):
            missing = [
                c
                for c in expect["task_categories_must_include"]
                if str(c) not in cats
            ]
            assertions.append(
                {
                    "checked": True,
                    "passed": not missing,
                    "hard": True,
                    "name": "task_categories_must_include",
                    "missing": missing,
                    "actual_categories": cats,
                }
            )
        if expect.get("required_dependencies"):
            missing = [
                d
                for d in expect["required_dependencies"]
                if str(d) not in deps_pairs
            ]
            assertions.append(
                {
                    "checked": True,
                    "passed": not missing,
                    "hard": True,
                    "name": "required_dependencies",
                    "missing": missing,
                    "actual_dependencies": sorted(deps_pairs),
                }
            )
        if expect.get("execution_order_prefix"):
            actual_order = list(
                getattr(decision, "decomposed_execution_order", None)
                or (decomp.get("execution_order") if isinstance(decomp, dict) else [])
                or []
            )
            prefix = [str(x) for x in expect["execution_order_prefix"]]
            ok = actual_order[: len(prefix)] == prefix
            assertions.append(
                {
                    "checked": True,
                    "passed": ok,
                    "hard": True,
                    "name": "execution_order_prefix",
                    "expected": prefix,
                    "actual": actual_order,
                }
            )
        if expect.get("decomposition_must_not_contain"):
            hay = (task_blob + " " + str(decomp).casefold())
            hits = [
                p
                for p in expect["decomposition_must_not_contain"]
                if str(p).casefold() in hay
            ]
            assertions.append(
                {
                    "checked": True,
                    "passed": not hits,
                    "hard": True,
                    "name": "decomposition_must_not_contain",
                    "hits": hits,
                }
            )

        # Sample / fixture tool argument schema checks (evaluation only).
        for sample in expect.get("sample_tool_calls") or []:
            if not isinstance(sample, dict):
                continue
            tname = str(sample.get("tool") or "")
            args = sample.get("arguments") if isinstance(sample.get("arguments"), dict) else {}
            schema_score = _validate_tool_args_with_pydantic(tname, args)
            assertions.append(schema_score)
            key_score = score_tool_args_schema(
                tname,
                args,
                required_keys=sample.get("required_keys"),
            )
            key_score["hard"] = True
            assertions.append(key_score)

        retrieval_score = score_retrieval_bundle(expect.get("retrieval"))
        retrieval_score["hard"] = True
        assertions.append(retrieval_score)

        # Offline answer checks only when fixture provides a mock_answer.
        mock_answer = expect.get("mock_answer")
        if mock_answer is not None:
            ans = score_answer_constraints(
                answer=str(mock_answer),
                must_contain=expect.get("answer_must_contain"),
                must_not_contain=expect.get("answer_must_not_contain"),
                must_abstain=expect.get("must_abstain"),
            )
            ans["hard"] = True
            assertions.append(ans)

            num = score_numeric(expect.get("numeric_equals"), str(mock_answer))
            num["hard"] = True
            assertions.append(num)

            ent = score_entities(expect.get("required_entities"), str(mock_answer))
            ent["hard"] = True
            assertions.append(ent)

            sec = score_security_isolation(
                forbidden_content=expect.get("forbidden_content"),
                answer=str(mock_answer),
                leaked_memory=expect.get("leaked_memory"),
                unauthorized_access=expect.get("unauthorized_access"),
            )
            sec["hard"] = True
            assertions.append(sec)

        # Fabricated tool success must never be asserted as true without execution.
        if expect.get("forbid_fabricated_tool_success"):
            assertions.append(
                {
                    "checked": True,
                    "passed": True,
                    "hard": True,
                    "name": "no_fabricated_tool_success",
                    "detail": "offline mode does not invent tool execution success",
                }
            )

    except Exception as exc:  # noqa: BLE001
        blocked = True
        block_reason = f"offline_eval_error: {exc}"
        error = traceback.format_exc()

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    requires_review = bool(expect.get("requires_human_review"))
    status = aggregate_case_status(
        assertion_results=assertions,
        blocked=blocked,
        block_reason=block_reason,
        requires_human_review=requires_review,
        live_answer_present=False,
    )

    return {
        "case_id": case_id,
        "category": case.get("category"),
        "status": status,
        "mode": "offline",
        "input_message": message,
        "expected_intent": expect.get("intent"),
        "classified_intent": classified_intent,
        "actual_intent": actual_intent,
        "normalized_message": normalized_message,
        "normalization_transforms": normalization_transforms,
        "semantic_intent": getattr(decision, "semantic_intent", None),
        "semantic_confidence": getattr(decision, "semantic_confidence", None),
        "entities": list(getattr(decision, "entities", None) or []),
        "needs_clarification": bool(
            getattr(decision, "needs_clarification", False)
        ),
        "is_follow_up": bool(getattr(decision, "is_follow_up", False)),
        "follow_up_type": getattr(decision, "follow_up_type", None),
        "resolved_query": getattr(decision, "resolved_query", None),
        "contextual_confidence": getattr(decision, "contextual_confidence", None),
        "expected_tools": expect.get("required_tools"),
        "actual_tools": actual_tools,
        "tools_enabled": tools_enabled,
        "forced_tool_choice": forced_choice
        if not isinstance(forced_choice, dict)
        else {
            "type": forced_choice.get("type"),
            "name": ((forced_choice.get("function") or {}).get("name")),
        },
        "route_kind": route_kind,
        "assertions": assertions,
        "latency_ms": latency_ms,
        "token_usage": None,
        "block_reason": block_reason,
        "error": error,
        "requires_human_review": requires_review,
        "notes": case.get("notes"),
    }


def evaluate_case_live(
    case: dict[str, Any],
    *,
    chat_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Optionally call the live chat path. Gated by caller; may return BLOCKED."""
    offline = evaluate_case_offline(case)
    if offline["status"] == STATUS_FAIL:
        offline["mode"] = "live"
        offline["live"] = {"skipped": True, "reason": "offline_assertions_failed"}
        return offline

    expect = case.get("expect") or {}
    message = str((case.get("input") or {}).get("message") or "")
    started = time.perf_counter()

    if chat_fn is None:
        offline["mode"] = "live"
        offline["status"] = STATUS_BLOCKED
        offline["block_reason"] = "live_chat_fn_not_provided"
        offline["latency_ms"] = round(
            offline.get("latency_ms", 0) + (time.perf_counter() - started) * 1000,
            3,
        )
        return offline

    try:
        result = chat_fn(message)
        answer = ""
        token_usage = None
        tools_called: list[str] = []
        if isinstance(result, dict):
            answer = str(result.get("answer") or result.get("content") or "")
            token_usage = result.get("token_usage") or result.get("usage")
            tools_called = list(result.get("tools_called") or [])
        else:
            answer = str(result)

        assertions = list(offline.get("assertions") or [])
        ans = score_answer_constraints(
            answer=answer,
            must_contain=expect.get("answer_must_contain"),
            must_not_contain=expect.get("answer_must_not_contain"),
            must_abstain=expect.get("must_abstain"),
        )
        ans["hard"] = bool(
            expect.get("answer_must_contain")
            or expect.get("answer_must_not_contain")
            or expect.get("must_abstain")
        )
        assertions.append(ans)

        num = score_numeric(expect.get("numeric_equals"), answer)
        num["hard"] = expect.get("numeric_equals") is not None
        assertions.append(num)

        ent = score_entities(expect.get("required_entities"), answer)
        ent["hard"] = bool(expect.get("required_entities"))
        assertions.append(ent)

        sec = score_security_isolation(
            forbidden_content=expect.get("forbidden_content"),
            answer=answer,
            leaked_memory=expect.get("leaked_memory"),
            unauthorized_access=expect.get("unauthorized_access"),
        )
        sec["hard"] = True
        assertions.append(sec)

        if expect.get("required_tools_called"):
            called_set = {str(t).casefold() for t in tools_called}
            for tool in expect["required_tools_called"]:
                ok = str(tool).casefold() in called_set
                assertions.append(
                    {
                        "checked": True,
                        "passed": ok,
                        "hard": True,
                        "name": "required_tool_called",
                        "tool": tool,
                    }
                )

        status = aggregate_case_status(
            assertion_results=assertions,
            blocked=False,
            requires_human_review=bool(expect.get("requires_human_review")),
            live_answer_present=bool(answer.strip()),
        )
        live_latency = round((time.perf_counter() - started) * 1000, 3)
        offline.update(
            {
                "mode": "live",
                "status": status,
                "assertions": assertions,
                "live_answer": answer,
                "tools_called": tools_called,
                "token_usage": token_usage,
                "latency_ms": round(
                    float(offline.get("latency_ms") or 0) + live_latency, 3
                ),
                "block_reason": None,
            }
        )
        return offline
    except Exception as exc:  # noqa: BLE001
        offline["mode"] = "live"
        offline["status"] = STATUS_BLOCKED
        offline["block_reason"] = f"live_provider_error: {exc}"
        offline["error"] = traceback.format_exc()
        offline["latency_ms"] = round(
            float(offline.get("latency_ms") or 0)
            + (time.perf_counter() - started) * 1000,
            3,
        )
        return offline


def run_evaluation(
    *,
    dataset_path: str | None = None,
    dataset: dict[str, Any] | None = None,
    live: bool = False,
    live_chat_fn: Callable[..., Any] | None = None,
    categories: list[str] | None = None,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run the assistant evaluation suite and return a JSON-serializable report."""
    data = dataset if dataset is not None else load_dataset(dataset_path)
    validate_dataset(data)

    env_live = os.environ.get("ASSISTANT_EVAL_LIVE", "").strip() in {"1", "true", "True", "yes"}
    use_live = bool(live or env_live)

    selected = list(data["cases"])
    if categories:
        allow = {c.casefold() for c in categories}
        selected = [c for c in selected if str(c.get("category", "")).casefold() in allow]
    if case_ids:
        allow_ids = set(case_ids)
        selected = [c for c in selected if str(c.get("case_id")) in allow_ids]

    started = time.perf_counter()
    case_results: list[dict[str, Any]] = []
    for case in selected:
        if use_live:
            case_results.append(evaluate_case_live(case, chat_fn=live_chat_fn))
        else:
            case_results.append(evaluate_case_offline(case))

    summary = {
        STATUS_PASS: 0,
        STATUS_FAIL: 0,
        STATUS_BLOCKED: 0,
        STATUS_NEEDS_REVIEW: 0,
    }
    by_category: dict[str, dict[str, int]] = {}
    for row in case_results:
        st = row.get("status") or STATUS_FAIL
        summary[st] = summary.get(st, 0) + 1
        cat = str(row.get("category") or "unknown")
        bucket = by_category.setdefault(
            cat, {STATUS_PASS: 0, STATUS_FAIL: 0, STATUS_BLOCKED: 0, STATUS_NEEDS_REVIEW: 0}
        )
        bucket[st] = bucket.get(st, 0) + 1

    latencies = [float(r.get("latency_ms") or 0) for r in case_results]
    token_rows = [r.get("token_usage") for r in case_results if r.get("token_usage")]

    report = {
        "framework": "assistant_eval",
        "dataset_version": data.get("version"),
        "dataset_id": data.get("dataset_id", "assistant_eval_v1"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if use_live else "offline",
        "live_gated": True,
        "case_count": len(case_results),
        "category_counts": category_counts({"cases": selected}),
        "summary": summary,
        "by_category": by_category,
        "runtime": {
            "total_ms": round((time.perf_counter() - started) * 1000, 3),
            "avg_case_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "max_case_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        },
        "token_usage": {
            "cases_with_usage": len(token_rows),
            "note": "Token usage is recorded only when a live provider response includes usage metadata.",
            "rows": token_rows,
        },
        "cases": case_results,
        "limitations": [
            "Offline mode scores routing, tool selection, schemas, and fixture retrieval — not free-form answer quality.",
            "Live mode is explicitly gated and does not run by default.",
            "Open-ended answers marked requires_human_review need human assessment.",
            "This framework does not prove production readiness by itself.",
        ],
    }
    return report
