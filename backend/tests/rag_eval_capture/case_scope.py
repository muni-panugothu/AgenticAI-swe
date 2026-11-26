"""Map golden cases to retrieve() ACL principals without inventing metadata."""

from __future__ import annotations

from typing import Any


class CaseScopeError(ValueError):
    """Golden case cannot be mapped to a safe retrieve scope."""


def _document_by_id(golden: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for doc in golden.get("documents") or []:
        doc_id = str(doc.get("document_id") or "")
        if doc_id:
            out[doc_id] = doc
    return out


def resolve_case_scope(case: dict[str, Any], golden: dict[str, Any]) -> dict[str, Any]:
    """Return user_id / tenant_id / project_id for retrieve().

    Rules (derived only from golden fields):
    - Default principal is alice / t1 when cases are about alice-readable docs.
    - metadata_filters.project_id / tenant_id override project/tenant when present.
    - acl_scenario private_owner → alice owning private policy.
    - acl_scenario shared_read → alice reading bob's shared handbook.
    - expect_acl_isolation → alice in t1 must not see other-tenant content.
    """
    if not isinstance(case, dict) or not case.get("case_id"):
        raise CaseScopeError("Case must include case_id")

    case_id = str(case["case_id"])
    filters = case.get("metadata_filters") or {}
    if filters is not None and not isinstance(filters, dict):
        raise CaseScopeError(f"metadata_filters must be an object for {case_id}")

    docs = _document_by_id(golden)
    user_id = "alice"
    tenant_id: str | None = "t1"
    project_id: str | None = None

    if "tenant_id" in filters:
        tenant_id = filters.get("tenant_id")
    if "project_id" in filters:
        project_id = filters.get("project_id")

    scenario = case.get("acl_scenario")
    if scenario == "private_owner":
        policy = docs.get("doc-policy-md")
        if not policy:
            raise CaseScopeError("private_owner requires doc-policy-md in golden documents")
        user_id = str(policy.get("owner_id") or "alice")
        tenant_id = policy.get("tenant_id")
        project_id = policy.get("project_id")
    elif scenario == "shared_read":
        handbook = docs.get("doc-handbook-pdf")
        if not handbook:
            raise CaseScopeError("shared_read requires doc-handbook-pdf in golden documents")
        shared_with = list(handbook.get("shared_with") or [])
        if "alice" not in shared_with:
            raise CaseScopeError("shared_read expects alice in handbook shared_with")
        user_id = "alice"
        tenant_id = handbook.get("tenant_id")
        project_id = handbook.get("project_id")
    elif scenario not in (None, ""):
        raise CaseScopeError(f"Unsupported acl_scenario {scenario!r} for {case_id}")

    # Isolation cases still query as alice/t1 so mallory/t2 rows must stay hidden.
    if case.get("expect_acl_isolation"):
        user_id = "alice"
        if "tenant_id" not in filters:
            tenant_id = "t1"
        if "project_id" not in filters:
            project_id = None

    return {
        "case_id": case_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "project_id": project_id,
        "question": str(case.get("question") or ""),
        "metadata_filters": dict(filters),
        "expect_acl_isolation": bool(case.get("expect_acl_isolation")),
        "expected_abstain": bool(case.get("expected_abstain", False)),
        "relevant_chunk_ids": [str(x) for x in (case.get("relevant_chunk_ids") or [])],
    }


def resolve_all_case_scopes(golden: dict[str, Any]) -> list[dict[str, Any]]:
    cases = golden.get("cases") or []
    if not cases:
        raise CaseScopeError("Golden dataset has no cases")
    return [resolve_case_scope(case, golden) for case in cases]
