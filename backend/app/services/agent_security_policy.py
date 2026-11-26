"""Central server-side security policy for Topics 48–53.

LLM suggestions never authorize. Unknown high-risk tools fail closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from app.validation.agent_security_models import RiskClass, ToolRiskMeta

POLICY_VERSION = os.getenv("AGENT_SECURITY_POLICY_VERSION", "1")

# Explicit baseline tools an authenticated chat user may invoke (deny-by-default
# outside this catalog unless a durable grant exists).
BASELINE_AUTHENTICATED_TOOLS: frozenset[str] = frozenset({
    "calculator",
    "get_current_time",
    "read_file",
    "search_code",
    "web_search",
    "search_memory",
    "search_memory_summaries",
    "expand_memory_summary",
    "inspect_memory_metadata",
    "detect_memory_conflict",
    "summarize_memories",
    "save_memory",
    "forget_memory",
    "reinforce_memory",
    "request_memory_update",
    "confirm_memory_update",
})

_TOOL_RISK: dict[str, ToolRiskMeta] = {
    "calculator": ToolRiskMeta(
        tool_name="calculator", risk_class=RiskClass.LOW, read_only=True
    ),
    "get_current_time": ToolRiskMeta(
        tool_name="get_current_time", risk_class=RiskClass.LOW, read_only=True
    ),
    "read_file": ToolRiskMeta(
        tool_name="read_file",
        risk_class=RiskClass.MEDIUM,
        read_only=True,
        requires_sandbox=True,
        resource_category="filesystem",
    ),
    "search_code": ToolRiskMeta(
        tool_name="search_code",
        risk_class=RiskClass.MEDIUM,
        read_only=True,
        requires_sandbox=True,
        resource_category="filesystem",
    ),
    "web_search": ToolRiskMeta(
        tool_name="web_search",
        risk_class=RiskClass.MEDIUM,
        read_only=True,
        resource_category="network",
    ),
    "search_memory": ToolRiskMeta(
        tool_name="search_memory", risk_class=RiskClass.MEDIUM, read_only=True,
        resource_category="memory",
    ),
    "search_memory_summaries": ToolRiskMeta(
        tool_name="search_memory_summaries",
        risk_class=RiskClass.MEDIUM,
        read_only=True,
        resource_category="memory",
    ),
    "expand_memory_summary": ToolRiskMeta(
        tool_name="expand_memory_summary",
        risk_class=RiskClass.MEDIUM,
        read_only=True,
        resource_category="memory",
    ),
    "inspect_memory_metadata": ToolRiskMeta(
        tool_name="inspect_memory_metadata",
        risk_class=RiskClass.MEDIUM,
        read_only=True,
        resource_category="memory",
    ),
    "detect_memory_conflict": ToolRiskMeta(
        tool_name="detect_memory_conflict",
        risk_class=RiskClass.MEDIUM,
        read_only=True,
        resource_category="memory",
    ),
    "summarize_memories": ToolRiskMeta(
        tool_name="summarize_memories",
        risk_class=RiskClass.MEDIUM,
        read_only=False,
        requires_approval=True,
        resource_category="memory",
    ),
    "save_memory": ToolRiskMeta(
        tool_name="save_memory",
        risk_class=RiskClass.HIGH,
        read_only=False,
        requires_approval=True,
        resource_category="memory",
    ),
    "forget_memory": ToolRiskMeta(
        tool_name="forget_memory",
        risk_class=RiskClass.CRITICAL,
        read_only=False,
        requires_approval=True,
        resource_category="memory",
    ),
    "reinforce_memory": ToolRiskMeta(
        tool_name="reinforce_memory",
        risk_class=RiskClass.HIGH,
        read_only=False,
        requires_approval=True,
        resource_category="memory",
    ),
    "request_memory_update": ToolRiskMeta(
        tool_name="request_memory_update",
        risk_class=RiskClass.HIGH,
        read_only=False,
        requires_approval=True,
        resource_category="memory",
    ),
    "confirm_memory_update": ToolRiskMeta(
        tool_name="confirm_memory_update",
        risk_class=RiskClass.CRITICAL,
        read_only=False,
        requires_approval=True,
        resource_category="memory",
    ),
}


def hitl_enabled() -> bool:
    return os.getenv("AGENT_HITL_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def get_tool_risk_meta(tool_name: str) -> ToolRiskMeta:
    meta = _TOOL_RISK.get(tool_name)
    if meta is not None:
        return meta
    # Unknown tools: fail closed — treat as critical + approval + sandbox.
    return ToolRiskMeta(
        tool_name=tool_name,
        risk_class=RiskClass.UNKNOWN,
        read_only=False,
        requires_approval=True,
        requires_sandbox=True,
        resource_category="unknown",
    )


def tool_requires_approval(tool_name: str) -> bool:
    if not hitl_enabled():
        return False
    return bool(get_tool_risk_meta(tool_name).requires_approval)


def normalize_tool_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic canonical form for approval binding digests."""
    raw = dict(arguments or {})
    # Never bind on untrusted identity fields.
    for key in (
        "user_id",
        "request_id",
        "project_id",
        "tenant_id",
        "owner_id",
        "shared_with",
        "permission_level",
        "approval_id",
        "approval_status",
        "role",
        "permissions",
        "sandbox",
    ):
        raw.pop(key, None)

    def _canon(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _canon(value[k]) for k in sorted(value.keys())}
        if isinstance(value, list):
            return [_canon(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return _canon(raw)  # type: ignore[return-value]


def arguments_digest(arguments: dict[str, Any] | None) -> str:
    normalized = normalize_tool_arguments(arguments)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def target_resource_for_tool(tool_name: str, arguments: dict[str, Any] | None) -> str:
    args = arguments or {}
    if tool_name in {"read_file", "search_code"}:
        return str(args.get("file_path") or args.get("path") or "")
    if "memory_id" in args:
        return str(args.get("memory_id") or "")
    if tool_name.startswith("search_memory") or tool_name == "search_memory":
        return str(args.get("query") or "")[:200]
    return tool_name
