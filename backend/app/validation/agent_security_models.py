"""Typed models for Topics 48–53 — HITL, permissions, approvals, provenance.

Security state is server-owned. LLM/client fields are never authoritative.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"
    CONSUMED = "consumed"
    EXECUTION_FAILED = "execution_failed"
    CANCELLED = "cancelled"


class PermissionEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class RiskClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ContentTrustLevel(str, Enum):
    SYSTEM = "system"
    SERVER_POLICY = "server_policy"
    AUTHENTICATED_USER = "authenticated_user"
    CONVERSATION_HISTORY = "conversation_history"
    RAG_DOCUMENT = "rag_document"
    WEB_RESULT = "web_result"
    FILE_CONTENT = "file_content"
    TOOL_OUTPUT = "tool_output"
    EXTERNAL = "external"
    GENERATED = "generated"


class ToolRiskMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    risk_class: RiskClass = RiskClass.MEDIUM
    read_only: bool = True
    requires_approval: bool = False
    requires_sandbox: bool = False
    resource_category: str = "general"
    permission_action: str = "invoke"


class ContentProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trust: ContentTrustLevel
    source: str = ""
    tool_name: str | None = None
    document_id: str | None = None


class PermissionGrant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grant_id: str
    effect: PermissionEffect
    user_id: str
    project_id: str | None = None
    tenant_id: str | None = None
    tool_name: str
    action: str = "invoke"
    resource_scope: str = "*"
    expires_at: datetime | None = None
    source: str = "policy"
    created_at: datetime
    revoked_at: datetime | None = None


class ApprovalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    status: ApprovalStatus
    user_id: str
    project_id: str | None = None
    tenant_id: str | None = None
    request_id: str | None = None
    task_id: str | None = None
    tool_name: str
    action: str = "invoke"
    arguments_digest: str
    normalized_arguments: dict[str, Any] = Field(default_factory=dict)
    target_resource: str = ""
    risk_class: RiskClass = RiskClass.HIGH
    reason: str = ""
    policy_version: str = "1"
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None
    consumed_at: datetime | None = None
    revoked_at: datetime | None = None
    rejection_reason: str | None = None


# Valid approval state transitions (to_status -> allowed from_status).
APPROVAL_TRANSITIONS: dict[ApprovalStatus, frozenset[ApprovalStatus]] = {
    ApprovalStatus.APPROVED: frozenset({ApprovalStatus.PENDING}),
    ApprovalStatus.REJECTED: frozenset({ApprovalStatus.PENDING}),
    ApprovalStatus.EXPIRED: frozenset({ApprovalStatus.PENDING, ApprovalStatus.APPROVED}),
    ApprovalStatus.REVOKED: frozenset(
        {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}
    ),
    ApprovalStatus.CONSUMED: frozenset({ApprovalStatus.APPROVED}),
    ApprovalStatus.EXECUTION_FAILED: frozenset({ApprovalStatus.CONSUMED}),
    ApprovalStatus.CANCELLED: frozenset(
        {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}
    ),
}


SecurityEventType = Literal[
    "permission_denied",
    "permission_allowed",
    "approval_created",
    "approval_required",
    "approval_approved",
    "approval_rejected",
    "approval_expired",
    "approval_revoked",
    "approval_consume_attempt",
    "approval_consume_success",
    "approval_consume_failed",
    "approval_replay",
    "scope_mismatch",
    "invalid_tool_invocation",
    "prompt_injection_rejected",
    "sandbox_violation",
    "resource_limit_violation",
    "execution_started",
    "execution_completed",
    "execution_failed",
]
