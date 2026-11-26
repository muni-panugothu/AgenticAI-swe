"""Human approval API (Topics 48/50) — authenticated, additive REST."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import AuthenticatedUser, get_current_user
from app.services import approval_service
from app.services.tool_permission_service import check_tool_permission

router = APIRouter(prefix="/api/approvals", tags=["approvals"])


class DecisionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approve: bool
    rejection_reason: str | None = Field(default=None, max_length=500)


class ExecuteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Client must not send identity; arguments come from the bound approval.
    pass


def _public_approval(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    return {
        "approval_id": doc.get("approval_id"),
        "status": doc.get("status"),
        "tool_name": doc.get("tool_name"),
        "target_resource": doc.get("target_resource"),
        "risk_class": doc.get("risk_class"),
        "reason": doc.get("reason"),
        "project_id": doc.get("project_id"),
        "tenant_id": doc.get("tenant_id"),
        "request_id": doc.get("request_id"),
        "expires_at": doc.get("expires_at"),
        "created_at": doc.get("created_at"),
        "decided_at": doc.get("decided_at"),
        "consumed_at": doc.get("consumed_at"),
        # Do not echo full arguments — digest only.
        "arguments_digest": doc.get("arguments_digest"),
        "policy_version": doc.get("policy_version"),
    }


@router.get("/{approval_id}")
async def get_approval(
    approval_id: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    doc = await run_in_threadpool(approval_service.get_approval, approval_id)
    if not doc or doc.get("user_id") != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Approval not found", "code": "NOT_FOUND"},
        )
    return {"success": True, "approval": _public_approval(doc)}


@router.post("/{approval_id}/decision")
async def decide(
    approval_id: str,
    body: DecisionBody,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    result = await run_in_threadpool(
        approval_service.decide_approval,
        approval_id=approval_id,
        reviewer_user_id=user.id,
        approve=body.approve,
        rejection_reason=body.rejection_reason,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": result.get("error") or "decision_failed",
                "code": result.get("error_code") or "APPROVAL_DECISION_FAILED",
            },
        )
    return {"success": True, "approval": _public_approval(result.get("approval"))}


@router.post("/{approval_id}/revoke")
async def revoke(
    approval_id: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    result = await run_in_threadpool(
        approval_service.revoke_approval,
        approval_id=approval_id,
        actor_user_id=user.id,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": result.get("error"), "code": "APPROVAL_REVOKE_FAILED"},
        )
    return {"success": True, "approval": _public_approval(result.get("approval"))}


@router.post("/{approval_id}/execute")
async def execute_approved(
    approval_id: str,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
):
    """Consume approval and run through authoritative ``execute_tool``."""
    doc = await run_in_threadpool(approval_service.get_approval, approval_id)
    if not doc or doc.get("user_id") != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Approval not found", "code": "NOT_FOUND"},
        )

    tool_name = str(doc.get("tool_name") or "")
    arguments = dict(doc.get("normalized_arguments") or {})
    project_id = doc.get("project_id")
    tenant_id = doc.get("tenant_id") or user.tenant_id

    # Permission must still allow — approval never overrides deny.
    perm = await run_in_threadpool(
        lambda: check_tool_permission(
            user_id=user.id,
            tool_name=tool_name,
            project_id=project_id,
            tenant_id=tenant_id,
            resource=str(doc.get("target_resource") or ""),
        )
    )
    if not perm.get("allowed"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Tool permission denied",
                "code": "TOOL_PERMISSION_DENIED",
                "reason": perm.get("reason"),
            },
        )

    def _run():
        from app.chatbot import execute_tool

        return execute_tool(
            tool_name,
            arguments,
            user_id=user.id,
            request_id=doc.get("request_id"),
            project_id=project_id,
            tenant_id=tenant_id,
            approval_id=approval_id,
            task_id=doc.get("task_id"),
        )

    result = await run_in_threadpool(_run)
    # Never claim success for pending/denied.
    return {
        "success": bool(result.get("success")),
        "tool_name": tool_name,
        "approval_id": approval_id,
        "result": result,
    }
