"""FastAPI dependencies — trusted identity from verified tokens only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.services.auth_service import get_user_by_id

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    email: str
    tenant_id: str | None
    status: str


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthenticatedUser:
    token = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Authentication required", "code": "UNAUTHORIZED"},
        )
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid or expired token", "code": "INVALID_TOKEN"},
        )
    user = get_user_by_id(str(payload["sub"]))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "User not found", "code": "USER_NOT_FOUND"},
        )
    if user.get("status") != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Account is disabled", "code": "ACCOUNT_DISABLED"},
        )
    return AuthenticatedUser(
        id=str(user["_id"]),
        email=user["email"],
        tenant_id=user.get("tenant_id"),
        status=user.get("status", "active"),
    )


def reject_untrusted_user_id_in_body(body: dict) -> None:
    """Raise if client tried to pass user_id / tenant_id as identity."""
    for key in (
        "user_id",
        "owner_id",
        "tenant_id",
        "project_id",
        "conversation_id",
    ):
        if key in body and body[key] is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"Do not send {key} in request body; use authentication",
                    "code": "UNTRUSTED_IDENTITY_FIELD",
                },
            )
