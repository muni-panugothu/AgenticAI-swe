"""Authentication routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.deps import AuthenticatedUser, get_current_user
from app.core.config import get_settings
from app.schemas.auth_schema import (
    AuthTokensResponse,
    LoginRequest,
    LogoutResponse,
    RefreshRequest,
    RegisterRequest,
    UserPublic,
)
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)
_settings = get_settings()


def _tokens_response(payload: dict) -> AuthTokensResponse:
    return AuthTokensResponse(
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token"),
        expires_in=int(payload["expires_in"]),
        user=payload["user"],
    )


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    settings = get_settings()
    max_age = settings.refresh_token_ttl_days * 86400
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=refresh_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,  # type: ignore[arg-type]
        max_age=max_age,
        path="/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path="/auth",
    )


@router.post("/register", response_model=AuthTokensResponse)
@limiter.limit(_settings.rate_limit_register)
async def register(
    request: Request,
    body: RegisterRequest,
    response: Response,
) -> AuthTokensResponse:
    result = auth_service.register_user(email=body.email, password=body.password)
    if not result.get("success"):
        code = result.get("code", "REGISTER_FAILED")
        status_code = status.HTTP_409_CONFLICT if code == "EMAIL_EXISTS" else 400
        raise HTTPException(
            status_code=status_code,
            detail={"error": result.get("error"), "code": code},
        )
    user_doc = auth_service.get_user_by_id(result["user"].id)
    tokens = auth_service.issue_tokens_for_user(user_doc)
    _set_refresh_cookie(response, tokens["refresh_token"])
    return _tokens_response(tokens)


@router.post("/login", response_model=AuthTokensResponse)
@limiter.limit(_settings.rate_limit_login)
async def login(
    request: Request,
    body: LoginRequest,
    response: Response,
) -> AuthTokensResponse:
    auth = auth_service.authenticate_user(email=body.email, password=body.password)
    if not auth.get("success"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": auth.get("error", "Invalid credentials"),
                "code": auth.get("code", "INVALID_CREDENTIALS"),
            },
        )
    tokens = auth_service.issue_tokens_for_user(auth["user"])
    _set_refresh_cookie(response, tokens["refresh_token"])
    return _tokens_response(tokens)


@router.post("/refresh", response_model=AuthTokensResponse)
@limiter.limit(_settings.rate_limit_refresh)
async def refresh_tokens(
    request: Request,
    body: RefreshRequest,
    response: Response,
) -> AuthTokensResponse:
    settings = get_settings()
    refresh = body.refresh_token or request.cookies.get(settings.refresh_cookie_name)
    if not refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Refresh token required", "code": "MISSING_REFRESH"},
        )
    result = auth_service.refresh_session(refresh_token=refresh)
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": result.get("error", "Invalid refresh"),
                "code": result.get("code", "INVALID_REFRESH"),
            },
        )
    _set_refresh_cookie(response, result["refresh_token"])
    return AuthTokensResponse(
        access_token=result["access_token"],
        refresh_token=result.get("refresh_token"),
        expires_in=result["expires_in"],
        user=result["user"],
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    response: Response,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> LogoutResponse:
    settings = get_settings()
    refresh = request.cookies.get(settings.refresh_cookie_name)
    if refresh:
        auth_service.revoke_refresh_token(refresh)
    _clear_refresh_cookie(response)
    return LogoutResponse()


@router.get("/me", response_model=UserPublic)
async def me(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> UserPublic:
    doc = auth_service.get_user_by_id(user.id)
    if not doc:
        raise HTTPException(status_code=404, detail={"error": "Not found", "code": "NOT_FOUND"})
    return UserPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        display_name=doc.get("display_name"),
        tenant_id=doc.get("tenant_id"),
        status=doc.get("status", "active"),
        created_at=(
            doc.get("created_at").isoformat() if doc.get("created_at") else None
        ),
    )
