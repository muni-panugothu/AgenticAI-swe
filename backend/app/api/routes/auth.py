"""Authentication routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.deps import AuthenticatedUser, _bearer, get_current_user
from app.core.security import decode_access_token
from app.core.config import get_settings
from app.schemas.auth_schema import (
    AuthTokensResponse,
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LegalVersionsResponse,
    LoginRequest,
    LogoutResponse,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    ResetPasswordResponse,
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


def _set_refresh_cookie(
    response: Response, refresh_token: str, *, ttl_days: int | None = None
) -> None:
    settings = get_settings()
    max_age = (ttl_days or settings.refresh_token_ttl_days) * 86400
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


@router.get("/legal-versions", response_model=LegalVersionsResponse)
async def legal_versions() -> LegalVersionsResponse:
    settings = get_settings()
    return LegalVersionsResponse(
        terms_version=settings.terms_version,
        privacy_version=settings.privacy_version,
    )


@router.post("/register", response_model=AuthTokensResponse)
@limiter.limit(_settings.rate_limit_register)
async def register(
    request: Request,
    body: RegisterRequest,
    response: Response,
) -> AuthTokensResponse:
    result = auth_service.register_user(
        email=body.email,
        password=body.password,
        display_name=body.display_name,
        accepted_terms=body.accepted_terms,
        accepted_privacy=body.accepted_privacy,
    )
    if not result.get("success"):
        code = result.get("code", "REGISTER_FAILED")
        status_code = status.HTTP_409_CONFLICT if code == "EMAIL_EXISTS" else 400
        raise HTTPException(
            status_code=status_code,
            detail={"error": result.get("error"), "code": code},
        )
    user_doc = auth_service.get_user_by_id(result["user"].id)
    tokens = auth_service.issue_tokens_for_user(user_doc, remember_me=False)
    _set_refresh_cookie(
        response,
        tokens["refresh_token"],
        ttl_days=tokens.get("refresh_ttl_days"),
    )
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
    tokens = auth_service.issue_tokens_for_user(
        auth["user"], remember_me=bool(body.remember_me)
    )
    _set_refresh_cookie(
        response,
        tokens["refresh_token"],
        ttl_days=tokens.get("refresh_ttl_days"),
    )
    return _tokens_response(tokens)


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
@limiter.limit(_settings.rate_limit_forgot_password)
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
) -> ForgotPasswordResponse:
    # Always generic — never reveal whether the email is registered.
    auth_service.request_password_reset(email=body.email)
    return ForgotPasswordResponse()


@router.post("/reset-password", response_model=ResetPasswordResponse)
@limiter.limit(_settings.rate_limit_reset_password)
async def reset_password(
    request: Request,
    body: ResetPasswordRequest,
) -> ResetPasswordResponse:
    result = auth_service.reset_password_with_token(
        token=body.token, password=body.password
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": result.get("error", "Unable to reset password"),
                "code": result.get("code", "RESET_FAILED"),
            },
        )
    return ResetPasswordResponse()


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
    body: RefreshRequest | None = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> LogoutResponse:
    """Revoke refresh session. CLI may send refresh_token only (expired access)."""
    settings = get_settings()
    refresh = request.cookies.get(settings.refresh_cookie_name)
    if body and body.refresh_token:
        refresh = body.refresh_token
    if refresh:
        auth_service.revoke_refresh_token(refresh)
        _clear_refresh_cookie(response)
        return LogoutResponse()
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Refresh token or bearer required", "code": "UNAUTHORIZED"},
        )
    payload = decode_access_token(credentials.credentials)
    if payload and payload.get("sub") and payload.get("sid"):
        auth_service.revoke_session_id(str(payload["sid"]), str(payload["sub"]))
    # Bearer-only logout (cookie refresh may still exist).
    cookie_refresh = request.cookies.get(settings.refresh_cookie_name)
    if cookie_refresh:
        auth_service.revoke_refresh_token(cookie_refresh)
    _clear_refresh_cookie(response)
    return LogoutResponse()


@router.get("/me", response_model=UserPublic)
async def me(user: Annotated[AuthenticatedUser, Depends(get_current_user)]) -> UserPublic:
    doc = auth_service.get_user_by_id(user.id)
    if not doc:
        raise HTTPException(status_code=404, detail={"error": "Not found", "code": "NOT_FOUND"})
    return auth_service._public_user(doc)
