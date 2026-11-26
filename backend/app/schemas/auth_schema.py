"""Pydantic models for authentication API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    # Optional — when provided must match password; UI no longer collects it.
    confirm_password: str | None = Field(default=None, max_length=128)
    # Must be JSON `true` — rejects false/omitted bypass attempts at the schema layer.
    accepted_terms: Literal[True]
    accepted_privacy: Literal[True]

    @field_validator("display_name")
    @classmethod
    def clean_display_name(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("Name is required")
        if not re.fullmatch(r"[\w][\w\s'\-]{0,79}", cleaned, flags=re.UNICODE):
            raise ValueError("Name may only contain letters, numbers, spaces, apostrophes, and hyphens")
        return cleaned

    @field_validator("password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must include an uppercase letter")
        if not re.search(r"[a-z]", value):
            raise ValueError("Password must include a lowercase letter")
        if not re.search(r"\d", value):
            raise ValueError("Password must include a digit")
        return value

    @model_validator(mode="after")
    def passwords_match(self) -> RegisterRequest:
        if (
            self.confirm_password is not None
            and self.confirm_password != self.password
        ):
            raise ValueError("Passwords do not match")
        return self


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
    remember_me: bool = False


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str | None = None


class ForgotPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr


class ResetPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=20, max_length=256)
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)

    @field_validator("password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        if not re.search(r"[A-Z]", value):
            raise ValueError("Password must include an uppercase letter")
        if not re.search(r"[a-z]", value):
            raise ValueError("Password must include a lowercase letter")
        if not re.search(r"\d", value):
            raise ValueError("Password must include a digit")
        return value

    @model_validator(mode="after")
    def passwords_match(self) -> ResetPasswordRequest:
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self


class ForgotPasswordResponse(BaseModel):
    success: bool = True
    message: str = (
        "If an account exists for that email, a password reset link has been sent."
    )


class ResetPasswordResponse(BaseModel):
    success: bool = True
    message: str = "Password updated. You can sign in with your new password."


class LegalVersionsResponse(BaseModel):
    terms_version: str
    privacy_version: str
    terms_title: str = "Terms of Use"
    privacy_title: str = "Privacy Policy"


class UserPublic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    email: EmailStr
    display_name: str | None = None
    tenant_id: str | None = None
    status: Literal["active", "disabled"] = "active"
    created_at: str | None = None
    terms_accepted_at: str | None = None
    privacy_accepted_at: str | None = None
    terms_version: str | None = None
    privacy_version: str | None = None


class AuthTokensResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic
    # Mobile/CLI store in secure storage; browsers may use HttpOnly cookie instead.
    refresh_token: str | None = None


class LogoutResponse(BaseModel):
    success: bool = True
    message: str = "Logged out"


class ApiError(BaseModel):
    error: str
    code: str
    details: list[dict] | None = None
