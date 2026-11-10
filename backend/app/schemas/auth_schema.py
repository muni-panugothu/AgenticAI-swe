"""Pydantic models for authentication API."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

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


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refresh_token: str | None = None


class UserPublic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    email: EmailStr
    display_name: str | None = None
    tenant_id: str | None = None
    status: Literal["active", "disabled"] = "active"
    created_at: str | None = None


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
