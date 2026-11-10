"""User registration, login, refresh rotation, and session revocation."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import ASCENDING
from pymongo.collection import Collection

from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token_plain,
    verify_password,
)
from app.schemas.auth_schema import UserPublic


def _users_collection() -> Collection:
    from app.services.memory_service import database

    settings = get_settings()
    return database[settings.auth_users_collection]


def _sessions_collection() -> Collection:
    from app.services.memory_service import database

    settings = get_settings()
    return database[settings.auth_sessions_collection]


def ensure_auth_indexes() -> None:
    users = _users_collection()
    users.create_index([("email", ASCENDING)], unique=True, name="auth_email_unique")
    users.create_index([("status", ASCENDING)], name="auth_status_idx")
    sessions = _sessions_collection()
    sessions.create_index(
        [("token_hash", ASCENDING)],
        unique=True,
        name="auth_session_token_hash_unique",
    )
    sessions.create_index(
        [("user_id", ASCENDING), ("revoked_at", ASCENDING)],
        name="auth_session_user_revoked_idx",
    )
    sessions.create_index(
        [("expires_at", ASCENDING)],
        expireAfterSeconds=0,
        name="auth_session_ttl",
    )


def _public_user(doc: dict) -> UserPublic:
    return UserPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        display_name=doc.get("display_name"),
        tenant_id=doc.get("tenant_id"),
        status=doc.get("status", "active"),
        created_at=(
            doc.get("created_at").isoformat()
            if doc.get("created_at")
            else None
        ),
    )


def register_user(*, email: str, password: str) -> dict[str, Any]:
    coll = _users_collection()
    normalized = email.strip().lower()
    if coll.find_one({"email": normalized}):
        return {
            "success": False,
            "code": "EMAIL_EXISTS",
            "error": "An account with this email already exists",
        }

    now = datetime.now(timezone.utc)
    user_id = f"usr_{uuid.uuid4().hex}"
    doc = {
        "_id": user_id,
        "email": normalized,
        "password_hash": hash_password(password),
        "display_name": normalized.split("@")[0],
        "tenant_id": f"ten_{uuid.uuid4().hex[:12]}",
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }
    coll.insert_one(doc)
    return {"success": True, "user": _public_user(doc)}


def authenticate_user(*, email: str, password: str) -> dict[str, Any]:
    coll = _users_collection()
    doc = coll.find_one({"email": email.strip().lower()})
    if not doc:
        return {"success": False, "code": "INVALID_CREDENTIALS", "error": "Invalid email or password"}
    if doc.get("status") != "active":
        return {"success": False, "code": "ACCOUNT_DISABLED", "error": "Account is disabled"}
    if not verify_password(password, doc.get("password_hash") or ""):
        return {"success": False, "code": "INVALID_CREDENTIALS", "error": "Invalid email or password"}
    return {"success": True, "user": doc}


def issue_tokens_for_user(user_doc: dict) -> dict[str, Any]:
    settings = get_settings()
    access = create_access_token(
        user_id=str(user_doc["_id"]),
        email=user_doc["email"],
        tenant_id=user_doc.get("tenant_id"),
    )
    refresh_plain = new_refresh_token_plain()
    refresh_hash = hash_refresh_token(refresh_plain)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=settings.refresh_token_ttl_days)
    _sessions_collection().insert_one(
        {
            "user_id": str(user_doc["_id"]),
            "token_hash": refresh_hash,
            "created_at": now,
            "expires_at": expires,
            "revoked_at": None,
            "rotated_from": None,
        }
    )
    return {
        "access_token": access,
        "refresh_token": refresh_plain,
        "expires_in": settings.access_token_ttl_minutes * 60,
        "user": _public_user(user_doc),
    }


def get_user_by_id(user_id: str) -> dict | None:
    return _users_collection().find_one({"_id": user_id})


def refresh_session(*, refresh_token: str) -> dict[str, Any]:
    token_hash = hash_refresh_token(refresh_token)
    sessions = _sessions_collection()
    session = sessions.find_one({"token_hash": token_hash})
    if not session or session.get("revoked_at") is not None:
        return {"success": False, "code": "INVALID_REFRESH", "error": "Invalid refresh token"}
    now = datetime.now(timezone.utc)
    if session.get("expires_at") and session["expires_at"] < now:
        return {"success": False, "code": "EXPIRED_REFRESH", "error": "Refresh token expired"}

    user = get_user_by_id(session["user_id"])
    if not user or user.get("status") != "active":
        return {"success": False, "code": "ACCOUNT_DISABLED", "error": "Account is disabled"}

    # Rotate: revoke old, issue new refresh.
    sessions.update_one(
        {"_id": session["_id"]},
        {"$set": {"revoked_at": now, "rotated_at": now}},
    )
    settings = get_settings()
    access = create_access_token(
        user_id=str(user["_id"]),
        email=user["email"],
        tenant_id=user.get("tenant_id"),
    )
    refresh_plain = new_refresh_token_plain()
    refresh_hash = hash_refresh_token(refresh_plain)
    expires = now + timedelta(days=settings.refresh_token_ttl_days)
    sessions.insert_one(
        {
            "user_id": str(user["_id"]),
            "token_hash": refresh_hash,
            "created_at": now,
            "expires_at": expires,
            "revoked_at": None,
            "rotated_from": session["_id"],
        }
    )
    return {
        "success": True,
        "access_token": access,
        "refresh_token": refresh_plain,
        "expires_in": settings.access_token_ttl_minutes * 60,
        "user": _public_user(user),
    }


def revoke_refresh_token(refresh_token: str) -> bool:
    token_hash = hash_refresh_token(refresh_token)
    now = datetime.now(timezone.utc)
    result = _sessions_collection().update_one(
        {"token_hash": token_hash, "revoked_at": None},
        {"$set": {"revoked_at": now}},
    )
    return result.matched_count > 0


def revoke_all_sessions(user_id: str) -> int:
    now = datetime.now(timezone.utc)
    result = _sessions_collection().update_many(
        {"user_id": user_id, "revoked_at": None},
        {"$set": {"revoked_at": now}},
    )
    return int(result.modified_count)
