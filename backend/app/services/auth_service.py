"""User registration, login, refresh rotation, and session revocation."""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

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
from app.services.email_service import send_password_reset_email

_REFRESH_RACE_GRACE_SECONDS = 3


def _users_collection() -> Collection:
    from app.services.memory_service import database

    settings = get_settings()
    return database[settings.auth_users_collection]


def _sessions_collection() -> Collection:
    from app.services.memory_service import database

    settings = get_settings()
    return database[settings.auth_sessions_collection]


def _password_resets_collection() -> Collection:
    from app.services.memory_service import database

    settings = get_settings()
    return database[settings.auth_password_reset_collection]


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
        [("family_id", ASCENDING), ("revoked_at", ASCENDING)],
        name="auth_session_family_revoked_idx",
    )
    sessions.create_index(
        [("expires_at", ASCENDING)],
        expireAfterSeconds=0,
        name="auth_session_ttl",
    )
    resets = _password_resets_collection()
    resets.create_index(
        [("token_hash", ASCENDING)],
        unique=True,
        name="auth_reset_token_hash_unique",
    )
    resets.create_index(
        [("user_id", ASCENDING), ("used_at", ASCENDING)],
        name="auth_reset_user_used_idx",
    )
    resets.create_index(
        [("expires_at", ASCENDING)],
        expireAfterSeconds=0,
        name="auth_reset_ttl",
    )


def _public_user(doc: dict) -> UserPublic:
    def _iso(value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    return UserPublic(
        id=str(doc["_id"]),
        email=doc["email"],
        display_name=doc.get("display_name"),
        tenant_id=doc.get("tenant_id"),
        status=doc.get("status", "active"),
        created_at=_iso(doc.get("created_at")),
        terms_accepted_at=_iso(doc.get("terms_accepted_at")),
        privacy_accepted_at=_iso(doc.get("privacy_accepted_at")),
        terms_version=doc.get("terms_version"),
        privacy_version=doc.get("privacy_version"),
    )


def register_user(
    *,
    email: str,
    password: str,
    display_name: str,
    accepted_terms: bool,
    accepted_privacy: bool,
) -> dict[str, Any]:
    """Create a user only when legal policies are explicitly accepted."""
    if accepted_terms is not True or accepted_privacy is not True:
        return {
            "success": False,
            "code": "LEGAL_NOT_ACCEPTED",
            "error": "Terms of Use and Privacy Policy must be accepted",
        }

    settings = get_settings()
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
        "display_name": display_name.strip(),
        "tenant_id": f"ten_{uuid.uuid4().hex[:12]}",
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "terms_version": settings.terms_version,
        "privacy_version": settings.privacy_version,
        "terms_accepted_at": now,
        "privacy_accepted_at": now,
        "legal_accepted_ip": None,
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


def issue_tokens_for_user(
    user_doc: dict, *, remember_me: bool = False
) -> dict[str, Any]:
    settings = get_settings()
    session_id = str(uuid.uuid4())
    refresh_plain = new_refresh_token_plain()
    refresh_hash = hash_refresh_token(refresh_plain)
    now = datetime.now(timezone.utc)
    ttl_days = (
        settings.remember_me_refresh_ttl_days
        if remember_me
        else settings.refresh_token_ttl_days
    )
    expires = now + timedelta(days=ttl_days)
    _sessions_collection().insert_one(
        {
            "_id": session_id,
            "user_id": str(user_doc["_id"]),
            "token_hash": refresh_hash,
            "created_at": now,
            "expires_at": expires,
            "revoked_at": None,
            "rotated_from": None,
            "family_id": session_id,
            "remember_me": bool(remember_me),
        }
    )
    access = create_access_token(
        user_id=str(user_doc["_id"]),
        email=user_doc["email"],
        session_id=session_id,
        tenant_id=user_doc.get("tenant_id"),
    )
    return {
        "access_token": access,
        "refresh_token": refresh_plain,
        "expires_in": settings.access_token_ttl_minutes * 60,
        "refresh_ttl_days": ttl_days,
        "user": _public_user(user_doc),
    }


def request_password_reset(*, email: str) -> dict[str, Any]:
    """
    Always returns a generic success payload. Creates a hashed, single-use
    token only when the account exists and is active.
    """
    settings = get_settings()
    normalized = email.strip().lower()
    user = _users_collection().find_one({"email": normalized, "status": "active"})
    if not user:
        return {"success": True}

    # Invalidate prior unused tokens for this user.
    now = datetime.now(timezone.utc)
    _password_resets_collection().update_many(
        {"user_id": str(user["_id"]), "used_at": None},
        {"$set": {"used_at": now, "invalidated_at": now}},
    )

    plain = secrets.token_urlsafe(32)
    token_hash = hash_refresh_token(plain)
    expires = now + timedelta(minutes=settings.password_reset_ttl_minutes)
    _password_resets_collection().insert_one(
        {
            "_id": f"rst_{uuid.uuid4().hex}",
            "user_id": str(user["_id"]),
            "token_hash": token_hash,
            "created_at": now,
            "expires_at": expires,
            "used_at": None,
        }
    )

    base = settings.password_reset_public_base_url.rstrip("/")
    sep = "&" if "?" in base else "?"
    reset_url = f"{base}{sep}{urlencode({'token': plain})}"
    send_password_reset_email(to_email=normalized, reset_url=reset_url)
    return {"success": True}


def reset_password_with_token(*, token: str, password: str) -> dict[str, Any]:
    """Consume a single-use reset token and set a new Argon2 password hash."""
    token_hash = hash_refresh_token(token.strip())
    resets = _password_resets_collection()
    row = resets.find_one({"token_hash": token_hash})
    if not row:
        return {
            "success": False,
            "code": "INVALID_RESET_TOKEN",
            "error": "Invalid or expired reset link",
        }
    if row.get("used_at") is not None:
        return {
            "success": False,
            "code": "RESET_TOKEN_USED",
            "error": "This reset link has already been used",
        }

    now = datetime.now(timezone.utc)
    expires_at = row.get("expires_at")
    if expires_at is None:
        return {
            "success": False,
            "code": "INVALID_RESET_TOKEN",
            "error": "Invalid or expired reset link",
        }
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < now:
        return {
            "success": False,
            "code": "RESET_TOKEN_EXPIRED",
            "error": "This reset link has expired",
        }

    consumed = resets.update_one(
        {"_id": row["_id"], "used_at": None},
        {"$set": {"used_at": now}},
    )
    if consumed.matched_count != 1:
        return {
            "success": False,
            "code": "RESET_TOKEN_USED",
            "error": "This reset link has already been used",
        }

    user_id = row["user_id"]
    user = get_user_by_id(user_id)
    if not user or user.get("status") != "active":
        return {
            "success": False,
            "code": "ACCOUNT_DISABLED",
            "error": "Account is disabled",
        }

    _users_collection().update_one(
        {"_id": user_id},
        {
            "$set": {
                "password_hash": hash_password(password),
                "updated_at": now,
                "password_changed_at": now,
            }
        },
    )
    # Force re-authentication on every device after a password change.
    revoke_all_sessions(user_id)
    return {"success": True}


def get_user_by_id(user_id: str) -> dict | None:
    return _users_collection().find_one({"_id": user_id})


def get_active_session(session_id: str, user_id: str) -> bool:
    """Validate the access token's owning, non-revoked refresh session."""
    now = datetime.now(timezone.utc)
    row = _sessions_collection().find_one({"_id": session_id, "user_id": user_id, "revoked_at": None})
    if not row:
        return False
    expiry = row.get("expires_at")
    if expiry is None:
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry > now


def refresh_session(*, refresh_token: str) -> dict[str, Any]:
    token_hash = hash_refresh_token(refresh_token)
    sessions = _sessions_collection()
    session = sessions.find_one({"token_hash": token_hash})
    if not session:
        return {"success": False, "code": "INVALID_REFRESH", "error": "Invalid refresh token"}
    if session.get("revoked_at") is not None:
        # Reuse of a rotated refresh token — revoke all active sessions for this user.
        if _is_recent_rotation(session):
            _record_concurrent_refresh(session)
            return {
                "success": False,
                "code": "REFRESH_RACE",
                "error": "A concurrent refresh already consumed this token",
            }
        _record_refresh_reuse(session)
        revoke_session_family(session)
        return {
            "success": False,
            "code": "REFRESH_REUSE",
            "error": "Refresh token reuse detected — sign in again",
        }
    now = datetime.now(timezone.utc)
    expires_at = session.get("expires_at")
    if expires_at is None:
        return {"success": False, "code": "EXPIRED_REFRESH", "error": "Refresh token expired"}
    if expires_at is not None:
        # PyMongo defaults to decoding BSON dates as naive UTC datetimes.
        # Normalize that representation before comparing with aware UTC now.
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < now:
            return {
                "success": False,
                "code": "EXPIRED_REFRESH",
                "error": "Refresh token expired",
            }

    user = get_user_by_id(session["user_id"])
    if not user or user.get("status") != "active":
        return {"success": False, "code": "ACCOUNT_DISABLED", "error": "Account is disabled"}

    # Rotate: revoke old, issue new refresh.
    rotation = sessions.update_one(
        {"_id": session["_id"], "revoked_at": None},
        {"$set": {"revoked_at": now, "rotated_at": now}},
    )
    # Rotation must be single use even when two refresh requests race. The
    # conditional update lets exactly one request consume this session.
    if rotation.matched_count != 1:
        current = sessions.find_one({"_id": session["_id"]}) or session
        if _is_recent_rotation(current):
            _record_concurrent_refresh(current)
            return {
                "success": False,
                "code": "REFRESH_RACE",
                "error": "A concurrent refresh already consumed this token",
            }
        _record_refresh_reuse(current)
        revoke_session_family(current)
        return {
            "success": False,
            "code": "REFRESH_REUSE",
            "error": "Refresh token reuse detected — sign in again",
        }
    settings = get_settings()
    refresh_plain = new_refresh_token_plain()
    refresh_hash = hash_refresh_token(refresh_plain)
    session_id = str(uuid.uuid4())
    expires = now + timedelta(days=settings.refresh_token_ttl_days)
    sessions.insert_one(
        {
            "_id": session_id,
            "user_id": str(user["_id"]),
            "token_hash": refresh_hash,
            "created_at": now,
            "expires_at": expires,
            "revoked_at": None,
            "rotated_from": session["_id"],
            "family_id": session.get("family_id") or session["_id"],
        }
    )
    # A competing worker may have recorded reuse before this child was
    # inserted. Recheck the parent after insertion so that ordering cannot
    # leave an unrevoked child behind.
    parent_after_insert = sessions.find_one(
        {"_id": session["_id"]}, {"reuse_detected_at": 1}
    )
    if parent_after_insert and parent_after_insert.get("reuse_detected_at"):
        revoke_session_family(session)
        return {
            "success": False,
            "code": "REFRESH_REUSE",
            "error": "Refresh token reuse detected; sign in again",
        }
    access = create_access_token(user_id=str(user["_id"]), email=user["email"], session_id=session_id, tenant_id=user.get("tenant_id"))
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


def revoke_session_id(session_id: str, user_id: str) -> bool:
    result = _sessions_collection().update_one(
        {"_id": session_id, "user_id": user_id, "revoked_at": None},
        {"$set": {"revoked_at": datetime.now(timezone.utc)}},
    )
    return result.matched_count > 0


def revoke_all_sessions(user_id: str) -> int:
    now = datetime.now(timezone.utc)
    result = _sessions_collection().update_many(
        {"user_id": user_id, "revoked_at": None},
        {"$set": {"revoked_at": now}},
    )
    return int(result.modified_count)


def revoke_session_family(session: dict) -> int:
    """Revoke only a session lineage, preserving other devices' sessions."""
    family_id = str(session.get("family_id") or session.get("_id") or "")
    if not family_id:
        return 0
    result = _sessions_collection().update_many(
        {
            "$or": [{"family_id": family_id}, {"_id": family_id}],
            "revoked_at": None,
        },
        {"$set": {"revoked_at": datetime.now(timezone.utc)}},
    )
    return int(result.modified_count)


def _record_refresh_reuse(session: dict) -> None:
    """Record token replay without ever persisting the presented secret."""
    session_id = session.get("_id")
    if not session_id:
        return
    _sessions_collection().update_one(
        {"_id": session_id},
        {
            "$set": {"reuse_detected_at": datetime.now(timezone.utc)},
            "$inc": {"reuse_count": 1},
        },
    )


def _is_recent_rotation(session: dict) -> bool:
    rotated_at = session.get("rotated_at")
    if not isinstance(rotated_at, datetime):
        return False
    if rotated_at.tzinfo is None:
        rotated_at = rotated_at.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - rotated_at).total_seconds()
    return 0 <= age <= _REFRESH_RACE_GRACE_SECONDS


def _record_concurrent_refresh(session: dict) -> None:
    """Record a rejected in-flight duplicate without invalidating its winner."""
    session_id = session.get("_id")
    if not session_id:
        return
    _sessions_collection().update_one(
        {"_id": session_id},
        {
            "$set": {"last_concurrent_refresh_at": datetime.now(timezone.utc)},
            "$inc": {"concurrent_refresh_count": 1},
        },
    )
