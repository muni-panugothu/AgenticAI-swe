"""Refresh rotation, reuse detection, and logout body refresh."""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.core.security import hash_refresh_token, new_refresh_token_plain  # noqa: E402
from app.core.security import decode_access_token  # noqa: E402
from app.services import auth_service  # noqa: E402


class TestRefreshLifecycle(unittest.TestCase):
    def setUp(self):
        self.sessions = MagicMock()
        self.users = MagicMock()

        def users_coll():
            return self.users

        def sessions_coll():
            return self.sessions

        patcher_u = patch("app.services.auth_service._users_collection", users_coll)
        patcher_s = patch("app.services.auth_service._sessions_collection", sessions_coll)
        self.addCleanup(patcher_u.stop)
        self.addCleanup(patcher_s.stop)
        patcher_u.start()
        patcher_s.start()

    def test_refresh_rotation_returns_new_tokens(self):
        plain = new_refresh_token_plain()
        now = datetime.now(timezone.utc)
        self.sessions.find_one.return_value = {
            "_id": "sess1",
            "user_id": "usr_a",
            "token_hash": hash_refresh_token(plain),
            "revoked_at": None,
            "expires_at": now + timedelta(days=1),
        }
        self.users.find_one.return_value = {
            "_id": "usr_a",
            "email": "a@test.com",
            "status": "active",
            "tenant_id": "ten_1",
        }
        self.sessions.update_one.return_value.matched_count = 1

        out = auth_service.refresh_session(refresh_token=plain)
        self.assertTrue(out.get("success"))
        self.assertIn("access_token", out)
        self.assertIn("refresh_token", out)
        self.assertEqual(decode_access_token(out["access_token"])["sub"], "usr_a")
        self.assertTrue(decode_access_token(out["access_token"])["sid"])
        self.assertEqual(out["user"].id, "usr_a")
        self.sessions.update_one.assert_called_once()

    def test_access_token_requires_issuer_audience_and_session_claims(self):
        from app.core.security import create_access_token

        token = create_access_token(user_id="usr_a", email="a@test.com", session_id="sess_a")
        claims = decode_access_token(token)
        self.assertEqual(claims["iss"], "agentic-ai-api")
        self.assertEqual(claims["aud"], "agentic-ai-cli")
        self.assertEqual(claims["sid"], "sess_a")

    def test_expired_refresh_record_without_expiry_is_rejected(self):
        plain = new_refresh_token_plain()
        self.sessions.find_one.return_value = {
            "_id": "sess_missing_expiry", "user_id": "usr_a",
            "token_hash": hash_refresh_token(plain), "revoked_at": None,
        }
        result = auth_service.refresh_session(refresh_token=plain)
        self.assertFalse(result["success"])
        self.assertEqual(result["code"], "EXPIRED_REFRESH")

    def test_unknown_refresh_token_is_rejected(self):
        self.sessions.find_one.return_value = None
        result = auth_service.refresh_session(refresh_token=new_refresh_token_plain())
        self.assertFalse(result["success"])
        self.assertEqual(result["code"], "INVALID_REFRESH")

    def test_access_session_must_belong_to_user_and_be_active(self):
        now = datetime.now(timezone.utc)
        self.sessions.find_one.return_value = {
            "_id": "sess_a", "user_id": "usr_a", "revoked_at": None,
            "expires_at": now + timedelta(days=1),
        }
        self.assertTrue(auth_service.get_active_session("sess_a", "usr_a"))
        self.sessions.find_one.assert_called_with({"_id": "sess_a", "user_id": "usr_a", "revoked_at": None})
        self.sessions.find_one.return_value = None
        self.assertFalse(auth_service.get_active_session("sess_a", "usr_b"))

    def test_refresh_accepts_future_mongo_naive_utc_expiry(self):
        plain = new_refresh_token_plain()
        self.sessions.find_one.return_value = {
            "_id": "sess_naive_future",
            "user_id": "usr_a",
            "token_hash": hash_refresh_token(plain),
            "revoked_at": None,
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(days=1)
            ).replace(tzinfo=None),
        }
        self.users.find_one.return_value = {
            "_id": "usr_a",
            "email": "a@test.com",
            "status": "active",
            "tenant_id": "ten_1",
        }
        self.sessions.update_one.return_value.matched_count = 1

        out = auth_service.refresh_session(refresh_token=plain)

        self.assertTrue(out.get("success"), out)
        self.assertEqual(decode_access_token(out["access_token"])["sub"], "usr_a")

    def test_expired_mongo_naive_utc_expiry_returns_expired_code(self):
        plain = new_refresh_token_plain()
        self.sessions.find_one.return_value = {
            "_id": "sess_naive_expired",
            "user_id": "usr_a",
            "token_hash": hash_refresh_token(plain),
            "revoked_at": None,
            "expires_at": (
                datetime.now(timezone.utc) - timedelta(days=1)
            ).replace(tzinfo=None),
        }

        out = auth_service.refresh_session(refresh_token=plain)

        self.assertFalse(out.get("success"))
        self.assertEqual(out.get("code"), "EXPIRED_REFRESH")

    def test_revoked_refresh_triggers_reuse_detection(self):
        plain = new_refresh_token_plain()
        self.sessions.find_one.return_value = {
            "_id": "sess1",
            "user_id": "usr_a",
            "token_hash": hash_refresh_token(plain),
            "revoked_at": datetime.now(timezone.utc),
        }
        self.sessions.update_many.return_value = MagicMock(modified_count=2)

        out = auth_service.refresh_session(refresh_token=plain)
        self.assertFalse(out.get("success"))
        self.assertEqual(out.get("code"), "REFRESH_REUSE")
        self.sessions.update_many.assert_called_once()

    def test_losing_concurrent_rotation_revokes_active_family(self):
        plain = new_refresh_token_plain()
        self.sessions.find_one.return_value = {
            "_id": "sess_race", "user_id": "usr_a",
            "token_hash": hash_refresh_token(plain), "revoked_at": None,
            "expires_at": datetime.now(timezone.utc) + timedelta(days=1),
        }
        self.users.find_one.return_value = {
            "_id": "usr_a", "email": "a@test.com", "status": "active",
        }
        self.sessions.update_one.return_value.matched_count = 0
        result = auth_service.refresh_session(refresh_token=plain)
        self.assertFalse(result["success"])
        self.assertEqual(result["code"], "REFRESH_REUSE")
        self.sessions.update_many.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
