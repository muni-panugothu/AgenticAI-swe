"""Authentication API tests (FastAPI TestClient)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure JWT is configured before app import.
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402


class TestAuthApi(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-32chars-min!!"
        self.client = TestClient(app)
        self.users = MagicMock()
        self.sessions = MagicMock()
        self.users.find_one.return_value = None
        self.users.insert_one.return_value = MagicMock(inserted_id="usr_test1")
        self.sessions.insert_one.return_value = MagicMock(inserted_id="sess1")
        self.sessions.find_one.return_value = None
        self.sessions.update_one.return_value = MagicMock(matched_count=1)

        def users_coll():
            return self.users

        def sessions_coll():
            return self.sessions

        patcher_u = patch("app.services.auth_service._users_collection", users_coll)
        patcher_s = patch("app.services.auth_service._sessions_collection", sessions_coll)
        patcher_idx = patch("app.services.auth_service.ensure_auth_indexes")
        self.addCleanup(patcher_u.stop)
        self.addCleanup(patcher_s.stop)
        self.addCleanup(patcher_idx.stop)
        patcher_u.start()
        patcher_s.start()
        patcher_idx.start()

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_register_success(self):
        from datetime import datetime, timezone

        doc = {
            "_id": "usr_test1",
            "email": "alice@example.com",
            "status": "active",
            "tenant_id": "ten_1",
            "display_name": "alice",
            "created_at": datetime.now(timezone.utc),
        }
        self.users.find_one.side_effect = [None, doc]
        res = self.client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "Secret1pass"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["user"]["email"], "alice@example.com")

    def test_register_duplicate(self):
        self.users.find_one.return_value = {"_id": "usr_x", "email": "alice@example.com"}
        res = self.client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "Secret1pass"},
        )
        self.assertEqual(res.status_code, 409)

    def test_register_weak_password(self):
        res = self.client.post(
            "/auth/register",
            json={"email": "weak@example.com", "password": "short"},
        )
        self.assertEqual(res.status_code, 422)

    def test_login_success(self):
        from app.core.security import hash_password

        self.users.find_one.return_value = {
            "_id": "usr_alice",
            "email": "alice@example.com",
            "password_hash": hash_password("Secret1pass"),
            "status": "active",
            "tenant_id": "ten_1",
            "display_name": "alice",
            "created_at": None,
        }
        res = self.client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "Secret1pass"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("access_token", res.json())

    def test_login_invalid_password(self):
        from app.core.security import hash_password

        self.users.find_one.return_value = {
            "_id": "usr_alice",
            "email": "alice@example.com",
            "password_hash": hash_password("Secret1pass"),
            "status": "active",
        }
        res = self.client.post(
            "/auth/login",
            json={"email": "alice@example.com", "password": "WrongPass1"},
        )
        self.assertEqual(res.status_code, 401)

    def test_chat_requires_auth(self):
        res = self.client.post(
            "/api/chat",
            json={"message": "hello"},
        )
        self.assertEqual(res.status_code, 401)

    def test_me_with_token(self):
        from app.core.security import create_access_token

        token = create_access_token(
            user_id="usr_alice",
            email="alice@example.com",
            tenant_id="ten_1",
        )
        self.users.find_one.return_value = {
            "_id": "usr_alice",
            "email": "alice@example.com",
            "status": "active",
            "tenant_id": "ten_1",
            "display_name": "alice",
            "created_at": None,
        }
        res = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["id"], "usr_alice")


class TestCliIdentityRejection(unittest.TestCase):
    def test_chat_rejects_untrusted_user_id_fields(self):
        from unittest.mock import MagicMock, patch

        from app.core.security import create_access_token

        users = MagicMock()
        users.find_one.return_value = {
            "_id": "usr_alice",
            "email": "alice@example.com",
            "status": "active",
            "tenant_id": "ten_1",
        }
        with patch("app.services.auth_service._users_collection", lambda: users):
            token = create_access_token(
                user_id="usr_alice",
                email="alice@example.com",
                tenant_id="ten_1",
            )
            res = TestClient(app).post(
                "/api/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "hi", "user_id": "imposter"},
            )
        # extra=forbid on ChatRequest — should 422, not impersonate
        self.assertIn(res.status_code, {400, 422})
