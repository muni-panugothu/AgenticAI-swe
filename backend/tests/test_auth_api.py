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
        patcher_active = patch("app.services.auth_service.get_active_session", return_value=True)
        self.addCleanup(patcher_u.stop)
        self.addCleanup(patcher_s.stop)
        self.addCleanup(patcher_idx.stop)
        self.addCleanup(patcher_active.stop)
        patcher_u.start()
        patcher_s.start()
        patcher_idx.start()
        patcher_active.start()

    def test_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def _register_payload(self, **overrides):
        payload = {
            "display_name": "Alice",
            "email": "alice@example.com",
            "password": "Secret1pass",
            "confirm_password": "Secret1pass",
            "accepted_terms": True,
            "accepted_privacy": True,
        }
        payload.update(overrides)
        return payload

    def test_register_success(self):
        from datetime import datetime, timezone

        doc = {
            "_id": "usr_test1",
            "email": "alice@example.com",
            "status": "active",
            "tenant_id": "ten_1",
            "display_name": "Alice",
            "created_at": datetime.now(timezone.utc),
            "terms_version": "2026-11-01",
            "privacy_version": "2026-11-01",
            "terms_accepted_at": datetime.now(timezone.utc),
            "privacy_accepted_at": datetime.now(timezone.utc),
        }
        self.users.find_one.side_effect = [None, doc]
        res = self.client.post("/auth/register", json=self._register_payload())
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["user"]["email"], "alice@example.com")

    def test_register_requires_legal_acceptance(self):
        res = self.client.post(
            "/auth/register",
            json=self._register_payload(accepted_terms=False, accepted_privacy=True),
        )
        self.assertEqual(res.status_code, 422)

    def test_register_duplicate(self):
        self.users.find_one.return_value = {"_id": "usr_x", "email": "alice@example.com"}
        res = self.client.post("/auth/register", json=self._register_payload())
        self.assertEqual(res.status_code, 409)

    def test_register_weak_password(self):
        res = self.client.post(
            "/auth/register",
            json=self._register_payload(password="short", confirm_password="short"),
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
            json={
                "email": "alice@example.com",
                "password": "Secret1pass",
                "remember_me": True,
            },
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
            session_id="sess_test",
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

    def test_logout_with_refresh_body_only(self):
        from app.services import auth_service

        with patch.object(
            auth_service, "revoke_refresh_token", return_value=True
        ) as revoke:
            res = self.client.post(
                "/auth/logout",
                json={"refresh_token": "opaque-refresh-token"},
            )
        self.assertEqual(res.status_code, 200, res.text)
        revoke.assert_called_once_with("opaque-refresh-token")

    def test_refresh_without_token_is_rejected(self):
        res = self.client.post("/auth/refresh", json={})
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["detail"]["code"], "MISSING_REFRESH")

    def test_bearer_logout_revokes_the_bound_access_session(self):
        from app.core.security import create_access_token
        from app.services import auth_service

        token = create_access_token(
            user_id="usr_alice", email="alice@example.com", tenant_id="ten_1", session_id="sess_logout"
        )
        with patch.object(auth_service, "revoke_session_id", return_value=True) as revoke:
            res = self.client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200, res.text)
        revoke.assert_called_once_with("sess_logout", "usr_alice")

    def test_forgot_password_always_generic(self):
        from app.services import auth_service

        with patch.object(
            auth_service, "request_password_reset", return_value={"success": True}
        ) as req:
            res = self.client.post(
                "/auth/forgot-password",
                json={"email": "nobody@example.com"},
            )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertTrue(res.json()["success"])
        req.assert_called_once()

    def test_reset_password_rejects_bad_token(self):
        from app.services import auth_service

        with patch.object(
            auth_service,
            "reset_password_with_token",
            return_value={
                "success": False,
                "code": "INVALID_RESET_TOKEN",
                "error": "Invalid or expired reset link",
            },
        ):
            res = self.client.post(
                "/auth/reset-password",
                json={
                    "token": "a" * 32,
                    "password": "Secret1pass",
                    "confirm_password": "Secret1pass",
                },
            )
        self.assertEqual(res.status_code, 400)

    def test_legal_versions(self):
        res = self.client.get("/auth/legal-versions")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("terms_version", body)
        self.assertIn("privacy_version", body)


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
        with patch("app.services.auth_service._users_collection", lambda: users), patch(
            "app.services.auth_service.get_active_session", return_value=True
        ):
            token = create_access_token(
                user_id="usr_alice",
                email="alice@example.com",
                tenant_id="ten_1",
            session_id="sess_test",
            )
            res = TestClient(app).post(
                "/api/chat",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": "hi", "user_id": "imposter"},
            )
        # extra=forbid on ChatRequest — should 422, not impersonate
        self.assertIn(res.status_code, {400, 422})


class TestChatRagRouting(unittest.TestCase):
    def setUp(self):
        from app.core.security import create_access_token

        self.client = TestClient(app)

        user = {
            "_id": "usr_rag_test",
            "email": "ragtest@example.com",
            "status": "active",
            "tenant_id": "ten_rag_test",
            "display_name": "ragtest",
        }

        self.user_patcher = patch(
            "app.api.deps.get_user_by_id",
            return_value=user,
        )
        self.session_patcher = patch(
            "app.api.deps.auth_service.get_active_session",
            return_value=True,
        )

        self.user_patcher.start()
        self.session_patcher.start()

        self.addCleanup(self.user_patcher.stop)
        self.addCleanup(self.session_patcher.stop)

        # Phase B: Redis chat rate limit — mock for unit isolation.
        self.rate_patcher = patch("app.api.routes.chat.enforce_chat_rate_limit")
        self.rate_patcher.start()
        self.addCleanup(self.rate_patcher.stop)

        token = create_access_token(
            user_id="usr_rag_test",
            email="ragtest@example.com",
            tenant_id="ten_rag_test",
            session_id="sess_rag_test",
        )

        self.headers = {"Authorization": f"Bearer {token}"}

    def _request(self):
        return self.client.post(
            "/api/chat",
            headers=self.headers,
            json={"message": "Explain the file memory_service.py"},
        )

    def test_no_readable_documents_continues_normal_chat(self):
        rag_result = {
            "has_readable_documents": False,
            "candidates": [],
            "context": "",
            "citations": [],
        }

        with patch(
            "app.api.routes.chat.retrieve",
            return_value=rag_result,
        ), patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": None},
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            return_value="Here is how memory_service.py works.",
        ) as mock_chat:
            response = self._request()

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertEqual(
            body["reply"],
            "Here is how memory_service.py works.",
        )
        self.assertTrue(body["ragSkipped"])
        self.assertFalse(body["contextUsed"])
        self.assertEqual(body["citations"], [])

        mock_chat.assert_called_once()
        self.assertIsNone(mock_chat.call_args.kwargs["rag_context"])

    def test_readable_documents_without_candidates_abstains(self):
        rag_result = {
            "has_readable_documents": True,
            "candidates": [],
            "context": "",
            "citations": [],
        }

        with patch(
            "app.api.routes.chat.retrieve",
            return_value=rag_result,
        ), patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": None},
        ), patch(
            "app.api.routes.chat.chat_with_ai",
        ) as mock_chat:
            response = self._request()

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertIn("couldn't find authorized document evidence", body["reply"])
        self.assertFalse(body["contextUsed"])
        self.assertFalse(body["ragSkipped"])
        self.assertEqual(body["citations"], [])

        mock_chat.assert_not_called()

    def test_relevant_candidates_continue_with_rag_context(self):
        citations = [
            {
                "document_id": "doc_test_1",
                "filename": "memory_service.py",
            }
        ]

        rag_result = {
            "has_readable_documents": True,
            "candidates": [{"document_id": "doc_test_1"}],
            "context": "The memory service stores conversation memories.",
            "citations": citations,
        }

        with patch(
            "app.api.routes.chat.retrieve",
            return_value=rag_result,
        ), patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": None},
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            return_value="The file stores conversation memories.",
        ) as mock_chat:
            response = self._request()

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertEqual(
            body["reply"],
            "The file stores conversation memories.",
        )
        self.assertTrue(body["contextUsed"])
        self.assertFalse(body["ragSkipped"])
        self.assertEqual(body["citations"], citations)

        mock_chat.assert_called_once()
        self.assertEqual(
            mock_chat.call_args.kwargs["rag_context"],
            rag_result["context"],
        )

    def test_chat_passes_history_to_retrieve_without_changing_acl_args(self):
        rag_result = {
            "has_readable_documents": True,
            "candidates": [{"document_id": "doc_test_1"}],
            "context": "evidence",
            "citations": [],
        }
        history = [
            {"role": "user", "content": "What does the leave policy say?"},
            {"role": "assistant", "content": "It covers vacation days."},
        ]

        with patch(
            "app.api.routes.chat.retrieve",
            return_value=rag_result,
        ) as mock_retrieve, patch(
            "app.services.memory_scope_service.get_active_project_scope",
            return_value={"project_id": "proj-rag"},
        ), patch(
            "app.api.routes.chat.chat_with_ai",
            return_value="ok",
        ):
            response = self.client.post(
                "/api/chat",
                headers=self.headers,
                json={
                    "message": "According to the document, what about its notice period?",
                    "history": history,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        mock_retrieve.assert_called_once()
        args, kwargs = mock_retrieve.call_args
        self.assertEqual(args[0], "According to the document, what about its notice period?")
        self.assertEqual(kwargs["user_id"], "usr_rag_test")
        self.assertEqual(kwargs["tenant_id"], "ten_rag_test")
        self.assertEqual(kwargs["project_id"], "proj-rag")
        self.assertEqual(kwargs["mode"], "hybrid")
        self.assertEqual(kwargs["conversation_history"], history)