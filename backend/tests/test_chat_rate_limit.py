"""Phase B — authenticated /api/chat per-user Redis rate limiting."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch  # noqa: F401 — MagicMock used below

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.security import create_access_token
from app.main import app
from app.services.chat_rate_limit_service import (
    ChatRateLimitExceeded,
    chat_rate_limit_key,
    enforce_chat_rate_limit,
)


class TestChatRateLimitService(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-32chars-min!!"
        os.environ["CHAT_RATE_LIMIT_COUNT"] = "3"
        os.environ["CHAT_RATE_LIMIT_WINDOW_SECONDS"] = "60"
        get_settings.cache_clear()

    def tearDown(self):
        get_settings.cache_clear()

    def test_key_is_stable_per_user(self):
        self.assertEqual(
            chat_rate_limit_key("usr_a"),
            chat_rate_limit_key(" usr_a "),
        )
        self.assertNotEqual(
            chat_rate_limit_key("usr_a"),
            chat_rate_limit_key("usr_b"),
        )

    def test_empty_user_rejected(self):
        with self.assertRaises(ValueError):
            enforce_chat_rate_limit("  ")

    def test_exceeded_when_count_over_limit(self):
        get_settings.cache_clear()
        with patch(
            "app.services.chat_rate_limit_service.redis"
        ) as mock_redis, patch(
            "app.services.chat_rate_limit_service.get_settings",
            return_value=MagicMock(
                chat_rate_limit_count=3,
                chat_rate_limit_window_seconds=60,
            ),
        ):
            mock_redis.eval.return_value = [4, 42]
            with self.assertRaises(ChatRateLimitExceeded) as ctx:
                enforce_chat_rate_limit("usr_limit")
            self.assertEqual(ctx.exception.retry_after_seconds, 42)
            mock_redis.eval.assert_called_once()
            call_kwargs = mock_redis.eval.call_args.kwargs
            self.assertEqual(
                call_kwargs.get("keys"),
                [chat_rate_limit_key("usr_limit")],
            )

    def test_allowed_under_limit(self):
        with patch(
            "app.services.chat_rate_limit_service.redis"
        ) as mock_redis:
            mock_redis.eval.return_value = [2, 55]
            enforce_chat_rate_limit("usr_ok")


class TestChatRateLimitApi(unittest.TestCase):
    def setUp(self):
        get_settings.cache_clear()
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-32chars-min!!"
        get_settings.cache_clear()
        self.client = TestClient(app)

        def _user(uid: str, email: str):
            return {
                "_id": uid,
                "email": email,
                "status": "active",
                "tenant_id": "ten_1",
                "display_name": email.split("@")[0],
            }

        self.users = {
            "usr_alice": _user("usr_alice", "alice@example.com"),
            "usr_bob": _user("usr_bob", "bob@example.com"),
        }

        def get_user(uid):
            return self.users.get(uid)

        self.patchers = [
            patch("app.api.deps.get_user_by_id", side_effect=get_user),
            patch("app.api.deps.auth_service.get_active_session", return_value=True),
            patch("app.api.routes.chat.chat_with_ai", return_value="ok"),
            patch(
                "app.api.routes.chat.is_document_question",
                return_value=False,
            ),
        ]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)

    def _token(self, user_id: str, email: str) -> str:
        return create_access_token(
            user_id=user_id,
            email=email,
            tenant_id="ten_1",
            session_id=f"sess_{user_id}",
        )

    def _chat(self, user_id: str, email: str, *, extra_json: dict | None = None):
        body = {"message": "hello from rate limit test"}
        if extra_json:
            body.update(extra_json)
        return self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {self._token(user_id, email)}"},
            json=body,
        )

    def test_same_user_shares_bucket_and_429(self):
        calls: list[str] = []

        def enforce(uid: str):
            calls.append(uid)
            if calls.count(uid) > 2:
                raise ChatRateLimitExceeded(17)

        with patch(
            "app.api.routes.chat.enforce_chat_rate_limit",
            side_effect=enforce,
        ):
            self.assertEqual(self._chat("usr_alice", "alice@example.com").status_code, 200)
            self.assertEqual(self._chat("usr_alice", "alice@example.com").status_code, 200)
            res = self._chat("usr_alice", "alice@example.com")
        self.assertEqual(res.status_code, 429)
        self.assertEqual(res.headers.get("Retry-After"), "17")
        self.assertEqual(res.json()["detail"]["code"], "CHAT_RATE_LIMIT_EXCEEDED")
        self.assertEqual(calls, ["usr_alice", "usr_alice", "usr_alice"])

    def test_different_users_independent_buckets(self):
        counts: dict[str, int] = {}

        def enforce(uid: str):
            counts[uid] = counts.get(uid, 0) + 1
            if counts[uid] > 1:
                raise ChatRateLimitExceeded(9)

        with patch(
            "app.api.routes.chat.enforce_chat_rate_limit",
            side_effect=enforce,
        ):
            self.assertEqual(self._chat("usr_alice", "alice@example.com").status_code, 200)
            self.assertEqual(self._chat("usr_bob", "bob@example.com").status_code, 200)
            # Alice's second request trips only Alice's bucket.
            res = self._chat("usr_alice", "alice@example.com")
        self.assertEqual(res.status_code, 429)
        self.assertEqual(counts["usr_alice"], 2)
        self.assertEqual(counts["usr_bob"], 1)

    def test_body_user_id_rejected_and_does_not_change_identity(self):
        seen: list[str] = []

        def enforce(uid: str):
            seen.append(uid)

        with patch(
            "app.api.routes.chat.enforce_chat_rate_limit",
            side_effect=enforce,
        ):
            # extra=forbid → 422 before handler; rate limit must not see attacker id.
            res = self.client.post(
                "/api/chat",
                headers={
                    "Authorization": f"Bearer {self._token('usr_alice', 'alice@example.com')}"
                },
                json={"message": "hi", "user_id": "usr_attacker"},
            )
        self.assertIn(res.status_code, {400, 422})
        # Rate limit must not be charged under the attacker's forged id.
        self.assertEqual(seen, [])
        self.assertNotIn("usr_attacker", seen)

    def test_rate_limit_uses_jwt_user_not_header_spoof(self):
        seen: list[str] = []

        def enforce(uid: str):
            seen.append(uid)

        with patch(
            "app.api.routes.chat.enforce_chat_rate_limit",
            side_effect=enforce,
        ):
            res = self.client.post(
                "/api/chat",
                headers={
                    "Authorization": f"Bearer {self._token('usr_alice', 'alice@example.com')}",
                    "X-User-Id": "usr_attacker",
                },
                json={"message": "hi"},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(seen, ["usr_alice"])

    def test_redis_failure_returns_503(self):
        with patch(
            "app.api.routes.chat.enforce_chat_rate_limit",
            side_effect=RuntimeError("redis down"),
        ), patch("app.api.routes.chat.chat_with_ai") as mock_chat:
            res = self._chat("usr_alice", "alice@example.com")
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["detail"]["code"], "CHAT_RATE_LIMIT_UNAVAILABLE")
        mock_chat.assert_not_called()

    def test_unauthenticated_still_401(self):
        res = self.client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(res.status_code, 401)


class TestGroqClientConfig(unittest.TestCase):
    def test_client_has_explicit_timeout_and_bounded_retries(self):
        from app import chatbot

        self.assertEqual(chatbot.GROQ_TIMEOUT_SECONDS, chatbot._groq_timeout_seconds())
        self.assertGreater(chatbot.GROQ_TIMEOUT_SECONDS, 0)
        self.assertGreaterEqual(chatbot.GROQ_MAX_RETRIES, 0)
        # Installed SDK exposes these on the client / http options.
        timeout = getattr(chatbot.client, "timeout", None)
        self.assertIsNotNone(timeout)
        # max_retries should match configured bound (default 0).
        self.assertEqual(
            getattr(chatbot.client, "max_retries", None),
            chatbot.GROQ_MAX_RETRIES,
        )

    def test_settings_validate_groq_and_chat_limits(self):
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-32chars-min!!"
        os.environ["GROQ_TIMEOUT_SECONDS"] = "45"
        os.environ["GROQ_MAX_RETRIES"] = "0"
        os.environ["CHAT_RATE_LIMIT_COUNT"] = "20"
        get_settings.cache_clear()
        settings = get_settings()
        settings.validate_startup()
        self.assertEqual(settings.groq_timeout_seconds, 45.0)
        self.assertEqual(settings.groq_max_retries, 0)
        self.assertEqual(settings.chat_rate_limit_count, 20)
        os.environ["GROQ_TIMEOUT_SECONDS"] = "60"
        get_settings.cache_clear()

    def test_invalid_timeout_rejected_at_startup(self):
        os.environ["JWT_SECRET"] = "test-jwt-secret-key-32chars-min!!"
        os.environ["GROQ_TIMEOUT_SECONDS"] = "0"
        get_settings.cache_clear()
        with self.assertRaises(ValueError):
            get_settings().validate_startup()
        os.environ["GROQ_TIMEOUT_SECONDS"] = "60"
        get_settings.cache_clear()

    def test_timeout_errors_abort_without_sdk_style_retry(self):
        """Timeouts enter the safe provider-failure path (abort, no TPM sleep)."""
        from groq import APITimeoutError

        from app.services.agent_retry import (
            ACTION_ABORT,
            ERROR_UNKNOWN,
            classify_provider_error,
            decide_llm_retry,
        )

        # Request object is unused by decide_llm_retry; keep message key-free.
        exc = APITimeoutError(request=MagicMock())
        self.assertEqual(classify_provider_error(exc), ERROR_UNKNOWN)
        decision = decide_llm_retry(
            exc=exc,
            tpm_retries=0,
            tpm_retry_limit=2,
            tools_enabled=True,
            has_evidence=True,
        )
        self.assertEqual(decision.action, ACTION_ABORT)
        self.assertEqual(decision.error_class, ERROR_UNKNOWN)
        self.assertEqual(decision.wait_seconds, 0.0)
        # No credential material in the exception string representation.
        self.assertNotIn("api_key", str(exc).lower())
        self.assertNotIn("gsk_", str(exc).lower())


class TestDocumentUploadScopeInWorker(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI

        from app.api.deps import AuthenticatedUser, get_current_user
        from app.api.routes.documents import router

        app = FastAPI()
        app.include_router(router)

        async def override_user():
            return AuthenticatedUser(
                id="owner-1",
                email="owner@example.com",
                tenant_id="ten-1",
                status="active",
            )

        app.dependency_overrides[get_current_user] = override_user
        self.client = TestClient(app)

    def test_project_scope_resolved_inside_ingest_worker(self):
        from app.api.routes import documents as docs_mod

        scope_calls: list[str] = []
        ingest_calls: list[dict] = []

        def fake_scope(uid: str):
            scope_calls.append(uid)
            return "proj-worker"

        def fake_ingest(**kwargs):
            ingest_calls.append(kwargs)
            return {"document_id": "doc1"}

        with patch.object(docs_mod, "_project_scope", side_effect=fake_scope), patch.object(
            docs_mod, "ingest_document", side_effect=fake_ingest
        ), patch.object(docs_mod, "enforce_upload_rate_limit"):
            res = self.client.post(
                "/api/documents/upload",
                files={"file": ("a.txt", b"hello", "text/plain")},
            )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(scope_calls, ["owner-1"])
        self.assertEqual(len(ingest_calls), 1)
        self.assertEqual(ingest_calls[0]["owner_id"], "owner-1")
        self.assertEqual(ingest_calls[0]["tenant_id"], "ten-1")
        self.assertEqual(ingest_calls[0]["project_id"], "proj-worker")

    def test_ingest_helper_binds_authenticated_identity(self):
        from app.api.routes.documents import _ingest_document_for_user

        with patch(
            "app.api.routes.documents._project_scope",
            return_value="p1",
        ) as scope, patch(
            "app.api.routes.documents.ingest_document",
            return_value={"ok": True},
        ) as ingest:
            out = _ingest_document_for_user(
                owner_id="u1",
                tenant_id="t1",
                filename="f.txt",
                content_type="text/plain",
                raw=b"x",
            )
        self.assertEqual(out, {"ok": True})
        scope.assert_called_once_with("u1")
        ingest.assert_called_once()
        self.assertEqual(ingest.call_args.kwargs["project_id"], "p1")
        self.assertEqual(ingest.call_args.kwargs["owner_id"], "u1")


if __name__ == "__main__":
    unittest.main()
