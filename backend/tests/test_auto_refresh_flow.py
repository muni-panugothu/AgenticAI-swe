"""Isolated end-to-end access-token refresh tests.

All auth users and sessions live in in-memory collections. The only test that
uses the real Windows Credential Manager writes a unique temporary test user
credential and removes it in a finally block.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import httpx
import jwt
from fastapi.testclient import TestClient

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

import cli_auth  # noqa: E402
from app.api.routes import chat as chat_route  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services import auth_service  # noqa: E402

_TEST_JWT_SECRET = "isolated-refresh-test-secret-32-chars-minimum"


class _MemoryCollection:
    """Small thread-safe collection fake for auth-only route tests."""

    def __init__(self):
        self.rows: list[dict] = []
        self.lock = threading.RLock()

    @staticmethod
    def _matches(row: dict, query: dict) -> bool:
        for key, expected in query.items():
            actual = row.get(key)
            if isinstance(expected, dict):
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$exists" in expected and ((key in row) != expected["$exists"]):
                    return False
            elif expected is None:
                if actual is not None:
                    return False
            elif actual != expected:
                return False
        return True

    def find_one(self, query: dict, projection: dict | None = None):
        del projection
        with self.lock:
            for row in self.rows:
                if self._matches(row, query):
                    return copy.deepcopy(row)
        return None

    def insert_one(self, row: dict):
        with self.lock:
            self.rows.append(copy.deepcopy(row))
        return MagicMock(inserted_id=row.get("_id"))

    def update_one(self, query: dict, update: dict):
        with self.lock:
            for row in self.rows:
                if self._matches(row, query):
                    row.update(copy.deepcopy(update.get("$set", {})))
                    return MagicMock(matched_count=1, modified_count=1)
        return MagicMock(matched_count=0, modified_count=0)

    def update_many(self, query: dict, update: dict):
        matched = 0
        with self.lock:
            for row in self.rows:
                if self._matches(row, query):
                    row.update(copy.deepcopy(update.get("$set", {})))
                    matched += 1
        return MagicMock(matched_count=matched, modified_count=matched)


@unittest.skipUnless(sys.platform == "win32", "Requires Windows Credential Manager")
class TestAutomaticRefreshFlow(unittest.TestCase):
    def setUp(self):
        self._env_patcher = patch.dict(
            os.environ,
            {
                "JWT_SECRET": _TEST_JWT_SECRET,
                "JWT_ISSUER": "isolated-refresh-test-issuer",
                "JWT_AUDIENCE": "isolated-refresh-test-audience",
            },
        )
        self._env_patcher.start()
        get_settings.cache_clear()
        self.settings = get_settings()
        self.original_access_ttl = self.settings.access_token_ttl_minutes
        self.settings.access_token_ttl_minutes = 5 / 60

        self.users = _MemoryCollection()
        self.sessions = _MemoryCollection()
        self._users_patcher = patch(
            "app.services.auth_service._users_collection", lambda: self.users
        )
        self._sessions_patcher = patch(
            "app.services.auth_service._sessions_collection", lambda: self.sessions
        )
        self._indexes_patcher = patch("app.services.auth_service.ensure_auth_indexes")
        self._chat_patcher = patch.object(
            chat_route,
            "chat_with_ai",
            side_effect=lambda message, user_id, **kwargs: f"authorized:{user_id}",
        )
        self._users_patcher.start()
        self._sessions_patcher.start()
        self._indexes_patcher.start()
        self._chat_patcher.start()
        self.addCleanup(self._chat_patcher.stop)
        self.addCleanup(self._indexes_patcher.stop)
        self.addCleanup(self._sessions_patcher.stop)
        self.addCleanup(self._users_patcher.stop)

        self.tmp = tempfile.TemporaryDirectory()
        self.session_path = Path(self.tmp.name) / "test-session.json"
        self.addCleanup(self.tmp.cleanup)
        self._path_patcher = patch.object(cli_auth, "SESSION_PATH", self.session_path)
        self._path_patcher.start()
        self.addCleanup(self._path_patcher.stop)

        self.credentials: dict[str, str] = {}
        self.test_keyring = MagicMock()
        self.test_keyring.get_password.side_effect = (
            lambda service, account: self.credentials.get(account)
        )
        self.test_keyring.set_password.side_effect = (
            lambda service, account, value: self.credentials.__setitem__(account, value)
        )
        self.test_keyring.delete_password.side_effect = (
            lambda service, account: self.credentials.pop(account, None)
        )
        self._keyring_patcher = patch(
            "cli_auth._credential_backend", return_value=self.test_keyring
        )
        self._keyring_patcher.start()
        self.addCleanup(self._keyring_patcher.stop)

        self.api = TestClient(app)
        self.transport_lock = threading.Lock()
        self.refresh_requests = 0
        self.chat_requests = 0

        self.user_id = f"usr_refresh_test_{uuid.uuid4().hex}"
        self.user = {
            "_id": self.user_id,
            "email": f"refresh-{uuid.uuid4().hex}@example.com",
            "status": "active",
            "tenant_id": f"ten_refresh_{uuid.uuid4().hex[:12]}",
            "display_name": "isolated refresh test",
            "created_at": None,
        }
        self.users.insert_one(self.user)
        issued = auth_service.issue_tokens_for_user(self.user)
        self.session = {
            "access_token": issued["access_token"],
            "refresh_token": issued["refresh_token"],
            "expires_in": issued["expires_in"],
            "user": issued["user"].model_dump(),
            "api_base": "http://testserver",
        }
        self.initial_access_token = self.session["access_token"]
        self.initial_refresh_token = self.session["refresh_token"]
        cli_auth.save_session(self.session)

        self.real_expiry_check = cli_auth._access_token_expired
        self._expiry_patcher = patch.object(
            cli_auth,
            "_access_token_expired",
            side_effect=lambda token, skew_seconds=60: self.real_expiry_check(
                token, skew_seconds=0
            ),
        )
        self._expiry_patcher.start()
        self.addCleanup(self._expiry_patcher.stop)
        self.addCleanup(self._restore_settings)
        self.addCleanup(self._env_patcher.stop)
        self.addCleanup(get_settings.cache_clear)

    def _restore_settings(self):
        self.settings.access_token_ttl_minutes = self.original_access_ttl

    def _transport(self, url: str, **kwargs):
        path = urlparse(url).path
        with self.transport_lock:
            if path == "/auth/refresh":
                self.refresh_requests += 1
            elif path == "/api/chat":
                self.chat_requests += 1
        return self.api.post(
            path,
            headers=kwargs.get("headers"),
            json=kwargs.get("json"),
        )

    def _post_as_cli(self, message: str):
        with patch("cli_auth.requests.post", side_effect=self._transport):
            return cli_auth.api_post_authenticated(
                self.session,
                "/api/chat",
                json_body={"message": message},
            )

    def test_five_second_access_token_refreshes_and_retries_authenticated_request(self):
        with patch("cli_auth.requests.post", side_effect=self._transport):
            session_before, body_before = cli_auth.api_post_authenticated(
                self.session,
                "/api/chat",
                json_body={"message": "before expiry"},
            )
        self.assertEqual(body_before["reply"], f"authorized:{self.user_id}")
        self.assertEqual(self.refresh_requests, 0)
        self.assertEqual(self.chat_requests, 1)

        old_claims = jwt.decode(
            self.initial_access_token,
            options={"verify_signature": False, "verify_exp": False},
            algorithms=["HS256"],
        )
        time.sleep(max(0.0, float(old_claims["exp"]) + 0.15 - time.time()))
        self.assertTrue(self.real_expiry_check(self.initial_access_token, skew_seconds=0))

        # First exercise the regular proactive path: an expired local token is
        # refreshed before the authenticated API request is sent.
        with patch("cli_auth.requests.post", side_effect=self._transport):
            refreshed, body_after_expiry = cli_auth.api_post_authenticated(
                session_before,
                "/api/chat",
                json_body={"message": "after expiry"},
            )
        self.assertEqual(body_after_expiry["reply"], f"authorized:{self.user_id}")
        self.assertEqual(self.refresh_requests, 1)
        self.assertEqual(self.chat_requests, 2)
        self.assertNotEqual(refreshed["access_token"], self.initial_access_token)
        self.assertNotEqual(refreshed["refresh_token"], self.initial_refresh_token)

        stored = cli_auth.load_session()
        self.assertEqual(stored["access_token"], refreshed["access_token"])
        self.assertEqual(stored["refresh_token"], refreshed["refresh_token"])
        metadata = self.session_path.read_text(encoding="utf-8")
        self.assertNotIn(refreshed["access_token"], metadata)
        self.assertNotIn(refreshed["refresh_token"], metadata)

        # Let the rotated token expire too, then exercise the actual server 401
        # path. The client must refresh once, retry the original request, and
        # return the authenticated response.
        second_claims = jwt.decode(
            refreshed["access_token"],
            options={"verify_signature": False, "verify_exp": False},
            algorithms=["HS256"],
        )
        time.sleep(max(0.0, float(second_claims["exp"]) + 0.15 - time.time()))
        expired_retry_token = refreshed["access_token"]

        def miss_expiry_for_one_token(token, skew_seconds=60):
            if token == expired_retry_token:
                return False
            return self.real_expiry_check(token, skew_seconds=0)

        with patch.object(
            cli_auth, "_access_token_expired", side_effect=miss_expiry_for_one_token
        ), patch("cli_auth.requests.post", side_effect=self._transport):
            retried, retry_body = cli_auth.api_post_authenticated(
                refreshed,
                "/api/chat",
                json_body={"message": "expired bearer retry"},
            )

        self.assertEqual(retry_body["reply"], f"authorized:{self.user_id}")
        self.assertEqual(self.refresh_requests, 2)
        self.assertEqual(self.chat_requests, 4)
        self.assertNotEqual(retried["access_token"], expired_retry_token)
        self.assertEqual(cli_auth.load_session()["refresh_token"], retried["refresh_token"])


@unittest.skipUnless(sys.platform == "win32", "Requires Windows Credential Manager")
class TestRefreshedSessionChildProcessRestore(unittest.TestCase):
    def test_rotated_credentials_restore_in_separate_process(self):
        backend = cli_auth._credential_backend()
        self.assertEqual(type(backend.get_keyring()).__name__, "WinVaultKeyring")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "refreshed-session.json"
            user_id = f"usr_refresh_child_{uuid.uuid4().hex}"
            email = "refreshed-child@example.com"
            api_base = "http://127.0.0.1:8787"
            session = {
                "access_token": "initial-test-access-placeholder",
                "refresh_token": f"initial-test-refresh-{uuid.uuid4().hex}",
                "expires_in": 5,
                "user": {"id": user_id, "email": email},
                "api_base": api_base,
            }
            account = cli_auth._credential_account(api_base, user_id)
            saved = False
            try:
                with patch.object(cli_auth, "SESSION_PATH", path):
                    cli_auth.save_session(session)
                    saved = True
                    refreshed = {
                        "access_token": cli_auth.jwt.encode(
                            {
                                "sub": user_id,
                                "email": email,
                                "type": "access",
                                "sid": f"sid_{uuid.uuid4().hex}",
                                "iss": "agentic-ai-api",
                                "aud": "agentic-ai-cli",
                                "iat": int(time.time()),
                                "exp": int(time.time()) + 300,
                            },
                            "test-jwt-secret-key-32chars-min!!",
                            algorithm="HS256",
                        ),
                        "refresh_token": f"rotated-test-refresh-{uuid.uuid4().hex}",
                        "expires_in": 300,
                        "user": session["user"],
                    }
                    cli_auth._apply_auth_response_to_session(session, refreshed)

                expected_access = hashlib.sha256(
                    refreshed["access_token"].encode("utf-8")
                ).hexdigest()
                expected_refresh = hashlib.sha256(
                    refreshed["refresh_token"].encode("utf-8")
                ).hexdigest()
                child_env = os.environ.copy()
                child_env["AGENTIC_CLI_SESSION"] = str(path)
                child_env["EXPECTED_ACCESS_SHA256"] = expected_access
                child_env["EXPECTED_REFRESH_SHA256"] = expected_refresh
                child_code = (
                    "import hashlib,json,os,cli_auth; "
                    "s=cli_auth.load_session(); "
                    "print(json.dumps({'restored':bool(s),"
                    "'access_matches':hashlib.sha256(s['access_token'].encode()).hexdigest()==os.environ['EXPECTED_ACCESS_SHA256'],"
                    "'refresh_matches':hashlib.sha256(s['refresh_token'].encode()).hexdigest()==os.environ['EXPECTED_REFRESH_SHA256'],"
                    "'user_id':s['user']['id']}))"
                )
                child = subprocess.run(
                    [sys.executable, "-c", child_code],
                    cwd=Path(__file__).resolve().parents[1],
                    env=child_env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(child.returncode, 0, child.stderr)
                result = json.loads(child.stdout.strip())
                self.assertTrue(result["restored"])
                self.assertTrue(result["access_matches"])
                self.assertTrue(result["refresh_matches"])
                self.assertEqual(result["user_id"], user_id)
            except RuntimeError as exc:
                if "1312" in str(exc) or "logon session does not exist" in str(exc).lower():
                    self.skipTest(
                        "Windows Credential Manager has no usable logon session here (WinError 1312)"
                    )
                raise
            finally:
                if saved:
                    with patch.object(cli_auth, "SESSION_PATH", path):
                        cli_auth.clear_session(session)
                    self.assertIsNone(backend.get_password("agentic-ai-cli", account))
                else:
                    path.unlink(missing_ok=True)
