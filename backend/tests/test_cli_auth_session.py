"""CLI session refresh and API retry behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt
import requests

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

import cli_auth  # noqa: E402

_real_credential_backend = cli_auth._credential_backend


def _expired_access_token() -> str:
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    payload = {
        "sub": "usr_a",
        "email": "a@test.com",
        "type": "access",
        "exp": int(past.timestamp()),
        "iat": int(past.timestamp()) - 60,
    }
    return jwt.encode(
        payload,
        "test-jwt-secret-key-32chars-min!!",
        algorithm="HS256",
    )


def _fresh_access_token(sub: str = "usr_a", email: str = "a@test.com") -> str:
    future = datetime.now(timezone.utc) + timedelta(minutes=10)
    payload = {
        "sub": sub,
        "email": email,
        "type": "access",
        "exp": int(future.timestamp()),
        "iat": int(datetime.now(timezone.utc).timestamp()),
    }
    return jwt.encode(
        payload,
        "test-jwt-secret-key-32chars-min!!",
        algorithm="HS256",
    )


class TestCliSession(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.session_path = Path(self._tmpdir.name) / "session.json"
        self.patcher = patch.object(cli_auth, "SESSION_PATH", self.session_path)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.addCleanup(self._tmpdir.cleanup)
        self.credentials = {}
        self.keyring_patcher = patch("cli_auth._credential_backend", return_value=MagicMock(
            get_password=lambda service, account: self.credentials.get(account),
            set_password=lambda service, account, value: self.credentials.__setitem__(account, value),
            delete_password=lambda service, account: self.credentials.pop(account, None),
        ))
        self.keyring_patcher.start()
        self.addCleanup(self.keyring_patcher.stop)

    def test_valid_saved_session_needs_no_refresh(self):
        session = {
            "access_token": _fresh_access_token(),
            "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        with patch("cli_auth.requests.post") as post_mock:
            out = cli_auth.ensure_fresh_session(session)
            post_mock.assert_not_called()
        self.assertEqual(out["access_token"], session["access_token"])

    def test_session_file_contains_metadata_only_and_credentials_are_account_scoped(self):
        session = {
            "access_token": _fresh_access_token(), "refresh_token": "very-secret-refresh",
            "user": {"id": "usr_a", "email": "a@test.com"}, "api_base": "http://localhost",
        }
        cli_auth.save_session(session)
        on_disk = self.session_path.read_text(encoding="utf-8")
        self.assertNotIn("very-secret-refresh", on_disk)
        self.assertNotIn(session["access_token"], on_disk)
        self.assertEqual(cli_auth.load_session()["refresh_token"], "very-secret-refresh")
        self.assertNotEqual(
            cli_auth._credential_account("http://localhost", "usr_a"),
            cli_auth._credential_account("http://localhost", "usr_b"),
        )

    def test_secure_storage_unavailable_fails_closed(self):
        session = {
            "access_token": _fresh_access_token(), "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        with patch("cli_auth._credential_backend", side_effect=RuntimeError("disabled")):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                cli_auth.save_session(session)
        self.assertFalse(self.session_path.exists())

    def test_missing_keyring_dependency_reports_install_command(self):
        session = {
            "access_token": _fresh_access_token(), "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        with patch("cli_auth._credential_backend", _real_credential_backend), patch.dict(
            "sys.modules", {"keyring": None}
        ):
            with self.assertRaisesRegex(RuntimeError, "keyring.*missing.*pip install"):
                cli_auth.save_session(session)
        self.assertFalse(self.session_path.exists())

    def test_loading_session_explains_unavailable_secure_backend(self):
        self.session_path.write_text(
            json.dumps({"user": {"id": "usr_a", "email": "a@test.com"}, "api_base": "http://localhost"}),
            encoding="utf-8",
        )
        with patch(
            "cli_auth._credential_backend",
            side_effect=RuntimeError("Windows Credential Manager backend unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Windows Credential Manager backend unavailable"):
                cli_auth.load_session()

    def test_continue_session_refreshes_expired_access(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)

        refresh_payload = {
            "access_token": _fresh_access_token(),
            "refresh_token": "rt_rotated",
            "expires_in": 900,
            "user": {"id": "usr_a", "email": "a@test.com"},
        }

        def fake_post(url, **kwargs):
            mock = MagicMock()
            if url.endswith("/auth/refresh"):
                mock.status_code = 200
                mock.json.return_value = refresh_payload
            return mock

        with patch("cli_auth.requests.post", side_effect=fake_post):
            updated = cli_auth.restore_saved_session()

        self.assertEqual(updated["refresh_token"], "rt_rotated")
        self.assertFalse(cli_auth._access_token_expired(updated["access_token"]))
        saved = json.loads(self.session_path.read_text(encoding="utf-8"))
        self.assertNotIn("refresh_token", saved)
        self.assertNotIn("access_token", saved)
        self.assertEqual(cli_auth.load_session()["refresh_token"], "rt_rotated")

    def test_expired_refresh_clears_session(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_old",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)

        def fake_post(url, **kwargs):
            mock = MagicMock()
            mock.status_code = 401
            mock.headers = {"content-type": "application/json"}
            mock.json.return_value = {
                "detail": {"code": "EXPIRED_REFRESH", "error": "expired"},
            }
            return mock

        with patch("cli_auth.requests.post", side_effect=fake_post):
            with self.assertRaises(cli_auth.SessionRefreshError) as ctx:
                cli_auth.refresh_session_tokens(session)
            self.assertEqual(ctx.exception.code, "EXPIRED_REFRESH")
        self.assertFalse(self.session_path.exists())

    def test_temporary_server_error_retains_saved_session(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_temporarily_unavailable",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)
        with patch(
            "cli_auth.refresh_session_tokens",
            side_effect=cli_auth.SessionRefreshError("HTTP_500"),
        ):
            with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                cli_auth.ensure_cli_session()
        self.assertTrue(self.session_path.exists())

    def test_expired_access_without_refresh_fails_without_an_api_request(self):
        session = {
            "access_token": _expired_access_token(),
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        with patch("cli_auth.requests.post") as post_mock:
            with self.assertRaises(cli_auth.SessionRefreshError) as ctx:
                cli_auth.ensure_fresh_session(session)
        self.assertEqual(ctx.exception.code, "MISSING_REFRESH")
        post_mock.assert_not_called()

    def test_preflight_refresh_network_failure_is_clear_and_retains_session(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_transient",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)
        with patch(
            "cli_auth.requests.post",
            side_effect=requests.ConnectionError("unavailable"),
        ) as post_mock:
            with self.assertRaisesRegex(RuntimeError, "Cannot reach the auth API"):
                cli_auth.api_post_authenticated(session, "/api/chat", json_body={"message": "test"})
        self.assertEqual(post_mock.call_count, 1)
        self.assertTrue(self.session_path.exists())
        self.assertEqual(cli_auth.load_session()["refresh_token"], "rt_transient")

    def test_revoked_refresh_clears_session(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_revoked",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }

        def fake_post(url, **kwargs):
            mock = MagicMock()
            mock.status_code = 401
            mock.headers = {"content-type": "application/json"}
            mock.json.return_value = {
                "detail": {"code": "INVALID_REFRESH", "error": "invalid"},
            }
            return mock

        with patch("cli_auth.requests.post", side_effect=fake_post):
            with self.assertRaises(cli_auth.SessionRefreshError):
                cli_auth.refresh_session_tokens(session)

    def test_user_mismatch_on_refresh_clears_session(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)

        def fake_post(url, **kwargs):
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {
                "access_token": _fresh_access_token(sub="usr_b"),
                "refresh_token": "rt_new",
                "expires_in": 900,
                "user": {"id": "usr_b", "email": "b@test.com"},
            }
            return mock

        with patch("cli_auth.requests.post", side_effect=fake_post):
            with self.assertRaises(cli_auth.SessionRefreshError) as ctx:
                cli_auth.refresh_session_tokens(session)
            self.assertEqual(ctx.exception.code, "USER_MISMATCH")
        self.assertFalse(self.session_path.exists())

    def test_refresh_requires_matching_user_and_access_subject(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)

        def fake_post(url, **kwargs):
            mock = MagicMock()
            mock.status_code = 200
            mock.json.return_value = {
                "access_token": _fresh_access_token(sub="usr_b"),
                "refresh_token": "rt_new",
                "expires_in": 900,
            }
            return mock

        with patch("cli_auth.requests.post", side_effect=fake_post):
            with self.assertRaises(cli_auth.SessionRefreshError) as ctx:
                cli_auth.refresh_session_tokens(session)
        self.assertEqual(ctx.exception.code, "USER_MISMATCH")
        self.assertFalse(self.session_path.exists())

    def test_network_error_does_not_clear_session(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)

        with patch(
            "cli_auth.requests.post",
            side_effect=requests.ConnectionError("down"),
        ):
            with self.assertRaises(cli_auth.SessionRefreshError) as ctx:
                cli_auth.refresh_session_tokens(session)
            self.assertEqual(ctx.exception.code, "NETWORK")
        self.assertTrue(self.session_path.exists())

    def test_api_post_authenticated_retries_after_401(self):
        session = {
            "access_token": _fresh_access_token(),
            "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        calls: list[str] = []

        def fake_post(url, **kwargs):
            calls.append(url)
            mock = MagicMock()
            if url.endswith("/api/chat"):
                if len([c for c in calls if c.endswith("/api/chat")]) == 1:
                    mock.status_code = 401
                    mock.json.return_value = {}
                    return mock
                mock.status_code = 200
                mock.json.return_value = {"reply": "ok"}
                return mock
            if url.endswith("/auth/refresh"):
                mock.status_code = 200
                mock.json.return_value = {
                    "access_token": _fresh_access_token(),
                    "refresh_token": "rt2",
                    "expires_in": 900,
                    "user": session["user"],
                }
            return mock

        with patch("cli_auth.requests.post", side_effect=fake_post):
            session, body = cli_auth.api_post_authenticated(
                session, "/api/chat", json_body={"message": "hi"}
            )

        self.assertEqual(body["reply"], "ok")
        self.assertIn("rt2", session.get("refresh_token", ""))

    def test_concurrent_401_requests_share_one_refresh_and_both_retry(self):
        session = {
            "access_token": _fresh_access_token(),
            "refresh_token": "rt_original",
            "user": {"id": "usr_a", "email": "a@test.com"},
            "api_base": "http://localhost",
        }
        cli_auth.save_session(session)
        first_two_requests = threading.Barrier(2)
        calls_lock = threading.Lock()
        counts = {"chat": 0, "refresh": 0}
        rotated = {
            "access_token": jwt.encode(
                {
                    "sub": "usr_a",
                    "email": "a@test.com",
                    "type": "access",
                    "iat": int(datetime.now(timezone.utc).timestamp()),
                    "exp": int((datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()),
                    "jti": uuid.uuid4().hex,
                },
                "test-jwt-secret-key-32chars-min!!",
                algorithm="HS256",
            ),
            "refresh_token": "rt_rotated_once",
            "expires_in": 900,
            "user": session["user"],
        }

        def response(status, body):
            result = MagicMock()
            result.status_code = status
            result.headers = {"content-type": "application/json"}
            result.json.return_value = body
            result.raise_for_status.return_value = None
            return result

        def fake_post(url, **kwargs):
            if url.endswith("/auth/refresh"):
                with calls_lock:
                    counts["refresh"] += 1
                return response(200, rotated)
            with calls_lock:
                counts["chat"] += 1
                chat_number = counts["chat"]
            if chat_number <= 2:
                first_two_requests.wait(timeout=5)
                return response(401, {"detail": {"code": "INVALID_TOKEN"}})
            return response(200, {"reply": "ok"})

        session_a = dict(session)
        session_b = dict(session)
        with patch("cli_auth.requests.post", side_effect=fake_post):
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_a = pool.submit(
                    cli_auth.api_post_authenticated,
                    session_a,
                    "/api/chat",
                    json_body={"message": "a"},
                )
                future_b = pool.submit(
                    cli_auth.api_post_authenticated,
                    session_b,
                    "/api/chat",
                    json_body={"message": "b"},
                )
                _, body_a = future_a.result(timeout=10)
                _, body_b = future_b.result(timeout=10)

        self.assertEqual(counts, {"chat": 4, "refresh": 1})
        self.assertEqual(body_a["reply"], "ok")
        self.assertEqual(body_b["reply"], "ok")
        self.assertEqual(cli_auth.load_session()["refresh_token"], "rt_rotated_once")

    def test_chat_transport_preserves_complete_long_response(self):
        session = {
            "access_token": _fresh_access_token(),
            "refresh_token": "rt_plain",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        complete_reply = "Full model response. " * 300
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"reply": complete_reply}
        with patch("cli_auth.requests.post", return_value=response):
            output = cli_auth.chat_via_api("please answer", session)
        self.assertEqual(output, complete_reply)

    def test_can_restore_false_without_refresh_when_access_expired(self):
        session = {
            "access_token": _expired_access_token(),
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        self.assertFalse(cli_auth.can_restore_saved_session(session))

    def test_logout_then_restore_fails(self):
        session = {
            "access_token": _expired_access_token(),
            "refresh_token": "rt_revoked",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)

        logout_requests = []

        def fake_post(url, **kwargs):
            mock = MagicMock()
            if url.endswith("/auth/logout"):
                logout_requests.append((url, kwargs))
                mock.status_code = 200
                mock.json.return_value = {"success": True}
                return mock
            mock.status_code = 401
            mock.headers = {"content-type": "application/json"}
            mock.json.return_value = {
                "detail": {"code": "INVALID_REFRESH"},
            }
            return mock

        with patch("cli_auth.requests.post", side_effect=fake_post):
            cli_auth.logout_server_session(session)
            cli_auth.clear_session(session)
            with self.assertRaises(cli_auth.SessionRefreshError):
                cli_auth.restore_saved_session()
        self.assertFalse(self.credentials)
        self.assertFalse(self.session_path.exists())
        self.assertEqual(len(logout_requests), 1)
        self.assertEqual(logout_requests[0][1]["json"]["refresh_token"], "rt_revoked")

    def test_failed_server_logout_retains_local_session_for_retry(self):
        session = {
            "access_token": _fresh_access_token(),
            "refresh_token": "rt_retry_logout",
            "user": {"id": "usr_a", "email": "a@test.com"},
        }
        cli_auth.save_session(session)
        with patch(
            "cli_auth.requests.post",
            side_effect=requests.ConnectionError("server unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not revoke the server-side session"):
                cli_auth.logout_server_session(session)
        self.assertTrue(self.session_path.exists())
        self.assertTrue(self.credentials)

    def test_logout_and_account_switch_uses_new_identity_for_chat(self):
        account_a = {
            "id": "usr_account_a", "email": "a@example.com",
        }
        account_b = {
            "id": "usr_account_b", "email": "b@example.com",
        }
        seen_chat_tokens = []

        def fake_post(url, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {"success": True}
            if url.endswith("/auth/login"):
                email = kwargs["json"]["email"]
                user = account_a if email == account_a["email"] else account_b
                response.json.return_value = {
                    "access_token": _fresh_access_token(user["id"], email),
                    "refresh_token": f"refresh-{user['id']}",
                    "expires_in": 900,
                    "user": user,
                }
            elif url.endswith("/api/chat"):
                seen_chat_tokens.append(kwargs["headers"]["Authorization"])
                response.json.return_value = {"reply": "memory lookup completed"}
            return response

        with (
            patch("cli_auth.check_api_reachable", return_value=True),
            patch("cli_auth._prompt_password", return_value="Secret1pass"),
            patch("cli_auth.requests.post", side_effect=fake_post),
            patch("builtins.input", side_effect=[account_a["email"], "2", account_b["email"]]),
        ):
            first = cli_auth.login_interactive()
            self.assertEqual(cli_auth.chat_via_api("show my preferences", first), "memory lookup completed")

            second = cli_auth.cli_logout_and_prompt()
            self.assertEqual(cli_auth.chat_via_api("show my preferences", second), "memory lookup completed")

        self.assertEqual(first["user"]["id"], account_a["id"])
        self.assertEqual(second["user"]["id"], account_b["id"])
        subjects = [
            jwt.decode(token.removeprefix("Bearer "), options={"verify_signature": False})["sub"]
            for token in seen_chat_tokens
        ]
        self.assertEqual(subjects, [account_a["id"], account_b["id"]])
        self.assertEqual(cli_auth.load_session()["user"]["id"], account_b["id"])


@unittest.skipUnless(sys.platform == "win32", "Requires Windows Credential Manager")
class TestWindowsCredentialManagerProcessRestore(unittest.TestCase):
    def test_credential_manager_restores_session_in_a_new_process(self):
        backend = cli_auth._credential_backend()
        self.assertEqual(type(backend.get_keyring()).__name__, "WinVaultKeyring")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"
            user_id = f"usr_cli_session_test_{uuid.uuid4().hex}"
            session = {
                "access_token": _fresh_access_token(
                    user_id, "credential-test@example.invalid"
                ),
                "refresh_token": f"refresh-test-{uuid.uuid4().hex}",
                "expires_in": 900,
                "user": {"id": user_id, "email": "credential-test@example.invalid"},
                "api_base": "http://127.0.0.1:8787",
            }
            account = cli_auth._credential_account(session["api_base"], user_id)
            credential_saved = False
            try:
                with patch.object(cli_auth, "SESSION_PATH", path):
                    try:
                        cli_auth.save_session(session)
                    except RuntimeError as exc:
                        if "1312" in str(exc) or "logon session does not exist" in str(exc).lower():
                            self.skipTest(
                                "Windows Credential Manager is present, but this "
                                "runner has no active Windows logon session (WinError 1312)"
                            )
                        raise
                    credential_saved = True
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    self.assertNotIn(session["access_token"], json.dumps(metadata))
                    self.assertNotIn(session["refresh_token"], json.dumps(metadata))

                child_env = os.environ.copy()
                child_env["AGENTIC_CLI_SESSION"] = str(path)
                child_code = (
                    "import builtins,json,cli_auth; "
                    "cli_auth.check_api_reachable=lambda: True; "
                    "builtins.input=lambda prompt: '3'; "
                    "s=cli_auth.startup_auth_menu(); "
                    "print(json.dumps({k:s[k] for k in "
                    "('access_token','refresh_token','user')}))"
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
                self.assertIn("[3] Continue as credential-test@example.invalid", child.stdout)
                self.assertIn("Using saved session for credential-test@example.invalid", child.stdout)
                restored = json.loads(child.stdout.strip().splitlines()[-1])
                self.assertEqual(restored["access_token"], session["access_token"])
                self.assertEqual(restored["refresh_token"], session["refresh_token"])
                self.assertEqual(restored["user"]["id"], user_id)
            finally:
                if credential_saved:
                    with patch.object(cli_auth, "SESSION_PATH", path):
                        cli_auth.clear_session(session)
                    self.assertIsNone(
                        backend.get_password("agentic-ai-cli", account)
                    )
                else:
                    path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
