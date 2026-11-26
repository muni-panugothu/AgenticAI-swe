"""CLI login helper — stores session locally without printing secrets."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Any

import jwt
import requests

SESSION_PATH = Path(
    os.getenv("AGENTIC_CLI_SESSION", Path.home() / ".agentic_cli_session.json")
)
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8787").rstrip("/")
_ACCESS_SKEW_SECONDS = 60
_MAX_AUTH_RETRIES = 1
_REFRESH_LOCK = threading.RLock()


def _session_api_base(session: dict | None) -> str:
    base = (session or {}).get("api_base") or API_BASE
    return str(base).rstrip("/")


def _session_user_id(session: dict) -> str | None:
    user = session.get("user") if isinstance(session.get("user"), dict) else {}
    uid = user.get("id")
    return str(uid) if uid else None


@dataclass
class SessionRefreshError(Exception):
    code: str | None = None

    def __str__(self) -> str:
        if self.code == "EXPIRED_REFRESH":
            return "Session expired — sign in again: python cli_auth.py login"
        return "Session expired — sign in again: python cli_auth.py login"


def _is_transient_refresh_error(code: str | None) -> bool:
    return code in {"NETWORK", "RATE_LIMITED"} or bool(
        code and code.startswith("HTTP_5")
    )


def _transient_refresh_message(code: str | None) -> str:
    if code == "NETWORK":
        return "Cannot reach the auth API; your saved session was retained."
    if code == "RATE_LIMITED":
        return "Too many refresh attempts; wait a moment and try again."
    return "The auth service is temporarily unavailable; your saved session was retained."


def load_session() -> dict | None:
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read saved CLI session metadata at {SESSION_PATH}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError("Saved CLI session metadata is invalid; sign in again")
    # Legacy files held bearer credentials in plaintext; never restore them.
    if "access_token" in data or "refresh_token" in data:
        uid = _session_user_id(data)
        if uid:
            try:
                _credential_delete(_credential_account(_session_api_base(data), uid))
            except RuntimeError:
                # Never leave legacy plaintext tokens behind because the OS
                # credential service is unavailable.
                pass
        SESSION_PATH.unlink(missing_ok=True)
        return None
    uid = data.get("user", {}).get("id") if isinstance(data.get("user"), dict) else None
    if not uid:
        raise RuntimeError("Saved CLI session has no account ID; sign in again")
    account = _credential_account(data.get("api_base", API_BASE), str(uid))
    secret = _credential_get(account)
    if not secret:
        # The metadata contains no tokens and cannot restore access by itself.
        # Remove this stale pointer so the normal login menu remains available.
        SESSION_PATH.unlink(missing_ok=True)
        return None
    try:
        tokens = json.loads(secret)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "The secure CLI credential is unreadable; sign in again"
        ) from exc
    if not isinstance(tokens, dict) or not tokens.get("access_token") or not tokens.get("refresh_token"):
        raise RuntimeError("The secure CLI credential is incomplete; sign in again")
    return {**data, **tokens}


def save_session(data: dict) -> None:
    uid = _session_user_id(data)
    if not uid or not data.get("access_token") or not data.get("refresh_token"):
        raise RuntimeError(
            "Cannot save CLI session securely: account ID, access token, or "
            "refresh token is missing"
        )
    account = _credential_account(_session_api_base(data), uid)
    credential_written = False
    try:
        secret = json.dumps({
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
            "expires_in": data.get("expires_in"),
        })
        _credential_set(account, secret)
        credential_written = True
        if _credential_get(account) != secret:
            raise RuntimeError("Credential Manager did not confirm the saved value")
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"user": data.get("user"), "api_base": _session_api_base(data)}
        temporary = SESSION_PATH.with_suffix(SESSION_PATH.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, SESSION_PATH)
    except Exception as exc:
        if credential_written:
            try:
                _credential_delete(account)
            except Exception:
                pass
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Could not save the secure CLI session: {exc}") from exc


def clear_session(session: dict | None = None) -> None:
    stored: dict = {}
    if SESSION_PATH.exists():
        try:
            value = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                stored = value
        except (OSError, json.JSONDecodeError):
            # There is no safe way to derive an account-specific credential key
            # from malformed metadata; still remove the non-secret pointer.
            stored = {}
    data = session or stored
    uid = _session_user_id(data)
    if uid:
        _credential_delete(_credential_account(_session_api_base(data), uid))
    stored_uid = _session_user_id(stored)
    # A failure from a stale session object must never unlink a different
    # account's current metadata pointer.
    if SESSION_PATH.exists() and (session is None or not stored_uid or stored_uid == uid):
        SESSION_PATH.unlink()


def _credential_account(api_base: str, user_id: str) -> str:
    identity = f"{api_base.rstrip('/')}|{user_id}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def _credential_backend():
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError(
            "Secure session storage is unavailable because the 'keyring' "
            f"package is missing from this Python environment ({sys.executable}). "
            f"Install project dependencies with: \"{sys.executable}\" -m pip install -r requirements.txt"
        ) from exc
    try:
        backend = keyring.get_keyring()
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialize the OS credential backend: {type(exc).__name__}: {exc}"
        ) from exc
    backend_type = type(backend)
    module = backend_type.__module__.lower()
    backend_name = f"{backend_type.__module__}.{backend_type.__name__}"
    try:
        priority = backend.priority
    except Exception as exc:
        raise RuntimeError(
            f"Credential backend {backend_name} is unavailable: {exc}"
        ) from exc
    if priority <= 0 or "fail" in module or "null" in module:
        raise RuntimeError(
            f"No usable OS credential backend was selected (selected {backend_name}, "
            f"priority {priority})."
        )
    if sys.platform == "win32" and not (
        "windows" in module and backend_type.__name__ == "WinVaultKeyring"
    ):
        raise RuntimeError(
            "Windows Credential Manager is unavailable: "
            f"keyring selected {backend_name}. Install 'keyring' with its "
            "Windows dependency into this interpreter and run as the same "
            "interactive Windows user that will restore the session."
        )
    return keyring


def _credential_get(account: str) -> str | None:
    try:
        return _credential_backend().get_password("agentic-ai-cli", account)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the secure CLI credential: {type(exc).__name__}: {exc}"
        ) from exc


def _credential_set(account: str, value: str) -> None:
    try:
        _credential_backend().set_password("agentic-ai-cli", account, value)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not write the secure CLI credential: {type(exc).__name__}: {exc}"
        ) from exc


def _credential_delete(account: str) -> None:
    keyring = _credential_backend()
    try:
        existing = keyring.get_password("agentic-ai-cli", account)
        if existing is None:
            return
        keyring.delete_password("agentic-ai-cli", account)
        if keyring.get_password("agentic-ai-cli", account) is not None:
            raise RuntimeError("Credential Manager still contains the CLI credential")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not delete the secure CLI credential: {type(exc).__name__}: {exc}"
        ) from exc


def _verify_refresh_user_unchanged(session: dict, payload: dict) -> None:
    prior = _session_user_id(session)
    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    new_id = user.get("id")
    # Require the server's public identity and the newly issued access token's
    # subject to agree with the saved identity before persisting rotated tokens.
    token = payload.get("access_token")
    token_sub = None
    if isinstance(token, str):
        try:
            claims = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
                algorithms=["HS256", "HS384", "HS512"],
            )
            token_sub = claims.get("sub")
        except jwt.PyJWTError:
            pass
    if (
        not prior
        or not new_id
        or str(new_id) != prior
        or not token_sub
        or str(token_sub) != prior
    ):
        clear_session(session)
        raise SessionRefreshError("USER_MISMATCH")


def _access_token_expired(token: str, *, skew_seconds: int = _ACCESS_SKEW_SECONDS) -> bool:
    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
            algorithms=["HS256", "HS384", "HS512"],
        )
    except jwt.PyJWTError:
        return True
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        return True
    return time.time() >= (float(exp) - skew_seconds)


def can_restore_saved_session(session: dict | None) -> bool:
    """True when Continue-as is possible (refresh or non-expired access)."""
    if not session or not session.get("user", {}).get("email"):
        return False
    if not session.get("access_token"):
        return False
    if session.get("refresh_token"):
        return True
    return not _access_token_expired(str(session["access_token"]))


def _apply_auth_response_to_session(session: dict, payload: dict) -> dict:
    session["access_token"] = payload["access_token"]
    if payload.get("refresh_token"):
        session["refresh_token"] = payload["refresh_token"]
    if payload.get("expires_in") is not None:
        session["expires_in"] = payload["expires_in"]
    if payload.get("user"):
        session["user"] = payload["user"]
    session["api_base"] = _session_api_base(session)
    save_session(session)
    return session


def _adopt_rotated_saved_session(session: dict, previous_access_token: str) -> bool:
    """Adopt a newer persisted session after another request refreshed it."""
    saved = load_session()
    if not saved:
        return False
    if (
        _session_user_id(saved) != _session_user_id(session)
        or _session_api_base(saved) != _session_api_base(session)
        or saved.get("access_token") == previous_access_token
    ):
        return False
    session.clear()
    session.update(saved)
    return True


def refresh_session_tokens(session: dict) -> dict:
    """Rotate refresh token and update the saved session; clear session on failure."""
    refresh = session.get("refresh_token")
    if not refresh:
        clear_session(session)
        raise SessionRefreshError("MISSING_REFRESH")

    api_base = _session_api_base(session)
    try:
        res = requests.post(
            f"{api_base}/auth/refresh",
            json={"refresh_token": refresh},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SessionRefreshError("NETWORK") from exc

    if res.status_code == 200:
        payload = res.json()
        _verify_refresh_user_unchanged(session, payload)
        return _apply_auth_response_to_session(session, payload)

    detail: dict[str, Any] = {}
    if res.headers.get("content-type", "").startswith("application/json"):
        body = res.json()
        if isinstance(body.get("detail"), dict):
            detail = body["detail"]
    code = detail.get("code")
    if res.status_code in {401, 403} or code in {
        "INVALID_REFRESH",
        "EXPIRED_REFRESH",
        "REFRESH_REUSE",
        "ACCOUNT_DISABLED",
        "MISSING_REFRESH",
    }:
        clear_session(session)
        raise SessionRefreshError(code or "INVALID_REFRESH")
    if res.status_code == 429:
        raise SessionRefreshError("RATE_LIMITED")
    raise SessionRefreshError(code or f"HTTP_{res.status_code}")


def ensure_fresh_session(session: dict) -> dict:
    """Refresh access token when expired or near expiry."""
    token = session.get("access_token")
    if not token:
        clear_session(session)
        raise SessionRefreshError("MISSING_ACCESS")
    if not _access_token_expired(str(token)):
        return session
    with _REFRESH_LOCK:
        # Another thread may have refreshed this shared dict while we waited.
        token = session.get("access_token")
        if token and not _access_token_expired(str(token)):
            return session
        if token:
            _adopt_rotated_saved_session(session, str(token))
            token = session.get("access_token")
            if token and not _access_token_expired(str(token)):
                return session
        if not session.get("refresh_token"):
            clear_session(session)
            raise SessionRefreshError("MISSING_REFRESH")
        return refresh_session_tokens(session)


def restore_saved_session() -> dict:
    """Load ~/.agentic_cli_session.json and refresh access token when needed."""
    session = load_session()
    if not session:
        raise SessionRefreshError("NO_SESSION")
    if not can_restore_saved_session(session):
        clear_session(session)
        raise SessionRefreshError("NOT_RESTORABLE")
    return ensure_fresh_session(session)


def _raise_cli_session_error(exc: SessionRefreshError) -> None:
    if _is_transient_refresh_error(exc.code):
        raise RuntimeError(_transient_refresh_message(exc.code)) from None
    raise RuntimeError(
        "Session expired — sign in again: python cli_auth.py login"
    ) from None


def api_post_authenticated(
    session: dict,
    path: str,
    *,
    json_body: dict | None = None,
    timeout: float = 120,
) -> tuple[dict, dict]:
    """
    POST with Bearer auth. On 401, refresh once and retry.
    Returns (updated_session, response_json).
    """
    try:
        session = ensure_fresh_session(session)
    except SessionRefreshError as exc:
        _raise_cli_session_error(exc)
    attempts = 0
    while True:
        token = session["access_token"]
        api_base = _session_api_base(session)
        res = requests.post(
            f"{api_base}{path}",
            headers={"Authorization": f"Bearer {token}"},
            json=json_body or {},
            timeout=timeout,
        )
        if res.status_code == 401 and attempts < _MAX_AUTH_RETRIES:
            attempts += 1
            try:
                with _REFRESH_LOCK:
                    # Concurrent requests may already have rotated and saved
                    # this session while this request was in flight.
                    if session.get("access_token") == token:
                        _adopt_rotated_saved_session(session, token)
                    if session.get("access_token") == token:
                        session = refresh_session_tokens(session)
                    session = ensure_fresh_session(session)
            except SessionRefreshError as exc:
                _raise_cli_session_error(exc)
            continue
        if res.status_code == 401:
            clear_session(session)
            raise RuntimeError(
                "Session expired — sign in again: python cli_auth.py login"
            )
        res.raise_for_status()
        return session, res.json()


def logout_server_session(session: dict | None) -> None:
    """Revoke refresh token server-side when possible."""
    session = session or load_session()
    if not session:
        return
    refresh = session.get("refresh_token")
    if not refresh and not session.get("access_token"):
        return
    api_base = _session_api_base(session)
    headers: dict[str, str] = {}
    token = session.get("access_token")
    if token and not _access_token_expired(str(token)):
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.post(
            f"{api_base}/auth/logout",
            headers=headers,
            json={"refresh_token": refresh},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not revoke the server-side session at {api_base}: {exc}. "
            "The local session was retained so logout can be retried."
        ) from exc


def _prompt_password() -> str:
    """
    Read a password in the terminal.

    Standard getpass hides all characters; in some Windows terminals (including
    integrated IDE shells) that mode accepts no keystrokes. We show '*' on Windows
    as feedback, and allow dev fallbacks via environment variables.
    """
    env_password = os.getenv("AGENTIC_CLI_PASSWORD")
    if env_password is not None:
        return env_password

    if os.getenv("AGENTIC_CLI_VISIBLE_PASSWORD", "").lower() in {"1", "true", "yes"}:
        print("Password (visible — dev only; unset AGENTIC_CLI_VISIBLE_PASSWORD in production):")
        return input().strip()

    print(
        "Password (hidden — you will not see characters; type anyway, then Enter):",
        flush=True,
    )

    if sys.platform == "win32":
        try:
            import msvcrt

            sys.stdout.write("Password: ")
            sys.stdout.flush()
            chars: list[str] = []
            while True:
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    break
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch in ("\x08", "\x7f"):
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                else:
                    chars.append(ch)
                    sys.stdout.write("*")
                    sys.stdout.flush()
            return "".join(chars)
        except ImportError:
            pass

    try:
        return getpass("Password: ")
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        print(
            "Password prompt failed. Set AGENTIC_CLI_PASSWORD or "
            "AGENTIC_CLI_VISIBLE_PASSWORD=true, or run from Windows Terminal / cmd.",
            file=sys.stderr,
        )
        raise


def check_api_reachable() -> bool:
    try:
        res = requests.get(f"{API_BASE}/health", timeout=3)
        return res.status_code == 200
    except requests.RequestException:
        return False


def _parse_api_error(res: requests.Response) -> tuple[str, str | None]:
    if not res.headers.get("content-type", "").startswith("application/json"):
        return res.text or f"HTTP {res.status_code}", None
    body = res.json()
    detail = body.get("detail")
    if isinstance(detail, dict):
        return str(detail.get("error", res.text)), detail.get("code")
    if isinstance(detail, list):
        return "Validation failed — check email/password rules", "VALIDATION_ERROR"
    return str(detail or body.get("error", res.text)), None


def _connection_help() -> str:
    return (
        f"\nCannot reach the API at {API_BASE} (connection refused).\n\n"
        "Start the backend in a **second** terminal, then run login again:\n\n"
        "  cd backend\n"
        "  python -m uvicorn app.main:app --host 127.0.0.1 --port 8787\n\n"
        "Ensure backend `.env` includes JWT_SECRET (32+ characters) and MONGO_URI.\n"
        "Or set API_BASE_URL if the server runs elsewhere."
    )


def _save_session_from_auth_response(payload: dict) -> dict:
    session = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token"),
        "expires_in": payload.get("expires_in"),
        "user": payload.get("user"),
        "api_base": API_BASE,
    }
    save_session(session)
    return session


def register_interactive() -> dict:
    if not check_api_reachable():
        raise RuntimeError(_connection_help())
    print(
        "Create an API account (first time only). Password rules: "
        "8+ characters, one uppercase, one lowercase, one digit."
    )
    display_name = input("Display name: ").strip() or "User"
    email = input("Email: ").strip()
    password = _prompt_password()
    print("By registering you accept the Terms of Use and Privacy Policy.")
    try:
        res = requests.post(
            f"{API_BASE}/auth/register",
            json={
                "display_name": display_name,
                "email": email,
                "password": password,
                "accepted_terms": True,
                "accepted_privacy": True,
            },
            timeout=30,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(_connection_help()) from exc
    if res.status_code >= 400:
        message, code = _parse_api_error(res)
        if code == "EMAIL_EXISTS":
            print("Account exists — try login instead.")
            return login_interactive()
        raise RuntimeError(message)
    session = _save_session_from_auth_response(res.json())
    print(f"Registered and signed in as {session['user']['email']}")
    return session


def login_interactive() -> dict:
    if not check_api_reachable():
        raise RuntimeError(_connection_help())
    email = input("Email: ").strip()
    password = _prompt_password()
    try:
        res = requests.post(
            f"{API_BASE}/auth/login",
            json={"email": email, "password": password},
            timeout=30,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(_connection_help()) from exc
    if res.status_code >= 400:
        message, code = _parse_api_error(res)
        if code == "INVALID_CREDENTIALS":
            raise RuntimeError(
                f"{message}\n\n"
                "This API account was not found, or the password does not match.\n"
                "New auth is separate from Mongo/Groq keys — register once:\n"
                "  python cli_auth.py register"
            )
        raise RuntimeError(message)
    session = _save_session_from_auth_response(res.json())
    print(f"Signed in as {session['user']['email']}")
    return session


def _auth_menu(*, force: bool, title: str) -> dict:
    """Show register / login / optional continue-with-saved-session."""
    if not check_api_reachable():
        raise RuntimeError(_connection_help())

    saved = load_session()
    has_saved = can_restore_saved_session(saved)
    has_stale_saved = bool(
        saved
        and saved.get("user", {}).get("email")
        and saved.get("access_token")
        and not has_saved
    )

    print(title)
    print("  [1] Register (new account)")
    print("  [2] Login")
    if (has_saved or has_stale_saved) and not force:
        email = saved["user"]["email"]
        if has_saved:
            print(f"  [3] Continue as {email} (saved session)")
        else:
            print(
                f"  [3] Continue as {email} "
                "(saved access expired — login required; no refresh token)"
            )
        print("  [4] Logout (forget saved session)")
        default = "3" if has_saved else "2"
        extra = ", 3, or 4"
    else:
        default = "1"
        extra = ""

    choice = input(f"Choose 1 or 2{extra} [{default}]: ").strip() or default

    if choice == "4" and has_saved and not force:
        logout_server_session(saved)
        clear_session(saved)
        print("Saved session cleared.")
        return _auth_menu(force=True, title="--- Sign in ---")

    if choice == "3" and (has_saved or has_stale_saved) and not force:
        if not has_saved:
            print("Saved session cannot be restored — sign in again.")
            clear_session()
            return _auth_menu(force=True, title="--- Sign in ---")
        try:
            saved = restore_saved_session()
            print(f"Using saved session for {saved['user']['email']}")
            return saved
        except SessionRefreshError as exc:
            if exc.code == "NETWORK":
                raise RuntimeError(
                    f"Cannot reach API at {_session_api_base(saved)} — check server and API_BASE_URL."
                ) from None
            if exc.code == "RATE_LIMITED":
                raise RuntimeError(
                    "Too many refresh attempts — wait a moment and try again."
                ) from None
            if exc.code and exc.code.startswith("HTTP_5"):
                raise RuntimeError(_transient_refresh_message(exc.code)) from None
            print("Saved session expired — sign in again.")
            clear_session()
            return _auth_menu(force=True, title="--- Sign in ---")
    if choice == "2":
        return login_interactive()
    if choice == "1":
        return register_interactive()
    if choice == "3":
        return login_interactive()
    print("Invalid choice — try again.")
    return _auth_menu(force=force, title=title)


def startup_auth_menu() -> dict:
    """Called when terminal chatbot starts."""
    return _auth_menu(
        force=False,
        title="\n--- Sign in ---",
    )


def ensure_cli_session() -> dict:
    """Prompt when no session is available (legacy entry)."""
    session = load_session()
    if session and session.get("access_token"):
        try:
            return ensure_fresh_session(session)
        except SessionRefreshError as exc:
            if _is_transient_refresh_error(exc.code):
                raise RuntimeError(_transient_refresh_message(exc.code)) from None
            clear_session()
    return _auth_menu(force=True, title="No saved CLI session.")


def cli_logout_and_prompt() -> dict:
    """Clear saved tokens and show register/login again."""
    session = load_session()
    logout_server_session(session)
    clear_session(session)
    print("\nSigned out.")
    return _auth_menu(
        force=True,
        title="--- Sign in again ---",
    )


def refresh_if_needed(session: dict) -> dict:
    """Backward-compatible alias for ensure_fresh_session."""
    return ensure_fresh_session(session)


def get_authenticated_user_id() -> str:
    session = load_session()
    if not session or not session.get("access_token"):
        raise RuntimeError(
            "Not signed in. Run: python cli_auth.py login"
        )
    user = session.get("user") or {}
    uid = user.get("id")
    if not uid:
        raise RuntimeError("Invalid session — login again")
    return str(uid)


def chat_via_api(message: str, session: dict | None = None) -> str:
    session = session or load_session()
    if not session:
        raise RuntimeError("Not signed in")
    _, body = api_post_authenticated(
        session,
        "/api/chat",
        json_body={"message": message},
    )
    return body.get("reply", "")


if __name__ == "__main__":
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "login").lower()
    if cmd in ("login", "signin"):
        login_interactive()
    elif cmd in ("register", "signup"):
        register_interactive()
    elif cmd == "logout":
        session = load_session()
        logout_server_session(session)
        clear_session(session)
        print("Signed out.")
    elif cmd == "whoami":
        s = load_session()
        print(json.dumps(s.get("user") if s else None, indent=2))
    else:
        print("Usage: python cli_auth.py [register|login|logout|whoami]")
