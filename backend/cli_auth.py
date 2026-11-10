"""CLI login helper — stores session locally without printing secrets."""

from __future__ import annotations

import json
import os
import sys
from getpass import getpass
from pathlib import Path

import requests

SESSION_PATH = Path(
    os.getenv("AGENTIC_CLI_SESSION", Path.home() / ".agentic_cli_session.json")
)
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8787").rstrip("/")


def load_session() -> dict | None:
    if not SESSION_PATH.exists():
        return None
    try:
        data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def save_session(data: dict) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        SESSION_PATH.chmod(0o600)
    except OSError:
        pass


def clear_session() -> None:
    if SESSION_PATH.exists():
        SESSION_PATH.unlink()


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
        # Last resort when getpass cannot attach to the console.
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
    email = input("Email: ").strip()
    password = _prompt_password()
    try:
        res = requests.post(
            f"{API_BASE}/auth/register",
            json={"email": email, "password": password},
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
    has_saved = bool(
        saved and saved.get("access_token") and saved.get("user", {}).get("email")
    )

    print(title)
    print("  [1] Register (new account)")
    print("  [2] Login")
    if has_saved and not force:
        email = saved["user"]["email"]
        print(f"  [3] Continue as {email} (saved session)")
        print("  [4] Logout (forget saved session)")
        default = "3"
        extra = ", 3, or 4"
    else:
        default = "1"
        extra = ""

    choice = input(f"Choose 1 or 2{extra} [{default}]: ").strip() or default

    if choice == "4" and has_saved and not force:
        clear_session()
        print("Saved session cleared.")
        return _auth_menu(force=True, title="--- Sign in ---")

    if choice == "3" and has_saved and not force:
        print(f"Using saved session for {saved['user']['email']}")
        return saved
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
    if load_session() and load_session().get("access_token"):
        return load_session()
    return _auth_menu(force=True, title="No saved CLI session.")


def cli_logout_and_prompt() -> dict:
    """Clear saved tokens and show register/login again."""
    clear_session()
    print("\nSigned out.")
    return _auth_menu(
        force=True,
        title="--- Sign in again ---",
    )


def refresh_if_needed(session: dict) -> dict:
    """Refresh access token using HttpOnly cookie when using requests.Session."""
    # Cookie jar must be used at login time for refresh; CLI stores access only.
    # Call /auth/refresh with stored refresh if present in session file.
    refresh = session.get("refresh_token")
    if not refresh:
        return session
    res = requests.post(
        f"{API_BASE}/auth/refresh",
        json={"refresh_token": refresh},
        timeout=30,
    )
    if res.status_code == 200:
        payload = res.json()
        session["access_token"] = payload["access_token"]
        session["user"] = payload.get("user", session.get("user"))
        save_session(session)
    return session


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
    token = session["access_token"]
    res = requests.post(
        f"{API_BASE}/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"message": message},
        timeout=120,
    )
    if res.status_code == 401:
        raise RuntimeError("Session expired — run: python cli_auth.py login")
    res.raise_for_status()
    return res.json().get("reply", "")


if __name__ == "__main__":
    import sys

    cmd = (sys.argv[1] if len(sys.argv) > 1 else "login").lower()
    if cmd in ("login", "signin"):
        login_interactive()
    elif cmd in ("register", "signup"):
        register_interactive()
    elif cmd == "logout":
        clear_session()
        print("Signed out.")
    elif cmd == "whoami":
        s = load_session()
        print(json.dumps(s.get("user") if s else None, indent=2))
    else:
        print("Usage: python cli_auth.py [register|login|logout|whoami]")
