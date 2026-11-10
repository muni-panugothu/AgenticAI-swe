
import os
import sys

from cli_auth import (
    chat_via_api,
    cli_logout_and_prompt,
    get_authenticated_user_id,
    load_session,
    startup_auth_menu,
)
from app.chatbot import chat_with_ai

USE_API_AUTH = os.getenv("USE_API_AUTH", "true").lower() in {"1", "true", "yes"}


def _resolve_user_id() -> str:
    if USE_API_AUTH:
        print(
            "Tip: API on port 8787. Startup: [4]=logout saved user. "
            "In chat: logout | exit.\n"
            ".env keys are not login — use Register/Login below."
        )
        try:
            startup_auth_menu()
        except RuntimeError as exc:
            print(exc)
            sys.exit(1)
        return get_authenticated_user_id()
    legacy = os.getenv("MEMORY_TEST_USER_ID") or os.getenv("DEFAULT_USER_ID")
    if legacy:
        print(
            "WARNING: Using legacy DEFAULT_USER_ID — not production-safe. "
            "Set USE_API_AUTH=true for API auth."
        )
        return legacy
    raise RuntimeError(
        "Authentication required. Set USE_API_AUTH=false with MEMORY_TEST_USER_ID (dev only)."
    )


def _print_banner(user_id: str) -> None:
    session = load_session() or {}
    user = session.get("user") or {}
    email = user.get("email", user_id)
    print("=" * 50)
    print("Terminal AI Chatbot")
    print(f"Auth mode: {'API' if USE_API_AUTH else 'legacy dev'}")
    print(f"Signed in as: {email}")
    print("Commands: logout  |  exit")
    print("=" * 50)


active_user_id = _resolve_user_id()
_print_banner(active_user_id)
print()

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    if user_input.lower() == "logout":
        if not USE_API_AUTH:
            print("\nLegacy auth mode — set USE_API_AUTH=true to use logout.\n")
            continue
        try:
            cli_logout_and_prompt()
            active_user_id = get_authenticated_user_id()
            _print_banner(active_user_id)
            print()
        except RuntimeError as exc:
            print(exc)
            sys.exit(1)
        continue

    if not user_input:
        print("\nPlease enter a message.\n")
        continue

    if USE_API_AUTH:
        try:
            response = chat_via_api(user_input, load_session())
        except Exception as exc:
            print(f"\n[Auth/API error]: {exc}\n")
            continue
    else:
        response = chat_with_ai(user_input, user_id=active_user_id)

    print("\nAI:", response)
