
from unittest.mock import Mock, patch

import app.chatbot as chatbot
from app.services.temporary_memory_service import (
    clear_task_context,
    get_task_context,
)

TEST_USER_ID = "muni_stage16c_integration_test"


def run_test():
    print("\n=== Stage 16C Integration Verification ===")

    # Use a dedicated test user to protect your normal Redis context.
    original_user_id = chatbot.USER_ID
    chatbot.USER_ID = TEST_USER_ID
    chatbot.conversation_history.clear()
    clear_task_context(TEST_USER_ID)

    try:
        # Build a mocked response matching the Groq SDK response structure.
        fake_message = Mock()
        fake_message.content = "This is a mocked assistant response."
        fake_message.tool_calls = None

        fake_choice = Mock()
        fake_choice.message = fake_message

        fake_response = Mock()
        fake_response.choices = [fake_choice]

        # Prevent real AI calls and long-term memory operations.
        with (
            patch(
                "app.chatbot.client.chat.completions.create",
                return_value=fake_response,
            ),
            patch(
                "app.chatbot.try_delete_explicit_memory",
                return_value=None,
            ),
            patch(
                "app.chatbot.try_update_explicit_memory",
                return_value=None,
            ),
            patch(
                "app.chatbot.try_save_explicit_memory",
                return_value=False,
            ),
            patch(
                "app.chatbot.load_long_term_memory",
                return_value=[],
            ),
        ):
            print("\n1. Testing one chatbot interaction...")

            answer = chatbot.chat_with_ai(
                "Stage 16C integration test message"
            )

        assert answer == "This is a mocked assistant response."
        print("PASS: Chatbot returned the mocked response.")

        # Check local conversation history.
        local_messages = chatbot.get_conversation_memory()

        assert len(local_messages) == 2, (
            f"Expected 2 local messages, got {len(local_messages)}"
        )

        assert local_messages[0] == {
            "role": "user",
            "content": "Stage 16C integration test message",
        }

        assert local_messages[1] == {
            "role": "assistant",
            "content": "This is a mocked assistant response.",
        }

        print(
            "PASS: Local history contains one user "
            "message and one assistant message."
        )

        # Check Redis task context.
        redis_messages = get_task_context(TEST_USER_ID)

        assert len(redis_messages) == 2, (
            f"Expected 2 Redis messages, got {len(redis_messages)}"
        )

        assert redis_messages[0]["role"] == "user"
        assert redis_messages[0]["content"] == (
            "Stage 16C integration test message"
        )
        assert isinstance(redis_messages[0].get("request_id"), str)
        assert redis_messages[0]["request_id"].strip()

        assert redis_messages[1]["role"] == "assistant"
        assert redis_messages[1]["content"] == (
            "This is a mocked assistant response."
        )
        assert (
            redis_messages[1]["request_id"]
            == redis_messages[0]["request_id"]
        )

        print(
            "PASS: Redis contains exactly two messages "
            "after one chatbot interaction."
        )

        # Reading must not append any messages.
        redis_messages_again = get_task_context(TEST_USER_ID)

        assert redis_messages_again == redis_messages

        print(
            "PASS: Repeated retrieval does not create duplicates."
        )

        # Check clearing of both local and Redis context.
        print("\n2. Testing clear_conversation_memory()...")

        chatbot.clear_conversation_memory()

        assert chatbot.get_conversation_memory() == [], (
            "Local conversation history was not cleared."
        )

        assert get_task_context(TEST_USER_ID) == [], (
            "Redis task context was not cleared."
        )

        print("PASS: Local conversation history is empty.")
        print("PASS: Redis task context is empty.")

        print("\nAll Stage 16C integration checks passed.")

    finally:
        # Clean up the dedicated test user's context.
        chatbot.conversation_history.clear()
        clear_task_context(TEST_USER_ID)

        # Restore the original user ID.
        chatbot.USER_ID = original_user_id

        print("Test cleanup completed.")


if __name__ == "__main__":
    run_test()
