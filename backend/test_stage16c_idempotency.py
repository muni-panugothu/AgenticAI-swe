from uuid import uuid4

import app.chatbot as chatbot
from app.services.temporary_memory_service import (
    MAX_TASK_MESSAGES,
    TASK_CONTEXT_TTL,
    TaskContextScriptError,
    _interpret_add_task_context_result,
    add_task_context_once,
    clear_task_context,
    get_task_context,
)
from app.services.temporary_memory_service import redis

TEST_USER_ID = "muni_stage16c_idempotency_test"
OTHER_USER_ID = "muni_stage16c_idempotency_test_other"
TRIM_USER_ID = "muni_stage16c_idempotency_test_trim"
BLOCKED_USER_ID = "muni_stage16c_idempotency_test_blocked"


def _assert_script_error(result, stage):
    try:
        _interpret_add_task_context_result(result, "user", "req")
    except TaskContextScriptError as exc:
        assert exc.stage == stage, exc.stage
        return
    raise AssertionError(f"Expected script error {stage}")


def check_script_replies():
    print("\n0. Testing script reply handling...")

    saved = _interpret_add_task_context_result(
        ["saved"],
        "user",
        "req",
    )
    assert saved["success"] is True
    assert saved["duplicate"] is False
    assert saved["request_id"] == "req"

    duplicate = _interpret_add_task_context_result(
        ["duplicate"],
        "assistant",
        "req",
    )
    assert duplicate["duplicate"] is True

    _assert_script_error(
        ["error", "rpush_failed_dedupe_cleared", "WRONGTYPE"],
        "rpush_failed_dedupe_cleared",
    )
    _assert_script_error(
        ["error", "rpush_failed_dedupe_remains", "DEL failed"],
        "rpush_failed_dedupe_remains",
    )
    _assert_script_error(
        ["error", "ltrim_failed_message_kept", "ERR"],
        "ltrim_failed_message_kept",
    )
    _assert_script_error(
        ["error", "expire_failed_message_kept", "EXPIRE did not set a timeout"],
        "expire_failed_message_kept",
    )
    _assert_script_error(None, "unexpected_reply")

    print("PASS: Script replies keep failed writes distinct from rollback.")


def run_test():
    print("\n=== Stage 16C Idempotency & Restart Test ===")

    check_script_replies()

    original_user_id = chatbot.USER_ID
    chatbot.USER_ID = TEST_USER_ID
    chatbot.conversation_history.clear()
    clear_task_context(TEST_USER_ID)
    clear_task_context(OTHER_USER_ID)
    clear_task_context(TRIM_USER_ID)
    clear_task_context(BLOCKED_USER_ID)

    try:
        request_id = str(uuid4())

        print("\n1. Testing duplicate user-message saving...")

        first = add_task_context_once(
            TEST_USER_ID,
            request_id,
            "user",
            "Test request",
        )

        second = add_task_context_once(
            TEST_USER_ID,
            request_id,
            "user",
            "Test request",
        )

        assert first["success"]
        assert first["duplicate"] is False
        assert second["success"]
        assert second["duplicate"] is True

        messages = get_task_context(TEST_USER_ID)
        assert len(messages) == 1, (
            f"Expected 1 message after duplicate save, got {len(messages)}"
        )
        print("PASS: Duplicate user-message save was skipped.")

        print("\n2. Testing assistant-message saving...")

        assistant_first = add_task_context_once(
            TEST_USER_ID,
            request_id,
            "assistant",
            "Test response",
        )

        assistant_second = add_task_context_once(
            TEST_USER_ID,
            request_id,
            "assistant",
            "Test response",
        )

        assert assistant_first["duplicate"] is False
        assert assistant_second["duplicate"] is True

        messages = get_task_context(TEST_USER_ID)
        assert len(messages) == 2
        assert messages[0]["content"] == "Test request"
        assert messages[0]["request_id"] == request_id
        assert messages[1]["content"] == "Test response"
        assert messages[1]["request_id"] == request_id

        list_key = f"agent:task_context:{TEST_USER_ID}"
        user_key = (
            f"agent:task_context:dedupe:{TEST_USER_ID}:{request_id}:user"
        )
        list_ttl = redis.ttl(list_key)
        dedupe_ttl = redis.ttl(user_key)
        assert 0 < list_ttl <= TASK_CONTEXT_TTL, list_ttl
        assert 0 < dedupe_ttl <= TASK_CONTEXT_TTL, dedupe_ttl

        print("PASS: Each role was saved once for the request ID.")
        print("PASS: List and dedupe key use the task-context TTL.")

        print("\n3. Simulating a chatbot process restart...")

        # Simulate a fresh process with no in-memory history.
        chatbot.conversation_history.clear()

        chatbot.restore_task_context_from_redis(TEST_USER_ID)

        restored = chatbot.get_conversation_memory()

        assert len(restored) == 2, (
            f"Expected 2 restored messages, got {len(restored)}"
        )
        assert restored[0]["content"] == "Test request"
        assert restored[1]["content"] == "Test response"

        print("PASS: Both messages were restored from Redis.")

        # Restoration must not append another copy to Redis.
        messages_after_restore = get_task_context(TEST_USER_ID)
        assert len(messages_after_restore) == 2

        print("PASS: Restoration did not duplicate Redis messages.")

        print("\n4. Clearing context must allow the same request to be saved again...")

        clear_request_id = str(uuid4())
        user_dedupe_key = (
            f"agent:task_context:dedupe:{TEST_USER_ID}:"
            f"{clear_request_id}:user"
        )
        assistant_dedupe_key = (
            f"agent:task_context:dedupe:{TEST_USER_ID}:"
            f"{clear_request_id}:assistant"
        )

        # A user whose ID only shares a prefix must survive the clear.
        other_saved = add_task_context_once(
            OTHER_USER_ID,
            clear_request_id,
            "user",
            "Other user context",
        )
        assert other_saved["duplicate"] is False

        saved = add_task_context_once(
            TEST_USER_ID,
            clear_request_id,
            "user",
            "Message before clear",
        )
        assistant_saved = add_task_context_once(
            TEST_USER_ID,
            clear_request_id,
            "assistant",
            "Reply before clear",
        )
        assert saved["duplicate"] is False
        assert assistant_saved["duplicate"] is False
        assert redis.get(user_dedupe_key) is not None
        assert redis.get(assistant_dedupe_key) is not None

        cleared = clear_task_context(TEST_USER_ID)
        assert cleared
        assert get_task_context(TEST_USER_ID) == []
        assert redis.get(user_dedupe_key) is None
        assert redis.get(assistant_dedupe_key) is None
        assert get_task_context(OTHER_USER_ID) == [
            {
                "role": "user",
                "content": "Other user context",
                "request_id": clear_request_id,
            }
        ]

        print("PASS: Clear removed the list and both dedupe keys.")

        retry = add_task_context_once(
            TEST_USER_ID,
            clear_request_id,
            "user",
            "Message after clear",
        )
        assert retry["success"]
        assert retry["duplicate"] is False

        messages_after_retry = get_task_context(TEST_USER_ID)
        assert len(messages_after_retry) == 1
        assert messages_after_retry[0]["content"] == "Message after clear"

        print("PASS: The same request ID can be saved again after clear.")

        print("\n5. Trimming the task-context list to the last 10 messages...")

        clear_task_context(TRIM_USER_ID)
        saved_request_ids = []
        for index in range(MAX_TASK_MESSAGES + 1):
            trimmed_request_id = str(uuid4())
            saved_request_ids.append(trimmed_request_id)
            trimmed = add_task_context_once(
                TRIM_USER_ID,
                trimmed_request_id,
                "user",
                f"Message {index}",
            )
            assert trimmed["duplicate"] is False

        trimmed_messages = get_task_context(TRIM_USER_ID)
        assert len(trimmed_messages) == MAX_TASK_MESSAGES
        assert trimmed_messages[0]["content"] == "Message 1"
        assert trimmed_messages[-1]["content"] == (
            f"Message {MAX_TASK_MESSAGES}"
        )
        assert trimmed_messages[0]["request_id"] == saved_request_ids[1]

        print("PASS: The list retained the 10 newest messages.")

        print("\n6. RPUSH failure must clear the dedupe key without rollback...")

        blocked_key = f"agent:task_context:{BLOCKED_USER_ID}"
        blocked_request_id = str(uuid4())
        blocked_dedupe_key = (
            f"agent:task_context:dedupe:{BLOCKED_USER_ID}:"
            f"{blocked_request_id}:user"
        )
        redis.set(blocked_key, "not-a-list")

        try:
            add_task_context_once(
                BLOCKED_USER_ID,
                blocked_request_id,
                "user",
                "This append should fail",
            )
            raise AssertionError("Expected RPUSH to fail")
        except TaskContextScriptError as exc:
            assert exc.stage == "rpush_failed_dedupe_cleared", exc.stage

        assert redis.get(blocked_dedupe_key) is None
        assert redis.get(blocked_key) == "not-a-list"

        print("PASS: Failed RPUSH removed the dedupe key and left the list key.")

        print("\nAll idempotency and restart checks passed.")

    finally:
        chatbot.conversation_history.clear()
        clear_task_context(TEST_USER_ID)
        clear_task_context(OTHER_USER_ID)
        clear_task_context(TRIM_USER_ID)
        clear_task_context(BLOCKED_USER_ID)
        chatbot.USER_ID = original_user_id
        print("Test cleanup completed.")


if __name__ == "__main__":
    run_test()
