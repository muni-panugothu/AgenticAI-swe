
import time

from app.services.temporary_memory_service import (
    add_task_context,
    get_task_context,
    clear_task_context,
)
from app.services.temporary_memory_service import redis

TEST_USER_ID = "muni_stage16c_test"
TEST_KEY = f"agent:task_context:{TEST_USER_ID}"

print("\n=== Stage 16C Verification ===")

# 1. Start with a clean, isolated test context.
clear_task_context(TEST_USER_ID)

# 2. Save exactly two messages.
add_task_context(
    TEST_USER_ID,
    "user",
    "I am testing Stage 16C.",
)
add_task_context(
    TEST_USER_ID,
    "assistant",
    "Let's verify Redis persistence.",
)

context = get_task_context(TEST_USER_ID)
assert len(context) == 2, f"Expected 2 messages, got {len(context)}"
print("PASS: Two messages saved and retrieved.")

# 3. Verify a subsequent read doesn't append messages.
context_again = get_task_context(TEST_USER_ID)
assert len(context_again) == 2
print("PASS: Reading context does not create duplicates.")

# 4. Check the configured expiration time.
ttl = redis.ttl(TEST_KEY)
assert ttl > 0, f"Expected a positive TTL, got {ttl}"
assert ttl <= 1800, f"TTL exceeds the configured 1800 seconds: {ttl}"
print(f"PASS: Task-context TTL is active ({ttl} seconds remaining).")

# 5. Verify clearing the task context.
cleared = clear_task_context(TEST_USER_ID)
assert cleared, "Expected the existing Redis key to be deleted."
assert get_task_context(TEST_USER_ID) == []
print("PASS: Task context cleared.")

# 6. Verify a short-lived key expires automatically.
expiry_key = f"agent:test:stage16c:{TEST_USER_ID}"
redis.set(expiry_key, "temporary test value", ex=2)
time.sleep(3)
assert redis.get(expiry_key) is None
print("PASS: Redis automatically expires a short-lived test key.")

print("\nAll Stage 16C service checks passed.")
