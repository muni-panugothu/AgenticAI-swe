
from app.services.temporary_memory_service import (
    add_task_context,
    clear_task_context,
    get_task_context,
)

USER_ID = "muni_ttl_test"

print("1. Clearing existing task context...")
clear_task_context(USER_ID)

print("\n2. Saving task context...")

add_task_context(
    USER_ID,
    "user",
    "I am debugging my AI agent loop.",
)

add_task_context(
    USER_ID,
    "assistant",
    "Let's inspect the tool-calling flow.",
)

print("\n3. Retrieving task context...")

context = get_task_context(USER_ID)

print("Raw context:", repr(context))
print("Context type:", type(context))
print("Message count:", len(context))

for message in context:
    print("Message:", repr(message))
