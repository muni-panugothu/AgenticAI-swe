
from app.services.temporary_memory_service import (
    save_temporary_memory,
    get_temporary_memory,
    delete_temporary_memory,
)


USER_ID= "muni"

saved = save_temporary_memory(
    user_id=USER_ID,
    memory="I am currently debugging the chatbot agent loop.",
    memory_type="temporary"
)


print("Saved:",saved)

memory_id = saved["memory_id"]

retrieved = get_temporary_memory(USER_ID, memory_id)
print("Retrieved:", retrieved)

deleted = delete_temporary_memory(USER_ID, memory_id)
print("Deleted:", deleted)

print(
    "After deletion:",
    get_temporary_memory(USER_ID, memory_id)
)