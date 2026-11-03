from app.services.memory_service import (
    check_memory_database,
    save_memory,
    get_memories
)


print()
print("=" * 60)
print("LONG-TERM MEMORY TEST")
print("=" * 60)


# ============================================================
# 1. TEST MONGODB CONNECTION
# ============================================================

print()
print("1. Testing MongoDB connection...")

connection_result = check_memory_database()

print(
    f"Connection result: {connection_result}"
)


# ============================================================
# STOP IF CONNECTION FAILED
# ============================================================

if not connection_result:

    print()
    print(
        "MongoDB connection failed."
    )

    print(
        "Check your MONGO_URI and Atlas configuration."
    )

    exit()


# ============================================================
# 2. SAVE MEMORY
# ============================================================

print()
print("2. Saving memory...")

save_result = save_memory(
    user_id="muni",
    memory="User's name is Muni"
)

print(
    save_result
)


# ============================================================
# 3. GET MEMORIES
# ============================================================

print()
print("3. Getting memories...")

memory_result = get_memories(
    user_id="muni"
)

print(
    memory_result
)


# ============================================================
# 4. DISPLAY MEMORIES
# ============================================================

print()
print("4. Stored memories:")

if memory_result.get("success"):

    memories = memory_result.get(
        "memories",
        []
    )

    for index, item in enumerate(
        memories,
        start=1
    ):

        print(
            f"{index}. "
            f"{item['memory']}"
        )

else:

    print(
        "Could not retrieve memories."
    )


print()
print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)