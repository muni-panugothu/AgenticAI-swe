from app.services.memory_service import (
    check_duplicate_memory
)

from app.services.embedding_service import (
    create_embedding
)


USER_ID = "muni"


print("=" * 60)
print("MEMORY DEDUPLICATION TEST")
print("=" * 60)


# ============================================================
# TEST MEMORY
# ============================================================

memory = "I am building an Agentic AI bug resolver"


print("\nTest memory:")
print(memory)


# ============================================================
# CREATE EMBEDDING
# ============================================================

print("\nCreating embedding...")

embedding = create_embedding(
    memory
)


# ============================================================
# CHECK DUPLICATE
# ============================================================

print("\nChecking duplicate...")

result = check_duplicate_memory(
    user_id=USER_ID,
    embedding=embedding,
    duplicate_threshold=0.90
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "-" * 60)

if not result["success"]:

    print("Duplicate check failed")
    print(result["error"])

elif result["is_duplicate"]:

    print("DUPLICATE MEMORY DETECTED")

    print(
        f"Existing memory: "
        f"{result['memory']}"
    )

    print(
        f"Similarity score: "
        f"{result['score']:.4f}"
    )

else:

    print("MEMORY IS UNIQUE")

    print(
        f"Similarity score: "
        f"{result['score']:.4f}"
    )