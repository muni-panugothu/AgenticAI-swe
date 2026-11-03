from app.services.memory_service import semantic_search_memories


USER_ID = "muni"


print("=" * 60)

print("SEMANTIC RETRIEVAL TEST")

print("=" * 60)


query = "What am I studying?"


print("\nQuery:")
print(query)


result = semantic_search_memories(
    user_id=USER_ID,
    query=query,
    limit=5
)


if result["success"]:

    print("\nSemantic search successfully")

    memories = result["memories"]

    print(f"Result found: {len(memories)}")

    print("\nRetrieved memories:")

    for index, memory in enumerate(memories, start=1):

        print(f"\n{index}. {memory['memory']}")
        print(f"  Score: {memory['score']}")

else:

    print("\nSemantic search failed!")

    print(result["error"])