

from app.services.memory_service import (
    semantic_search_memories
)


USER_ID = "muni"

print("=" * 60)
print("MEMORY QUALITY TEST")
print("=" * 60)


queries = [
    "What am I studying?",
    "What technologies am I learning?",
    "What is my name",
    "What is my favorite food"
]



for query in queries:

    print("\n" + "-" * 60)
    
    print(
        f"Query: {query}"
    )

    result = semantic_search_memories(
        user_id=USER_ID,
        query=query,
        limit=5,
        min_score=0.70
    )

    if not result["success"]:

        print(
            "Search failed:"
        )


        print(
            result["error"]
        )

        continue


    memories = result["memories"]


    if not memories:

        print(
            "No relevant memories found."
        )

        continue

    print(
        f"Relevant memories: "
        f"{len(memories)}"
    )


    for index, memory in enumerate(memories,start=1):
        print(
            f"\n{index}. "
            f"{memory['memory']}"
        )


        print(
            f"  Score: "
            f"{memory['score']:.4f}"
        )


        

  