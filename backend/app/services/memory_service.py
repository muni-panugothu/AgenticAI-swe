import os

from datetime import datetime, timezone

from uuid import uuid4

from dotenv import load_dotenv

from pymongo import MongoClient


# SEMATIC SEARCH IMPORT

from app.services.embedding_service import create_embedding

from app.schemas.memory_schema import MemoryDocument



# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# MONGODB CONFIGURATION
# ============================================================

MONGO_URI = os.getenv("MONGO_URI")

MEMORY_DATABASE = os.getenv(
    "MEMORY_DATABASE",
    "agentic_ai_memory"
)

MEMORY_COLLECTION = os.getenv(
    "MEMORY_COLLECTION",
    "memories_history"
)


# ============================================================
# VALIDATE MONGO URI
# ============================================================

if not MONGO_URI:

    raise ValueError(
        "MONGO_URI is not configured in .env"
    )


# ============================================================
# MONGODB CLIENT
# ============================================================

client = MongoClient(
    MONGO_URI
)


# ============================================================
# DATABASE
# ============================================================

database = client[
    MEMORY_DATABASE
]


# ============================================================
# COLLECTION
# ============================================================

memory_collection = database[
    MEMORY_COLLECTION
]


# ============================================================
# CHECK DATABASE CONNECTION
# ============================================================

def check_memory_database():

    try:

        client.admin.command(
            "ping"
        )

        print(
            "[MongoDB] Connection successful"
        )

        print(
            f"[Database] {MEMORY_DATABASE}"
        )

        print(
            f"[Collection] {MEMORY_COLLECTION}"
        )

        return True

    except Exception as e:

        print(
            f"[MongoDB] Connection failed: {e}"
        )

        return False


# ============================================================
# SAVE MEMORY
# ============================================================


def save_memory(
    user_id,
    memory,
    embedding=None,
    importance=3,
    confidence=0.75,
    status="active",
    fact_key=None,
    conflict_type="none",
    conflict_group_id=None,
    conflict_with_memory_id=None,
    supersedes_memory_id=None,
    superseded_by_memory_id=None,
    version=1,
):
    try:
        current_time = datetime.now(timezone.utc)

        validated_memory = MemoryDocument(
            user_id=user_id,
            memory=memory,
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            created_at=current_time,
            updated_at=current_time,
            status=status,
            fact_key=fact_key,
            conflict_type=conflict_type,
            conflict_group_id=conflict_group_id,
            conflict_with_memory_id=conflict_with_memory_id,
            supersedes_memory_id=supersedes_memory_id,
            superseded_by_memory_id=superseded_by_memory_id,
            version=version,
        )

        memory_document = validated_memory.model_dump()
        result = memory_collection.insert_one(memory_document)

        print(f"[Memory saved] {result.inserted_id}")
        print(f"[Memory status]: {status}")
        print(f"[Memory importance]: {importance}/5")

        return {
            "success": True,
            "memory_id": str(result.inserted_id),
            "memory": memory,
            "importance": importance,
            "status": status,
            "version": version,
        }

    except Exception as e:
        print(f"[Memory save error]: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# GET MEMORIES
# ============================================================

def get_memories(
    user_id,
    limit=10
):

    try:

        memories = memory_collection.find(
            {
                "user_id": user_id,
                "$or": [
                    {"status": "active"},
                    {"status": {"$exists": False}},  # legacy documents
                ],
            },
            {
                "_id": 1,
                "memory": 1,
                "created_at": 1,
                "updated_at": 1,
                "status": 1,
                "version": 1,
                "fact_key": 1,
            },
        ).sort("updated_at", -1).limit(limit)

        memory_list = []

        for item in memories:

            memory_list.append({

                "memory_id": str(
                    item["_id"]
                ),

                "memory": item["memory"],

                "created_at": item.get(
                    "created_at"
                ),

                "updated_at": item.get(
                    "updated_at"
                ),

                "status": item.get("status", "active"),

                "version": item.get("version", 1),

                "fact_key": item.get("fact_key"),

            })

        return {
            "success": True,
            "memories": memory_list
        }

    except Exception as e:

        print(
            f"[Memory retrieval error]: {e}"
        )

        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
# SEMANTIC SEARCH MEMORIES
# ============================================================

def semantic_search_memories(
    user_id,
    query,
    limit=5,
    min_score=0.80
):

    try:

        # --------------------------------------------------
        # 1. Create embedding for the user's query
        # --------------------------------------------------

        query_embedding = create_embedding(
            query
        )

        # --------------------------------------------------
        # 2. Vector search pipeline
        # --------------------------------------------------

        pipeline = [

            {
                "$vectorSearch": {

                    "index": "memory_vector_index",

                    "path": "embedding",

                    "queryVector": query_embedding,

                    "numCandidates": max(
                        100,
                        limit * 20
                    ),

                    "limit": max(limit * 5, 30),

                    "filter": {
                        "user_id": user_id
                    }
                }
            },

            # --------------------------------------------------
            # 3. Drop superseded and review-only documents
            # --------------------------------------------------

            {
                "$match": {
                    "$or": [
                        {"status": "active"},
                        {"status": {"$exists": False}},  # legacy documents
                    ]
                }
            },

            # --------------------------------------------------
            # 4. Return useful fields + similarity score
            # --------------------------------------------------

            {
                "$project": {

                    "_id": 1,

                    "memory": 1,

                    "importance": 1,

                    "confidence": 1,

                    "created_at": 1,

                    "updated_at": 1,

                    "status": 1,

                    "fact_key": 1,

                    "conflict_group_id": 1,

                    "version": 1,

                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }
        ]

        # --------------------------------------------------
        # 5. Execute vector search
        # --------------------------------------------------

        results = memory_collection.aggregate(
            pipeline
        )

        memories = []

        # --------------------------------------------------
        # 6. Apply memory quality filter
        # --------------------------------------------------

        for item in results:

            score = item.get(
                "score",
                0
            )

            print(
                f"[Memory candidate] "
                f"Score: {score:.4f}"
            )

            if score < min_score:

                print(
                    f"[Memory rejected] "
                    f"Score {score:.4f} "
                    f"< threshold {min_score}"
                )

                continue

            memories.append({

                "memory_id": str(
                    item["_id"]
                ),

                "memory": item["memory"],

                "score": score,

                "importance": item.get(
                    "importance",
                    3
                ),

                "confidence": item.get(
                    "confidence",
                    0.75
                ),

                "created_at": item.get(
                    "created_at"
                ),

                "updated_at": item.get(
                    "updated_at"
                ),

                "status": item.get(
                    "status",
                    "active"
                ),

                "fact_key": item.get(
                    "fact_key"
                ),

                "conflict_group_id": item.get(
                    "conflict_group_id"
                ),

                "version": item.get(
                    "version",
                    1
                ),
            })

        # --------------------------------------------------
        # 7. Return relevant memories
        # --------------------------------------------------

        print(
            f"[Semantic search] "
            f"Query: {query} | "
            f"Relevant results: "
            f"{len(memories)}"
        )

        memories.sort(
            key=lambda item: (
                item["score"] * 0.75
                + (item.get("importance", 3) / 5) * 0.15
                + item.get("confidence", 0.75) * 0.10
            ),
            reverse=True,
        )

        memories = memories[:limit]

        print("[Memory retrieval ranking]")
        for item in memories:
            weighted_score = (
                item["score"] * 0.75
                + (item.get("importance", 3) / 5) * 0.15
                + item.get("confidence", 0.75) * 0.10
            )

            print(
                f"Importance: {item.get('importance', 3)}/5 | "
                f"Confidence: {item.get('confidence', 0.75):.2f} | "
                f"Similarity: {item['score']:.4f} | "
                f"Weighted: {weighted_score:.4f} | "
                f"Memory: {item['memory']}"
            )

        return {"success": True, "memories": memories}

    except Exception as e:

        print(
            f"[Semantic search error]: {e}"
        )

        return {
            "success": False,
            "error": str(e)
        }




# ============================================================
# UPDATE MEMORY
# ============================================================

def update_memory_importance(user_id, memory_id, importance):
    """Update importance without changing the memory text or embedding."""
    from bson import ObjectId

    try:
        importance = max(1, min(5, int(importance)))

        result = memory_collection.update_one(
            {
                "_id": ObjectId(memory_id),
                "user_id": user_id,
            },
            {
                "$set": {
                    "importance": importance,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

        if result.matched_count == 0:
            return {
                "success": False,
                "error": "Memory not found for this user",
            }

        print(f"[Memory importance updated] {memory_id}: {importance}/5")

        return {
            "success": True,
            "memory_id": memory_id,
            "importance": importance,
        }

    except Exception as e:
        print(f"[Memory importance update error]: {e}")
        return {"success": False, "error": str(e)}


# ============================================================
# CHECK DUPLICATE MEMORY
# ============================================================

def check_duplicate_memory(
    user_id,
    embedding,
    duplicate_threshold=0.90,
    candidate_limit=10,
):
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "memory_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": max(100, candidate_limit * 20),
                    "limit": max(candidate_limit * 5, 30),
                    "filter": {"user_id": user_id},
                }
            },
            {
                "$match": {
                    "$or": [
                        {"status": "active"},
                        {"status": {"$exists": False}},
                    ]
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "memory": 1,
                    "importance": 1,
                    "confidence": 1,
                    "status": 1,
                    "fact_key": 1,
                    "conflict_group_id": 1,
                    "version": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = list(memory_collection.aggregate(pipeline))[:candidate_limit]
        candidates = []

        for item in results:
            candidates.append({
                "memory_id": str(item["_id"]),
                "memory": item.get("memory", ""),
                "importance": item.get("importance", 3),
                "confidence": item.get("confidence", 0.75),
                "status": item.get("status", "active"),
                "fact_key": item.get("fact_key"),
                "conflict_group_id": item.get("conflict_group_id"),
                "version": item.get("version", 1),
                "score": float(item.get("score", 0.0)),
            })

        if not candidates:
            return {
                "success": True,
                "is_duplicate": False,
                "memory": None,
                "score": None,
                "candidates": [],
            }

        best = candidates[0]
        is_duplicate = best["score"] >= duplicate_threshold

        return {
            "success": True,
            "is_duplicate": is_duplicate,
            "memory_id": best["memory_id"],
            "memory": best["memory"],
            "importance": best["importance"],
            "score": best["score"],
            "candidates": candidates,
        }

    except Exception as e:
        print(f"[Duplicate check error]: {e}")
        return {"success": False, "error": str(e)}





def update_memory(user_id, memory_id, memory, embedding):
    from bson import ObjectId

    try:
        result = memory_collection.update_one(
            {
                "_id": ObjectId(memory_id),
                "user_id": user_id
            },
            {
                "$set": {
                    "memory": memory,
                    "embedding": embedding,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )

        if result.matched_count == 0:
            return {
                "success": False,
                "error": "Memory not found for this user"
            }

        print(f"[Memory updated] {memory_id}")

        return {
            "success": True,
            "memory_id": memory_id,
            "memory": memory
        }

    except Exception as e:
        print(f"[Memory update error]: {e}")

        return {
            "success": False,
            "error": str(e)
        }


def supersede_memory(user_id, memory_id, memory, embedding, fact_key=None):
    from bson import ObjectId

    try:
        old_object_id = ObjectId(memory_id)
        current_time = datetime.now(timezone.utc)

        with client.start_session() as session:
            with session.start_transaction():
                existing = memory_collection.find_one(
                    {"_id": old_object_id, "user_id": user_id},
                    session=session,
                )

                if not existing:
                    return {
                        "success": False,
                        "error": "Memory not found for this user",
                    }

                if existing.get("status", "active") != "active":
                    return {
                        "success": False,
                        "error": "Only an active memory can be superseded",
                    }

                conflict_group_id = (
                    existing.get("conflict_group_id") or uuid4().hex
                )
                next_version = int(existing.get("version", 1)) + 1

                new_document = MemoryDocument(
                    user_id=user_id,
                    memory=memory,
                    embedding=embedding,
                    importance=existing.get("importance", 3),
                    confidence=existing.get("confidence", 0.75),
                    created_at=current_time,
                    updated_at=current_time,
                    status="active",
                    fact_key=fact_key or existing.get("fact_key"),
                    conflict_type="confirmed_update",
                    conflict_group_id=conflict_group_id,
                    supersedes_memory_id=str(old_object_id),
                    version=next_version,
                ).model_dump()

                inserted = memory_collection.insert_one(
                    new_document,
                    session=session,
                )
                new_memory_id = str(inserted.inserted_id)

                old_update = memory_collection.update_one(
                    {
                        "_id": old_object_id,
                        "user_id": user_id,
                        "$or": [
                            {"status": "active"},
                            {"status": {"$exists": False}},
                        ],
                    },
                    {
                        "$set": {
                            "status": "superseded",
                            "superseded_by_memory_id": new_memory_id,
                            "conflict_group_id": conflict_group_id,
                            "updated_at": current_time,
                        }
                    },
                    session=session,
                )

                if old_update.matched_count != 1:
                    raise RuntimeError(
                        "The previous memory changed before it could be superseded"
                    )

                # Resolve pending review records that were linked to this old fact.
                memory_collection.update_many(
                    {
                        "user_id": user_id,
                        "status": "needs_review",
                        "conflict_with_memory_id": str(old_object_id),
                    },
                    {
                        "$set": {
                            "status": "resolved",
                            "updated_at": current_time,
                        }
                    },
                    session=session,
                )

        print(
            f"[Memory version created] old={memory_id}, "
            f"new={new_memory_id}, version={next_version}"
        )

        return {
            "success": True,
            "old_memory_id": memory_id,
            "memory_id": new_memory_id,
            "memory": memory,
            "version": next_version,
            "status": "active",
        }

    except Exception as e:
        print(f"[Memory supersede error]: {e}")
        return {"success": False, "error": str(e)}
        











# ============================================================
# DELETE MEMORY



# ============================================================

def delete_memory(
    memory_id
):

    try:

        from bson import ObjectId

        result = memory_collection.delete_one(
            {
                "_id": ObjectId(
                    memory_id
                )
            }
        )

        if result.deleted_count == 0:

            return {
                "success": False,
                "error": "Memory not found"
            }

        return {
            "success": True,
            "message": "Memory deleted"
        }

    except Exception as e:

        print(
            f"[Memory delete error]: {e}"
        )

        return {
            "success": False,
            "error": str(e)
        }