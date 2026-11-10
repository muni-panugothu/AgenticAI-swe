"""
Stage 18 MongoDB Atlas lifecycle integration test.

Uses a uniquely named temporary collection only.
Does not modify the production memories_history collection.
Makes no Groq or Gemini API calls.
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bson import ObjectId

from app.services import memory_service
from app.services.memory_lifecycle_service import identify_stale_memory


def main() -> None:
    client = memory_service.client
    database = memory_service.database

    hello = client.admin.command("hello")
    transaction_supported = bool(
        hello.get("setName") or hello.get("msg") == "isdbgrid"
    )
    if not transaction_supported:
        raise RuntimeError(
            "MongoDB does not report a replica-set or mongos topology."
        )

    collection_name = f"stage18_lifecycle_test_{uuid4().hex}"
    user_id = f"stage18-test-{uuid4().hex}"
    other_user = f"stage18-other-{uuid4().hex}"
    original_collection = memory_service.memory_collection
    collection_created = False

    print("=== Stage 18 MongoDB Lifecycle Integration Test ===")
    print(f"Temporary collection: {collection_name}")

    try:
        test_collection = database.create_collection(collection_name)
        collection_created = True
        memory_service.memory_collection = test_collection

        operation_id = f"stage18-save-{uuid4().hex}"
        saved = memory_service.save_memory(
            user_id=user_id,
            memory="My preferred language is Python",
            embedding=None,
            importance=3,
            confidence=0.95,
            status="active",
            fact_key="preferred_language",
            normalized_value="python",
            version=1,
            operation_id=operation_id,
        )
        assert saved.get("success"), saved
        memory_id = saved["memory_id"]
        print("PASS: Memory inserted.")

        again = memory_service.save_memory(
            user_id=user_id,
            memory="My preferred language is Python",
            embedding=None,
            importance=3,
            confidence=0.95,
            status="active",
            operation_id=operation_id,
        )
        assert again.get("success") and again.get("idempotent") is True, again
        assert again.get("memory_id") == memory_id
        print("PASS: Idempotent save replayed safely.")

        access_result = memory_service._record_memory_access(
            user_id,
            [{"memory_id": memory_id}],
        )
        assert access_result is None
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("access_count") == 1
        assert isinstance(doc.get("last_accessed_at"), datetime)
        print("PASS: Access tracking persisted.")

        reinforced = memory_service.reinforce_memory(user_id, memory_id)
        assert reinforced.get("success"), reinforced
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("reinforcement_count") == 1
        assert isinstance(doc.get("last_reinforced_at"), datetime)
        print("PASS: Reinforcement persisted.")

        # Concurrent-style user isolation: other user cannot reinforce/delete.
        cross = memory_service.reinforce_memory(other_user, memory_id)
        assert not cross.get("success"), cross
        print("PASS: Reinforcement is user-scoped.")

        forgotten = memory_service.delete_memory(
            user_id=user_id,
            memory_id=memory_id,
            soft=True,
        )
        assert forgotten.get("success"), forgotten
        doc = test_collection.find_one({"_id": ObjectId(memory_id), "user_id": user_id})
        assert doc.get("status") == "forgotten"
        print("PASS: Soft-forget marks forgotten without hard delete.")

        # Stale identification remains non-destructive.
        stale_probe = {
            "memory_id": "probe",
            "importance": 1,
            "created_at": datetime.now(timezone.utc) - timedelta(days=400),
        }
        stale = identify_stale_memory(stale_probe)
        assert stale["is_stale"] is True
        print("PASS: Stale identification flags old facts without deleting.")

        print("\nAll Stage 18 MongoDB lifecycle checks passed.")

    finally:
        memory_service.memory_collection = original_collection
        if collection_created:
            database.drop_collection(collection_name)
            print(f"Cleanup: dropped temporary collection {collection_name}")


if __name__ == "__main__":
    main()
