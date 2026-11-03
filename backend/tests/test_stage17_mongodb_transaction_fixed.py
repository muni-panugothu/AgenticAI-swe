"""
Stage 17 MongoDB Atlas transaction integration test.

Uses a uniquely named temporary collection only.
Does not modify the production memories_history collection.
Makes no Groq or Gemini API calls.
"""

from uuid import uuid4

from bson import ObjectId

from app.services import memory_service


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

    collection_name = f"stage17_transaction_test_{uuid4().hex}"
    user_id = f"stage17-test-{uuid4().hex}"
    original_collection = memory_service.memory_collection
    collection_created = False

    print("=== Stage 17 MongoDB Transaction Integration Test ===")
    print(f"Temporary collection: {collection_name}")
    print("Production collection: untouched")

    try:
        test_collection = database.create_collection(collection_name)
        collection_created = True
        memory_service.memory_collection = test_collection

        # Save the original active memory.
        original = memory_service.save_memory(
            user_id=user_id,
            memory="My role is Junior Developer.",
            embedding=None,
            importance=4,
            confidence=0.95,
            status="active",
            fact_key="role",
            conflict_type="none",
            version=1,
        )
        assert original.get("success"), (
            f"Could not save original memory: {original}"
        )
        old_id = ObjectId(original["memory_id"])
        print("PASS: Original active memory inserted.")

        # Save the conflicting statement for review.
        # Valid schema value: "needs_review" (not "pending").
        pending = memory_service.save_memory(
            user_id=user_id,
            memory="My role is Software Engineer.",
            embedding=None,
            importance=4,
            confidence=0.95,
            status="needs_review",
            fact_key="role",
            conflict_type="possible_contradiction",
            conflict_group_id=uuid4().hex,
            conflict_with_memory_id=str(old_id),
            version=1,
        )
        assert pending.get("success"), (
            f"Could not save pending conflict: {pending}"
        )
        pending_id = ObjectId(pending["memory_id"])
        print("PASS: Conflicting memory inserted with needs_review status.")

        # Confirm the update through the real MongoDB transaction.
        updated = memory_service.supersede_memory(
            user_id=user_id,
            memory_id=str(old_id),
            memory="My role is Software Engineer.",
            embedding=None,
            fact_key="role",
        )
        assert updated.get("success"), (
            f"supersede_memory failed: {updated}"
        )
        new_id = ObjectId(updated["memory_id"])
        print("PASS: supersede_memory transaction returned success.")

        # Verify the original, new, and pending records.
        old_doc = test_collection.find_one(
            {"_id": old_id, "user_id": user_id}
        )
        new_doc = test_collection.find_one(
            {"_id": new_id, "user_id": user_id}
        )
        pending_doc = test_collection.find_one(
            {"_id": pending_id, "user_id": user_id}
        )

        assert old_doc is not None, "Original memory disappeared."
        assert old_doc.get("status") == "superseded", (
            f"Old status is {old_doc.get('status')!r}; expected 'superseded'."
        )
        assert old_doc.get("superseded_by_memory_id") == str(new_id), (
            "Old memory does not reference the new version."
        )
        print("PASS: Original memory preserved as superseded.")

        assert new_doc is not None, "New memory version was not inserted."
        assert new_doc.get("status") == "active", (
            f"New status is {new_doc.get('status')!r}; expected 'active'."
        )
        assert new_doc.get("supersedes_memory_id") == str(old_id), (
            "New memory does not reference the previous version."
        )
        assert new_doc.get("version") == 2, (
            f"Version is {new_doc.get('version')!r}; expected 2."
        )
        assert new_doc.get("conflict_type") == "confirmed_update", (
            "New version is not marked confirmed_update."
        )
        print("PASS: New active version created with version=2.")

        assert pending_doc is not None, "Pending conflict record disappeared."
        assert pending_doc.get("status") == "resolved", (
            f"Pending status is {pending_doc.get('status')!r}; "
            "expected 'resolved'."
        )
        print("PASS: Linked pending conflict marked resolved.")

        print("\nAll Stage 17 MongoDB transaction checks passed.")

    finally:
        memory_service.memory_collection = original_collection
        if collection_created:
            database.drop_collection(collection_name)
            print(f"Cleanup: dropped temporary collection {collection_name}")


if __name__ == "__main__":
    main()
