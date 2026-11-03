"""Mocked service-level tests for Stage 17 memory conflict resolution.

Run from the backend directory:
    python -m unittest tests.test_memory_stage17_service -v

These tests replace MongoDB collection/session operations with mocks. They do
not insert, update, or delete documents in the real MongoDB database.
"""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from bson import ObjectId

from app.services import memory_service


class FakeContext:
    """Minimal context manager used for mocked MongoDB sessions/transactions."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def start_transaction(self):
        return FakeContext()


class TestStage17MemoryService(unittest.TestCase):
    def test_save_memory_persists_review_metadata(self):
        inserted_id = ObjectId()
        collection = MagicMock()
        collection.insert_one.return_value = SimpleNamespace(
            inserted_id=inserted_id
        )

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.save_memory(
                user_id="stage17-test-user",
                memory="My role is Software Engineer",
                embedding=[0.1, 0.2, 0.3],
                status="needs_review",
                fact_key="role",
                conflict_type="possible_contradiction",
                conflict_group_id="test-conflict-group",
                conflict_with_memory_id=str(ObjectId()),
                version=1,
            )

            self.assertTrue(result["success"], result)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["memory_id"], str(inserted_id))

        saved_document = collection.insert_one.call_args.args[0]
        self.assertEqual(saved_document["status"], "needs_review")
        self.assertEqual(saved_document["fact_key"], "role")
        self.assertEqual(
            saved_document["conflict_type"],
            "possible_contradiction",
        )
        self.assertEqual(
            saved_document["conflict_group_id"],
            "test-conflict-group",
        )
        self.assertIsNotNone(saved_document["conflict_with_memory_id"])

    def test_duplicate_check_returns_active_candidates(self):
        first_id = ObjectId()
        second_id = ObjectId()
        collection = MagicMock()
        collection.aggregate.return_value = iter([
            {
                "_id": first_id,
                "memory": "My role is Junior Developer",
                "importance": 4,
                "confidence": 0.95,
                "status": "active",
                "fact_key": "role",
                "version": 1,
                "score": 0.88,
            },
            {
                "_id": second_id,
                "memory": "I am learning Python",
                "importance": 4,
                "confidence": 0.95,
                "status": "active",
                "fact_key": None,
                "version": 1,
                "score": 0.73,
            },
        ])

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.check_duplicate_memory(
                user_id="stage17-test-user",
                embedding=[0.1, 0.2, 0.3],
                duplicate_threshold=0.90,
                candidate_limit=10,
            )

        self.assertTrue(result["success"], result)
        self.assertFalse(result["is_duplicate"])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][0]["memory_id"], str(first_id))

        pipeline = collection.aggregate.call_args.args[0]
        self.assertIn("$vectorSearch", pipeline[0])
        self.assertIn("$match", pipeline[1])

    def test_semantic_search_pipeline_excludes_review_and_superseded(self):
        collection = MagicMock()
        collection.aggregate.return_value = iter([
            {
                "_id": ObjectId(),
                "memory": "My role is Software Engineer",
                "importance": 4,
                "confidence": 0.95,
                "status": "active",
                "fact_key": "role",
                "score": 0.95,
            }
        ])

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "create_embedding", return_value=[0.1, 0.2]),
        ):
            result = memory_service.semantic_search_memories(
                user_id="stage17-test-user",
                query="What is my role?",
                limit=5,
                min_score=0.80,
            )

            self.assertTrue(result["success"], result)
            self.assertEqual(len(result["memories"]), 1)
            self.assertEqual(result["memories"][0]["status"], "active")

        pipeline = collection.aggregate.call_args.args[0]
        status_match = pipeline[1]["$match"]["$or"]
        self.assertIn({"status": "active"}, status_match)
        self.assertIn({"status": {"$exists": False}}, status_match)

    def test_supersede_memory_preserves_old_version_and_creates_new_version(self):
        old_id = ObjectId()
        new_id = ObjectId()
        collection = MagicMock()
        collection.find_one.return_value = {
            "_id": old_id,
            "user_id": "stage17-test-user",
            "memory": "My role is Junior Developer",
            "status": "active",
            "importance": 4,
            "confidence": 0.95,
            "fact_key": "role",
            "version": 1,
        }

        collection.insert_one.return_value = SimpleNamespace(inserted_id=new_id)
        collection.update_one.return_value = SimpleNamespace(matched_count=1)
        collection.update_many.return_value = SimpleNamespace(matched_count=1)

        fake_session = FakeContext()
        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(
                memory_service.client,
                "start_session",
                return_value=fake_session,
            ),
        ):
            result = memory_service.supersede_memory(
                user_id="stage17-test-user",
                memory_id=str(old_id),
                memory="My role is Software Engineer",
                embedding=[0.4, 0.5, 0.6],
                fact_key="role",
            )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["old_memory_id"], str(old_id))
        self.assertEqual(result["memory_id"], str(new_id))
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["status"], "active")

        inserted_document = collection.insert_one.call_args.args[0]
        self.assertEqual(inserted_document["memory"], "My role is Software Engineer")
        self.assertEqual(inserted_document["status"], "active")
        self.assertEqual(inserted_document["version"], 2)
        self.assertEqual(inserted_document["supersedes_memory_id"], str(old_id))
        self.assertEqual(inserted_document["conflict_type"], "confirmed_update")

        old_update_filter = collection.update_one.call_args.args[0]
        old_update_changes = collection.update_one.call_args.args[1]["$set"]
        self.assertEqual(old_update_filter["_id"], old_id)
        self.assertEqual(old_update_changes["status"], "superseded")
        self.assertEqual(old_update_changes["superseded_by_memory_id"], str(new_id))

        # Pending conflict records linked to the prior active memory are resolved.
        pending_filter = collection.update_many.call_args.args[0]
        pending_changes = collection.update_many.call_args.args[1]["$set"]
        self.assertEqual(pending_filter["status"], "needs_review")
        self.assertEqual(pending_filter["conflict_with_memory_id"], str(old_id))
        self.assertEqual(pending_changes["status"], "resolved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
