
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from bson import ObjectId

from app.services import memory_service


class TestMemoryLifecycleIntegration(unittest.TestCase):
    def test_record_memory_access_increments_counter_and_sets_timestamp(self):
        collection = Mock()
        memory_id = ObjectId()

        with patch.object(memory_service, "memory_collection", collection):
            memory_service._record_memory_access(
                "stage18-test-user", [{"memory_id": str(memory_id)}]
            )

        collection.update_many.assert_called_once()
        query, update = collection.update_many.call_args.args
        self.assertEqual(query["user_id"], "stage18-test-user")
        self.assertEqual(query["_id"]["$in"], [memory_id])
        self.assertEqual(update["$inc"], {"access_count": 1})
        self.assertIsInstance(update["$set"]["last_accessed_at"], datetime)

    def test_reinforce_memory_updates_target_users_active_memory(self):
        collection = Mock()
        collection.update_one.return_value.matched_count = 1
        memory_id = ObjectId()
        collection.find_one.return_value = {
            "_id": memory_id,
            "user_id": "stage18-test-user",
            "reinforcement_count": 1,
            "last_reinforced_at": datetime.now(timezone.utc),
        }

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "audit_collection", Mock()),
        ):
            result = memory_service.reinforce_memory(
                "stage18-test-user", str(memory_id)
            )

        self.assertTrue(result["success"])
        query, update = collection.update_one.call_args.args
        self.assertEqual(query["_id"], memory_id)
        self.assertEqual(query["user_id"], "stage18-test-user")
        self.assertEqual(update["$inc"], {"reinforcement_count": 1})
        self.assertIsInstance(update["$set"]["last_reinforced_at"], datetime)

    def test_reinforcement_is_scoped_to_user(self):
        collection = Mock()
        collection.update_one.return_value.matched_count = 0
        memory_id = ObjectId()

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.reinforce_memory(
                "wrong-user", str(memory_id)
            )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["error"], "Active memory not found for this user"
        )


if __name__ == "__main__":
    unittest.main()
