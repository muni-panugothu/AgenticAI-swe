"""Stage 19 — Memory lifecycle, idempotency, and reliability tests.

Covers:
- Repeated reinforcement idempotency.
- Duplicate request retry with same operation ID.
- Access tracking does not affect freshness.
- Stale memory identification and recovery.
- Save idempotency with operation_id.
- Concurrent conflicting updates (mocked).
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from bson import ObjectId

from app.services import memory_service
from app.services.memory_lifecycle_service import (
    calculate_memory_freshness_score,
    identify_stale_memory,
)


class TestReinforcementIdempotency(unittest.TestCase):
    """Reinforcing the same memory with the same operation_id is idempotent."""

    def test_idempotent_reinforcement_with_operation_id(self):
        collection = MagicMock()
        audit = MagicMock()
        memory_id = ObjectId()
        op_id = "reinforce-test-op-001"

        # First call: no prior audit event for this operation_id.
        audit.find_one.return_value = None
        collection.update_one.return_value.matched_count = 1
        collection.find_one.return_value = {
            "_id": memory_id,
            "user_id": "test-user",
            "reinforcement_count": 1,
        }

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "audit_collection", audit),
        ):
            result1 = memory_service.reinforce_memory(
                "test-user", str(memory_id), operation_id=op_id
            )

        self.assertTrue(result1["success"])
        self.assertFalse(result1.get("idempotent", False))

    def test_replay_reinforcement_is_detected(self):
        collection = MagicMock()
        audit = MagicMock()
        memory_id = ObjectId()
        op_id = "reinforce-test-op-002"

        # Replay is detected via the audit trail for this operation_id.
        audit.find_one.return_value = {
            "user_id": "test-user",
            "memory_id": str(memory_id),
            "operation_id": op_id,
            "event_type": "memory_reinforced",
        }
        collection.find_one.return_value = {
            "_id": memory_id,
            "user_id": "test-user",
            "last_reinforced_at": datetime.now(timezone.utc),
            "reinforcement_count": 1,
        }

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "audit_collection", audit),
        ):
            result = memory_service.reinforce_memory(
                "test-user", str(memory_id), operation_id=op_id
            )

        self.assertTrue(result["success"])
        self.assertTrue(result.get("idempotent"))
        collection.update_one.assert_not_called()


class TestSaveIdempotency(unittest.TestCase):
    """Saving with the same operation_id should be idempotent."""

    def test_replay_save_returns_original(self):
        collection = MagicMock()
        op_id = "save-test-op-001"
        original_id = ObjectId()

        collection.find_one.return_value = {
            "_id": original_id,
            "user_id": "test-user",
            "memory": "My role is Engineer",
            "importance": 3,
            "status": "active",
            "version": 1,
            "operation_id": op_id,
        }

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.save_memory(
                user_id="test-user",
                memory="My role is Engineer",
                embedding=[0.1],
                operation_id=op_id,
            )

        self.assertTrue(result["success"])
        self.assertTrue(result.get("idempotent"))
        self.assertEqual(result["memory_id"], str(original_id))


class TestAccessTrackingDoesNotAffectFreshness(unittest.TestCase):
    """Access count and last_accessed_at must not change freshness score."""

    def test_access_does_not_change_freshness(self):
        now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
        base = {
            "importance": 3,
            "created_at": now - timedelta(days=90),
        }
        accessed = {
            **base,
            "last_accessed_at": now,
            "access_count": 1000,
        }
        self.assertEqual(
            calculate_memory_freshness_score(base, now=now),
            calculate_memory_freshness_score(accessed, now=now),
        )


class TestStaleMemoryRecovery(unittest.TestCase):
    """Reinforcement should clear stale flags."""

    def test_reinforce_clears_stale_flag(self):
        collection = MagicMock()
        audit = MagicMock()
        memory_id = ObjectId()
        audit.find_one.return_value = None
        collection.update_one.return_value.matched_count = 1
        collection.find_one.return_value = {
            "_id": memory_id,
            "user_id": "test-user",
            "reinforcement_count": 1,
        }

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "audit_collection", audit),
        ):
            result = memory_service.reinforce_memory(
                "test-user", str(memory_id)
            )

        self.assertTrue(result["success"])
        update_doc = collection.update_one.call_args.args[1]
        self.assertFalse(update_doc["$set"]["is_stale"])
        self.assertIsNone(update_doc["$set"]["stale_reason"])

    def test_stale_identification_on_old_memory(self):
        now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
        old = {
            "memory_id": "stale-1",
            "importance": 1,
            "created_at": now - timedelta(days=400),
        }
        result = identify_stale_memory(old, now=now)
        self.assertTrue(result["is_stale"])
        self.assertIn("Freshness score", result["stale_reason"])

    def test_recent_memory_not_stale(self):
        now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
        recent = {
            "memory_id": "fresh-1",
            "importance": 5,
            "created_at": now - timedelta(days=1),
        }
        result = identify_stale_memory(recent, now=now)
        self.assertFalse(result["is_stale"])


class TestConcurrentUpdateDetection(unittest.TestCase):
    """supersede_memory should detect when the target was already superseded."""

    def test_supersede_fails_if_already_superseded(self):
        collection = MagicMock()
        old_id = ObjectId()

        # The memory exists but is already superseded.
        collection.find_one.return_value = {
            "_id": old_id,
            "user_id": "test-user",
            "memory": "Old fact",
            "status": "superseded",
            "importance": 3,
            "confidence": 0.75,
            "version": 1,
        }

        fake_session = MagicMock()
        fake_session.__enter__ = MagicMock(return_value=fake_session)
        fake_session.__exit__ = MagicMock(return_value=False)
        fake_session.start_transaction.return_value = fake_session

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service.client, "start_session", return_value=fake_session),
        ):
            result = memory_service.supersede_memory(
                user_id="test-user",
                memory_id=str(old_id),
                memory="New fact",
                embedding=[0.1],
            )

        self.assertFalse(result["success"])
        self.assertIn("active", result["error"].lower())


class TestPartialFailureHandling(unittest.TestCase):
    """Operations should return clean error responses on exceptions."""

    def test_save_returns_error_on_exception(self):
        collection = MagicMock()
        collection.find_one.return_value = None
        collection.insert_one.side_effect = Exception("DB connection lost")

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.save_memory(
                user_id="test-user",
                memory="Test memory",
                embedding=[0.1],
            )

        self.assertFalse(result["success"])
        self.assertIn("DB connection lost", result["error"])

    def test_delete_returns_error_on_invalid_id(self):
        result = memory_service.delete_memory(
            user_id="test-user",
            memory_id="not-valid",
            soft=True,
        )
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
