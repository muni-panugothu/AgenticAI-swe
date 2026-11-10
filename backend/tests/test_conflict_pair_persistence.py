"""Bilateral conflict-metadata persistence regression tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch, call
from bson import ObjectId

from app.services.memory_service import (
    compare_conflict_group_ids,
    persist_memory_conflict_pair,
)


class TestCompareConflictInvariant(unittest.TestCase):
    def test_consistent_when_reciprocal(self):
        result = compare_conflict_group_ids(
            [
                {
                    "memory_id": "a",
                    "conflict_group_id": "cg",
                    "conflict_with_memory_id": "b",
                },
                {
                    "memory_id": "b",
                    "conflict_group_id": "cg",
                    "conflict_with_memory_id": "a",
                },
            ]
        )
        self.assertEqual(result["verdict"], "consistent")
        self.assertTrue(result["same_conflict_group"])
        self.assertTrue(result["reciprocal_references"])

    def test_asymmetric_references_detected(self):
        result = compare_conflict_group_ids(
            [
                {
                    "memory_id": "a",
                    "conflict_group_id": "cg",
                    "conflict_with_memory_id": None,
                },
                {
                    "memory_id": "b",
                    "conflict_group_id": "cg",
                    "conflict_with_memory_id": "a",
                },
            ]
        )
        self.assertEqual(result["verdict"], "asymmetric_references")
        self.assertFalse(result["same_conflict_group"])


class TestPersistMemoryConflictPair(unittest.TestCase):
    def _docs(self):
        left_id = ObjectId()
        right_id = ObjectId()
        left = {
            "_id": left_id,
            "user_id": "u1",
            "status": "active",
            "fact_key": None,
            "normalized_value": None,
            "conflict_group_id": None,
            "conflict_type": "none",
            "conflict_with_memory_id": None,
            "memory": "my preferred database is MongoDB",
        }
        right = {
            "_id": right_id,
            "user_id": "u1",
            "status": "needs_review",
            "fact_key": "preferred_database",
            "normalized_value": "postgresql",
            "conflict_group_id": "cg-existing",
            "conflict_type": "possible_contradiction",
            "conflict_with_memory_id": str(left_id),
            "memory": "my preferred database is PostgreSQL",
        }
        return left_id, right_id, left, right

    def test_dry_run_plans_bilateral_patch(self):
        left_id, right_id, left, right = self._docs()
        mock_coll = MagicMock()
        mock_coll.find_one.side_effect = [left, right]

        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ), patch("app.services.memory_service.client"):
            result = persist_memory_conflict_pair(
                "u1",
                str(left_id),
                str(right_id),
                fact_key="preferred_database",
                conflict_group_id="cg-existing",
                dry_run=True,
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["dry_run"])
        plan = result["plan"]
        self.assertEqual(plan["conflict_group_id"], "cg-existing")
        self.assertEqual(plan["left_set"]["fact_key"], "preferred_database")
        self.assertEqual(plan["left_set"]["normalized_value"], "mongodb")
        self.assertEqual(plan["left_set"]["conflict_group_id"], "cg-existing")
        self.assertEqual(plan["left_set"]["conflict_type"], "possible_contradiction")
        self.assertEqual(plan["left_set"]["conflict_with_memory_id"], str(right_id))
        self.assertEqual(plan["left_status_preserved"], "active")
        self.assertEqual(plan["right_status_preserved"], "needs_review")
        mock_coll.update_one.assert_not_called()

    def test_apply_updates_both_and_verifies(self):
        left_id, right_id, left, right = self._docs()
        left_after = dict(left)
        left_after.update(
            {
                "fact_key": "preferred_database",
                "normalized_value": "mongodb",
                "conflict_group_id": "cg-existing",
                "conflict_type": "possible_contradiction",
                "conflict_with_memory_id": str(right_id),
            }
        )
        right_after = dict(right)

        mock_coll = MagicMock()
        # Initial load, then post-write reads.
        mock_coll.find_one.side_effect = [left, right, left_after, right_after]
        mock_coll.update_one.return_value = MagicMock(matched_count=1)

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.start_transaction.return_value.__enter__ = MagicMock(
            return_value=None
        )
        mock_session.start_transaction.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_client = MagicMock()
        mock_client.start_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_client.start_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ), patch(
            "app.services.memory_service.client", mock_client
        ), patch(
            "app.services.memory_service.record_audit_event"
        ):
            result = persist_memory_conflict_pair(
                "u1",
                str(left_id),
                str(right_id),
                fact_key="preferred_database",
                conflict_group_id="cg-existing",
                dry_run=False,
            )

        self.assertTrue(result["success"], result)
        self.assertTrue(result.get("verified"))
        self.assertEqual(result["left"]["status"], "active")
        self.assertEqual(result["right"]["status"], "needs_review")
        self.assertEqual(result["left"]["conflict_group_id"], "cg-existing")
        self.assertEqual(
            result["left"]["conflict_with_memory_id"], str(right_id)
        )

    def test_rejects_cross_user(self):
        left_id = ObjectId()
        right_id = ObjectId()
        mock_coll = MagicMock()
        mock_coll.find_one.side_effect = [
            {
                "_id": left_id,
                "user_id": "u1",
                "memory": "my preferred database is MongoDB",
                "status": "active",
            },
            None,
        ]
        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ):
            result = persist_memory_conflict_pair(
                "u1",
                str(left_id),
                str(right_id),
                fact_key="preferred_database",
                dry_run=False,
            )
        self.assertFalse(result["success"])

    def test_idempotent_retry_reuses_group(self):
        left_id, right_id, left, right = self._docs()
        # Already consistent.
        left.update(
            {
                "fact_key": "preferred_database",
                "normalized_value": "mongodb",
                "conflict_group_id": "cg-existing",
                "conflict_type": "possible_contradiction",
                "conflict_with_memory_id": str(right_id),
            }
        )
        mock_coll = MagicMock()
        mock_coll.find_one.side_effect = [left, right, left, right]
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.start_transaction.return_value.__enter__ = MagicMock(
            return_value=None
        )
        mock_session.start_transaction.return_value.__exit__ = MagicMock(
            return_value=False
        )
        mock_client = MagicMock()
        mock_client.start_session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        mock_client.start_session.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ), patch(
            "app.services.memory_service.client", mock_client
        ), patch(
            "app.services.memory_service.record_audit_event"
        ):
            result = persist_memory_conflict_pair(
                "u1",
                str(left_id),
                str(right_id),
                fact_key="preferred_database",
                conflict_group_id="cg-new-should-be-ignored",
                dry_run=False,
            )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["conflict_group_id"], "cg-existing")


if __name__ == "__main__":
    unittest.main()
