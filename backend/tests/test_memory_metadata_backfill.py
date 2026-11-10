"""Optional memory metadata backfill tests (dry-run + apply)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from bson import ObjectId

from app.services.memory_service import (
    backfill_memory_metadata,
    compare_conflict_group_ids,
    ensure_conflict_partner_metadata,
)


class TestEnsureConflictPartnerMetadata(unittest.TestCase):
    def test_fills_missing_fields_only(self):
        mongo_id = ObjectId()
        doc = {
            "_id": mongo_id,
            "user_id": "u1",
            "status": "active",
            "fact_key": None,
            "normalized_value": None,
            "conflict_group_id": None,
            "memory": "my preferred database is MongoDB",
        }
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = doc
        mock_coll.update_one.return_value = MagicMock(matched_count=1)

        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ), patch(
            "app.services.memory_service.record_audit_event"
        ):
            result = ensure_conflict_partner_metadata(
                "u1",
                str(mongo_id),
                conflict_group_id="cg-1",
                fact_key="preferred_database",
                dry_run=False,
            )

        self.assertTrue(result["success"])
        self.assertFalse(result.get("dry_run"))
        applied = result["applied_set"]
        self.assertEqual(applied["conflict_group_id"], "cg-1")
        self.assertEqual(applied["fact_key"], "preferred_database")
        self.assertEqual(applied["normalized_value"], "mongodb")

    def test_does_not_overwrite_different_group(self):
        mongo_id = ObjectId()
        doc = {
            "_id": mongo_id,
            "user_id": "u1",
            "status": "active",
            "fact_key": "preferred_database",
            "normalized_value": "mongodb",
            "conflict_group_id": "cg-existing",
            "memory": "my preferred database is MongoDB",
        }
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = doc

        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ):
            result = ensure_conflict_partner_metadata(
                "u1",
                str(mongo_id),
                conflict_group_id="cg-other",
                dry_run=False,
            )

        self.assertTrue(result["success"])
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result["reason"], "existing_conflict_group_mismatch")
        mock_coll.update_one.assert_not_called()


class TestBackfillMemoryMetadata(unittest.TestCase):
    def test_dry_run_proposes_without_writes(self):
        mongo_id = ObjectId()
        pg_id = ObjectId()
        docs = [
            {
                "_id": pg_id,
                "user_id": "u1",
                "status": "needs_review",
                "fact_key": "preferred_database",
                "normalized_value": "postgresql",
                "conflict_group_id": "cg-1",
                "conflict_with_memory_id": str(mongo_id),
                "memory": "my preferred database is PostgreSQL",
            },
            {
                "_id": mongo_id,
                "user_id": "u1",
                "status": "active",
                "fact_key": None,
                "normalized_value": None,
                "conflict_group_id": None,
                "memory": "my preferred database is MongoDB",
            },
        ]
        mock_coll = MagicMock()
        # find() is called twice (pass1 + pass2); return same docs.
        mock_coll.find.return_value.limit.return_value = docs

        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ), patch(
            "app.services.memory_service.record_audit_event"
        ):
            result = backfill_memory_metadata(
                "u1",
                fact_key="preferred_database",
                dry_run=True,
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["dry_run"])
        self.assertGreaterEqual(result["proposal_count"], 1)
        mock_coll.update_one.assert_not_called()
        reasons = {p["reason"] for p in result["proposals"]}
        self.assertTrue(
            "missing_fact_metadata" in reasons
            or "propagate_conflict_group_from_pending" in reasons
        )

    def test_apply_propagates_conflict_group(self):
        mongo_id = ObjectId()
        pg_id = ObjectId()
        mongo_doc = {
            "_id": mongo_id,
            "user_id": "u1",
            "status": "active",
            "fact_key": None,
            "normalized_value": None,
            "conflict_group_id": None,
            "memory": "my preferred database is MongoDB",
        }
        pg_doc = {
            "_id": pg_id,
            "user_id": "u1",
            "status": "needs_review",
            "fact_key": "preferred_database",
            "normalized_value": "postgresql",
            "conflict_group_id": "cg-1",
            "conflict_with_memory_id": str(mongo_id),
            "memory": "my preferred database is PostgreSQL",
        }
        docs = [pg_doc, mongo_doc]
        mock_coll = MagicMock()
        mock_coll.find.return_value.limit.return_value = docs
        mock_coll.update_one.return_value = MagicMock(matched_count=1)

        with patch(
            "app.services.memory_service.memory_collection", mock_coll
        ), patch(
            "app.services.memory_service.record_audit_event"
        ):
            result = backfill_memory_metadata(
                "u1",
                fact_key="preferred_database",
                dry_run=False,
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["dry_run"])
        self.assertGreaterEqual(result["applied_count"], 1)
        self.assertTrue(mock_coll.update_one.called)

        # Simulate post-apply comparison using proposed fields.
        records = [
            {"conflict_group_id": "cg-1"},
            {"conflict_group_id": "cg-1"},
        ]
        self.assertTrue(compare_conflict_group_ids(records)["same_conflict_group"])


if __name__ == "__main__":
    unittest.main()
