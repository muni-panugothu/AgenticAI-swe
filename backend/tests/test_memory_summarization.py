"""Unit tests for conflict-aware memory summarization (no Mongo / no LLM)."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from bson import ObjectId

from app.services.memory_summary_service import (
    build_conflict_references,
    classify_summary_type,
    compute_source_hash,
    synthesize_summary_text,
    validate_summary_text,
    generate_summary_for_type,
    search_memory_summaries,
)
from app.chatbot import build_memory_context


def _mem(text, *, status="active", fact_key=None, value=None, mid=None, group=None):
    oid = mid or ObjectId()
    return {
        "_id": oid if isinstance(oid, ObjectId) else ObjectId(oid),
        "user_id": "u1",
        "owner_id": "u1",
        "permission_level": "private",
        "shared_with": [],
        "memory": text,
        "status": status,
        "fact_key": fact_key,
        "normalized_value": value,
        "conflict_group_id": group,
        "updated_at": datetime.now(timezone.utc),
    }


class TestSummaryClassification(unittest.TestCase):
    def test_preferences_and_project(self):
        self.assertEqual(
            classify_summary_type(
                {"fact_key": "preferred_database", "memory": "x"}
            ),
            "preferences",
        )
        self.assertEqual(
            classify_summary_type(
                {"fact_key": "primary_project", "memory": "x"}
            ),
            "project",
        )
        self.assertEqual(
            classify_summary_type({"fact_key": "role", "memory": "x"}),
            "user",
        )


class TestConflictAwareSynthesis(unittest.TestCase):
    def test_unresolved_database_conflict_not_confirmed(self):
        mongo = _mem(
            "my preferred database is MongoDB",
            status="active",
            fact_key="preferred_database",
            value="mongodb",
            group="g1",
        )
        pg = _mem(
            "my preferred database is PostgreSQL",
            status="needs_review",
            fact_key="preferred_database",
            value="postgresql",
            group="g1",
        )
        refs = build_conflict_references([mongo, pg])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["conflict_kind"], "active_vs_pending")

        text = synthesize_summary_text("preferences", [mongo, pg], refs)
        lower = text.lower()
        self.assertIn("unresolved", lower)
        self.assertIn("mongodb", lower)
        self.assertIn("postgresql", lower)
        self.assertIn("clarification", lower)
        self.assertNotRegex(lower, r"\bprefers?\s+mongodb\b")
        self.assertNotRegex(lower, r"\bprefers?\s+postgresql\b")

        ok, issues = validate_summary_text(text, [mongo, pg], refs)
        self.assertTrue(ok, issues)

    def test_rejects_confirmed_language_under_conflict(self):
        refs = [{
            "fact_key": "preferred_database",
            "conflict_kind": "active_vs_pending",
            "active_values": ["mongodb"],
            "pending_values": ["postgresql"],
        }]
        bad = "The user prefers MongoDB as their database."
        ok, issues = validate_summary_text(bad, [], refs)
        self.assertFalse(ok)
        self.assertTrue(issues)

    def test_duplicate_actives_deduped(self):
        a = _mem(
            "my preferred code editor is VS Code",
            fact_key="editor",
            value="vs code",
        )
        b = _mem(
            "my preferred code editor is VS Code",
            fact_key="editor",
            value="vs code",
        )
        text = synthesize_summary_text("preferences", [a, b], [])
        self.assertEqual(text.lower().count("vs code"), 1)

    def test_source_hash_changes_with_status(self):
        a = _mem("my role is Engineer", fact_key="role", value="engineer")
        h1 = compute_source_hash([a])
        a2 = dict(a)
        a2["status"] = "needs_review"
        h2 = compute_source_hash([a2])
        self.assertNotEqual(h1, h2)


class TestSummaryPersistenceMocks(unittest.TestCase):
    def test_generate_idempotent_on_same_hash(self):
        mid = ObjectId()
        source = _mem(
            "my preferred language is Python",
            fact_key="preferred_language",
            value="python",
            mid=mid,
        )
        source_hash = compute_source_hash([source])
        active = {
            "_id": ObjectId(),
            "user_id": "u1",
            "summary_type": "preferences",
            "scope_key": "preferred_language",
            "summary_text": "Confirmed preferences: preferred language: python.",
            "source_memory_ids": [str(mid)],
            "source_memory_count": 1,
            "source_hash": source_hash,
            "version": 1,
            "status": "active",
            "conflict_references": [],
            "metadata": {"scope_key": "preferred_language"},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        mock_sum = MagicMock()
        mock_sum.find_one.return_value = active
        mock_mem = MagicMock()
        mock_mem.find.return_value.limit.return_value = [source]

        with patch(
            "app.services.memory_summary_service._summaries",
            return_value=mock_sum,
        ), patch(
            "app.services.memory_summary_service._memories",
            return_value=mock_mem,
        ), patch(
            "app.services.memory_summary_service.verify_summary_currency",
            return_value={"current": True, "issues": []},
        ):
            result = generate_summary_for_type(
                "u1", "preferences", scope_key="preferred_language"
            )
        self.assertTrue(result["success"])
        self.assertTrue(result.get("unchanged"))
        mock_sum.insert_one.assert_not_called()

    def test_search_excludes_outdated(self):
        mock_sum = MagicMock()
        mock_sum.find.return_value.limit.return_value = [
            {
                "_id": ObjectId(),
                "user_id": "u1",
                "summary_type": "preferences",
                "summary_text": "database preference unresolved mongodb postgresql",
                "status": "outdated",
                "source_memory_ids": [],
                "source_memory_count": 0,
                "source_hash": "x",
                "version": 1,
                "conflict_references": [],
                "metadata": {},
            },
            {
                "_id": ObjectId(),
                "user_id": "u1",
                "summary_type": "preferences",
                "summary_text": "database preference unresolved mongodb postgresql",
                "status": "active",
                "source_memory_ids": ["a"],
                "source_memory_count": 1,
                "source_hash": "y",
                "version": 2,
                "conflict_references": [],
                "metadata": {},
            },
        ]
        with patch(
            "app.services.memory_summary_service._summaries",
            return_value=mock_sum,
        ), patch(
            "app.services.memory_summary_service.verify_summary_currency",
            return_value={"current": True, "issues": []},
        ):
            result = search_memory_summaries(
                "u1", "preferred database", include_outdated=False
            )
        self.assertTrue(result["success"])
        self.assertEqual(len(result["summaries"]), 1)
        self.assertEqual(result["summaries"][0]["status"], "active")

    def test_cross_user_generate_uses_user_filter(self):
        mock_sum = MagicMock()
        mock_sum.find_one.return_value = None
        mock_mem = MagicMock()
        mock_mem.find.return_value.limit.return_value = []
        inserted = ObjectId()
        mock_sum.insert_one.return_value = MagicMock(inserted_id=inserted)
        stored = {
            "_id": inserted,
            "user_id": "u1",
            "summary_type": "general",
            "summary_text": "No confirmed general facts are available from the current source memories.",
            "source_memory_ids": [],
            "source_memory_count": 0,
            "source_hash": compute_source_hash([]),
            "version": 1,
            "status": "active",
            "conflict_references": [],
            "metadata": {},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        # active lookup → None; post-insert verify → stored
        mock_sum.find_one.side_effect = [None, stored]

        with patch(
            "app.services.memory_summary_service._summaries",
            return_value=mock_sum,
        ), patch(
            "app.services.memory_summary_service._memories",
            return_value=mock_mem,
        ), patch(
            "app.services.memory_summary_service.record_audit_event"
        ):
            result = generate_summary_for_type("u1", "general", force=True)
        self.assertTrue(result["success"], result)
        # Memory load queried with user_id.
        mock_mem.find.assert_called()
        args = mock_mem.find.call_args[0][0]
        self.assertEqual(args["user_id"], "u1")


class TestSummaryContextIntegration(unittest.TestCase):
    def test_build_context_includes_summaries_not_outdated(self):
        context = build_memory_context(
            memories=[
                {
                    "memory": "my preferred database is MongoDB",
                    "status": "active",
                    "importance": 4,
                    "confidence": 0.9,
                }
            ],
            summaries=[
                {
                    "summary_type": "preferences",
                    "summary_text": (
                        "The user's preferred database is unresolved. "
                        "mongodb is recorded as active, while postgresql is "
                        "pending review. Clarification is required."
                    ),
                    "status": "active",
                    "source_memory_count": 2,
                    "conflict_references": [{"fact_key": "preferred_database"}],
                },
                {
                    "summary_type": "preferences",
                    "summary_text": "outdated junk prefers mongodb",
                    "status": "outdated",
                    "source_memory_count": 1,
                    "conflict_references": [],
                },
            ],
        )
        self.assertIn("MEMORY_SUMMARIES", context)
        self.assertIn("unresolved", context.lower())
        self.assertNotIn("outdated junk", context)


if __name__ == "__main__":
    unittest.main()
