"""Regression tests for scoped memory summarization isolation."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from bson import ObjectId

from app.services.memory_fact_extraction import (
    extract_facts_from_message,
    is_summary_artifact_text,
)
from app.services.memory_summary_service import (
    build_conflict_references,
    filter_memories_for_scope,
    generate_summary_for_type,
    memory_scope_keys,
    synthesize_summary_text,
    validate_sources_in_scope,
    verify_summary_currency,
)


def _doc(text, *, status="active", fact_key=None, value=None, user="u1", mid=None):
    return {
        "_id": mid or ObjectId(),
        "user_id": user,
        "memory": text,
        "status": status,
        "fact_key": fact_key,
        "normalized_value": value,
        "updated_at": datetime.now(timezone.utc),
    }


class TestProjectScopeIsolation(unittest.TestCase):
    def test_listifys_excludes_unrelated_projects(self):
        docs = [
            _doc(
                "Listifys is my classified marketplace project",
                fact_key="project_name",
                value="listifys (classified marketplace)",
            ),
            _doc("My project is Alpha", fact_key="primary_project", value="alpha"),
            _doc("my project is Beta", fact_key="primary_project", value="beta"),
            _doc(
                "Freshcart is my grocery marketplace project",
                fact_key="project_name",
                value="freshcart (grocery marketplace)",
            ),
            _doc("my goal is to finish a marathon in under four hours", fact_key="goal"),
        ]
        scoped = filter_memories_for_scope(
            docs, summary_type="project", scope_key="listifys"
        )
        self.assertEqual(len(scoped), 1)
        self.assertIn("Listifys", scoped[0]["memory"])
        text = synthesize_summary_text(
            "project", scoped, build_conflict_references(scoped, scope_key="listifys")
        ).lower()
        self.assertIn("listifys", text)
        for foreign in ("alpha", "beta", "freshcart", "marathon", "nova", "orion"):
            self.assertNotIn(foreign, text)

    def test_primary_conflicts_isolated_from_named_project(self):
        alpha = _doc(
            "my primary project is Alpha",
            fact_key="primary_project",
            value="alpha",
        )
        alpha["conflict_group_id"] = "g1"
        nova = _doc(
            "my primary project is Nova-CRM",
            status="needs_review",
            fact_key="primary_project",
            value="nova crm",
        )
        nova["conflict_group_id"] = "g1"
        listify = _doc(
            "Listifys is my classified marketplace project",
            fact_key="project_name",
            value="listifys (classified marketplace)",
        )
        self.assertEqual(memory_scope_keys(alpha, "project"), {"primary"})
        self.assertEqual(memory_scope_keys(listify, "project"), {"listifys"})

        primary = filter_memories_for_scope(
            [alpha, nova, listify], summary_type="project", scope_key="primary"
        )
        refs = build_conflict_references(primary, scope_key="primary")
        self.assertTrue(refs)
        self.assertEqual(refs[0]["conflict_kind"], "active_vs_pending")
        text = synthesize_summary_text("project", primary, refs).lower()
        self.assertIn("unresolved", text)
        self.assertNotIn("listifys", text)


class TestSourceValidation(unittest.TestCase):
    def test_rejects_out_of_scope_and_artifacts(self):
        good = _doc(
            "Listifys is my classified marketplace project",
            fact_key="project_name",
            value="listifys (classified marketplace)",
        )
        bad = _doc("My project is Alpha", fact_key="primary_project", value="alpha")
        artifact = _doc(
            "Summary: The user has a classified marketplace project named Listifys. "
            "Source memory ID(s): abc"
        )
        clean, rejected = validate_sources_in_scope(
            [good, bad, artifact],
            user_id="u1",
            summary_type="project",
            scope_key="listifys",
        )
        self.assertEqual(len(clean), 1)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(len({_ for _ in [str(good["_id"])]}), 1)

    def test_summary_artifact_detection(self):
        self.assertTrue(
            is_summary_artifact_text(
                "Summary: The user has a classified marketplace project named Listifys"
            )
        )


class TestIngestContamination(unittest.TestCase):
    def test_splits_and_drops_appended_question(self):
        result = extract_facts_from_message(
            "Remember that Freshcart is my grocery marketplace project. "
            "Remember that I am training for a marathon. "
            "What grocery marketplace project am I building?"
        )
        statements = [f.statement.lower() for f in result.facts]
        texts = " ".join(statements)
        self.assertTrue(any("freshcart" in s for s in statements))
        self.assertTrue(any("marathon" in s for s in statements))
        self.assertFalse(any(s.strip().startswith("what ") for s in statements))
        self.assertNotIn("what grocery marketplace project am i building", texts)


class TestGenerateScopedSummary(unittest.TestCase):
    def test_persists_matching_count_and_scope(self):
        mid = ObjectId()
        source = _doc(
            "Listifys is my classified marketplace project",
            fact_key="project_name",
            value="listifys (classified marketplace)",
            mid=mid,
        )
        mock_sum = MagicMock()
        mock_sum.find_one.side_effect = [None, None, None]
        inserted = ObjectId()
        mock_sum.insert_one.return_value = MagicMock(inserted_id=inserted)

        stored = {
            "_id": inserted,
            "user_id": "u1",
            "summary_type": "project",
            "scope_key": "listifys",
            "summary_text": "Project facts: project name: listifys (classified marketplace).",
            "source_memory_ids": [str(mid)],
            "source_memory_count": 1,
            "source_hash": "will-replace",
            "version": 1,
            "status": "active",
            "conflict_references": [],
            "metadata": {"scope_key": "listifys"},
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        mock_mem = MagicMock()
        mock_mem.find.return_value.limit.return_value = [
            source,
            _doc("My project is Alpha", fact_key="primary_project", value="alpha"),
        ]

        def _find_one(query, *args, **kwargs):
            if query.get("_id") == inserted:
                # After insert, return stored with live hash filled by caller path.
                return stored
            if query.get("operation_id"):
                return None
            if query.get("status") == "active":
                return None
            if query.get("_id") == mid:
                return source
            return None

        mock_sum.find_one.side_effect = _find_one

        with patch(
            "app.services.memory_summary_service._summaries",
            return_value=mock_sum,
        ), patch(
            "app.services.memory_summary_service._memories",
            return_value=mock_mem,
        ), patch(
            "app.services.memory_summary_service.record_audit_event"
        ), patch(
            "app.services.memory_summary_service.verify_summary_currency",
            return_value={"current": True, "issues": []},
        ):
            result = generate_summary_for_type(
                "u1", "project", scope_key="listifys", force=True
            )

        self.assertTrue(result["success"], result)
        payload = mock_sum.insert_one.call_args[0][0]
        self.assertEqual(payload["scope_key"], "listifys")
        self.assertEqual(payload["source_memory_count"], 1)
        self.assertEqual(payload["source_memory_ids"], [str(mid)])
        self.assertNotIn("alpha", payload["summary_text"].lower())


if __name__ == "__main__":
    unittest.main()
