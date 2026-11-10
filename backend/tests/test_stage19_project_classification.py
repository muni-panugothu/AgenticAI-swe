"""Project fact classification, retrieval intent, and primary update tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from uuid import uuid4

from bson import ObjectId

from app import chatbot
from app.services.memory_conflict_service import (
    classify_project_fact,
    detect_memory_conflict,
    extract_exclusive_fact,
    find_conflicting_memories,
)
from app.services.memory_service import (
    _project_intent_boost,
    detect_conflicting_active_facts,
)


class TestProjectClassification(unittest.TestCase):
    def test_primary_project_exclusive(self):
        self.assertEqual(
            classify_project_fact("My project is Beta."),
            ("primary_project", "beta"),
        )
        self.assertEqual(
            extract_exclusive_fact("My primary project is Orion-QA."),
            ("primary_project", "orion qa"),
        )

    def test_named_project_not_exclusive(self):
        self.assertEqual(
            classify_project_fact(
                "Freshcart is my grocery marketplace project."
            )[0],
            "project_name",
        )
        self.assertIsNone(
            extract_exclusive_fact(
                "Freshcart is my grocery marketplace project."
            )
        )

    def test_description_not_exclusive(self):
        self.assertEqual(
            classify_project_fact(
                "I am building an Agentic AI bug resolver."
            ),
            ("project_description", "agentic ai bug resolver"),
        )
        self.assertIsNone(
            extract_exclusive_fact(
                "I am building an Agentic AI bug resolver."
            )
        )

    def test_two_named_projects_do_not_exclusive_conflict(self):
        conflicts = find_conflicting_memories(
            "Freshcart is my grocery marketplace project.",
            [
                {
                    "memory_id": "1",
                    "memory": "Orion is my QA tooling project.",
                }
            ],
        )
        self.assertEqual(conflicts, [])

    def test_two_primary_values_conflict(self):
        conflicts = find_conflicting_memories(
            "My primary project is Freshcart.",
            [{"memory_id": "1", "memory": "My project is Beta"}],
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["matched_fact_key"], "primary_project")

    def test_freshcart_vs_agentic_not_primary_conflict(self):
        conflicts = detect_conflicting_active_facts(
            [
                {
                    "memory": "Freshcart is my grocery marketplace project.",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "freshcart",
                },
                {
                    "memory": "I am building an Agentic AI bug resolver.",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "agentic ai bug resolver",
                },
            ]
        )
        self.assertEqual(conflicts, [])

    def test_legacy_alpha_beta_still_conflict_as_primary(self):
        conflicts = detect_conflicting_active_facts(
            [
                {
                    "memory": "my project is Alpha",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                },
                {
                    "memory": "my project is Beta",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "beta",
                },
            ]
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["fact_key"], "primary_project")


class TestProjectRetrievalIntent(unittest.TestCase):
    def test_primary_query_prefers_primary_fact(self):
        primary = {
            "fact_key": "primary_project",
            "memory": "My project is Beta",
        }
        named = {
            "fact_key": "project_name",
            "memory": "Freshcart is my grocery marketplace project",
        }
        q = "What is my current primary project?"
        self.assertGreater(
            _project_intent_boost(q, primary),
            _project_intent_boost(q, named),
        )

    def test_grocery_query_prefers_named_project(self):
        primary = {
            "fact_key": "primary_project",
            "memory": "My project is Beta",
        }
        named = {
            "fact_key": "project_name",
            "memory": "Freshcart is my grocery marketplace project",
        }
        q = "What grocery marketplace project am I building?"
        self.assertGreater(
            _project_intent_boost(q, named),
            _project_intent_boost(q, primary),
        )

    def test_agentic_query_prefers_description(self):
        desc = {
            "fact_key": "project_description",
            "memory": "I am building an Agentic AI bug resolver.",
        }
        primary = {
            "fact_key": "primary_project",
            "memory": "My project is Beta",
        }
        q = "What is my Agentic AI project?"
        self.assertGreater(
            _project_intent_boost(q, desc),
            _project_intent_boost(q, primary),
        )


class TestPrimaryProjectUpdate(unittest.TestCase):
    @patch.object(chatbot, "retire_sibling_exclusive_actives")
    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "_collect_update_candidates")
    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    def test_update_primary_retires_multiple_primaries(
        self, _embed, collect, supersede, retire
    ):
        collect.return_value = (
            [
                {
                    "memory_id": "a",
                    "memory": "My project is Alpha",
                    "status": "active",
                    "fact_key": "primary_project",
                    "normalized_value": "alpha",
                    "reinforcement_count": 1,
                },
                {
                    "memory_id": "b",
                    "memory": "My project is Beta",
                    "status": "active",
                    "fact_key": "primary_project",
                    "normalized_value": "beta",
                    "reinforcement_count": 2,
                },
            ],
            [],
        )
        supersede.return_value = {
            "success": True,
            "memory_id": "fresh",
            "old_memory_id": "b",
            "version": 2,
        }
        retire.return_value = {"success": True, "retired_count": 1}

        result = chatbot.try_update_explicit_memory(
            "Update my primary project to Freshcart.",
            "user-1",
        )
        self.assertTrue(result["success"])
        supersede.assert_called_once()
        retire.assert_called_once()

    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "_collect_update_candidates")
    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    def test_generic_update_still_asks_when_multiple_primaries(
        self, _embed, collect, supersede
    ):
        collect.return_value = (
            [
                {
                    "memory_id": "a",
                    "memory": "My project is Alpha",
                    "status": "active",
                    "normalized_value": "alpha",
                    "reinforcement_count": 1,
                },
                {
                    "memory_id": "b",
                    "memory": "My project is Beta",
                    "status": "active",
                    "normalized_value": "beta",
                    "reinforcement_count": 1,
                },
            ],
            [],
        )
        result = chatbot.try_update_explicit_memory(
            "Update my project to Freshcart.",
            "user-1",
        )
        self.assertFalse(result["success"])
        self.assertIn("primary project", result["message"].lower())
        supersede.assert_not_called()


class TestLegacyRepairDryRun(unittest.TestCase):
    def test_detects_combined_and_requires_approval_path(self):
        from scripts.repair_combined_memories import (
            _is_combined_candidate,
            propose_repairs,
        )

        self.assertTrue(
            _is_combined_candidate(
                "my preferred code editor is VS Code."
                "I prefer IntelliJ IDEA as my code editor"
            )
        )
        self.assertTrue(
            _is_combined_candidate(
                "Freshcart is my grocery marketplace project."
                "Remember that I am training for a marathon."
            )
        )


if __name__ == "__main__":
    unittest.main()
