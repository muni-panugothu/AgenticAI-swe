"""Regression tests for MongoDB update failure + no-whitespace extraction."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from bson import ObjectId

from app import chatbot
from app.services import memory_service
from app.services.memory_conflict_service import detect_memory_conflict
from app.services.memory_fact_extraction import (
    extract_facts_from_message,
    split_statements,
)
from app.utils.json_safe import dumps_json_safe, is_json_serializable


class TestNoWhitespaceExtraction(unittest.TestCase):
    def test_split_period_without_space(self):
        parts = split_statements(
            "my preferred code editor is VS Code.I prefer IntelliJ IDEA as my code editor"
        )
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(any("VS Code" in p for p in parts))
        self.assertTrue(any("IntelliJ" in p for p in parts))

    def test_extract_two_editor_facts_without_space(self):
        result = extract_facts_from_message(
            "Remember that my preferred code editor is VS Code."
            "I prefer IntelliJ IDEA as my code editor."
        )
        self.assertEqual(len(result.facts), 2)
        self.assertEqual(result.facts[0].fact_key, "editor")
        self.assertEqual(result.facts[1].fact_key, "editor")
        values = {f.normalized_value for f in result.facts}
        self.assertEqual(values, {"vs code", "intellij idea"})

    def test_preserves_decimals_and_abbreviations(self):
        parts = split_statements("Remember that version 11.0 is stable.")
        self.assertEqual(len(parts), 1)
        parts2 = split_statements("Remember that I use Dr. Smith as mentor.")
        self.assertEqual(len(parts2), 1)

    def test_multiple_remember_without_space(self):
        result = extract_facts_from_message(
            "Remember that my primary project is Orion-QA."
            "Remember that my primary project is Nova-CRM."
        )
        self.assertEqual(len(result.facts), 2)
        values = {f.normalized_value for f in result.facts}
        self.assertEqual(values, {"orion qa", "nova crm"})

    def test_editor_conflict_detection(self):
        conflict = detect_memory_conflict(
            "I prefer IntelliJ IDEA as my code editor.",
            [
                {
                    "memory_id": "1",
                    "memory": "My preferred code editor is VS Code.",
                    "fact_key": "editor",
                    "normalized_value": "vs code",
                }
            ],
        )
        self.assertTrue(conflict.get("conflicts"))


class TestExplicitUpdatePaths(unittest.TestCase):
    @patch.object(chatbot, "activate_confirmed_memory")
    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    @patch.object(chatbot, "_collect_update_candidates")
    def test_pending_only_uses_activate_not_supersede(
        self, collect, _embed, activate
    ):
        collect.return_value = (
            [],
            [
                {
                    "memory_id": "pending-1",
                    "memory": "my project is Alpha",
                    "status": "needs_review",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                }
            ],
        )
        activate.return_value = {
            "success": True,
            "memory_id": "new-active",
            "version": 1,
            "activated_from_pending": True,
        }

        with patch.object(chatbot, "supersede_memory") as supersede:
            result = chatbot.try_update_explicit_memory(
                "Update my project to Freshcart.",
                "user-1",
            )
            supersede.assert_not_called()

        self.assertTrue(result["success"])
        self.assertIn("Freshcart", result["message"])
        activate.assert_called_once()
        self.assertTrue(is_json_serializable(result))

    @patch.object(chatbot, "retire_sibling_exclusive_actives")
    @patch.object(chatbot, "supersede_memory")
    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    @patch.object(chatbot, "_collect_update_candidates")
    def test_active_path_supersedes(
        self, collect, _embed, supersede, retire
    ):
        collect.return_value = (
            [
                {
                    "memory_id": "active-1",
                    "memory": "my project is Alpha",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                    "reinforcement_count": 1,
                }
            ],
            [],
        )
        supersede.return_value = {
            "success": True,
            "memory_id": "new-1",
            "old_memory_id": "active-1",
            "version": 2,
        }
        retire.return_value = {"success": True, "retired_count": 0}

        result = chatbot.try_update_explicit_memory(
            "Update my project to Freshcart.",
            "user-1",
        )
        self.assertTrue(result["success"])
        supersede.assert_called_once()
        self.assertEqual(result.get("version"), 2)
        dumps_json_safe(result)

    @patch.object(chatbot, "create_embedding", return_value=[0.1] * 8)
    @patch.object(chatbot, "_collect_update_candidates")
    def test_surfaces_supersede_phase_on_failure(self, collect, _embed):
        collect.return_value = (
            [
                {
                    "memory_id": "active-1",
                    "memory": "my project is Alpha",
                    "status": "active",
                    "fact_key": "project",
                    "normalized_value": "alpha",
                    "reinforcement_count": 0,
                }
            ],
            [],
        )
        with patch.object(
            chatbot,
            "supersede_memory",
            return_value={
                "success": False,
                "error": "Only an active memory can be superseded",
                "error_code": "not_active",
                "phase": "load_existing",
            },
        ):
            result = chatbot.try_update_explicit_memory(
                "Update my project to Freshcart.",
                "user-1",
            )
        self.assertFalse(result["success"])
        self.assertIn("phase=load_existing", result["message"])


class TestSupersedeServiceDiagnostics(unittest.TestCase):
    def test_not_active_returns_structured_error(self):
        collection = MagicMock()
        memory_id = ObjectId()
        collection.find_one.return_value = {
            "_id": memory_id,
            "user_id": "u1",
            "status": "needs_review",
            "version": 1,
        }

        fake_session = MagicMock()
        fake_session.__enter__ = MagicMock(return_value=fake_session)
        fake_session.__exit__ = MagicMock(return_value=False)
        fake_session.start_transaction.return_value = fake_session

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "_find_by_operation_id", return_value=None),
            patch.object(
                memory_service.client, "start_session", return_value=fake_session
            ),
        ):
            result = memory_service.supersede_memory(
                user_id="u1",
                memory_id=str(memory_id),
                memory="My project is Freshcart",
                embedding=[0.1],
                fact_key="project",
                normalized_value="freshcart",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result.get("error_code"), "not_active")
        self.assertIn("needs_review", result.get("error", ""))

    def test_missing_memory_structured_error(self):
        collection = MagicMock()
        collection.find_one.return_value = None
        memory_id = ObjectId()
        fake_session = MagicMock()
        fake_session.__enter__ = MagicMock(return_value=fake_session)
        fake_session.__exit__ = MagicMock(return_value=False)
        fake_session.start_transaction.return_value = fake_session

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "_find_by_operation_id", return_value=None),
            patch.object(
                memory_service.client, "start_session", return_value=fake_session
            ),
        ):
            result = memory_service.supersede_memory(
                user_id="u1",
                memory_id=str(memory_id),
                memory="My project is Freshcart",
                embedding=None,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result.get("error_code"), "not_found")

    def test_cross_user_not_found(self):
        collection = MagicMock()
        # find_one filters by user_id so wrong user ⇒ None
        collection.find_one.return_value = None
        memory_id = ObjectId()
        fake_session = MagicMock()
        fake_session.__enter__ = MagicMock(return_value=fake_session)
        fake_session.__exit__ = MagicMock(return_value=False)
        fake_session.start_transaction.return_value = fake_session

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "_find_by_operation_id", return_value=None),
            patch.object(
                memory_service.client, "start_session", return_value=fake_session
            ),
        ):
            result = memory_service.supersede_memory(
                user_id="attacker",
                memory_id=str(memory_id),
                memory="hacked",
                embedding=None,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result.get("error_code"), "not_found")


class TestLegacyCombinedHandling(unittest.TestCase):
    def test_repair_proposer_detects_combined_editor(self):
        from scripts.repair_combined_memories import _is_combined_candidate

        self.assertTrue(
            _is_combined_candidate(
                "my preferred code editor is VS Code."
                "I prefer IntelliJ IDEA as my code editor"
            )
        )
        self.assertFalse(
            _is_combined_candidate("my preferred code editor is VS Code")
        )


class TestUpdateParsingRegression(unittest.TestCase):
    def test_update_value_only(self):
        parsed = chatbot._parse_explicit_update(
            "Update my project to Freshcart. Then verify reinforcement_count should be 3"
        )
        self.assertEqual(parsed[1], "Freshcart")


if __name__ == "__main__":
    unittest.main()
