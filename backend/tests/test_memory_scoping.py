"""Topic 20 — Memory scoping: user / project / tenant isolation."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

from bson import ObjectId

from app.services import memory_service
from app.services.memory_scope_service import (
    build_scope_filter,
    clear_active_project_scope,
    document_matches_scope,
    filter_documents_by_scope,
    get_active_project_scope,
    set_active_project_scope,
    validate_scope_access,
)


def _doc(
    *,
    user_id="user-a",
    memory="My preferred language is Python",
    project_id=None,
    tenant_id=None,
    status="active",
):
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "owner_id": user_id,
        "memory": memory,
        "status": status,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "scope": "project" if project_id else "user",
        "shared_with": [],
        "permission_level": "private",
        "importance": 3,
        "confidence": 0.9,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "version": 1,
    }


class TestScopeValidation(unittest.TestCase):
    def test_validate_requires_user_id(self):
        result = validate_scope_access(user_id="")
        self.assertFalse(result["success"])

    def test_validate_normalizes_blank_project(self):
        result = validate_scope_access(user_id="u1", project_id="  ")
        self.assertFalse(result["success"])

    def test_document_matches_user_scope_legacy(self):
        doc = _doc(project_id=None, tenant_id=None)
        self.assertTrue(
            document_matches_scope(doc, user_id="user-a")
        )
        self.assertFalse(
            document_matches_scope(doc, user_id="user-b")
        )

    def test_project_isolation(self):
        alpha = _doc(project_id="alpha")
        beta = _doc(project_id="beta")
        self.assertTrue(
            document_matches_scope(alpha, user_id="user-a", project_id="alpha")
        )
        self.assertFalse(
            document_matches_scope(alpha, user_id="user-a", project_id="beta")
        )
        self.assertFalse(
            document_matches_scope(beta, user_id="user-a", project_id="alpha")
        )

    def test_tenant_isolation(self):
        org1 = _doc(tenant_id="org1")
        org2 = _doc(tenant_id="org2")
        self.assertTrue(
            document_matches_scope(org1, user_id="user-a", tenant_id="org1")
        )
        self.assertFalse(
            document_matches_scope(org1, user_id="user-a", tenant_id="org2")
        )
        self.assertFalse(
            document_matches_scope(org2, user_id="user-a", tenant_id="org1")
        )

    def test_missing_tenant_never_matches_tagged_tenant(self):
        tagged = _doc(tenant_id="org1")
        self.assertFalse(
            document_matches_scope(tagged, user_id="user-a", tenant_id=None)
        )

    def test_unscoped_query_excludes_project_memories(self):
        project_mem = _doc(project_id="alpha")
        user_mem = _doc(project_id=None)
        filtered = filter_documents_by_scope(
            [project_mem, user_mem],
            user_id="user-a",
            project_id=None,
        )
        self.assertEqual(len(filtered), 1)
        self.assertIsNone(filtered[0].get("project_id"))


class TestScopeFilterQuery(unittest.TestCase):
    def test_build_scope_filter_includes_user(self):
        filt = build_scope_filter(user_id="u1")
        self.assertIn("$and", filt)

    def test_get_memories_applies_project_filter(self):
        alpha = _doc(project_id="alpha", memory="alpha fact")
        beta = _doc(project_id="beta", memory="beta fact")
        collection = MagicMock()
        collection.find.return_value.sort.return_value.limit.return_value = [
            alpha,
            beta,
        ]
        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.get_memories(
                "user-a",
                limit=10,
                project_id="alpha",
            )
        self.assertTrue(result["success"])
        memories = result["memories"]
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["project_id"], "alpha")


class TestSaveMemoryScopeFields(unittest.TestCase):
    def test_save_memory_persists_project_and_tenant(self):
        collection = MagicMock()
        collection.find_one.return_value = None
        inserted = MagicMock()
        inserted.inserted_id = ObjectId()
        collection.insert_one.return_value = inserted

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "record_audit_event", return_value={}),
            patch.object(memory_service, "schedule_summary_refresh"),
        ):
            result = memory_service.save_memory(
                user_id="user-a",
                memory="My preferred database is MongoDB",
                embedding=[0.1] * 8,
                project_id="alpha",
                tenant_id="org1",
                operation_id=f"scope-save-{uuid4().hex}",
            )

        self.assertTrue(result["success"])
        saved = collection.insert_one.call_args[0][0]
        self.assertEqual(saved["project_id"], "alpha")
        self.assertEqual(saved["tenant_id"], "org1")
        self.assertEqual(saved["scope"], "project")
        self.assertEqual(saved["owner_id"], "user-a")
        self.assertEqual(saved["permission_level"], "private")

    def test_legacy_save_defaults_remain_user_scoped(self):
        collection = MagicMock()
        collection.find_one.return_value = None
        inserted = MagicMock()
        inserted.inserted_id = ObjectId()
        collection.insert_one.return_value = inserted

        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "record_audit_event", return_value={}),
            patch.object(memory_service, "schedule_summary_refresh"),
        ):
            result = memory_service.save_memory(
                user_id="user-a",
                memory="My role is Engineer",
                embedding=[0.1] * 8,
                operation_id=f"legacy-save-{uuid4().hex}",
            )

        self.assertTrue(result["success"])
        saved = collection.insert_one.call_args[0][0]
        self.assertIsNone(saved.get("project_id"))
        self.assertIsNone(saved.get("tenant_id"))
        self.assertEqual(saved.get("scope"), "user")


class TestActiveSessionScope(unittest.TestCase):
    def test_set_and_get_active_scope_without_redis(self):
        with patch(
            "app.services.memory_scope_service._redis",
            None,
        ):
            set_result = set_active_project_scope(
                "session-user",
                project_id="listifys",
                tenant_id="org1",
            )
            self.assertTrue(set_result["success"])
            get_result = get_active_project_scope("session-user")
            # Without Redis, get returns empty defaults.
            self.assertTrue(get_result["success"])
            clear_active_project_scope("session-user")


if __name__ == "__main__":
    unittest.main(verbosity=2)
