"""Security regressions for Topics 20–21 (scope + permissions)."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from bson import ObjectId

from app.services import memory_service
from app.services.memory_permission_service import (
    check_read_permission,
    check_write_permission,
    filter_readable_documents,
    memory_owner_id,
)
from app.services.memory_scope_service import document_matches_scope
from app.services.memory_service import semantic_search_memories


def _doc(
    *,
    user_id="alice",
    owner_id=None,
    shared_with=None,
    permission_level="private",
    project_id=None,
    tenant_id=None,
    memory="alice private fact",
    memory_id=None,
):
    return {
        "_id": memory_id or ObjectId(),
        "user_id": user_id,
        "owner_id": owner_id if owner_id is not None else user_id,
        "memory": memory,
        "status": "active",
        "shared_with": list(shared_with or []),
        "permission_level": permission_level,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "scope": "project" if project_id else "user",
        "importance": 4,
        "confidence": 0.9,
        "fact_key": "preferred_database",
        "normalized_value": "mongodb",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


class TestCrossUserIsolation(unittest.TestCase):
    def test_user_a_cannot_read_user_b_private_memory(self):
        doc = _doc(user_id="bob", memory="bob secret")
        result = check_read_permission("alice", doc, audit=False)
        self.assertFalse(result["allowed"])
        self.assertIsNone(memory_owner_id({"memory": "orphan"}))

    def test_user_a_cannot_modify_user_b_memory(self):
        doc = _doc(user_id="bob")
        collection = MagicMock()
        collection.find_one.return_value = doc
        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(
                memory_service,
                "record_audit_event",
                return_value={"success": True},
            ),
        ):
            deleted = memory_service.delete_memory(
                user_id="alice",
                memory_id=str(doc["_id"]),
                soft=True,
            )
            reinforced = memory_service.reinforce_memory(
                user_id="alice",
                memory_id=str(doc["_id"]),
            )
        self.assertFalse(deleted["success"])
        self.assertFalse(reinforced["success"])
        self.assertNotIn("bob secret", str(deleted))
        self.assertNotIn("bob secret", str(reinforced))


class TestCrossTenantProjectIsolation(unittest.TestCase):
    def test_cross_tenant_access_is_denied(self):
        doc = _doc(user_id="alice", tenant_id="org1")
        self.assertFalse(
            check_read_permission(
                "alice", doc, tenant_id="org2", audit=False
            )["allowed"]
        )
        self.assertFalse(
            document_matches_scope(doc, user_id="alice", tenant_id="org2")
        )

    def test_cross_project_access_is_denied(self):
        doc = _doc(user_id="alice", project_id="alpha", tenant_id="org1")
        self.assertFalse(
            check_read_permission(
                "alice",
                doc,
                project_id="beta",
                tenant_id="org1",
                audit=False,
            )["allowed"]
        )
        self.assertFalse(
            document_matches_scope(
                doc, user_id="alice", project_id="beta", tenant_id="org1"
            )
        )

    def test_tenantless_legacy_row_is_allowed_for_its_private_owner_only(self):
        legacy = _doc(user_id="alice", tenant_id=None)
        self.assertTrue(
            check_read_permission(
                "alice", legacy, tenant_id="org1", audit=False
            )["allowed"]
        )
        self.assertFalse(
            check_read_permission(
                "bob", legacy, tenant_id="org1", audit=False
            )["allowed"]
        )
        wrong_tenant = _doc(user_id="alice", tenant_id="org2")
        self.assertFalse(
            check_read_permission(
                "alice", wrong_tenant, tenant_id="org1", audit=False
            )["allowed"]
        )


class TestMissingOwnershipNeverPublic(unittest.TestCase):
    def test_missing_ownership_does_not_grant_public_access(self):
        orphan = {
            "memory": "leaked?",
            "permission_level": "public_read",
            "shared_with": [],
            "status": "active",
        }
        self.assertIsNone(memory_owner_id(orphan))
        self.assertFalse(
            check_read_permission("alice", orphan, audit=False)["allowed"]
        )
        self.assertFalse(
            check_write_permission("alice", orphan, audit=False)["allowed"]
        )
        # Ensure identity helper does not invent ownership for the caller.
        filled = memory_service._ensure_identity_fields(
            dict(orphan), fallback_user_id="alice"
        )
        self.assertFalse(filled.get("user_id"))
        self.assertFalse(filled.get("owner_id"))
        # Never invent an owner for the caller even if level says public_read.
        self.assertFalse(
            check_read_permission("alice", filled, audit=False)["allowed"]
        )
        self.assertFalse(
            document_matches_scope(orphan, user_id="alice")
        )


class TestSharedMemoryAccess(unittest.TestCase):
    def test_authorized_shared_memory_reads_work(self):
        doc = _doc(user_id="bob", shared_with=["alice"], memory="shared ok")
        self.assertTrue(check_read_permission("alice", doc)["allowed"])
        readable = filter_readable_documents([doc], user_id="alice")
        self.assertEqual(len(readable), 1)

    def test_unauthorized_shared_memory_reads_fail(self):
        doc = _doc(user_id="bob", shared_with=["carol"], memory="not for alice")
        self.assertFalse(
            check_read_permission("alice", doc, audit=False)["allowed"]
        )
        readable = filter_readable_documents([doc], user_id="alice")
        self.assertEqual(readable, [])

    def test_unauthorized_writes_fail(self):
        doc = _doc(user_id="bob", shared_with=["alice"])
        # Sharees are read-only.
        self.assertFalse(
            check_write_permission("alice", doc, audit=False)["allowed"]
        )
        self.assertFalse(
            check_write_permission("stranger", doc, audit=False)["allowed"]
        )


class TestSemanticRetrievalPermissions(unittest.TestCase):
    def test_semantic_retrieval_respects_permissions(self):
        alice_doc = {
            "memory_id": "a1",
            "user_id": "alice",
            "owner_id": "alice",
            "memory": "my preferred database is MongoDB",
            "score": 0.95,
            "status": "active",
            "fact_key": "preferred_database",
            "normalized_value": "mongodb",
            "importance": 4,
            "confidence": 0.9,
            "permission_level": "private",
            "shared_with": [],
        }
        bob_private = {
            "memory_id": "b1",
            "user_id": "bob",
            "owner_id": "bob",
            "memory": "bob secret database is Postgres",
            "score": 0.99,
            "status": "active",
            "fact_key": "preferred_database",
            "normalized_value": "postgresql",
            "importance": 5,
            "confidence": 0.95,
            "permission_level": "private",
            "shared_with": [],
        }
        orphan = {
            "memory_id": "o1",
            "memory": "unowned public-looking memory",
            "score": 0.98,
            "status": "active",
            "permission_level": "public_read",
            "shared_with": [],
        }

        def fake_vector(user_id, embedding, *, status_filter, limit, min_score, **_):
            if status_filter == {"status": "needs_review"}:
                return []
            # Simulate a buggy vector hit that includes foreign/orphan docs.
            return [dict(alice_doc), dict(bob_private), dict(orphan)]

        with patch(
            "app.services.memory_service.create_embedding",
            return_value=[0.1] * 8,
        ), patch(
            "app.services.memory_service._vector_search_memories",
            side_effect=fake_vector,
        ), patch(
            "app.services.memory_service.find_memories_by_fact_key",
            return_value={"success": True, "memories": []},
        ), patch(
            "app.services.memory_service.annotate_memories_with_lifecycle",
            side_effect=lambda xs: xs,
        ), patch(
            "app.services.memory_service._record_memory_access",
        ), patch(
            "app.services.memory_service.record_audit_event",
        ), patch(
            "app.services.memory_service.search_memory_summaries",
            return_value={"success": True, "summaries": []},
        ), patch(
            "app.services.memory_service.memory_collection"
        ) as mock_coll:
            mock_coll.find.return_value.limit.return_value = []
            result = semantic_search_memories(
                "alice",
                "Which database do I prefer?",
                limit=5,
                min_score=0.80,
                include_needs_review=False,
            )

        self.assertTrue(result["success"])
        texts = [m["memory"] for m in result["memories"]]
        self.assertTrue(any("MongoDB" in t for t in texts))
        self.assertFalse(any("bob secret" in t for t in texts))
        self.assertFalse(any("unowned" in t for t in texts))

    def test_semantic_search_includes_owned_legacy_rows_but_rejects_other_tenants(self):
        legacy = {
            "memory_id": "legacy-owned",
            "user_id": "alice",
            "owner_id": "alice",
            "memory": "My preferred language is Python",
            "score": 0.95,
            "status": "active",
            "fact_key": "preferred_language",
            "normalized_value": "python",
            "importance": 4,
            "confidence": 0.9,
            "permission_level": "private",
            "shared_with": [],
            "tenant_id": None,
        }
        wrong_tenant = dict(legacy)
        wrong_tenant.update(
            memory_id="wrong-tenant",
            memory="Other tenant secret",
            tenant_id="org2",
        )

        def fake_vector(user_id, embedding, *, status_filter, limit, min_score, **_):
            return [legacy, wrong_tenant] if status_filter != {"status": "needs_review"} else []

        with patch(
            "app.services.memory_service.create_embedding",
            return_value=[0.1] * 8,
        ), patch(
            "app.services.memory_service._vector_search_memories",
            side_effect=fake_vector,
        ), patch(
            "app.services.memory_service.find_memories_by_fact_key",
            return_value={"success": True, "memories": []},
        ), patch(
            "app.services.memory_service.annotate_memories_with_lifecycle",
            side_effect=lambda xs: xs,
        ), patch(
            "app.services.memory_service._record_memory_access",
        ), patch(
            "app.services.memory_service.record_audit_event",
        ), patch(
            "app.services.memory_service.search_memory_summaries",
            return_value={"success": True, "summaries": []},
        ), patch("app.services.memory_service.memory_collection") as mock_coll:
            mock_coll.find.return_value.limit.return_value = []
            result = semantic_search_memories(
                "alice",
                "What is my preferred language?",
                tenant_id="org1",
                include_needs_review=False,
            )

        texts = [row["memory"] for row in result["memories"]]
        self.assertIn("My preferred language is Python", texts)
        self.assertNotIn("Other tenant secret", texts)


class TestExistingMemoryOpsRemainFunctional(unittest.TestCase):
    def test_existing_memory_operations_remain_functional(self):
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
            saved = memory_service.save_memory(
                user_id="alice",
                memory="My preferred database is MongoDB",
                embedding=[0.1] * 8,
                operation_id="sec-save-1",
            )
        self.assertTrue(saved["success"])
        payload = collection.insert_one.call_args[0][0]
        self.assertEqual(payload["user_id"], "alice")
        self.assertEqual(payload["owner_id"], "alice")
        self.assertEqual(payload["permission_level"], "private")

        owned = _doc(user_id="alice")
        collection.find_one.return_value = owned
        collection.update_one.return_value.matched_count = 1
        with (
            patch.object(memory_service, "memory_collection", collection),
            patch.object(memory_service, "record_audit_event", return_value={}),
            patch.object(
                memory_service,
                "audit_collection",
                None,
            ),
        ):
            reinforced = memory_service.reinforce_memory(
                user_id="alice",
                memory_id=str(owned["_id"]),
                operation_id="sec-reinf-1",
            )
        self.assertTrue(reinforced["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
