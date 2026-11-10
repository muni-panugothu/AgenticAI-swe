"""
Production security matrix for Topics 20–21.

Maps to the Phase 8 checklist (user / project / tenant / ownership / shared /
semantic / API impersonation). Uses mocks — never touches production Mongo data.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from bson import ObjectId
from fastapi.testclient import TestClient

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.core.security import create_access_token  # noqa: E402
from app.main import app  # noqa: E402
from app.services import memory_service  # noqa: E402
from app.services.memory_permission_service import (  # noqa: E402
    check_read_permission,
    check_write_permission,
    filter_readable_documents,
    grant_access,
    memory_owner_id,
    revoke_access,
)
from app.services.memory_scope_service import document_matches_scope  # noqa: E402
from app.services.memory_service import semantic_search_memories  # noqa: E402


def _doc(**kwargs):
    base = {
        "_id": ObjectId(),
        "user_id": "user_a",
        "owner_id": "user_a",
        "memory": "fact",
        "status": "active",
        "shared_with": [],
        "permission_level": "private",
        "importance": 3,
        "confidence": 0.9,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    base.update(kwargs)
    if "owner_id" not in kwargs:
        base["owner_id"] = base.get("user_id")
    return base


class TestUserIsolation(unittest.TestCase):
    def test_a1_user_b_private_not_readable(self):
        doc = _doc(user_id="user_b", memory="secret")
        self.assertFalse(check_read_permission("user_a", doc, audit=False)["allowed"])

    def test_a2_semantic_excludes_user_b(self):
        alice = {
            "memory_id": "a1",
            "user_id": "user_a",
            "owner_id": "user_a",
            "memory": "alice db",
            "score": 0.9,
            "status": "active",
            "permission_level": "private",
            "shared_with": [],
        }
        bob = dict(alice)
        bob.update(memory_id="b1", user_id="user_b", owner_id="user_b", memory="bob db")

        with patch(
            "app.services.memory_service.create_embedding",
            return_value=[0.1] * 8,
        ), patch(
            "app.services.memory_service._vector_search_memories",
            return_value=[alice, bob],
        ), patch(
            "app.services.memory_service.find_memories_by_fact_key",
            return_value={"success": True, "memories": []},
        ), patch(
            "app.services.memory_service.annotate_memories_with_lifecycle",
            side_effect=lambda x: x,
        ), patch("app.services.memory_service._record_memory_access"), patch(
            "app.services.memory_service.record_audit_event"
        ), patch(
            "app.services.memory_service.search_memory_summaries",
            return_value={"success": True, "summaries": []},
        ):
            res = semantic_search_memories("user_a", "database", limit=5)
        texts = " ".join(m["memory"] for m in res["memories"])
        self.assertIn("alice", texts)
        self.assertNotIn("bob db", texts)


class TestProjectIsolation(unittest.TestCase):
    def test_b3_cross_project_denied(self):
        doc = _doc(project_id="proj_a", tenant_id="ten_1")
        self.assertFalse(
            document_matches_scope(
                doc, user_id="user_a", project_id="proj_b", tenant_id="ten_1"
            )
        )

    def test_b4_project_membership_required(self):
        doc = _doc(project_id="proj_a", permission_level="project_read")
        self.assertFalse(
            check_read_permission(
                "user_a", doc, project_id=None, tenant_id="ten_1", audit=False
            )["allowed"]
        )

    def test_b5_authorized_project_read(self):
        doc = _doc(
            user_id="owner",
            project_id="proj_a",
            permission_level="project_read",
            tenant_id="ten_1",
        )
        self.assertTrue(
            check_read_permission(
                "user_a", doc, project_id="proj_a", tenant_id="ten_1", audit=False
            )["allowed"]
        )


class TestTenantIsolation(unittest.TestCase):
    def test_c6_cross_tenant_read_denied(self):
        doc = _doc(tenant_id="ten_a")
        self.assertFalse(
            check_read_permission("user_a", doc, tenant_id="ten_b", audit=False)[
                "allowed"
            ]
        )

    def test_c7_cross_tenant_update_denied(self):
        doc = _doc(user_id="user_b", tenant_id="ten_b")
        coll = MagicMock()
        coll.find_one.return_value = doc
        with patch.object(memory_service, "memory_collection", coll), patch.object(
            memory_service, "record_audit_event", return_value={}
        ):
            out = memory_service.reinforce_memory(
                "user_a", memory_id=str(doc["_id"]), tenant_id="ten_a"
            )
        self.assertFalse(out["success"])

    def test_c8_cross_tenant_delete_denied(self):
        doc = _doc(user_id="user_b", tenant_id="ten_b")
        coll = MagicMock()
        coll.find_one.return_value = doc
        with patch.object(memory_service, "memory_collection", coll), patch.object(
            memory_service, "record_audit_event", return_value={}
        ):
            out = memory_service.delete_memory(
                "user_a", str(doc["_id"]), tenant_id="ten_a"
            )
        self.assertFalse(out["success"])


class TestOwnershipProtection(unittest.TestCase):
    def test_d9_non_owner_cannot_update(self):
        doc = _doc(user_id="owner")
        self.assertFalse(check_write_permission("user_a", doc, audit=False)["allowed"])

    def test_d10_non_owner_cannot_delete(self):
        doc = _doc(user_id="owner")
        coll = MagicMock()
        coll.find_one.return_value = doc
        with patch.object(memory_service, "memory_collection", coll), patch.object(
            memory_service, "record_audit_event", return_value={}
        ):
            out = memory_service.delete_memory("user_a", str(doc["_id"]))
        self.assertFalse(out["success"])

    def test_d11_caller_cannot_reassign_owner_on_save(self):
        coll = MagicMock()
        coll.find_one.return_value = None
        coll.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        with patch.object(memory_service, "memory_collection", coll), patch.object(
            memory_service, "record_audit_event", return_value={}
        ), patch.object(memory_service, "schedule_summary_refresh"):
            saved = memory_service.save_memory(
                user_id="user_a",
                memory="fact",
                embedding=[0.1] * 8,
                owner_id="user_b",
            )
        self.assertTrue(saved["success"])
        payload = coll.insert_one.call_args[0][0]
        self.assertEqual(payload["owner_id"], "user_a")

    def test_d12_missing_owner_not_public(self):
        orphan = {"memory": "x", "permission_level": "public_read"}
        self.assertIsNone(memory_owner_id(orphan))
        self.assertFalse(check_read_permission("user_a", orphan, audit=False)["allowed"])


class TestSharedMemory(unittest.TestCase):
    def test_e13_shared_read_ok(self):
        doc = _doc(user_id="owner", shared_with=["user_a"])
        self.assertTrue(check_read_permission("user_a", doc)["allowed"])

    def test_e14_unauthorized_shared_read_fail(self):
        doc = _doc(user_id="owner", shared_with=["other"])
        self.assertFalse(check_read_permission("user_a", doc, audit=False)["allowed"])

    def test_e15_shared_cannot_modify(self):
        doc = _doc(user_id="owner", shared_with=["user_a"])
        self.assertFalse(check_write_permission("user_a", doc, audit=False)["allowed"])

    def test_e16_shared_cannot_delete(self):
        doc = _doc(user_id="owner", shared_with=["user_a"])
        coll = MagicMock()
        coll.find_one.return_value = doc
        with patch.object(memory_service, "memory_collection", coll), patch.object(
            memory_service, "record_audit_event", return_value={}
        ):
            out = memory_service.delete_memory("user_a", str(doc["_id"]))
        self.assertFalse(out["success"])

    def test_e17_revoked_loses_access(self):
        doc = _doc(user_id="owner", shared_with=["user_a"])
        coll = MagicMock()
        coll.update_one.return_value = MagicMock(matched_count=1)
        revoke_access(
            owner_id="owner",
            memory_doc=doc,
            target_user_id="user_a",
            memory_collection=coll,
        )
        doc["shared_with"] = []
        self.assertFalse(check_read_permission("user_a", doc, audit=False)["allowed"])


class TestSemanticAndSummaries(unittest.TestCase):
    def test_f18_unauthorized_not_in_context(self):
        self.assertEqual(
            filter_readable_documents(
                [_doc(user_id="other")], user_id="user_a"
            ),
            [],
        )

    def test_f19_summary_path_filtered(self):
        from app.services.memory_scope_service import filter_documents_by_scope

        summaries = [_doc(user_id="other", tenant_id="ten_b")]
        scoped = filter_documents_by_scope(
            summaries, user_id="user_a", tenant_id="ten_a"
        )
        self.assertEqual(scoped, [])

    def test_f20_shared_respected_in_filter(self):
        doc = _doc(user_id="owner", shared_with=["user_a"])
        self.assertEqual(
            filter_readable_documents([doc], user_id="user_a"),
            [doc],
        )

    def test_f21_missing_scope_no_broad_retrieval(self):
        doc = {"memory": "leak", "permission_level": "public_read"}
        self.assertFalse(document_matches_scope(doc, user_id="user_a"))


class TestConversationScope(unittest.TestCase):
    def test_conversation_tagged_hidden_without_context(self):
        doc = _doc(conversation_id="conv-1")
        self.assertFalse(document_matches_scope(doc, user_id="user_a"))

    def test_conversation_tagged_visible_with_match(self):
        doc = _doc(conversation_id="conv-1")
        self.assertTrue(
            document_matches_scope(
                doc, user_id="user_a", conversation_id="conv-1"
            )
        )

    def test_forged_conversation_id_denied(self):
        doc = _doc(conversation_id="conv-1")
        self.assertFalse(
            check_read_permission(
                "user_a",
                doc,
                conversation_id="conv-evil",
                audit=False,
            )["allowed"]
        )


class TestDirectApiImpersonation(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        users = MagicMock()
        users.find_one.return_value = {
            "_id": "usr_alice",
            "email": "a@test.com",
            "status": "active",
            "tenant_id": "ten_1",
        }
        self._patch_users = patch(
            "app.services.auth_service._users_collection", lambda: users
        )
        self._patch_users.start()

    def tearDown(self):
        self._patch_users.stop()

    def test_g22_forged_user_id_rejected(self):
        token = create_access_token(
            user_id="usr_alice", email="a@test.com", tenant_id="ten_1"
        )
        res = self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "hi", "user_id": "imposter"},
        )
        self.assertIn(res.status_code, {400, 422})

    def test_g23_forged_tenant_id_rejected(self):
        token = create_access_token(
            user_id="usr_alice", email="a@test.com", tenant_id="ten_1"
        )
        res = self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "hi", "tenant_id": "ten_evil"},
        )
        self.assertIn(res.status_code, {400, 422})

    def test_g24_forged_project_id_rejected(self):
        token = create_access_token(
            user_id="usr_alice", email="a@test.com", tenant_id="ten_1"
        )
        res = self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "hi", "project_id": "proj_evil"},
        )
        self.assertIn(res.status_code, {400, 422})

    def test_g25_forged_conversation_id_rejected(self):
        token = create_access_token(
            user_id="usr_alice", email="a@test.com", tenant_id="ten_1"
        )
        res = self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "hi", "conversation_id": "conv_evil"},
        )
        self.assertIn(res.status_code, {400, 422})

    def test_g26_unauthenticated_chat_rejected(self):
        res = self.client.post("/api/chat", json={"message": "hi"})
        self.assertEqual(res.status_code, 401)


if __name__ == "__main__":
    unittest.main(verbosity=2)
