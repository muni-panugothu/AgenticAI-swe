"""Topic 21 — Memory ownership, sharing, and permission enforcement."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from bson import ObjectId

from app import chatbot
from app.services import memory_service
from app.services.memory_permission_service import (
    check_read_permission,
    check_write_permission,
    default_permission_fields,
    filter_readable_documents,
    grant_access,
    revoke_access,
)


def _memory_doc(
    *,
    user_id="owner",
    owner_id=None,
    shared_with=None,
    permission_level="private",
    project_id=None,
    tenant_id=None,
    memory="secret fact",
):
    return {
        "_id": ObjectId(),
        "user_id": user_id,
        "owner_id": owner_id or user_id,
        "memory": memory,
        "status": "active",
        "shared_with": list(shared_with or []),
        "permission_level": permission_level,
        "project_id": project_id,
        "tenant_id": tenant_id,
        "scope": "project" if project_id else "user",
        "importance": 3,
        "confidence": 0.9,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


class TestPermissionDefaults(unittest.TestCase):
    def test_default_fields_are_private_owner(self):
        fields = default_permission_fields(user_id="u1")
        self.assertEqual(fields["owner_id"], "u1")
        self.assertEqual(fields["permission_level"], "private")
        self.assertEqual(fields["shared_with"], [])


class TestReadWritePermissions(unittest.TestCase):
    def test_owner_can_read_and_write(self):
        doc = _memory_doc()
        self.assertTrue(check_read_permission("owner", doc)["allowed"])
        self.assertTrue(check_write_permission("owner", doc)["allowed"])

    def test_stranger_denied_private(self):
        doc = _memory_doc()
        read = check_read_permission("stranger", doc, audit=False)
        write = check_write_permission("stranger", doc, audit=False)
        self.assertFalse(read["allowed"])
        self.assertFalse(write["allowed"])

    def test_shared_with_can_read_not_write(self):
        doc = _memory_doc(shared_with=["alice"])
        self.assertTrue(check_read_permission("alice", doc)["allowed"])
        self.assertFalse(check_write_permission("alice", doc)["allowed"])

    def test_project_read_same_project(self):
        doc = _memory_doc(
            permission_level="project_read",
            project_id="alpha",
            tenant_id="org1",
        )
        self.assertTrue(
            check_read_permission(
                "teammate",
                doc,
                project_id="alpha",
                tenant_id="org1",
            )["allowed"]
        )
        self.assertFalse(
            check_write_permission(
                "teammate",
                doc,
                project_id="alpha",
                tenant_id="org1",
            )["allowed"]
        )

    def test_project_write_same_project(self):
        doc = _memory_doc(
            permission_level="project_write",
            project_id="alpha",
            tenant_id="org1",
        )
        self.assertTrue(
            check_write_permission(
                "teammate",
                doc,
                project_id="alpha",
                tenant_id="org1",
            )["allowed"]
        )

    def test_public_read_same_tenant(self):
        doc = _memory_doc(
            permission_level="public_read",
            tenant_id="org1",
        )
        self.assertTrue(
            check_read_permission("other", doc, tenant_id="org1")["allowed"]
        )
        self.assertFalse(
            check_write_permission("other", doc, tenant_id="org1")["allowed"]
        )
        self.assertFalse(
            check_read_permission("other", doc, tenant_id="org2")["allowed"]
        )

    def test_filter_readable_documents(self):
        private = _memory_doc(user_id="owner", memory="private")
        shared = _memory_doc(
            user_id="owner",
            shared_with=["alice"],
            memory="shared",
        )
        readable = filter_readable_documents(
            [private, shared],
            user_id="alice",
        )
        self.assertEqual(len(readable), 1)
        self.assertEqual(readable[0]["memory"], "shared")


class TestGrantRevokeAccess(unittest.TestCase):
    def test_grant_and_revoke(self):
        doc = _memory_doc()
        collection = MagicMock()
        collection.update_one.return_value.matched_count = 1

        with patch(
            "app.services.memory_permission_service.record_audit_event",
            return_value={"success": True},
        ) as audit:
            granted = grant_access(
                owner_id="owner",
                memory_doc=doc,
                target_user_id="alice",
                memory_collection=collection,
            )
            self.assertTrue(granted["success"])
            revoked = revoke_access(
                owner_id="owner",
                memory_doc=doc,
                target_user_id="alice",
                memory_collection=collection,
            )
            self.assertTrue(revoked["success"])
            events = [c.kwargs["event_type"] for c in audit.call_args_list]
            self.assertIn("permission_granted", events)
            self.assertIn("permission_revoked", events)

    def test_non_owner_cannot_grant(self):
        doc = _memory_doc()
        collection = MagicMock()
        with patch(
            "app.services.memory_permission_service.record_audit_event",
            return_value={"success": True},
        ):
            result = grant_access(
                owner_id="imposter",
                memory_doc=doc,
                target_user_id="alice",
                memory_collection=collection,
            )
        self.assertFalse(result["success"])


class TestMutationPermissionEnforcement(unittest.TestCase):
    def test_delete_denies_non_owner(self):
        doc = _memory_doc(user_id="owner")
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
            result = memory_service.delete_memory(
                user_id="attacker",
                memory_id=str(doc["_id"]),
                soft=True,
            )
        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"].lower())

    def test_reinforce_denies_non_owner(self):
        doc = _memory_doc(user_id="owner")
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
            result = memory_service.reinforce_memory(
                user_id="attacker",
                memory_id=str(doc["_id"]),
            )
        self.assertFalse(result["success"])

    def test_permission_denied_does_not_leak_memory_text(self):
        doc = _memory_doc(memory="TOP SECRET VALUE")
        with patch(
            "app.services.memory_permission_service.record_audit_event",
            return_value={"success": True},
        ) as audit:
            check_read_permission("attacker", doc, audit=True)
            details = audit.call_args.kwargs.get("details") or {}
        self.assertNotIn("TOP SECRET", str(details))
        self.assertNotIn("TOP SECRET", str(audit.call_args))


class TestToolScopeInjection(unittest.TestCase):
    def test_llm_cannot_inject_project_or_tenant(self):
        captured = {}

        def fake_search_memory(
            user_id,
            query,
            limit=5,
            *,
            project_id=None,
            tenant_id=None,
        ):
            captured.update(
                {
                    "user_id": user_id,
                    "query": query,
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                }
            )
            return {"success": True, "memories": []}

        with patch.dict(chatbot.tools, {"search_memory": fake_search_memory}):
            chatbot.execute_tool(
                "search_memory",
                {
                    "query": "test",
                    "user_id": "injected",
                    "project_id": "evil-project",
                    "tenant_id": "evil-tenant",
                },
                user_id="real-user",
                project_id="alpha",
                tenant_id="org1",
            )
        self.assertEqual(captured["user_id"], "real-user")
        self.assertEqual(captured.get("project_id"), "alpha")
        self.assertEqual(captured.get("tenant_id"), "org1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
