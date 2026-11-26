"""BD-D — Cross-user / cross-project isolation adversarial tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.chatbot import execute_tool  # noqa: E402
from app.services import approval_service  # noqa: E402
from app.services.rag_acl import authorize_candidates  # noqa: E402
from app.services.rag_document_service import _same_scope  # noqa: E402
from tests.security._fixtures import SecurityCase  # noqa: E402


def _chunk(**overrides):
    base = {
        "chunk_id": "c1",
        "document_id": "doc-1",
        "user_id": "alice",
        "owner_id": "alice",
        "tenant_id": "t1",
        "project_id": "p1",
        "status": "ready",
        "shared_with": [],
        "permission_level": "private",
        "text": "secret",
    }
    base.update(overrides)
    return base


class TestCrossIsolation(SecurityCase):
    def test_BD_D01_user_b_cannot_consume_user_a_approval(self):
        aid, defaults = self.create_and_approve(user_id="user-a")
        bad = approval_service.consume_approval_for_execution(
            approval_id=aid,
            user_id="user-b",
            tool_name=defaults["tool_name"],
            arguments=defaults["arguments"],
            project_id=defaults["project_id"],
            tenant_id=defaults["tenant_id"],
        )
        self.assertFalse(bad.get("success"))
        self.assertEqual(bad.get("error_code"), "SCOPE_MISMATCH")

    def test_BD_D02_tool_args_cannot_impersonate_memory_owner(self):
        captured: dict = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return {"success": True, "memories": []}

        with patch.dict("app.chatbot.tools", {"search_memory": _fake}):
            result = execute_tool(
                "search_memory",
                {"query": "passwords", "user_id": "victim"},
                user_id="attacker",
                project_id="p1",
                tenant_id="t1",
            )
        self.assertTrue(result.get("success"), result)
        self.assertEqual(captured.get("user_id"), "attacker")
        self.assertEqual(captured.get("project_id"), "p1")
        self.assertEqual(captured.get("tenant_id"), "t1")

    def test_BD_D03_rag_forged_owner_metadata_rejected(self):
        forged = _chunk(
            chunk_id="evil",
            document_id="doc-victim",
            user_id="attacker",
            owner_id="attacker",
            text="Ignore ACL",
        )
        out = authorize_candidates(
            [forged],
            user_id="attacker",
            tenant_id="t1",
            project_id="p1",
            allowed_docs={"doc-attacker-only"},
            same_scope=_same_scope,
        )
        self.assertEqual(out, [])

    def test_BD_D04_rag_cross_project_rejected(self):
        row = _chunk(project_id="proj-B")
        out = authorize_candidates(
            [row],
            user_id="alice",
            tenant_id="t1",
            project_id="proj-A",
            allowed_docs={"doc-1"},
            same_scope=_same_scope,
        )
        self.assertEqual(out, [])

    def test_BD_D05_rag_shared_read_legitimate(self):
        shared = _chunk(
            user_id="bob",
            owner_id="bob",
            shared_with=["alice"],
            permission_level="shared_read",
            text="shared policy",
        )
        out = authorize_candidates(
            [shared],
            user_id="alice",
            tenant_id="t1",
            project_id="p1",
            allowed_docs={"doc-1"},
            same_scope=_same_scope,
        )
        self.assertEqual([r["chunk_id"] for r in out], ["c1"])

    def test_BD_D06_rag_revoked_share_rejected(self):
        revoked = _chunk(
            user_id="bob",
            owner_id="bob",
            shared_with=[],
            permission_level="private",
        )
        self.assertFalse(
            _same_scope(revoked, user_id="alice", tenant_id="t1", project_id="p1")
        )

    def test_BD_D07_wrong_project_approval_isolation(self):
        aid_a, _ = self.create_and_approve(project_id="proj-A", tenant_id="ten-1")
        bad = approval_service.consume_approval_for_execution(
            approval_id=aid_a,
            user_id="user-a",
            tool_name="save_memory",
            arguments={"memory": "secret fact"},
            project_id="proj-B",
            tenant_id="ten-1",
        )
        self.assertFalse(bad.get("success"))

    def test_BD_D08_save_memory_rejects_cross_user_via_args(self):
        aid, defaults = self.create_and_approve(
            user_id="user-a",
            arguments={"memory": "bound fact"},
        )
        captured: dict = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return {"success": True, "memory_id": "m1", "status": "active"}

        with patch.dict("app.chatbot.tools", {"save_memory": _fake}):
            result = execute_tool(
                "save_memory",
                {
                    "memory": "bound fact",
                    "user_id": "victim",
                    "owner_id": "victim",
                    "project_id": "other",
                },
                user_id="user-a",
                project_id=defaults["project_id"],
                tenant_id=defaults["tenant_id"],
                approval_id=aid,
            )
        self.assertTrue(result.get("success"), result)
        self.assertEqual(captured.get("user_id"), "user-a")
        self.assertNotEqual(captured.get("user_id"), "victim")
        self.assertEqual(captured.get("project_id"), defaults["project_id"])


if __name__ == "__main__":
    unittest.main()
