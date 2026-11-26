"""Batch C — MongoDB auth / memory / approval persistence (gated)."""

from __future__ import annotations

import os
import threading
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest import mock
from unittest.mock import patch

from tests.integration.safety import mongo_memory_gate

_GATE = mongo_memory_gate()


@unittest.skipUnless(_GATE.enabled, _GATE.reason)
class TestBatchCMongoIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")
        test_db = _GATE.database
        assert test_db
        os.environ["MEMORY_DATABASE"] = test_db
        from app.core.config import get_settings
        from app.services import memory_audit_service, memory_service
        from app.services import memory_summary_service
        from app.services import approval_service, tool_permission_service

        get_settings.cache_clear()
        cls._bindings = (
            memory_service.database,
            memory_service.memory_collection,
            memory_service.audit_collection,
            memory_service.summary_collection,
        )
        db = memory_service.client[test_db]
        memory_service.database = db
        memory_service.memory_collection = db[memory_service.MEMORY_COLLECTION]
        memory_service.audit_collection = db[memory_service.MEMORY_AUDIT_COLLECTION]
        memory_service.summary_collection = db[
            memory_service.MEMORY_SUMMARY_COLLECTION
        ]
        memory_audit_service.configure_audit_collection(
            memory_service.audit_collection
        )
        memory_summary_service.configure_summary_collections(
            memory_service.summary_collection,
            memory_service.memory_collection,
        )
        cls.approvals = db[
            os.getenv("AGENT_APPROVALS_COLLECTION", "agent_approvals")
        ]
        cls.perms = db[
            os.getenv(
                "AGENT_TOOL_PERMISSIONS_COLLECTION", "agent_tool_permissions"
            )
        ]
        approval_service.configure_approval_collection(cls.approvals)
        tool_permission_service.configure_permission_collection(cls.perms)
        tool_permission_service.invalidate_permission_cache()

    @classmethod
    def tearDownClass(cls):
        from app.services import memory_audit_service, memory_service
        from app.services import memory_summary_service
        from app.services import approval_service, tool_permission_service

        (
            memory_service.database,
            memory_service.memory_collection,
            memory_service.audit_collection,
            memory_service.summary_collection,
        ) = cls._bindings
        memory_audit_service.configure_audit_collection(
            memory_service.audit_collection
        )
        memory_summary_service.configure_summary_collections(
            memory_service.summary_collection,
            memory_service.memory_collection,
        )
        approval_service.configure_approval_collection(None)
        tool_permission_service.configure_permission_collection(None)
        tool_permission_service.invalidate_permission_cache()

    def test_BC_M01_auth_session_ownership_and_jwt(self):
        from app.core.security import (
            create_access_token,
            decode_access_token,
            hash_password,
            hash_refresh_token,
        )
        from app.services import auth_service

        suffix = uuid.uuid4().hex[:8]
        user_id = f"usr_bc_m01_{suffix}"
        email = f"bc_m01_{suffix}@example.com"
        users = auth_service._users_collection()
        self.addCleanup(users.delete_many, {"_id": user_id})

        users.insert_one(
            {
                "_id": user_id,
                "email": email,
                "password_hash": hash_password("BatchC-Test-Pass-1!"),
                "status": "active",
                "tenant_id": f"ten_{suffix}",
                "display_name": "Batch C",
                "created_at": datetime.now(timezone.utc),
            }
        )
        user = users.find_one({"_id": user_id})
        issued = auth_service.issue_tokens_for_user(user)
        self.assertTrue(issued.get("access_token"))
        claims = decode_access_token(issued["access_token"])
        self.assertIsNotNone(claims)
        self.assertEqual(claims["sub"], user_id)
        self.assertTrue(
            auth_service.get_active_session(claims["sid"], user_id)
        )
        self.assertFalse(
            auth_service.get_active_session(claims["sid"], "usr_other")
        )

        refreshed = auth_service.refresh_session(
            refresh_token=issued["refresh_token"]
        )
        self.assertTrue(refreshed.get("success"), refreshed)
        self.assertTrue(
            auth_service.revoke_refresh_token(refreshed["refresh_token"])
        )
        reused = auth_service.refresh_session(
            refresh_token=refreshed["refresh_token"]
        )
        self.assertFalse(reused.get("success"))

        # Expired refresh
        again = auth_service.issue_tokens_for_user(user)
        auth_service._sessions_collection().update_one(
            {"token_hash": hash_refresh_token(again["refresh_token"])},
            {
                "$set": {
                    "expires_at": datetime.now(timezone.utc) - timedelta(days=1)
                }
            },
        )
        expired = auth_service.refresh_session(
            refresh_token=again["refresh_token"]
        )
        self.assertFalse(expired.get("success"))
        self.addCleanup(
            auth_service._sessions_collection().delete_many,
            {"user_id": user_id},
        )

    def test_BC_M02_memory_crud_isolation_and_delete(self):
        from app.services import memory_service

        suffix = uuid.uuid4().hex[:8]
        user_a = f"usr_bc_a_{suffix}"
        user_b = f"usr_bc_b_{suffix}"
        self.addCleanup(
            memory_service.memory_collection.delete_many,
            {"user_id": {"$in": [user_a, user_b]}},
        )
        embedding = [0.01] * 16
        with mock.patch(
            "app.services.memory_service.create_embedding",
            return_value=embedding,
        ), mock.patch("app.services.memory_service.schedule_summary_refresh"):
            saved = memory_service.save_memory(
                user_a,
                "BatchC favorite editor is Neovim",
                embedding=embedding,
                fact_key="preferred_editor",
                normalized_value="neovim",
            )
            self.assertTrue(saved.get("success"), saved)
            mid = saved.get("memory_id")
            self.assertTrue(mid)

            found = memory_service.semantic_search_memories(
                user_id=user_a,
                query="favorite editor",
                limit=5,
            )
            texts = " ".join(
                str(m.get("memory") or "") for m in (found.get("memories") or [])
            )
            self.assertIn("Neovim", texts)

            other = memory_service.semantic_search_memories(
                user_id=user_b,
                query="favorite editor",
                limit=5,
            )
            other_texts = " ".join(
                str(m.get("memory") or "") for m in (other.get("memories") or [])
            )
            self.assertNotIn("Neovim", other_texts)

            deleted = memory_service.delete_memory(
                user_id=user_a,
                memory_id=str(mid),
            )
            self.assertTrue(deleted.get("success") or deleted.get("deleted"), deleted)

    def test_BC_M03_approval_atomic_consume_and_replay(self):
        from app.chatbot import execute_tool
        from app.services import approval_service

        suffix = uuid.uuid4().hex[:8]
        user_id = f"usr_bc_h_{suffix}"
        self.addCleanup(self.approvals.delete_many, {"user_id": user_id})

        with patch(
            "app.services.agent_security_policy.hitl_enabled", return_value=True
        ):
            created = approval_service.create_approval_request(
                user_id=user_id,
                tool_name="save_memory",
                arguments={"memory": "batchc blue"},
                project_id=f"proj_{suffix}",
                tenant_id=f"ten_{suffix}",
            )
            self.assertTrue(created.get("success"), created)
            aid = created["approval"]["approval_id"]
            decided = approval_service.decide_approval(
                approval_id=aid,
                reviewer_user_id=user_id,
                approve=True,
            )
            self.assertTrue(decided.get("success"), decided)

            calls = {"n": 0}

            def _fake_save(**kwargs):
                calls["n"] += 1
                return {"success": True, "memory_id": f"m_{suffix}"}

            with patch.dict("app.chatbot.tools", {"save_memory": _fake_save}):
                first = execute_tool(
                    "save_memory",
                    {"memory": "batchc blue"},
                    user_id=user_id,
                    project_id=f"proj_{suffix}",
                    tenant_id=f"ten_{suffix}",
                    approval_id=aid,
                )
                second = execute_tool(
                    "save_memory",
                    {"memory": "batchc blue"},
                    user_id=user_id,
                    project_id=f"proj_{suffix}",
                    tenant_id=f"ten_{suffix}",
                    approval_id=aid,
                )
        self.assertTrue(first.get("success"), first)
        self.assertEqual(calls["n"], 1)
        self.assertFalse(second.get("success"), second)

    def test_BC_M04_concurrent_approval_consume_once(self):
        from app.chatbot import execute_tool
        from app.services import approval_service

        suffix = uuid.uuid4().hex[:8]
        user_id = f"usr_bc_race_{suffix}"
        self.addCleanup(self.approvals.delete_many, {"user_id": user_id})

        with patch(
            "app.services.agent_security_policy.hitl_enabled", return_value=True
        ):
            created = approval_service.create_approval_request(
                user_id=user_id,
                tool_name="save_memory",
                arguments={"memory": "race"},
                project_id=f"proj_{suffix}",
                tenant_id=f"ten_{suffix}",
            )
            aid = created["approval"]["approval_id"]
            approval_service.decide_approval(
                approval_id=aid, reviewer_user_id=user_id, approve=True
            )
            hits = {"n": 0}
            lock = threading.Lock()

            def _fake_save(**kwargs):
                with lock:
                    hits["n"] += 1
                return {"success": True, "memory_id": "m_race"}

            results: list[dict] = []

            def worker():
                with patch.dict("app.chatbot.tools", {"save_memory": _fake_save}):
                    results.append(
                        execute_tool(
                            "save_memory",
                            {"memory": "race"},
                            user_id=user_id,
                            project_id=f"proj_{suffix}",
                            tenant_id=f"ten_{suffix}",
                            approval_id=aid,
                        )
                    )

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
        successes = [r for r in results if r.get("success")]
        self.assertEqual(len(successes), 1)
        self.assertEqual(hits["n"], 1)


if __name__ == "__main__":
    unittest.main()
