"""Batch E — disposable Atlas Mongo persistence (auth/memory/approval/commit)."""

from __future__ import annotations

import os
import threading
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tests.integration.batch_e_safety import mongo_gate, new_batch_e_run_id

_GATE = mongo_gate()


@unittest.skipUnless(_GATE.enabled, _GATE.reason)
class TestBatchEMongoLive(unittest.TestCase):
    """LIVE Atlas only — never use InMemoryCollection here."""

    run_id: str = ""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")
        test_db = _GATE.database
        assert test_db
        os.environ["MEMORY_DATABASE"] = test_db
        from app.core.config import get_settings
        from app.services import (
            approval_service,
            memory_audit_service,
            memory_service,
            memory_summary_service,
            tool_permission_service,
        )

        get_settings.cache_clear()
        cls.run_id = new_batch_e_run_id()
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
        cls.db = db
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
        cls._owned_user_ids: list[str] = []
        cls._owned_memory_ids: list[str] = []
        cls._owned_approval_ids: list[str] = []

    @classmethod
    def tearDownClass(cls):
        from app.services import (
            approval_service,
            memory_audit_service,
            memory_service,
            memory_summary_service,
            tool_permission_service,
        )
        from app.services import auth_service

        # Cleanup only Batch E run-owned records.
        try:
            users = auth_service._users_collection()
            if cls._owned_user_ids:
                users.delete_many({"_id": {"$in": cls._owned_user_ids}})
                users.delete_many({"batch_e_run_id": cls.run_id})
        except Exception:
            pass
        try:
            if cls._owned_memory_ids:
                memory_service.memory_collection.delete_many(
                    {"_id": {"$in": cls._owned_memory_ids}}
                )
            memory_service.memory_collection.delete_many(
                {"batch_e_run_id": cls.run_id}
            )
        except Exception:
            pass
        try:
            if cls._owned_approval_ids:
                cls.approvals.delete_many(
                    {"approval_id": {"$in": cls._owned_approval_ids}}
                )
            cls.approvals.delete_many({"batch_e_run_id": cls.run_id})
            cls.perms.delete_many({"batch_e_run_id": cls.run_id})
        except Exception:
            pass

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

    def test_BE_M01_auth_jwt_and_session_isolation(self):
        from app.core.security import decode_access_token, hash_password
        from app.services import auth_service

        suffix = uuid.uuid4().hex[:8]
        user_id = f"usr_be_m01_{suffix}"
        email = f"be_m01_{suffix}@example.com"
        users = auth_service._users_collection()
        self.__class__._owned_user_ids.append(user_id)
        users.insert_one(
            {
                "_id": user_id,
                "email": email,
                "password_hash": hash_password("BatchE-Test-Pass-1!"),
                "status": "active",
                "tenant_id": f"ten_{suffix}",
                "display_name": "Batch E",
                "batch_e_run_id": self.run_id,
                "created_at": datetime.now(timezone.utc),
            }
        )
        user = users.find_one({"_id": user_id})
        issued = auth_service.issue_tokens_for_user(user)
        claims = decode_access_token(issued["access_token"])
        self.assertEqual(claims["sub"], user_id)
        self.assertTrue(auth_service.get_active_session(claims["sid"], user_id))
        self.assertFalse(
            auth_service.get_active_session(claims["sid"], "other-user")
        )

    def test_BE_M02_memory_crud_ownership_and_isolation(self):
        from app.services import memory_service

        uid_a = f"be_mem_a_{uuid.uuid4().hex[:8]}"
        uid_b = f"be_mem_b_{uuid.uuid4().hex[:8]}"
        emb = [0.01] * 8
        saved = memory_service.save_memory(
            user_id=uid_a,
            memory=f"batch-e fact {self.run_id}",
            embedding=emb,
            project_id="proj-a",
            tenant_id="ten-a",
            operation_id=f"be-save-{self.run_id}-{uuid.uuid4().hex[:6]}",
        )
        self.assertTrue(saved.get("success"), saved)
        mid = saved["memory_id"]
        self.__class__._owned_memory_ids.append(mid)
        # Tag for cleanup (best-effort).
        try:
            from bson import ObjectId

            memory_service.memory_collection.update_one(
                {"_id": ObjectId(mid)},
                {"$set": {"batch_e_run_id": self.run_id}},
            )
        except Exception:
            pass

        got = memory_service.memory_collection.find_one(
            {"_id": __import__("bson").ObjectId(mid), "user_id": uid_a}
        )
        self.assertIsNotNone(got)
        stranger = memory_service.memory_collection.find_one(
            {"_id": __import__("bson").ObjectId(mid), "user_id": uid_b}
        )
        self.assertIsNone(stranger)

        deleted = memory_service.delete_memory(
            user_id=uid_a,
            memory_id=mid,
            soft=True,
            project_id="proj-a",
            tenant_id="ten-a",
            operation_id=f"be-del-{uuid.uuid4().hex[:6]}",
        )
        self.assertTrue(deleted.get("success"), deleted)

    def test_BE_M03_approval_atomic_consume_and_replay(self):
        from app.services import approval_service
        from app.validation.agent_security_models import ApprovalStatus

        os.environ["AGENT_HITL_ENABLED"] = "true"
        os.environ["AGENT_SECURITY_POLICY_VERSION"] = "1"
        created = approval_service.create_approval_request(
            user_id="be_user_a",
            tool_name="save_memory",
            arguments={"memory": f"approved {self.run_id}"},
            project_id="proj-e",
            tenant_id="ten-e",
        )
        self.assertTrue(created.get("success"), created)
        aid = created["approval"]["approval_id"]
        self.__class__._owned_approval_ids.append(aid)
        self.approvals.update_one(
            {"approval_id": aid},
            {"$set": {"batch_e_run_id": self.run_id}},
        )
        approval_service.decide_approval(
            approval_id=aid,
            reviewer_user_id="be_user_a",
            approve=True,
        )
        first = approval_service.consume_approval_for_execution(
            approval_id=aid,
            user_id="be_user_a",
            tool_name="save_memory",
            arguments={"memory": f"approved {self.run_id}"},
            project_id="proj-e",
            tenant_id="ten-e",
        )
        self.assertTrue(first.get("success"), first)
        second = approval_service.consume_approval_for_execution(
            approval_id=aid,
            user_id="be_user_a",
            tool_name="save_memory",
            arguments={"memory": f"approved {self.run_id}"},
            project_id="proj-e",
            tenant_id="ten-e",
        )
        self.assertFalse(second.get("success"))
        self.assertEqual(second.get("error_code"), "APPROVAL_REPLAY")
        doc = self.approvals.find_one({"approval_id": aid})
        self.assertEqual(doc.get("status"), ApprovalStatus.CONSUMED.value)

    def test_BE_M04_concurrent_approval_consume_single_winner(self):
        from app.services import approval_service

        created = approval_service.create_approval_request(
            user_id="be_user_a",
            tool_name="save_memory",
            arguments={"memory": f"race {self.run_id}"},
            project_id="proj-e",
            tenant_id="ten-e",
        )
        aid = created["approval"]["approval_id"]
        self.__class__._owned_approval_ids.append(aid)
        self.approvals.update_one(
            {"approval_id": aid},
            {"$set": {"batch_e_run_id": self.run_id}},
        )
        approval_service.decide_approval(
            approval_id=aid, reviewer_user_id="be_user_a", approve=True
        )
        results: list[dict] = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait(timeout=5)
            results.append(
                approval_service.consume_approval_for_execution(
                    approval_id=aid,
                    user_id="be_user_a",
                    tool_name="save_memory",
                    arguments={"memory": f"race {self.run_id}"},
                    project_id="proj-e",
                    tenant_id="ten-e",
                )
            )

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(sum(1 for r in results if r.get("success")), 1)

    def test_BE_M05_wrong_project_approval_rejected(self):
        from app.services import approval_service

        created = approval_service.create_approval_request(
            user_id="be_user_a",
            tool_name="save_memory",
            arguments={"memory": "scope"},
            project_id="proj-e",
            tenant_id="ten-e",
        )
        aid = created["approval"]["approval_id"]
        self.__class__._owned_approval_ids.append(aid)
        approval_service.decide_approval(
            approval_id=aid, reviewer_user_id="be_user_a", approve=True
        )
        bad = approval_service.consume_approval_for_execution(
            approval_id=aid,
            user_id="be_user_a",
            tool_name="save_memory",
            arguments={"memory": "scope"},
            project_id="other-project",
            tenant_id="ten-e",
        )
        self.assertFalse(bad.get("success"))
        self.assertEqual(bad.get("error_code"), "SCOPE_MISMATCH")

    def test_BE_M06_commit_boundary_with_real_approval_store(self):
        """Lease invalidation before commit rejects; does not claim OS kill."""
        from app.services.execution_lifecycle import (
            activate_execution,
            bind_execution,
            execution_public_state,
            invalidate_execution,
            reserve_execution,
            reset_execution_lifecycle_for_tests,
            try_begin_commit,
            unbind_execution,
        )

        reset_execution_lifecycle_for_tests()
        eid = reserve_execution(
            tool_name="save_memory",
            approval_id=f"be_appr_{uuid.uuid4().hex[:8]}",
            user_id="be_user_a",
            request_id=self.run_id,
        )
        self.assertTrue(activate_execution(eid, tool_name="save_memory"))
        tokens = bind_execution(eid)
        try:
            invalidate_execution(eid, reason="timeout")
            self.assertFalse(try_begin_commit())
            state = execution_public_state(eid)
            self.assertEqual(state["state"], "invalidated")
            self.assertEqual(state["commit_count"], 0)
        finally:
            unbind_execution(tokens)


if __name__ == "__main__":
    unittest.main()
