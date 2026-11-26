"""
Cross-session memory persistence (real MongoDB).

NOT RUN unless MEMORY_PERSISTENCE_INTEGRATION=1 and MONGO_URI point at a disposable DB.
Never run against production data.
"""

from __future__ import annotations

import os
import unittest
import uuid
from unittest import mock

_RUN = os.getenv("MEMORY_PERSISTENCE_INTEGRATION", "").lower() in {
    "1",
    "true",
    "yes",
}


@unittest.skipUnless(
    _RUN,
    "NOT RUN: set MEMORY_PERSISTENCE_INTEGRATION=1 and MEMORY_TEST_DATABASE for Atlas integration",
)
class TestMemoryPersistenceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")
        test_db = os.getenv("MEMORY_TEST_DATABASE", "agentic_ai_memory_test")
        os.environ["MEMORY_DATABASE"] = test_db
        from app.core.config import get_settings
        from app.services import memory_audit_service, memory_service
        from app.services import memory_summary_service

        get_settings.cache_clear()
        cls._original_bindings = (
            memory_service.database,
            memory_service.memory_collection,
            memory_service.audit_collection,
            memory_service.summary_collection,
            memory_audit_service.audit_collection,
            memory_summary_service.summary_collection,
            memory_summary_service.memory_collection_ref,
        )
        test_database = memory_service.client[test_db]
        memory_service.database = test_database
        memory_service.memory_collection = test_database[
            memory_service.MEMORY_COLLECTION
        ]
        memory_service.audit_collection = test_database[
            memory_service.MEMORY_AUDIT_COLLECTION
        ]
        memory_service.summary_collection = test_database[
            memory_service.MEMORY_SUMMARY_COLLECTION
        ]
        memory_audit_service.configure_audit_collection(
            memory_service.audit_collection
        )
        memory_summary_service.configure_summary_collections(
            memory_service.summary_collection,
            memory_service.memory_collection,
        )

    @classmethod
    def tearDownClass(cls):
        from app.services import memory_audit_service, memory_service
        from app.services import memory_summary_service

        (
            memory_service.database,
            memory_service.memory_collection,
            memory_service.audit_collection,
            memory_service.summary_collection,
            audit_collection,
            summary_collection,
            summary_memories,
        ) = cls._original_bindings
        memory_audit_service.configure_audit_collection(audit_collection)
        memory_summary_service.configure_summary_collections(
            summary_collection,
            summary_memories,
        )

    def test_two_users_isolated_across_save_and_fetch(self):
        from app.core.security import (
            create_access_token,
            hash_password,
            hash_refresh_token,
        )
        from app.services import auth_service, memory_service
        from datetime import datetime, timedelta, timezone

        suffix = uuid.uuid4().hex[:8]
        user_a_id = f"usr_test_a_{suffix}"
        user_b_id = f"usr_test_b_{suffix}"
        tenant_a_id = f"ten_a_{suffix}"
        tenant_b_id = f"ten_b_{suffix}"
        self.addCleanup(
            memory_service.memory_collection.delete_many,
            {"user_id": {"$in": [user_a_id, user_b_id]}},
        )
        self.addCleanup(
            memory_service.summary_collection.delete_many,
            {"user_id": {"$in": [user_a_id, user_b_id]}},
        )
        self.addCleanup(
            memory_service.audit_collection.delete_many,
            {"user_id": {"$in": [user_a_id, user_b_id]}},
        )
        self.addCleanup(
            auth_service._users_collection().delete_many,
            {"_id": {"$in": [user_a_id, user_b_id]}},
        )
        self.addCleanup(
            auth_service._sessions_collection().delete_many,
            {"user_id": {"$in": [user_a_id, user_b_id]}},
        )

        users = auth_service._users_collection()
        users.insert_one(
            {
                "_id": user_a_id,
                "email": f"a_{suffix}@example.com",
                "password_hash": hash_password("Secret1pass"),
                "status": "active",
                "tenant_id": tenant_a_id,
            }
        )
        users.insert_one(
            {
                "_id": user_b_id,
                "email": f"b_{suffix}@example.com",
                "password_hash": hash_password("Secret1pass"),
                "status": "active",
                "tenant_id": tenant_b_id,
            }
        )

        # Exercise refresh rotation against BSON dates actually decoded by
        # PyMongo, then verify logout revocation prevents restoration.
        user_a = users.find_one({"_id": user_a_id})
        issued = auth_service.issue_tokens_for_user(user_a)
        refreshed = auth_service.refresh_session(
            refresh_token=issued["refresh_token"]
        )
        self.assertTrue(refreshed.get("success"), refreshed)
        from app.core.security import decode_access_token

        self.assertEqual(
            decode_access_token(refreshed["access_token"])["sub"], user_a_id
        )
        self.assertEqual(refreshed["user"].id, user_a_id)
        self.assertNotEqual(
            issued["refresh_token"], refreshed["refresh_token"]
        )
        self.assertTrue(
            auth_service.revoke_refresh_token(refreshed["refresh_token"])
        )
        restored_after_logout = auth_service.refresh_session(
            refresh_token=refreshed["refresh_token"]
        )
        self.assertFalse(restored_after_logout.get("success"))
        self.assertEqual(restored_after_logout.get("code"), "REFRESH_REUSE")

        expired_tokens = auth_service.issue_tokens_for_user(user_a)
        auth_service._sessions_collection().update_one(
            {"token_hash": hash_refresh_token(expired_tokens["refresh_token"])},
            {"$set": {"expires_at": datetime.now(timezone.utc) - timedelta(days=1)}},
        )
        expired_refresh = auth_service.refresh_session(
            refresh_token=expired_tokens["refresh_token"]
        )
        self.assertFalse(expired_refresh.get("success"))
        self.assertEqual(expired_refresh.get("code"), "EXPIRED_REFRESH")

        embedding = [0.01] * 16
        with mock.patch(
            "app.services.memory_service.create_embedding",
            return_value=embedding,
        ), mock.patch(
            "app.services.memory_service.schedule_summary_refresh",
        ):
            save_a = memory_service.save_memory(
                user_a_id,
                "My preferred programming language is JavaScript",
                embedding=embedding,
                fact_key="preferred_language",
                normalized_value="javascript",
            )
            save_a_project = memory_service.save_memory(
                user_a_id,
                "My project is FreshCart",
                embedding=embedding,
                fact_key="primary_project",
                normalized_value="freshcart",
            )
            save_a_database = memory_service.save_memory(
                user_a_id,
                "My preferred database is MongoDB",
                embedding=embedding,
                fact_key="preferred_database",
                normalized_value="mongodb",
            )
            save_a_ai_project = memory_service.save_memory(
                user_a_id,
                "My AI project is AgenticAI-swe",
                embedding=embedding,
                fact_key="project_name",
                normalized_value="agenticai swe",
            )
            save_b = memory_service.save_memory(
                user_b_id,
                "My preferred programming language is Python",
                embedding=embedding,
                fact_key="preferred_language",
                normalized_value="python",
            )

        self.assertTrue(save_a.get("success"))
        self.assertTrue(save_a_project.get("success"))
        self.assertTrue(save_a_database.get("success"))
        self.assertTrue(save_a_ai_project.get("success"))
        self.assertTrue(save_b.get("success"))

        from bson import ObjectId

        doc_a = memory_service.memory_collection.find_one(
            {"_id": ObjectId(save_a["memory_id"])}
        )
        self.assertIsNotNone(doc_a)
        self.assertEqual(doc_a.get("user_id"), user_a_id)
        self.assertEqual(doc_a.get("owner_id"), user_a_id)

        # Simulate re-login: same ids, new access token.
        token = create_access_token(
            user_id=user_b_id,
            email=f"b_{suffix}@example.com",
            tenant_id=tenant_b_id,
            session_id="integration-test-session",
        )
        from app.core.security import decode_access_token

        self.assertEqual(decode_access_token(token)["sub"], user_b_id)

        # Documents saved before tenant metadata was populated must remain
        # retrievable under the user's authenticated tenant, without crossing
        # into another user's private memories.
        b_rows = memory_service.get_memories(
            user_b_id, limit=10, tenant_id=tenant_b_id
        )
        self.assertTrue(b_rows["success"])
        self.assertEqual(
            [row["memory"] for row in b_rows["memories"]],
            ["My preferred programming language is Python"],
        )
        cross_rows = memory_service.get_memories(
            user_b_id, limit=10, tenant_id=tenant_b_id
        )
        self.assertFalse(
            any("FreshCart" in row["memory"] for row in cross_rows["memories"])
        )

        # Updating the language supersedes that fact only; the other project
        # memory remains active and visible.
        with mock.patch(
            "app.services.memory_service.schedule_summary_refresh",
        ):
            update = memory_service.supersede_memory(
                user_id=user_a_id,
                memory_id=save_a["memory_id"],
                memory="My preferred programming language is Python",
                embedding=embedding,
                fact_key="preferred_language",
                normalized_value="python",
                operation_id=f"integration-update-{suffix}",
                tenant_id=tenant_a_id,
            )
        self.assertTrue(update.get("success"), update)
        a_rows = memory_service.get_memories(
            user_a_id, limit=10, tenant_id=tenant_a_id
        )
        self.assertTrue(a_rows["success"])
        a_texts = [row["memory"] for row in a_rows["memories"]]
        self.assertIn("My preferred programming language is Python", a_texts)
        self.assertIn("My project is FreshCart", a_texts)
        self.assertIn("My preferred database is MongoDB", a_texts)
        self.assertIn("My AI project is AgenticAI-swe", a_texts)
        self.assertNotIn("My preferred programming language is JavaScript", a_texts)

        with mock.patch(
            "app.services.memory_service.create_embedding",
            return_value=embedding,
        ), mock.patch(
            "app.services.memory_service._vector_search_memories",
            side_effect=lambda uid, *a, **k: (
                [
                    {
                        "memory_id": save_b["memory_id"],
                        "user_id": user_b_id,
                        "owner_id": user_b_id,
                        "memory": save_b.get("memory", ""),
                        "score": 0.9,
                        "status": "active",
                        "permission_level": "private",
                        "shared_with": [],
                    },
                    {
                        "memory_id": save_a_project["memory_id"],
                        "user_id": user_a_id,
                        "owner_id": user_a_id,
                        "tenant_id": tenant_a_id,
                        "memory": "My project is FreshCart",
                        "score": 0.99,
                        "status": "active",
                        "permission_level": "private",
                        "shared_with": [],
                    },
                ]
                if uid == user_b_id
                else []
            ),
        ), mock.patch(
            "app.services.memory_service.find_memories_by_fact_key",
            return_value={"success": True, "memories": []},
        ), unittest.mock.patch(
            "app.services.memory_service.search_memory_summaries",
            return_value={"success": True, "summaries": []},
        ):
            search_b = memory_service.semantic_search_memories(
                user_b_id, "Freshcart JavaScript", limit=5
            )
            search_cross = memory_service.semantic_search_memories(
                user_b_id, "FreshCart JavaScript", limit=5
            )

        self.assertTrue(search_b.get("success"))
        self.assertGreaterEqual(len(search_b.get("memories", [])), 1)
        self.assertTrue(search_cross.get("success"))
        texts = " ".join(m.get("memory", "") for m in search_cross.get("memories", []))
        self.assertNotIn("FreshCart", texts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
