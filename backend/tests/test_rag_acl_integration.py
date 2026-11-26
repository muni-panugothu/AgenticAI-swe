"""Live RAG ACL integration against an isolated Atlas test database.

NOT RUN unless explicitly enabled:

    RAG_ACL_INTEGRATION=1
    RAG_TEST_DATABASE=<disposable database name>

Safety gates:
- Never uses MEMORY_DATABASE / agentic_ai_memory as the test target.
- Never silently falls back to the normal application database.
- Cleans up only documents created by this module.

Requires a READY Atlas vectorSearch index on the test DB ``rag_chunks``
collection matching production filter/dimension/similarity settings, plus
working Gemini embeddings for ingestion/query.
"""

from __future__ import annotations

import os
import time
import unittest
import uuid
from dataclasses import replace
from unittest.mock import patch

_RUN = os.getenv("RAG_ACL_INTEGRATION", "").lower() in {"1", "true", "yes"}
_TEST_DB = (os.getenv("RAG_TEST_DATABASE") or "").strip()
_PROD_DB = (os.getenv("MEMORY_DATABASE") or "agentic_ai_memory").strip()
_FORBIDDEN = {_PROD_DB, "agentic_ai_memory", "production", "prod"}
_ISOLATED = bool(_RUN and _TEST_DB and _TEST_DB not in _FORBIDDEN)


def _skip_reason() -> str:
    if not _RUN:
        return (
            "NOT RUN: set RAG_ACL_INTEGRATION=1 and RAG_TEST_DATABASE to an "
            "isolated disposable Atlas database (not MEMORY_DATABASE)"
        )
    if not _TEST_DB:
        return "NOT RUN: RAG_TEST_DATABASE is required for RAG ACL integration"
    if _TEST_DB in _FORBIDDEN:
        return (
            f"NOT RUN: RAG_TEST_DATABASE={_TEST_DB!r} is forbidden; "
            "use a disposable database distinct from production"
        )
    return "NOT RUN: isolated RAG ACL environment is not configured"


@unittest.skipUnless(_ISOLATED, _skip_reason())
class TestRagAclIntegration(unittest.TestCase):
    """Exercise production ingest/share/retrieve ACL against isolated Atlas."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")
        from app.services import memory_service
        from app.services import rag_document_service as rag
        from app.services.rag_config import settings

        cls.rag = rag
        cls.settings = settings
        cls._original_database = memory_service.database
        cls._test_database = memory_service.client[_TEST_DB]
        memory_service.database = cls._test_database
        cls.collection = cls._test_database[settings.collection_name]
        try:
            rag.verify_rag_index()
        except Exception as exc:
            memory_service.database = cls._original_database
            raise unittest.SkipTest(
                "RAG ACL integration skipped: test database vector index is "
                f"missing or not ready ({type(exc).__name__})"
            ) from exc

    @classmethod
    def tearDownClass(cls):
        from app.services import memory_service

        memory_service.database = cls._original_database

    def setUp(self):
        suffix = uuid.uuid4().hex[:12]
        self.suffix = suffix
        self.owner_id = f"rag_acl_owner_{suffix}"
        self.peer_id = f"rag_acl_peer_{suffix}"
        self.stranger_id = f"rag_acl_stranger_{suffix}"
        self.tenant_a = f"ten_acl_a_{suffix}"
        self.tenant_b = f"ten_acl_b_{suffix}"
        self.project_a = f"proj_acl_a_{suffix}"
        self.project_b = f"proj_acl_b_{suffix}"
        self.marker = f"RAGACLMARKER{suffix}"
        self.secret = f"SECRETVALUE{suffix}"
        self.created_document_ids: list[str] = []
        self.created_owner_ids: list[str] = [self.owner_id]
        self.addCleanup(self._cleanup_created)

    def _cleanup_created(self):
        clauses = [
            {"user_id": {"$in": list(self.created_owner_ids)}},
            {"text": {"$regex": self.marker}},
        ]
        if self.created_document_ids:
            clauses.insert(0, {"document_id": {"$in": list(self.created_document_ids)}})
        query = {"$or": clauses}
        result = self.collection.delete_many(query)
        remaining = self.collection.count_documents(query)
        if remaining:
            raise AssertionError(
                f"RAG ACL cleanup left {remaining} temporary chunk(s) "
                f"after deleting matched_count={result.deleted_count}"
            )

    def _doc_bytes(self) -> bytes:
        body = (
            f"According to the uploaded document {self.marker}, "
            f"the notice period token is {self.secret}. "
            f"This temporary ACL fixture belongs only to {self.owner_id}."
        )
        return body.encode("utf-8")

    def _ingest(self, *, tenant_id: str, project_id: str | None):
        result = self.rag.ingest_document(
            owner_id=self.owner_id,
            tenant_id=tenant_id,
            project_id=project_id,
            filename=f"acl_{self.suffix}.txt",
            content_type="text/plain",
            raw=self._doc_bytes(),
        )
        self.created_document_ids.append(result["document_id"])
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result.get("duplicate"))
        return result

    def _query(self) -> str:
        return (
            f"According to the uploaded document, what is the notice period "
            f"token for {self.marker}?"
        )

    def _retrieve(
        self,
        *,
        user_id: str,
        tenant_id: str | None,
        project_id: str | None,
        mode: str = "hybrid",
    ) -> dict:
        return self.rag.retrieve(
            self._query(),
            user_id=user_id,
            tenant_id=tenant_id,
            project_id=project_id,
            mode=mode,
        )

    def _contains_marker(self, result: dict) -> bool:
        blob = self._evidence_blob(result)
        return self.marker in blob and self.secret in blob

    def _evidence_blob(self, result: dict) -> str:
        parts = [str(result.get("context") or "")]
        for row in result.get("candidates") or []:
            parts.append(str(row.get("text") or ""))
            parts.append(str(row.get("chunk_id") or ""))
            parts.append(str(row.get("document_id") or ""))
        for row in result.get("citations") or []:
            parts.append(str(row))
        return "\n".join(parts)

    def _wait_for_authorized_hit(
        self,
        *,
        user_id: str,
        tenant_id: str | None,
        project_id: str | None,
        mode: str = "hybrid",
        timeout_s: float = 90.0,
    ) -> dict:
        deadline = time.time() + timeout_s
        last: dict | None = None
        while time.time() < deadline:
            last = self._retrieve(
                user_id=user_id,
                tenant_id=tenant_id,
                project_id=project_id,
                mode=mode,
            )
            if self._contains_marker(last):
                return last
            time.sleep(2.0)
        self.fail(
            f"Authorized retrieval never surfaced fixture marker for mode={mode} "
            f"candidates={0 if not last else len(last.get('candidates') or [])}"
        )

    def _assert_authorized(self, result: dict):
        self.assertTrue(result.get("has_readable_documents"))
        self.assertTrue(result.get("candidates"))
        joined = self._evidence_blob(result)
        self.assertIn(self.marker, joined)
        self.assertIn(self.secret, joined)
        for row in result.get("citations") or []:
            self.assertTrue(str(row.get("document_id") or ""))
        messages = self.rag.rag_evidence_messages(str(result.get("context") or ""))
        self.assertIn(self.marker, messages[0]["content"])
        self.assertIn(self.secret, messages[0]["content"])

    def _assert_unauthorized(self, result: dict):
        joined = self._evidence_blob(result)
        self.assertNotIn(self.marker, joined)
        self.assertNotIn(self.secret, joined)
        owned_doc = self.created_document_ids[0] if self.created_document_ids else None
        for row in result.get("candidates") or []:
            self.assertNotIn(self.marker, str(row.get("text") or ""))
            self.assertNotIn(self.secret, str(row.get("text") or ""))
            if owned_doc:
                self.assertNotEqual(str(row.get("document_id") or ""), owned_doc)
        for row in result.get("citations") or []:
            self.assertNotIn(self.marker, str(row))
            self.assertNotIn(self.secret, str(row))
            if owned_doc:
                self.assertNotEqual(str(row.get("document_id") or ""), owned_doc)
        messages = self.rag.rag_evidence_messages(str(result.get("context") or ""))
        self.assertNotIn(self.marker, messages[0]["content"])
        self.assertNotIn(self.secret, messages[0]["content"])

    def test_owner_can_retrieve_private_document_stranger_cannot(self):
        self._ingest(tenant_id=self.tenant_a, project_id=None)
        owner_hit = self._wait_for_authorized_hit(
            user_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=None,
            mode="hybrid",
        )
        self._assert_authorized(owner_hit)

        stranger = self._retrieve(
            user_id=self.stranger_id,
            tenant_id=self.tenant_a,
            project_id=None,
            mode="hybrid",
        )
        self._assert_unauthorized(stranger)

    def test_share_grants_and_revoke_removes_access(self):
        ingested = self._ingest(tenant_id=self.tenant_a, project_id=None)
        self._wait_for_authorized_hit(
            user_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=None,
        )
        shared = self.rag.set_document_shares(
            document_id=ingested["document_id"],
            owner_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=None,
            shared_with=[self.peer_id],
        )
        self.assertTrue(shared)

        peer_hit = self._wait_for_authorized_hit(
            user_id=self.peer_id,
            tenant_id=self.tenant_a,
            project_id=None,
        )
        self._assert_authorized(peer_hit)

        unrelated = self._retrieve(
            user_id=self.stranger_id,
            tenant_id=self.tenant_a,
            project_id=None,
        )
        self._assert_unauthorized(unrelated)

        revoked = self.rag.set_document_shares(
            document_id=ingested["document_id"],
            owner_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=None,
            shared_with=[],
        )
        self.assertTrue(revoked)
        after_revoke = self._retrieve(
            user_id=self.peer_id,
            tenant_id=self.tenant_a,
            project_id=None,
        )
        self._assert_unauthorized(after_revoke)

    def test_tenant_isolation_even_with_overlapping_identifiers(self):
        self._ingest(tenant_id=self.tenant_a, project_id=None)
        self._wait_for_authorized_hit(
            user_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=None,
        )
        cross_tenant = self._retrieve(
            user_id=self.owner_id,
            tenant_id=self.tenant_b,
            project_id=None,
        )
        self._assert_unauthorized(cross_tenant)

    def test_project_isolation_for_vector_and_lexical(self):
        self._ingest(tenant_id=self.tenant_a, project_id=self.project_a)
        self._wait_for_authorized_hit(
            user_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=self.project_a,
            mode="hybrid",
        )
        for mode in ("hybrid", "vector", "lexical"):
            with self.subTest(mode=mode):
                wrong_project = self._retrieve(
                    user_id=self.owner_id,
                    tenant_id=self.tenant_a,
                    project_id=self.project_b,
                    mode=mode,
                )
                self._assert_unauthorized(wrong_project)
                none_project = self._retrieve(
                    user_id=self.owner_id,
                    tenant_id=self.tenant_a,
                    project_id=None,
                    mode=mode,
                )
                self._assert_unauthorized(none_project)

    def test_hybrid_vector_lexical_and_rerank_cannot_surface_unauthorized(self):
        self._ingest(tenant_id=self.tenant_a, project_id=None)
        self._wait_for_authorized_hit(
            user_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=None,
            mode="hybrid",
        )
        for mode in ("vector", "lexical", "hybrid"):
            with self.subTest(mode=mode, rerank=False):
                denied = self._retrieve(
                    user_id=self.stranger_id,
                    tenant_id=self.tenant_a,
                    project_id=None,
                    mode=mode,
                )
                self._assert_unauthorized(denied)

        with patch.object(
            self.rag,
            "settings",
            replace(self.settings, enable_reranking=True),
        ):
            denied_rerank = self._retrieve(
                user_id=self.stranger_id,
                tenant_id=self.tenant_a,
                project_id=None,
                mode="hybrid",
            )
            self._assert_unauthorized(denied_rerank)

    def test_cleanup_removes_temporary_chunks(self):
        ingested = self._ingest(tenant_id=self.tenant_a, project_id=None)
        before = self.collection.count_documents({"document_id": ingested["document_id"]})
        self.assertGreater(before, 0)
        deleted = self.rag.delete_document(
            document_id=ingested["document_id"],
            owner_id=self.owner_id,
            tenant_id=self.tenant_a,
            project_id=None,
        )
        self.assertGreater(deleted, 0)
        remaining = self.collection.count_documents({"document_id": ingested["document_id"]})
        self.assertEqual(remaining, 0)
        self.created_document_ids.clear()
