"""Offline unit tests for RAG golden capture tooling (no Atlas / Gemini)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.rag_config import settings
from app.services.rag_evaluation import evaluate
from tests.rag_eval_capture import case_scope, corpus, safety, seed
from tests.rag_eval_capture.capture import (
    allowed_fixture_ids_for_scope,
    build_capture_payload,
    capture_cases,
    observation_from_retrieve_result,
    patched_rag_settings,
    run_capture,
)
from tests.rag_eval_capture.safety import EvalCaptureSafetyError, eval_database_redirect


def _ready_index_rows():
    return [
        {
            "name": settings.vector_index,
            "type": "vectorSearch",
            "status": "READY",
            "queryable": True,
            "latestDefinition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": settings.embedding_dimensions,
                        "similarity": settings.vector_similarity,
                    },
                    {"type": "filter", "path": "user_id"},
                    {"type": "filter", "path": "project_id"},
                    {"type": "filter", "path": "tenant_id"},
                    {"type": "filter", "path": "document_id"},
                    {"type": "filter", "path": "status"},
                ]
            },
        }
    ]


def _eval_env():
    return {
        "RAG_EVAL_CAPTURE": "1",
        "RAG_EVAL_DATABASE": "rag_eval_capture_disposable_v1",
        "MEMORY_DATABASE": "agentic_ai_memory",
    }


def _acl_eval_env():
    return {
        "RAG_EVAL_CAPTURE": "1",
        "RAG_EVAL_DATABASE": "agentic_ai_memory_rag_acl_test",
        "MEMORY_DATABASE": "agentic_ai_memory",
    }


class TestCorpusAndGolden(unittest.TestCase):
    def test_golden_and_corpus_validate(self):
        golden = corpus.load_golden_dataset()
        manifest = corpus.load_corpus_manifest()
        report = corpus.validate_corpus_against_golden(golden, manifest)
        self.assertEqual(report["case_count"], 9)
        self.assertIn("doc-policy-md:2", report["chunk_ids"])
        self.assertIn("doc-handbook-pdf:1", report["chunk_ids"])
        self.assertIn("doc-roster-csv:0", report["chunk_ids"])

    def test_missing_chunk_mapping_rejected(self):
        golden = corpus.load_golden_dataset()
        manifest = corpus.load_corpus_manifest()
        for doc in manifest["documents"]:
            if doc["document_id"] == "doc-policy-md":
                doc["chunks"] = [c for c in doc["chunks"] if c["chunk_id"] != "doc-policy-md:2"]
        with self.assertRaises(corpus.CorpusMappingError):
            corpus.validate_corpus_against_golden(golden, manifest)

    def test_case_scopes_cover_all_cases(self):
        golden = corpus.load_golden_dataset()
        scopes = case_scope.resolve_all_case_scopes(golden)
        self.assertEqual(len(scopes), 9)
        by_id = {s["case_id"]: s for s in scopes}
        self.assertEqual(by_id["csv-roster-count"]["project_id"], "proj-hr")
        self.assertEqual(by_id["metadata-project-filter"]["tenant_id"], "t1")
        self.assertEqual(by_id["acl-shared-read"]["user_id"], "alice")
        self.assertTrue(by_id["acl-other-tenant"]["expect_acl_isolation"])


class TestSafetyGates(unittest.TestCase):
    def test_forbidden_database_names(self):
        self.assertTrue(safety.is_forbidden_database_name("agentic_ai_memory"))
        self.assertTrue(safety.is_forbidden_database_name("my_production_db"))
        self.assertTrue(safety.is_forbidden_database_name("prod"))
        self.assertTrue(safety.is_forbidden_database_name("MEMORY_DATABASE"))
        self.assertTrue(safety.is_forbidden_database_name("agentic_ai_memory_rag_eval_only"))
        self.assertFalse(safety.is_forbidden_database_name("rag_eval_capture_disposable_v1"))

    def test_exact_production_database_rejected(self):
        self.assertTrue(safety.is_forbidden_database_name("agentic_ai_memory"))
        with self.assertRaises(EvalCaptureSafetyError):
            safety.assert_safe_eval_database(
                environ={"RAG_EVAL_CAPTURE": "1", "RAG_EVAL_DATABASE": "agentic_ai_memory"}
            )

    def test_configured_memory_database_rejected(self):
        env = {
            "RAG_EVAL_CAPTURE": "1",
            "RAG_EVAL_DATABASE": "agentic_ai_memory",
            "MEMORY_DATABASE": "agentic_ai_memory",
        }
        self.assertTrue(safety.is_forbidden_database_name("agentic_ai_memory"))
        with self.assertRaises(EvalCaptureSafetyError):
            safety.assert_safe_eval_database(environ=env)

    def test_allowlisted_acl_test_database_accepted(self):
        self.assertFalse(safety.is_forbidden_database_name("agentic_ai_memory_rag_acl_test"))
        name = safety.assert_safe_eval_database(environ=_acl_eval_env())
        self.assertEqual(name, "agentic_ai_memory_rag_acl_test")
        self.assertEqual(
            safety.ALLOWED_EVAL_DATABASES,
            frozenset({"agentic_ai_memory_rag_acl_test"}),
        )

    def test_non_allowlisted_agentic_ai_memory_suffix_rejected(self):
        self.assertTrue(safety.is_forbidden_database_name("agentic_ai_memory_rag_eval_only"))
        with self.assertRaises(EvalCaptureSafetyError):
            safety.assert_safe_eval_database(
                environ={
                    "RAG_EVAL_CAPTURE": "1",
                    "RAG_EVAL_DATABASE": "agentic_ai_memory_rag_eval_only",
                    "MEMORY_DATABASE": "agentic_ai_memory",
                }
            )

    def test_unrelated_database_with_test_suffix_rejected(self):
        # `_test` alone does not grant an allowlist exception.
        unrelated = "agentic_ai_memory_unrelated_test"
        self.assertNotIn(unrelated.casefold(), {n.casefold() for n in safety.ALLOWED_EVAL_DATABASES})
        self.assertTrue(safety.is_forbidden_database_name(unrelated))
        with self.assertRaises(EvalCaptureSafetyError):
            safety.assert_safe_eval_database(
                environ={
                    "RAG_EVAL_CAPTURE": "1",
                    "RAG_EVAL_DATABASE": unrelated,
                    "MEMORY_DATABASE": "agentic_ai_memory",
                }
            )

    def test_production_and_prod_rejected(self):
        self.assertTrue(safety.is_forbidden_database_name("production"))
        self.assertTrue(safety.is_forbidden_database_name("prod"))
        self.assertTrue(safety.is_forbidden_database_name("my_production_db"))
        for name in ("production", "prod", "customer_prod_mirror"):
            with self.assertRaises(EvalCaptureSafetyError):
                safety.assert_safe_eval_database(
                    environ={
                        "RAG_EVAL_CAPTURE": "1",
                        "RAG_EVAL_DATABASE": name,
                        "MEMORY_DATABASE": "agentic_ai_memory",
                    }
                )

    def test_missing_opt_in_rejected(self):
        with self.assertRaises(EvalCaptureSafetyError):
            safety.assert_eval_capture_enabled(environ={})

    def test_missing_eval_database_rejected(self):
        env = {"RAG_EVAL_CAPTURE": "1"}
        with self.assertRaises(EvalCaptureSafetyError):
            safety.assert_safe_eval_database(environ=env)

    def test_forbidden_eval_database_rejected(self):
        env = {"RAG_EVAL_CAPTURE": "1", "RAG_EVAL_DATABASE": "agentic_ai_memory"}
        with self.assertRaises(EvalCaptureSafetyError):
            safety.assert_safe_eval_database(environ=env)

    def test_approved_database_accepted(self):
        name = safety.assert_safe_eval_database(environ=_eval_env())
        self.assertEqual(name, "rag_eval_capture_disposable_v1")

    def test_injected_collection_missing_db_name_rejected(self):
        collection = MagicMock()
        collection.database = None
        with self.assertRaises(EvalCaptureSafetyError):
            safety.require_collection_database_name(
                collection, approved_name="rag_eval_capture_disposable_v1"
            )

    def test_injected_collection_mismatched_db_rejected(self):
        collection = MagicMock()
        collection.database.name = "agentic_ai_memory"
        with self.assertRaises(EvalCaptureSafetyError):
            safety.require_collection_database_name(
                collection, approved_name="rag_eval_capture_disposable_v1"
            )


class TestEvalDatabaseRedirect(unittest.TestCase):
    def test_redirect_restores_on_success_and_exception(self):
        memory = MagicMock()
        original = MagicMock(name="original_db")
        memory.database = original
        test_db = MagicMock()
        test_db.name = "rag_eval_capture_disposable_v1"
        memory.client.__getitem__.return_value = test_db
        test_db.__getitem__.return_value = MagicMock(database=test_db)

        with patch.dict("os.environ", _eval_env(), clear=False):
            with eval_database_redirect(memory_service=memory, settings=settings) as ctx:
                self.assertIs(memory.database, test_db)
                ctx.assert_still_active()
            self.assertIs(memory.database, original)

            with self.assertRaises(RuntimeError):
                with eval_database_redirect(memory_service=memory, settings=settings):
                    self.assertIs(memory.database, test_db)
                    raise RuntimeError("boom")
            self.assertIs(memory.database, original)

    def test_assert_still_active_detects_pointer_change(self):
        memory = MagicMock()
        original = MagicMock(name="original_db")
        memory.database = original
        test_db = MagicMock()
        test_db.name = "rag_eval_capture_disposable_v1"
        memory.client.__getitem__.return_value = test_db
        test_db.__getitem__.return_value = MagicMock(database=test_db)

        with patch.dict("os.environ", _eval_env(), clear=False):
            with eval_database_redirect(memory_service=memory, settings=settings) as ctx:
                other = MagicMock()
                other.name = "agentic_ai_memory"
                memory.database = other
                with self.assertRaises(EvalCaptureSafetyError):
                    ctx.assert_still_active()

    def test_allowlisted_redirect_verifies_collection_database_identity(self):
        memory = MagicMock()
        original = MagicMock(name="original_db")
        memory.database = original
        test_db = MagicMock()
        test_db.name = "agentic_ai_memory_rag_acl_test"
        memory.client.__getitem__.return_value = test_db
        collection = MagicMock()
        collection.database = test_db
        test_db.__getitem__.return_value = collection

        with patch.dict("os.environ", _acl_eval_env(), clear=False):
            with eval_database_redirect(memory_service=memory, settings=settings) as ctx:
                self.assertEqual(ctx.approved_db, "agentic_ai_memory_rag_acl_test")
                self.assertIs(memory.database, test_db)
                self.assertIs(ctx.collection, collection)
                safety.require_collection_database_name(
                    collection, approved_name="agentic_ai_memory_rag_acl_test"
                )
                ctx.assert_still_active()
            self.assertIs(memory.database, original)

        # Mismatched injected collection identity still fails closed.
        wrong = MagicMock()
        wrong.database.name = "agentic_ai_memory"
        with self.assertRaises(EvalCaptureSafetyError):
            safety.require_collection_database_name(
                wrong, approved_name="agentic_ai_memory_rag_acl_test"
            )


class TestSeedDryRunAndCleanup(unittest.TestCase):
    def test_dry_run_plan_zero_writes(self):
        with patch.dict("os.environ", _eval_env(), clear=False):
            plan = seed.plan_seed(capture_run_id="capture01")
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["writes"], 0)
        self.assertEqual(plan["chunk_count"], 7)

    def test_cleanup_limited_to_marker(self):
        collection = MagicMock()
        collection.database.name = "rag_eval_capture_disposable_v1"
        collection.count_documents.side_effect = [3, 0]
        collection.delete_many.return_value = MagicMock(deleted_count=3)
        with patch.dict("os.environ", _eval_env(), clear=False):
            report = seed.cleanup_capture_run(
                capture_run_id="capture01",
                dry_run=False,
                collection=collection,
                settings=settings,
            )
        collection.delete_many.assert_called_once_with({seed.MARKER_FIELD: "capture01"})
        self.assertEqual(report["deleted_count"], 3)
        self.assertEqual(report["remaining"], 0)

    def test_cleanup_marker_scoped_on_allowlisted_db(self):
        collection = MagicMock()
        collection.database.name = "agentic_ai_memory_rag_acl_test"
        collection.count_documents.side_effect = [2, 0]
        collection.delete_many.return_value = MagicMock(deleted_count=2)
        with patch.dict("os.environ", _acl_eval_env(), clear=False):
            report = seed.cleanup_capture_run(
                capture_run_id="capture01",
                dry_run=False,
                collection=collection,
                settings=settings,
            )
        collection.delete_many.assert_called_once_with({seed.MARKER_FIELD: "capture01"})
        self.assertEqual(report["database"], "agentic_ai_memory_rag_acl_test")
        self.assertEqual(report["deleted_count"], 2)

    def test_cleanup_leaves_unmarked_acl_rows_untouched(self):
        collection = MagicMock()
        collection.database.name = "agentic_ai_memory_rag_acl_test"
        collection.count_documents.side_effect = [1, 0]
        collection.delete_many.return_value = MagicMock(deleted_count=1)
        with patch.dict("os.environ", _acl_eval_env(), clear=False):
            seed.cleanup_capture_run(
                capture_run_id="capture01",
                dry_run=False,
                collection=collection,
                settings=settings,
            )
        # Only the capture marker filter is used — unmarked ACL leftovers are never targeted.
        args, _kwargs = collection.delete_many.call_args
        self.assertEqual(args[0], {seed.MARKER_FIELD: "capture01"})
        self.assertNotIn("user_id", args[0])
        self.assertNotIn("_id", args[0])
        self.assertEqual(collection.delete_many.call_count, 1)

    def test_cleanup_rejects_collection_without_db_name(self):
        collection = MagicMock()
        collection.database = object()  # no .name
        with patch.dict("os.environ", _eval_env(), clear=False):
            with self.assertRaises(EvalCaptureSafetyError):
                seed.cleanup_capture_run(
                    capture_run_id="capture01",
                    dry_run=False,
                    collection=collection,
                    settings=settings,
                )

    def test_seed_same_marker_idempotent(self):
        collection = MagicMock()
        collection.database.name = "rag_eval_capture_disposable_v1"
        collection.aggregate.return_value = _ready_index_rows()
        collection.find.return_value = []  # no foreign markers
        collection.bulk_write.return_value = MagicMock(upserted_count=0)
        collection.count_documents.return_value = 7

        def fake_embed(texts, task_type, **kwargs):
            return [[0.01] * settings.embedding_dimensions for _ in texts]

        with patch.dict("os.environ", _eval_env(), clear=False):
            report = seed.seed_corpus(
                capture_run_id="capture01",
                dry_run=False,
                collection=collection,
                settings=settings,
                embed_many=fake_embed,
                manage_redirect=True,
            )
        self.assertEqual(report["writes"], 7)
        collection.bulk_write.assert_called_once()

    def test_seed_different_marker_fails_closed(self):
        collection = MagicMock()
        collection.database.name = "rag_eval_capture_disposable_v1"
        collection.aggregate.return_value = _ready_index_rows()
        collection.find.return_value = [
            {"_id": "doc-policy-md:2", seed.MARKER_FIELD: "oldermark"}
        ]

        def fake_embed(texts, task_type, **kwargs):
            return [[0.01] * settings.embedding_dimensions for _ in texts]

        with patch.dict("os.environ", _eval_env(), clear=False):
            with self.assertRaises(EvalCaptureSafetyError) as ctx:
                seed.seed_corpus(
                    capture_run_id="capture01",
                    dry_run=False,
                    collection=collection,
                    settings=settings,
                    embed_many=fake_embed,
                )
        self.assertIn("another capture marker", str(ctx.exception))
        collection.bulk_write.assert_not_called()

    def test_seed_after_cleanup_permits_new_marker(self):
        collection = MagicMock()
        collection.database.name = "rag_eval_capture_disposable_v1"
        collection.aggregate.return_value = _ready_index_rows()
        # After cleanup, no foreign markers remain.
        collection.find.return_value = []
        collection.bulk_write.return_value = MagicMock(upserted_count=7)
        collection.count_documents.return_value = 7

        def fake_embed(texts, task_type, **kwargs):
            return [[0.01] * settings.embedding_dimensions for _ in texts]

        with patch.dict("os.environ", _eval_env(), clear=False):
            report = seed.seed_corpus(
                capture_run_id="capture02",
                dry_run=False,
                collection=collection,
                settings=settings,
                embed_many=fake_embed,
            )
        self.assertEqual(report["capture_run_id"], "capture02")

    def test_seed_live_restores_database_pointer(self):
        memory = MagicMock()
        original = MagicMock(name="original_db")
        memory.database = original
        test_db = MagicMock()
        test_db.name = "rag_eval_capture_disposable_v1"
        memory.client.__getitem__.return_value = test_db
        collection = MagicMock()
        collection.database = test_db
        test_db.__getitem__.return_value = collection
        collection.aggregate.return_value = _ready_index_rows()
        collection.find.return_value = []
        collection.bulk_write.return_value = MagicMock(upserted_count=7)
        collection.count_documents.return_value = 7

        def fake_embed(texts, task_type, **kwargs):
            return [[0.01] * settings.embedding_dimensions for _ in texts]

        with patch.dict("os.environ", _eval_env(), clear=False):
            report = seed.seed_corpus(
                capture_run_id="capture01",
                dry_run=False,
                memory_service=memory,
                settings=settings,
                embed_many=fake_embed,
            )
        self.assertFalse(report["dry_run"])
        self.assertIs(memory.database, original)
        collection.bulk_write.assert_called_once()


class TestCaptureObservations(unittest.TestCase):
    def test_empty_retrieval_abstains(self):
        obs = observation_from_retrieve_result(
            {
                "candidates": [],
                "citations": [],
                "latency_ms": 12,
                "embedding_calls": 1,
                "rerank_calls": 0,
                "diagnostics": {},
            },
            allowed_chunk_ids=set(),
            allowed_document_ids=set(),
        )
        self.assertTrue(obs["abstained"])
        self.assertEqual(obs["answer"], "")
        self.assertEqual(obs["retrieved_chunk_ids"], [])
        self.assertFalse(obs["acl_violation"])

    def test_packed_candidates_mapped(self):
        obs = observation_from_retrieve_result(
            {
                "candidates": [
                    {
                        "chunk_id": "doc-policy-md:2",
                        "document_id": "doc-policy-md",
                        "text": "notice 14 days",
                    },
                ],
                "citations": [{"chunk_id": "doc-policy-md:2"}],
                "latency_ms": 5,
                "embedding_calls": 1,
                "rerank_calls": 0,
                "diagnostics": {"lexical_backend_used": "overlap"},
            },
            allowed_chunk_ids={"doc-policy-md:2"},
            allowed_document_ids={"doc-policy-md"},
        )
        self.assertFalse(obs["abstained"])
        self.assertEqual(obs["retrieved_chunk_ids"], ["doc-policy-md:2"])
        self.assertEqual(obs["citation_chunk_ids"], ["doc-policy-md:2"])
        self.assertEqual(obs["evidence_texts"], ["notice 14 days"])
        self.assertFalse(obs["acl_violation"])

    def test_unauthorized_candidate_reported(self):
        obs = observation_from_retrieve_result(
            {
                "candidates": [
                    {
                        "chunk_id": "doc-other-tenant:0",
                        "document_id": "doc-other-tenant",
                        "text": "SECRET",
                    },
                    {
                        "chunk_id": "doc-policy-md:2",
                        "document_id": "doc-policy-md",
                        "text": "14 days",
                    },
                ],
                "citations": [],
                "latency_ms": 1,
                "embedding_calls": 0,
                "rerank_calls": 0,
                "diagnostics": {},
            },
            allowed_chunk_ids={"doc-policy-md:2", "doc-policy-md:0", "doc-policy-md:1"},
            allowed_document_ids={"doc-policy-md"},
        )
        self.assertTrue(obs["acl_violation"])
        self.assertEqual(obs["unauthorized_chunk_ids"], ["doc-other-tenant:0"])
        # Unauthorized rows remain in retrieved_chunk_ids (diagnostic, not stripped).
        self.assertIn("doc-other-tenant:0", obs["retrieved_chunk_ids"])
        self.assertIn("doc-policy-md:2", obs["retrieved_chunk_ids"])

    def test_allowed_fixture_scope_excludes_other_tenant(self):
        manifest = corpus.load_corpus_manifest()
        scope = {
            "user_id": "alice",
            "tenant_id": "t1",
            "project_id": None,
        }
        allowed = allowed_fixture_ids_for_scope(scope, manifest)
        self.assertIn("doc-policy-md", allowed["document_ids"])
        self.assertIn("doc-handbook-pdf", allowed["document_ids"])
        self.assertNotIn("doc-other-tenant", allowed["document_ids"])
        self.assertNotIn("doc-roster-csv", allowed["document_ids"])

    def test_payload_compatible_with_evaluate(self):
        golden = corpus.load_golden_dataset()
        observations = {
            str(case["case_id"]): {
                "retrieved_chunk_ids": list(case.get("relevant_chunk_ids") or []),
                "citation_chunk_ids": list(case.get("relevant_chunk_ids") or []),
                "answer": "",
                "abstained": not bool(case.get("relevant_chunk_ids")),
                "evidence_texts": [],
                "unauthorized_chunk_ids": [],
                "acl_violation": False,
                "latency_ms": 0,
                "embedding_calls": 0,
                "rerank_calls": 0,
                "diagnostics": {},
            }
            for case in golden["cases"]
        }
        payload = build_capture_payload(
            golden=golden,
            observations=observations,
            reranker="none",
            top_k=6,
            capture_run_id="capture01",
            database="rag_eval_capture_disposable_v1",
            dry_run=True,
        )
        self.assertEqual(len(payload["cases"]), 9)
        report = evaluate(golden, {"cases": payload["cases"]}, k=5)
        self.assertEqual(report["case_count"], 9)
        self.assertFalse(report["llm_judge"]["enabled"])

    def test_reranker_patch_does_not_change_env(self):
        from app.services import rag_document_service as rag
        from app.services import rag_rerank

        before_env = settings.reranker
        with patched_rag_settings(reranker="heuristic"):
            self.assertEqual(rag.settings.reranker, "heuristic")
            self.assertEqual(rag_rerank.settings.reranker, "heuristic")
        self.assertEqual(rag.settings.reranker, before_env)
        self.assertEqual(rag_rerank.settings.reranker, before_env)

    def test_capture_cases_calls_retrieve_with_scopes(self):
        golden = corpus.load_golden_dataset()
        manifest = corpus.load_corpus_manifest()
        calls = []

        def fake_retrieve(question, *, user_id, tenant_id, project_id, mode="hybrid"):
            calls.append(
                {
                    "question": question,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "mode": mode,
                }
            )
            return {
                "candidates": [],
                "citations": [],
                "latency_ms": 1,
                "embedding_calls": 0,
                "rerank_calls": 0,
                "diagnostics": {},
            }

        observations = capture_cases(
            golden=golden,
            manifest=manifest,
            reranker="none",
            retrieve_fn=fake_retrieve,
        )
        self.assertEqual(len(observations), 9)
        self.assertEqual(len(calls), 9)
        self.assertTrue(all(c["mode"] == "hybrid" for c in calls))
        roster = next(c for c in calls if "roster" in c["question"].lower())
        self.assertEqual(roster["project_id"], "proj-hr")

    def test_run_capture_dry_run_writes_output_without_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "captured_results.json"
            with patch.dict("os.environ", _eval_env(), clear=False):
                payload = run_capture(
                    dataset_path=corpus.DEFAULT_GOLDEN,
                    output_path=out,
                    reranker="none",
                    top_k=None,
                    dry_run=True,
                    capture_run_id="capture01",
                    seed=True,
                )
            self.assertTrue(out.exists())
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(set(loaded["cases"]), set(payload["cases"]))
            self.assertEqual(len(loaded["cases"]), 9)
            self.assertTrue(loaded["meta"]["dry_run"])
            self.assertEqual(loaded["meta"]["reranker"], "none")

    def test_live_capture_keeps_eval_db_for_every_retrieve(self):
        memory = MagicMock()
        original = MagicMock(name="original_db")
        memory.database = original
        test_db = MagicMock()
        test_db.name = "rag_eval_capture_disposable_v1"
        memory.client.__getitem__.return_value = test_db
        collection = MagicMock()
        collection.database = test_db
        test_db.__getitem__.return_value = collection
        collection.aggregate.return_value = _ready_index_rows()
        collection.find.return_value = []
        collection.bulk_write.return_value = MagicMock(upserted_count=7)
        collection.count_documents.return_value = 7
        seen_dbs: list[str] = []

        def fake_embed(texts, task_type, **kwargs):
            return [[0.01] * settings.embedding_dimensions for _ in texts]

        def fake_retrieve(question, *, user_id, tenant_id, project_id, mode="hybrid"):
            seen_dbs.append(getattr(memory.database, "name", None))
            return {
                "candidates": [],
                "citations": [],
                "latency_ms": 1,
                "embedding_calls": 0,
                "rerank_calls": 0,
                "diagnostics": {},
            }

        with patch.dict("os.environ", _eval_env(), clear=False):
            run_capture(
                dataset_path=corpus.DEFAULT_GOLDEN,
                output_path=None,
                reranker="none",
                top_k=None,
                dry_run=False,
                capture_run_id="capture01",
                seed=True,
                retrieve_fn=fake_retrieve,
                memory_service=memory,
                settings=settings,
                embed_many=fake_embed,
            )
        self.assertEqual(len(seen_dbs), 9)
        self.assertTrue(all(name == "rag_eval_capture_disposable_v1" for name in seen_dbs))
        self.assertIs(memory.database, original)

    def test_live_capture_no_seed_still_uses_eval_db(self):
        memory = MagicMock()
        original = MagicMock(name="original_db")
        memory.database = original
        test_db = MagicMock()
        test_db.name = "rag_eval_capture_disposable_v1"
        memory.client.__getitem__.return_value = test_db
        collection = MagicMock()
        collection.database = test_db
        test_db.__getitem__.return_value = collection
        seen_dbs: list[str] = []

        def fake_retrieve(question, *, user_id, tenant_id, project_id, mode="hybrid"):
            seen_dbs.append(getattr(memory.database, "name", None))
            return {
                "candidates": [],
                "citations": [],
                "latency_ms": 1,
                "embedding_calls": 0,
                "rerank_calls": 0,
                "diagnostics": {},
            }

        with patch.dict("os.environ", _eval_env(), clear=False):
            run_capture(
                dataset_path=corpus.DEFAULT_GOLDEN,
                output_path=None,
                reranker="none",
                top_k=None,
                dry_run=False,
                capture_run_id="capture01",
                seed=False,
                retrieve_fn=fake_retrieve,
                memory_service=memory,
                settings=settings,
            )
        self.assertEqual(len(seen_dbs), 9)
        self.assertTrue(all(name == "rag_eval_capture_disposable_v1" for name in seen_dbs))
        self.assertIs(memory.database, original)

    def test_pointer_change_between_retrieves_aborts(self):
        memory = MagicMock()
        original = MagicMock(name="original_db")
        memory.database = original
        test_db = MagicMock()
        test_db.name = "rag_eval_capture_disposable_v1"
        memory.client.__getitem__.return_value = test_db
        collection = MagicMock()
        collection.database = test_db
        test_db.__getitem__.return_value = collection
        calls = {"n": 0}

        def fake_retrieve(question, *, user_id, tenant_id, project_id, mode="hybrid"):
            calls["n"] += 1
            if calls["n"] == 2:
                other = MagicMock()
                other.name = "agentic_ai_memory"
                memory.database = other
            return {
                "candidates": [],
                "citations": [],
                "latency_ms": 1,
                "embedding_calls": 0,
                "rerank_calls": 0,
                "diagnostics": {},
            }

        with patch.dict("os.environ", _eval_env(), clear=False):
            with self.assertRaises(EvalCaptureSafetyError):
                run_capture(
                    dataset_path=corpus.DEFAULT_GOLDEN,
                    output_path=None,
                    reranker="none",
                    top_k=None,
                    dry_run=False,
                    capture_run_id="capture01",
                    seed=False,
                    retrieve_fn=fake_retrieve,
                    memory_service=memory,
                    settings=settings,
                )
        # Outer finally must restore even after abort.
        self.assertIs(memory.database, original)


if __name__ == "__main__":
    unittest.main()
