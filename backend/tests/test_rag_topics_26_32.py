"""Focused unit tests for RAG Topics 26–32 (no live Atlas / provider calls)."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services import rag_document_service as rag
from app.services import rag_evaluation as evaluation
from app.services import rag_lexical
from app.services import rag_query_rewrite
from app.services import rag_rerank
from app.services.rag_acl import authorize_candidates
from app.services.rag_config import resolved_num_candidates, settings, validate_settings
from app.services.rag_context import (
    compress_evidence_llm,
    evidence_wrapper_overhead_tokens,
    is_near_duplicate,
    select_context_token_budget,
)
from app.services.rag_tokens import count_tokens


def _acl_row(**overrides):
    base = {
        "chunk_id": "c1",
        "document_id": "doc-1",
        "filename": "a.md",
        "chunk_index": 0,
        "start_line": 1,
        "end_line": 2,
        "text": "leave policy notice period is 14 days",
        "score": 0.5,
        "retrieval_score": 0.5,
        "retrieval_source": "vector",
        "user_id": "alice",
        "owner_id": "alice",
        "tenant_id": "t1",
        "project_id": None,
        "shared_with": [],
        "permission_level": "private",
        "status": "ready",
    }
    base.update(overrides)
    return base


class TestTopic26VectorSearch(unittest.TestCase):
    def test_vector_search_success_and_min_score(self):
        collection = MagicMock()
        collection.aggregate.return_value = [
            _acl_row(chunk_id="keep", score=0.9, text="keep me"),
            _acl_row(chunk_id="drop", score=0.1, text="too low"),
        ]
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "_embed", return_value=[0.0] * 768
        ), patch.object(rag, "settings", replace(settings, min_vector_score=0.25)):
            rows = rag._vector_candidates(
                "notice",
                user_id="alice",
                tenant_id="t1",
                project_id=None,
                allowed_docs={"doc-1"},
                allowed_owners=["alice"],
            )
        self.assertEqual([r["chunk_id"] for r in rows], ["keep"])
        stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        self.assertEqual(stage["numCandidates"], resolved_num_candidates())
        self.assertEqual(stage["limit"], settings.candidate_limit)

    def test_num_candidates_config_validation(self):
        from app.services import rag_config

        bad = replace(rag_config.settings, num_candidates=5, candidate_limit=80)
        with patch.object(rag_config, "settings", bad), self.assertRaises(ValueError):
            rag_config.validate_settings()

    def test_invalid_embedding_dimension_rejected(self):
        with patch.object(rag, "_embed", return_value=[0.0] * 16), self.assertRaises(rag.RagError):
            rag._vector_candidates(
                "q",
                user_id="alice",
                tenant_id="t1",
                project_id=None,
                allowed_docs={"doc-1"},
                allowed_owners=["alice"],
            )

    def test_duplicate_chunk_ids_removed(self):
        collection = MagicMock()
        collection.aggregate.return_value = [
            _acl_row(chunk_id="dup", score=0.9),
            _acl_row(chunk_id="dup", score=0.8),
        ]
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "_embed", return_value=[0.0] * 768
        ):
            rows = rag._vector_candidates(
                "q",
                user_id="alice",
                tenant_id="t1",
                project_id=None,
                allowed_docs={"doc-1"},
                allowed_owners=["alice"],
            )
        self.assertEqual(len(rows), 1)

    def test_empty_retrieval(self):
        collection = MagicMock()
        collection.aggregate.return_value = []
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "_embed", return_value=[0.0] * 768
        ):
            rows = rag._vector_candidates(
                "q",
                user_id="alice",
                tenant_id="t1",
                project_id=None,
                allowed_docs={"doc-1"},
                allowed_owners=["alice"],
            )
        self.assertEqual(rows, [])

    def test_missing_index_still_fail_closed(self):
        collection = MagicMock()
        collection.aggregate.return_value = []
        with patch.object(rag, "_collection", return_value=collection), self.assertRaises(rag.RagError):
            rag.verify_rag_index()


class TestTopic27AclMetadata(unittest.TestCase):
    def test_authorize_mixed_candidates(self):
        ok = _acl_row(chunk_id="ok")
        bad_tenant = _acl_row(chunk_id="bad", tenant_id="t2")
        revoked = _acl_row(
            chunk_id="revoked",
            user_id="bob",
            owner_id="bob",
            shared_with=[],
            permission_level="private",
        )
        shared = _acl_row(
            chunk_id="shared",
            user_id="bob",
            owner_id="bob",
            shared_with=["alice"],
            permission_level="shared_read",
        )
        out = authorize_candidates(
            [ok, bad_tenant, revoked, shared],
            user_id="alice",
            tenant_id="t1",
            project_id=None,
            allowed_docs={"doc-1"},
            same_scope=rag._same_scope,
        )
        self.assertEqual([r["chunk_id"] for r in out], ["ok", "shared"])

    def test_project_and_ready_filters(self):
        wrong_project = _acl_row(chunk_id="p", project_id="other")
        not_ready = _acl_row(chunk_id="n", status="ingesting")
        out = authorize_candidates(
            [wrong_project, not_ready],
            user_id="alice",
            tenant_id="t1",
            project_id=None,
            allowed_docs={"doc-1"},
            same_scope=rag._same_scope,
        )
        self.assertEqual(out, [])

    def test_authorization_after_rerank_and_rewrite_paths(self):
        unauthorized = _acl_row(
            chunk_id="x",
            user_id="bob",
            owner_id="bob",
            shared_with=[],
            permission_level="private",
            text="secret",
            score=0.99,
        )
        authorized = _acl_row(chunk_id="y", text="policy", score=0.1)
        reranked, _diag = rag_rerank.rerank(
            "policy",
            [unauthorized, authorized],
            mode="heuristic",
        )
        # Heuristic may put unauthorized first; ACL must still drop it.
        final = authorize_candidates(
            reranked,
            user_id="alice",
            tenant_id="t1",
            project_id=None,
            allowed_docs={"doc-1"},
            same_scope=rag._same_scope,
        )
        self.assertEqual([r["chunk_id"] for r in final], ["y"])


class TestTopic28HybridBm25(unittest.TestCase):
    def test_overlap_fallback_and_bm25_success(self):
        collection = MagicMock()
        collection.find.return_value.limit.return_value = iter([
            _acl_row(text="notice period fourteen days leave policy"),
        ])
        rows = rag_lexical.overlap_candidates(
            "notice period",
            collection=collection,
            user_id="alice",
            tenant_id="t1",
            project_id=None,
            allowed_docs={"doc-1"},
            same_scope=rag._same_scope,
            candidate_fn=rag._candidate,
        )
        self.assertEqual(rows[0]["retrieval_source"], "lexical")

        collection.aggregate.side_effect = [
            [{"name": "rag_text_index", "type": "search", "status": "READY", "queryable": True}],
            [_acl_row(chunk_id="bm", score=3.2, text="notice")],
        ]
        bm25 = rag_lexical.bm25_candidates(
            "notice",
            collection=collection,
            user_id="alice",
            tenant_id="t1",
            project_id=None,
            allowed_docs={"doc-1"},
            allowed_owners=["alice"],
            same_scope=rag._same_scope,
            candidate_fn=rag._candidate,
        )
        self.assertEqual(bm25[0]["retrieval_source"], "bm25")

    def test_bm25_unavailable_falls_back(self):
        collection = MagicMock()
        collection.aggregate.return_value = []  # missing search index
        collection.find.return_value.limit.return_value = iter([
            _acl_row(text="alpha beta gamma"),
        ])
        with patch.object(rag, "settings", replace(settings, lexical_backend="bm25")), patch.object(
            rag, "_collection", return_value=collection
        ):
            rows, diag = rag._lexical_candidates(
                "alpha",
                user_id="alice",
                tenant_id="t1",
                project_id=None,
                allowed_docs={"doc-1"},
                allowed_owners=["alice"],
            )
        self.assertTrue(diag["lexical_fallback"])
        self.assertEqual(diag["lexical_backend_used"], "overlap")
        self.assertEqual(rows[0]["retrieval_source"], "lexical")

    def test_weighted_rrf_and_dedupe(self):
        left = [_acl_row(chunk_id="a", retrieval_source="vector", score=0.9)]
        right = [
            _acl_row(chunk_id="a", retrieval_source="bm25", score=0.2),
            _acl_row(chunk_id="b", retrieval_source="bm25", score=0.1),
        ]
        fused = rag._reciprocal_rank_fusion([left, right], weights=[2.0, 1.0])
        self.assertEqual([r["chunk_id"] for r in fused], ["a", "b"])
        self.assertIn("vector", fused[0]["retrieval_source"])
        self.assertIn("bm25", fused[0]["retrieval_source"])

    def test_both_backends_unavailable_hybrid(self):
        with patch.object(rag, "_readable_documents", return_value=({"doc-1"}, ["alice"])), patch.object(
            rag, "verify_rag_index", side_effect=rag.RagError("missing")
        ), patch.object(
            rag,
            "_lexical_candidates",
            return_value=([], {"lexical_unavailable": True, "lexical_fallback": True, "lexical_backend_used": "overlap", "lexical_backend_requested": "bm25"}),
        ):
            result = rag.retrieve("policy", user_id="alice", tenant_id="t1", mode="hybrid")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["candidates"], [])


class TestTopic29Rerank(unittest.TestCase):
    def setUp(self):
        rag_rerank.reset_cross_encoder_for_tests()

    def test_heuristic_ranking_and_top_n(self):
        rows = [
            _acl_row(chunk_id="low", text="unrelated text", score=0.9),
            _acl_row(chunk_id="high", text="leave policy notice", score=0.1),
            _acl_row(chunk_id="mid", text="leave unrelated", score=0.5),
        ]
        ranked, diag = rag_rerank.rerank("leave policy", rows, mode="heuristic")
        self.assertEqual(ranked[0]["chunk_id"], "high")
        self.assertTrue(diag["rerank_applied"])
        self.assertIn("retrieval_score", ranked[0])
        self.assertIn("rerank_score", ranked[0])

        limited = replace(settings, rerank_top_n=1, reranker="heuristic")
        with patch.object(rag_rerank, "settings", limited):
            ranked2, _ = rag_rerank.rerank("leave policy", rows, mode="heuristic")
        # Only first candidate is rescored; order of head changes, tail preserved.
        self.assertEqual(len(ranked2), 3)

    def test_cross_encoder_missing_falls_back(self):
        rows = [
            _acl_row(chunk_id="a", text="leave policy"),
            _acl_row(chunk_id="b", text="other"),
        ]
        with patch.object(rag_rerank, "_load_cross_encoder", side_effect=RuntimeError("missing")):
            ranked, diag = rag_rerank.rerank("leave", rows, mode="cross_encoder")
        self.assertEqual(diag["rerank_fallback"], "heuristic")
        self.assertEqual(ranked[0]["chunk_id"], "a")

    def test_metadata_preserved(self):
        rows = [_acl_row(chunk_id="a", text="leave"), _acl_row(chunk_id="b", text="x")]
        ranked, _ = rag_rerank.rerank("leave", rows, mode="heuristic")
        self.assertEqual(ranked[0]["document_id"], "doc-1")
        self.assertEqual(ranked[0]["filename"], "a.md")


class TestTopic30QueryRewrite(unittest.TestCase):
    def test_original_preserved_and_limits(self):
        q = "What does the policy say about leave and what is the notice period?"
        rewritten = rag.rewrite_queries(q)
        self.assertEqual(rewritten[0], q)
        self.assertGreaterEqual(len(rewritten), 2)
        limited = replace(settings, max_rewritten_queries=1, enable_query_decomposition=True)
        with patch.object(rag, "settings", limited):
            self.assertEqual(rag.rewrite_queries(q), [q])

    def test_multi_query_and_llm_failure(self):
        with patch.object(rag, "settings", replace(settings, enable_multi_query=True, max_rewritten_queries=3)), patch(
            "app.services.rag_query_rewrite._llm_multi_queries", return_value=["alt query one", "What does the policy say about leave and what is the notice period?"]
        ):
            out = rag.rewrite_queries("base question here")
        self.assertEqual(out[0], "base question here")
        self.assertIn("alt query one", out)

        with patch.object(rag, "settings", replace(settings, enable_multi_query=True)), patch(
            "app.services.rag_query_rewrite._llm_multi_queries", return_value=[]
        ):
            self.assertEqual(rag.rewrite_queries("only original query"), ["only original query"])

    def test_hyde_not_evidence(self):
        with patch("app.chatbot.client") as client:
            client.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content="Hypothetical leave policy says 14 days."))
            ]
            passage = rag_query_rewrite.generate_hyde_passage("notice?", enabled=True)
        self.assertIn("Hypothetical", passage)

        calls = {"n": 0}

        def fake_vector(*_args, **kwargs):
            calls["n"] += 1
            source = kwargs.get("retrieval_source", "vector")
            if source == "hyde":
                return [_acl_row(chunk_id="hyde-hit", retrieval_source="hyde", text="real chunk")]
            return []

        with patch.object(rag, "_readable_documents", return_value=({"doc-1"}, ["alice"])), patch.object(
            rag, "verify_rag_index"
        ), patch.object(rag, "_vector_candidates", side_effect=fake_vector), patch.object(
            rag, "settings", replace(settings, enable_hyde=True)
        ), patch(
            "app.services.rag_document_service.generate_hyde_passage", return_value="hypo text"
        ), patch.object(rag, "_embed", return_value=[0.0] * 768):
            result = rag.retrieve("notice period", user_id="alice", tenant_id="t1", mode="vector")
        self.assertTrue(result["diagnostics"]["hyde_used"])
        self.assertFalse(result["diagnostics"]["hyde_is_evidence"])
        self.assertNotIn("hypo text", result.get("context") or "")


class TestTopic31ContextCompression(unittest.TestCase):
    def test_token_budget_and_wrapper(self):
        overhead = evidence_wrapper_overhead_tokens(
            rag.RAG_EVIDENCE_USER_PREFIX,
            rag.RAG_EVIDENCE_ASSISTANT_ACK,
        )
        self.assertGreater(overhead, 0)
        rows = [
            _acl_row(chunk_id="a", text="alpha evidence"),
            _acl_row(chunk_id="b", text="beta evidence"),
        ]
        context, selected = select_context_token_budget(
            rows, budget_tokens=overhead + 500, wrapper_overhead=overhead, drop_near_duplicates=False
        )
        self.assertTrue(selected)
        self.assertLessEqual(count_tokens(context) + overhead, overhead + 500)
        # Tiny budget cannot fit any chunk after wrapper overhead.
        empty_ctx, empty_sel = select_context_token_budget(
            rows, budget_tokens=overhead + 1, wrapper_overhead=overhead, drop_near_duplicates=False
        )
        self.assertEqual(empty_ctx, "")
        self.assertEqual(empty_sel, [])

    def test_near_duplicate_removal_and_citations(self):
        self.assertTrue(is_near_duplicate("leave policy notice period", "leave policy notice period"))
        a = _acl_row(chunk_id="a", text="leave policy notice period fourteen days")
        b = _acl_row(chunk_id="b", text="leave policy notice period fourteen days")
        context, selected = rag.select_context(
            [a, b], prefer_tokens=True, budget_tokens=2000, drop_near_duplicates=True
        )
        self.assertEqual([r["chunk_id"] for r in selected], ["a"])
        self.assertIn("source_id", context)

    def test_compression_failure_falls_back(self):
        rows = [_acl_row(text="long evidence about leave")]
        self.assertIsNone(compress_evidence_llm("q", rows, enabled=False))
        with patch("app.chatbot.client", None):
            self.assertIsNone(compress_evidence_llm("q", rows, enabled=True))


class TestTopic32Evaluation(unittest.TestCase):
    def test_metrics_and_golden_dataset(self):
        metrics = evaluation.retrieval_metrics(["a", "b"], ["x", "b", "a"], k=3)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["mrr@3"], 0.5)
        self.assertEqual(evaluation.citation_metrics(["a"], ["a", "z"])["citation_precision"], 0.5)
        self.assertEqual(evaluation.faithfulness_heuristic("fourteen days leave", ["leave fourteen days"]), 1.0)

        golden = json.loads(
            (Path(__file__).parent / "fixtures" / "rag_golden_v2.json").read_text(encoding="utf-8")
        )
        evaluation.validate_golden_dataset(golden)
        results = {
            "cases": {
                case["case_id"]: {
                    "retrieved_chunk_ids": case.get("relevant_chunk_ids", []),
                    "citation_chunk_ids": case.get("relevant_chunk_ids", []),
                    "answer": " ".join(case.get("expected_answer_keywords") or ["cannot find"]),
                    "abstained": case.get("expected_abstain", False),
                    "evidence_texts": ["fourteen days quarterly engineer leave"],
                    "unauthorized_chunk_ids": [],
                }
                for case in golden["cases"]
            }
        }
        report = evaluation.evaluate(golden, results, k=5)
        self.assertEqual(report["case_count"], len(golden["cases"]))
        self.assertEqual(report["llm_judge"]["enabled"], False)
        self.assertIsNotNone(report["retrieval_metrics"])
        # Repeatability
        report2 = evaluation.evaluate(golden, results, k=5)
        self.assertEqual(report, report2)

    def test_malformed_dataset_and_incorrect_citations(self):
        with self.assertRaises(ValueError):
            evaluation.validate_golden_dataset({"version": "x", "cases": [], "review": {}})
        cite = evaluation.citation_metrics(["a"], ["z"])
        self.assertEqual(cite["citation_precision"], 0.0)
        self.assertEqual(cite["citation_recall"], 0.0)

    def test_legacy_fixture_still_scores(self):
        fixture = Path(__file__).parent / "fixtures" / "rag_eval_v1.json"
        dataset = json.loads(fixture.read_text(encoding="utf-8"))
        report = evaluation.evaluate(
            dataset,
            {
                "cases": {
                    "retrieval-ttl": {
                        "retrieved_chunk_ids": ["fixture-auth-ttl"],
                        "citation_chunk_ids": ["fixture-auth-ttl"],
                        "answer": "15 minute",
                        "abstained": False,
                    },
                    "retrieval-model": {
                        "retrieved_chunk_ids": ["fixture-embedding-config"],
                        "citation_chunk_ids": ["fixture-embedding-config"],
                        "answer": "gemini-embedding-001 with 768",
                        "abstained": False,
                    },
                    "unanswerable": {
                        "retrieved_chunk_ids": [],
                        "citation_chunk_ids": [],
                        "answer": "I cannot find evidence.",
                        "abstained": True,
                    },
                    "permission-isolation": {
                        "retrieved_chunk_ids": [],
                        "citation_chunk_ids": [],
                        "answer": "I cannot access that.",
                        "abstained": True,
                    },
                }
            },
        )
        self.assertEqual(report["abstention_accuracy"], 1.0)


class TestConfigValidation(unittest.TestCase):
    def test_defaults_validate(self):
        validate_settings()
        self.assertGreaterEqual(resolved_num_candidates(), settings.candidate_limit)


if __name__ == "__main__":
    unittest.main()
