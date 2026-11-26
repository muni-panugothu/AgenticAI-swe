"""Unit tests for the Phase A assistant evaluation framework."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.eval.assistant_eval.loader import category_counts, load_dataset
from app.eval.assistant_eval.runner import (
    evaluate_case_live,
    evaluate_case_offline,
    run_evaluation,
)
from app.eval.assistant_eval.schema import KNOWN_CATEGORIES, validate_dataset
from app.eval.assistant_eval.scorer import (
    STATUS_BLOCKED,
    STATUS_FAIL,
    STATUS_PASS,
    score_intent,
    score_numeric,
    score_tools,
    score_retrieval_bundle,
)
from app.services.rag_evaluation import retrieval_metrics


FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "assistant_eval_v1" / "dataset.json"
)


class AssistantEvalSchemaTests(unittest.TestCase):
    def test_dataset_loads_and_validates(self):
        dataset = load_dataset(FIXTURE)
        self.assertEqual(dataset["version"], "1.4.0")
        self.assertTrue(dataset["cases"])
        validate_dataset(dataset)

    def test_all_eighteen_categories_present(self):
        dataset = load_dataset(FIXTURE)
        counts = category_counts(dataset)
        self.assertEqual(set(counts), set(KNOWN_CATEGORIES))
        for cat, n in counts.items():
            self.assertGreaterEqual(n, 2, msg=f"{cat} needs multiple examples")

    def test_rejects_duplicate_case_ids(self):
        bad = {
            "version": "1.0.0",
            "review": {"status": "draft"},
            "cases": [
                {
                    "case_id": "x",
                    "category": "mathematics",
                    "input": {"message": "1+1"},
                },
                {
                    "case_id": "x",
                    "category": "mathematics",
                    "input": {"message": "2+2"},
                },
            ],
        }
        with self.assertRaises(ValueError):
            validate_dataset(bad)


class AssistantEvalScorerTests(unittest.TestCase):
    def test_intent_and_tools_deterministic(self):
        self.assertTrue(score_intent("calculation", "calculation")["passed"])
        self.assertFalse(score_intent("calculation", "general_conversation")["passed"])
        tools = score_tools(
            required_tools=["calculator"],
            forbidden_tools=["web_search"],
            actual_tools=["calculator"],
            tools_enabled_expected=True,
            tools_enabled_actual=True,
        )
        self.assertTrue(tools["passed"])

    def test_numeric_and_retrieval_reuse_rag_metrics(self):
        self.assertTrue(score_numeric(391, "answer is 391")["passed"])
        self.assertFalse(score_numeric(391, "answer is 390")["passed"])
        bundle = score_retrieval_bundle(
            {
                "relevant_chunk_ids": ["a:1"],
                "retrieved_chunk_ids": ["a:1", "b:1"],
            }
        )
        self.assertTrue(bundle["passed"])
        # Same primitive as RAG evaluation module.
        metrics = retrieval_metrics(["a:1"], ["a:1", "b:1"], k=5)
        self.assertEqual(bundle["metrics"]["recall@5"], metrics["recall@5"])


class AssistantEvalRunnerTests(unittest.TestCase):
    def test_offline_math_case_passes(self):
        dataset = load_dataset(FIXTURE)
        case = next(c for c in dataset["cases"] if c["case_id"] == "math-01")
        result = evaluate_case_offline(case)
        self.assertEqual(result["status"], STATUS_PASS)
        self.assertEqual(result["actual_intent"], "calculation")
        self.assertIn("calculator", result["actual_tools"])

    def test_offline_current_cm_routes_web_search(self):
        dataset = load_dataset(FIXTURE)
        case = next(c for c in dataset["cases"] if c["case_id"] == "curr-cm-tg-01")
        result = evaluate_case_offline(case)
        self.assertEqual(result["actual_intent"], "current_information")
        self.assertIn("web_search", result["actual_tools"])
        self.assertEqual(result["status"], STATUS_PASS)

    def test_rag_context_disables_tools(self):
        dataset = load_dataset(FIXTURE)
        case = next(c for c in dataset["cases"] if c["case_id"] == "rag-hit-01")
        result = evaluate_case_offline(case)
        self.assertFalse(result["tools_enabled"])
        self.assertEqual(result["route_kind"], "rag")
        self.assertEqual(result["status"], STATUS_PASS)

    def test_live_without_chat_fn_is_blocked(self):
        case = {
            "case_id": "live-block",
            "category": "mathematics",
            "input": {"message": "Calculate 1+1"},
            "expect": {"intent": "calculation", "tools_enabled": True, "required_tools": ["calculator"]},
        }
        result = evaluate_case_live(case, chat_fn=None)
        self.assertEqual(result["status"], STATUS_BLOCKED)

    def test_live_with_mock_chat_records_usage(self):
        case = {
            "case_id": "live-mock",
            "category": "mathematics",
            "input": {"message": "Calculate 2+2"},
            "expect": {
                "intent": "calculation",
                "tools_enabled": True,
                "required_tools": ["calculator"],
                "numeric_equals": 4,
                "answer_must_contain": ["4"],
            },
        }

        def chat_fn(_msg: str):
            return {
                "answer": "4",
                "tools_called": ["calculator"],
                "token_usage": {"total_tokens": 12},
            }

        result = evaluate_case_live(case, chat_fn=chat_fn)
        self.assertEqual(result["status"], STATUS_PASS)
        self.assertEqual(result["token_usage"]["total_tokens"], 12)

    def test_run_evaluation_writes_json_report_shape(self):
        report = run_evaluation(dataset_path=str(FIXTURE), live=False)
        self.assertEqual(report["framework"], "assistant_eval")
        self.assertEqual(report["dataset_version"], "1.4.0")
        self.assertEqual(report["mode"], "offline")
        self.assertIn("summary", report)
        self.assertIn("cases", report)
        self.assertGreaterEqual(report["case_count"], 83)
        # Framework itself must produce machine-readable JSON.
        payload = json.dumps(report)
        self.assertIn("PASS", payload)

    def test_intent_mismatch_fails(self):
        case = {
            "case_id": "fail-intent",
            "category": "mathematics",
            "input": {"message": "What is Python?"},
            "expect": {"intent": "calculation"},
        }
        result = evaluate_case_offline(case)
        self.assertEqual(result["status"], STATUS_FAIL)

    def test_cli_module_importable(self):
        from scripts import run_assistant_eval as cli

        self.assertTrue(callable(cli.main))


class AssistantEvalCliSmokeTests(unittest.TestCase):
    def test_cli_offline_writes_report(self):
        from scripts.run_assistant_eval import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            code = main(
                [
                    "--dataset",
                    str(FIXTURE),
                    "--output",
                    str(out),
                    "--case-id",
                    "math-01",
                    "--case-id",
                    "curr-cm-tg-01",
                ]
            )
            self.assertEqual(code, 0)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["case_count"], 2)
            self.assertEqual(data["summary"]["FAIL"], 0)


if __name__ == "__main__":
    unittest.main()
