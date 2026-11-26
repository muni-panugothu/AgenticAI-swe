"""Batch E offline gate documentation tests (always run; no live services)."""

from __future__ import annotations

import os
import unittest

from tests.integration import batch_e_safety as be


class TestBatchEOfflineGates(unittest.TestCase):
    def test_BE_OFF01_default_flags_skip_live_groups(self):
        # Ensure ordinary suite does not enable Batch E live paths.
        for flag in (
            "BATCH_E_MONGO_INTEGRATION",
            "BATCH_E_REDIS_INTEGRATION",
            "BATCH_E_RAG_INTEGRATION",
            "BATCH_E_LIVE_GEMINI",
            "BATCH_E_LIVE_GROQ",
            "BATCH_E_LIVE_TAVILY",
        ):
            self.assertFalse(
                (os.getenv(flag) or "").strip().lower() in {"1", "true", "yes", "on"},
                msg=f"{flag} must remain off in default offline runs",
            )

    def test_BE_OFF02_gates_report_explicit_skip_reasons(self):
        self.assertFalse(be.mongo_gate().enabled)
        self.assertIn("BATCH_E_MONGO_INTEGRATION", be.mongo_gate().reason)
        self.assertFalse(be.redis_gate().enabled)
        self.assertIn("BATCH_E_REDIS_INTEGRATION", be.redis_gate().reason)
        self.assertFalse(be.rag_gate().enabled)
        self.assertFalse(be.live_gemini_gate().enabled)
        self.assertFalse(be.live_groq_gate().enabled)
        self.assertFalse(be.live_tavily_gate().enabled)

    def test_BE_OFF03_disposable_db_name_rules(self):
        self.assertTrue(be.looks_like_disposable_database("agentic_ai_memory_batch_e_test"))
        self.assertFalse(be.looks_like_disposable_database("agentic_ai_memory"))
        self.assertFalse(be.looks_like_disposable_database("production"))

    def test_BE_OFF04_provider_budget_enforced(self):
        budget = be.ProviderBudget("demo", max_requests=2)
        budget.consume(1)
        budget.consume(1)
        with self.assertRaises(RuntimeError):
            budget.consume(1)
        self.assertEqual(budget.used, 2)

    def test_BE_OFF05_run_id_unique_prefix(self):
        a = be.new_batch_e_run_id()
        b = be.new_batch_e_run_id()
        self.assertTrue(a.startswith("be_"))
        self.assertNotEqual(a, b)

    def test_BE_OFF06_rate_limit_fail_closed_on_redis_error(self):
        """Offline contract: Redis eval errors must not silently allow chat."""
        from unittest.mock import patch

        from app.services import chat_rate_limit_service as crl

        with patch.object(
            crl.redis,
            "eval",
            side_effect=RuntimeError("redis down"),
        ):
            with self.assertRaises(Exception):
                crl.enforce_chat_rate_limit("be_offline_user")


if __name__ == "__main__":
    unittest.main()
