"""Batch C — web/current vs repository investigation routing & finalization."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JWT_SECRET", "e2e-batch-c-jwt-secret-key-32chars!!")

from app import chatbot
from app.services.agent_orchestration import (
    INTENT_CODEBASE,
    INTENT_CURRENT,
    INTENT_DEBUG,
    derive_investigation_requirements,
    scope_investigation_requirements_for_intent,
)
from app.services.agent_routing import route_request
from tests.e2e.harness import (
    api_e2e_session,
    text_completion,
    tool_completion,
)


class BatchCRoutingFinalizationTests(unittest.TestCase):
    def test_BC_R01_web_queries_route_current_not_investigation(self):
        cases = [
            "Search the web for the latest React news.",
            "Find current information about FastAPI.",
            "Search the web for MongoDB Atlas pricing.",
            "What is new in Python this month?",
        ]
        for message in cases:
            with self.subTest(message=message):
                decision = route_request(
                    user_message=message,
                    all_tool_definitions=chatbot.tool_definitions,
                    multi_tool_fn=chatbot.is_multi_tool_request,
                )
                self.assertEqual(decision.intent, INTENT_CURRENT, message)
                names = {
                    (t.get("function") or {}).get("name")
                    for t in (decision.tool_definitions or [])
                }
                self.assertIn("web_search", names, message)
                self.assertNotIn("read_file", names, message)
                self.assertNotIn("search_code", names, message)

    def test_BC_R02_repo_queries_remain_investigation(self):
        cases = [
            (
                "Find the function that handles authentication in my repository.",
                {INTENT_CODEBASE, INTENT_DEBUG},
            ),
            (
                "Investigate why this repository's tests are failing.",
                {INTENT_CODEBASE, INTENT_DEBUG},
            ),
        ]
        for message, allowed in cases:
            with self.subTest(message=message):
                decision = route_request(
                    user_message=message,
                    all_tool_definitions=chatbot.tool_definitions,
                    multi_tool_fn=chatbot.is_multi_tool_request,
                )
                self.assertIn(decision.intent, allowed, message)
                names = {
                    (t.get("function") or {}).get("name")
                    for t in (decision.tool_definitions or [])
                }
                self.assertTrue(names & {"read_file", "search_code"}, message)

    def test_BC_R03_scope_clears_fastapi_targets_for_current(self):
        raw = derive_investigation_requirements(
            "Search the web for FastAPI lifespan events"
        )
        self.assertTrue(
            any(
                str(t.get("id")) == "FastAPI"
                for t in (raw.get("required_targets") or [])
            )
        )
        scoped = scope_investigation_requirements_for_intent(
            INTENT_CURRENT, raw
        )
        self.assertEqual(scoped.get("required_targets"), [])
        self.assertFalse(scoped.get("needs_search"))
        self.assertFalse(scoped.get("needs_source_reads"))

    def test_BC_R04_web_fastapi_does_not_emit_code_evidence_ledger(self):
        def fake_web_search(query: str = "", **_kwargs):
            return {
                "success": True,
                "results": [
                    {
                        "title": "FastAPI lifespan",
                        "content": "Prefer lifespan context managers over on_event.",
                        "url": "https://example.test/fastapi-lifespan",
                    }
                ],
                "query": query,
            }

        with api_e2e_session(
            user_id="bc_r04",
            fake_web_search=fake_web_search,
            script=[
                tool_completion(
                    "web_search", {"query": "FastAPI lifespan events"}
                ),
                text_completion(
                    "Prefer lifespan context managers instead of on_event."
                ),
                text_completion(
                    "Prefer lifespan context managers instead of on_event."
                ),
            ],
        ) as s:
            result = s.post_chat("Search the web for FastAPI lifespan events")
        self.assertEqual(result.status_code, 200, result.raw_text)
        self.assertIn("web_search", s.recorder.tool_names)
        self.assertNotIn("evidence ledger", result.reply.lower())
        self.assertNotIn("missing_source_evidence", result.reply.lower())
        self.assertRegex(
            result.reply.lower(),
            r"lifespan|on_event|fastapi|search",
        )


if __name__ == "__main__":
    unittest.main()
