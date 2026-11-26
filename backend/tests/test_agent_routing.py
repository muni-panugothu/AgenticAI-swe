"""Topic 36 — typed agent routing (RouteDecision / route_request)."""

from __future__ import annotations

import unittest

from app import chatbot
from app.services import agent_orchestration as orch
from app.services.agent_routing import (
    ROUTE_DIRECT,
    ROUTE_INVESTIGATION,
    ROUTE_MULTI_TOOL,
    ROUTE_RAG,
    memory_tools_in_definitions,
    route_request,
)


def _names(tool_defs: list) -> set[str]:
    return {
        (t.get("function") or {}).get("name")
        for t in tool_defs
        if (t.get("function") or {}).get("name")
    }


class TestAgentRouting(unittest.TestCase):
    def test_greeting_remains_tool_free(self):
        decision = route_request(
            user_message="Hi, how are you?",
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertEqual(decision.intent, orch.INTENT_GENERAL)
        self.assertFalse(decision.tools_enabled)
        self.assertEqual(decision.tool_definitions, [])
        self.assertEqual(decision.route_kind, ROUTE_DIRECT)
        self.assertEqual(decision.confidence, "high")
        self.assertFalse(decision.fallback_applied)

    def test_intent_to_allowlist_codebase(self):
        decision = route_request(
            user_message="Find my agent loop implementation",
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertEqual(decision.intent, orch.INTENT_CODEBASE)
        self.assertTrue(decision.tools_enabled)
        self.assertEqual(_names(decision.tool_definitions), {"search_code", "read_file"})
        self.assertEqual(decision.route_kind, ROUTE_INVESTIGATION)

    def test_codebase_requests_retain_restricted_tools(self):
        """Codebase intent must not widen to the full catalog or memory tools."""
        decision = route_request(
            user_message="Find my agent loop implementation",
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertEqual(decision.intent, orch.INTENT_CODEBASE)
        names = _names(decision.tool_definitions)
        self.assertEqual(names, {"search_code", "read_file"})
        self.assertNotIn("calculator", names)
        self.assertNotIn("web_search", names)
        self.assertFalse(memory_tools_in_definitions(decision.tool_definitions))

    def test_multi_tool_keeps_full_catalog(self):
        msg = "Search the code for calculator and also calculate 2+2"
        self.assertTrue(chatbot.is_multi_tool_request(msg))
        decision = route_request(
            user_message=msg,
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertTrue(decision.multi_tool)
        self.assertTrue(decision.tools_enabled)
        self.assertEqual(decision.route_kind, ROUTE_MULTI_TOOL)
        names = _names(decision.tool_definitions)
        catalog = _names(chatbot.tool_definitions)
        self.assertEqual(names, catalog)
        self.assertIn("calculator", names)
        self.assertIn("search_code", names)

    def test_rag_disables_tools(self):
        decision = route_request(
            user_message="What does the uploaded doc say about RAG?",
            rag_context="Evidence chunk about retrieval-augmented generation.",
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertEqual(decision.intent, orch.INTENT_TECHNICAL)
        self.assertFalse(decision.tools_enabled)
        self.assertEqual(decision.tool_definitions, [])
        self.assertEqual(decision.route_kind, ROUTE_RAG)
        self.assertEqual(decision.reason, "rag_context_forces_technical_no_tools")

    def test_empty_input_fallback(self):
        for msg in ("", "   ", None):
            with self.subTest(msg=msg):
                decision = route_request(
                    user_message=msg,
                    all_tool_definitions=chatbot.tool_definitions,
                    multi_tool_fn=chatbot.is_multi_tool_request,
                )
                self.assertEqual(decision.intent, orch.INTENT_GENERAL)
                self.assertFalse(decision.tools_enabled)
                self.assertEqual(decision.tool_definitions, [])
                self.assertEqual(decision.confidence, "low")
                self.assertTrue(decision.fallback_applied)
                self.assertEqual(decision.route_kind, ROUTE_DIRECT)

    def test_memory_tools_not_exposed_to_unauthorized_intents(self):
        cases = [
            ("Hi there", orch.INTENT_GENERAL),
            ("What is RAG?", orch.INTENT_TECHNICAL),
            ("Find my agent loop implementation", orch.INTENT_CODEBASE),
            ("Calculate 25 * 40", orch.INTENT_CALC),
            ("What is the current time?", orch.INTENT_CURRENT),
        ]
        for msg, expected_intent in cases:
            with self.subTest(msg=msg):
                decision = route_request(
                    user_message=msg,
                    all_tool_definitions=chatbot.tool_definitions,
                    multi_tool_fn=chatbot.is_multi_tool_request,
                )
                self.assertEqual(decision.intent, expected_intent)
                self.assertFalse(
                    memory_tools_in_definitions(decision.tool_definitions),
                    f"memory tools leaked for intent={decision.intent}",
                )

    def test_memory_intent_keeps_memory_allowlist_only(self):
        decision = route_request(
            user_message="What do you remember about me?",
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertEqual(decision.intent, orch.INTENT_MEMORY)
        names = _names(decision.tool_definitions)
        allow = orch.INTENT_TOOL_ALLOWLIST[orch.INTENT_MEMORY]
        self.assertTrue(names)
        self.assertTrue(names.issubset(allow))
        self.assertNotIn("search_code", names)
        self.assertNotIn("read_file", names)

    def test_matches_legacy_allowlist_helpers(self):
        """route_request tool set matches tools_for_intent for single-tool intents."""
        msg = "Investigate this error in my backend and fix the bug"
        intent = orch.classify_intent(msg)
        legacy = orch.tools_for_intent(intent, chatbot.tool_definitions)
        decision = route_request(
            user_message=msg,
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertFalse(decision.multi_tool)
        self.assertEqual(decision.intent, intent)
        self.assertEqual(_names(decision.tool_definitions), _names(legacy))
        self.assertEqual(
            decision.tools_enabled,
            orch.intent_allows_tools(intent) and bool(legacy),
        )


if __name__ == "__main__":
    unittest.main()
