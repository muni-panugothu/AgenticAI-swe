"""Regression: live/current facts must enable web_search, not tool-free tech."""

from __future__ import annotations

import unittest

from app import chatbot
from app.services import agent_orchestration as orch
from app.services.agent_routing import route_request


def _names(defs: list[dict]) -> set[str]:
    out: set[str] = set()
    for item in defs:
        name = (item.get("function") or {}).get("name")
        if name:
            out.add(name)
    return out


class CurrentInformationRoutingTests(unittest.TestCase):
    def test_current_officeholder_routes_to_web_search(self):
        msg = "Who is the current Chief Minister of Telangana?"
        intent = orch.classify_intent(msg)
        self.assertEqual(intent, orch.INTENT_CURRENT)
        decision = route_request(
            user_message=msg,
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertTrue(decision.tools_enabled)
        self.assertIn("web_search", _names(decision.tool_definitions))

    def test_latest_ai_developments_is_current(self):
        msg = "What are the latest developments in artificial intelligence?"
        self.assertEqual(orch.classify_intent(msg), orch.INTENT_CURRENT)

    def test_exchange_rate_is_current(self):
        msg = "What is the current USD to INR exchange rate?"
        self.assertEqual(orch.classify_intent(msg), orch.INTENT_CURRENT)
        choice, nudge = chatbot.resolve_tool_choice(
            user_message=msg,
            step=0,
            max_steps=8,
            calculator_used=False,
        )
        self.assertIsInstance(choice, dict)
        self.assertEqual(choice["function"]["name"], "web_search")
        self.assertIn("web_search", nudge or "")

    def test_stable_concept_stays_tool_free(self):
        msg = "What is Python?"
        self.assertEqual(orch.classify_intent(msg), orch.INTENT_TECHNICAL)
        decision = route_request(
            user_message=msg,
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )
        self.assertFalse(decision.tools_enabled)
        self.assertEqual(decision.tool_definitions, [])

    def test_coding_request_not_web_search(self):
        msg = "Write a React login page using Tailwind CSS."
        intent = orch.classify_intent(msg)
        self.assertNotEqual(intent, orch.INTENT_CURRENT)

    def test_clock_only_stays_auto_not_forced_web_search(self):
        msg = "What is the current time?"
        self.assertEqual(orch.classify_intent(msg), orch.INTENT_CURRENT)
        choice, _ = chatbot.resolve_tool_choice(
            user_message=msg,
            step=0,
            max_steps=8,
            calculator_used=False,
        )
        self.assertEqual(choice, "auto")

    def test_repo_current_state_not_web_search(self):
        msg = "What is the current state of authentication in app/services?"
        # Path / app/ target → not CURRENT
        self.assertNotEqual(orch.classify_intent(msg), orch.INTENT_CURRENT)


if __name__ == "__main__":
    unittest.main()
