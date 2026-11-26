"""Phase B1 — deterministic input normalization unit tests."""

from __future__ import annotations

import unittest

from app import chatbot
from app.services.agent_orchestration import INTENT_CALC, INTENT_CURRENT, INTENT_TECHNICAL
from app.services.agent_routing import route_request
from app.services.input_normalization import normalize_user_input, routing_text


class InputNormalizationCoreTests(unittest.TestCase):
    def test_preserves_original(self):
        raw = "whos cm of andhraprdesh"
        result = normalize_user_input(raw)
        self.assertEqual(result.original, raw)
        self.assertNotEqual(result.normalized, raw)
        self.assertTrue(result.transforms)

    def test_cm_shortcuts_and_typos(self):
        cases = {
            "cm ap": "Andhra Pradesh",
            "who cm ts": "Telangana",
            "whos cm of andhraprdesh": "Andhra Pradesh",
            "who is telangana cm currently": "Telangana",
        }
        for raw, place in cases.items():
            with self.subTest(raw=raw):
                norm = normalize_user_input(raw).normalized
                self.assertIn("chief minister", norm.casefold())
                self.assertIn(place.casefold(), norm.casefold())

    def test_does_not_expand_ambiguous_cm_unit(self):
        result = normalize_user_input("the board is 5 cm long")
        self.assertEqual(result.normalized, "the board is 5 cm long")
        self.assertFalse(any(t.startswith("cm_context") for t in result.transforms))

    def test_weather_city_and_calc(self):
        weather = normalize_user_input("hyd weather")
        self.assertIn("Hyderabad", weather.normalized)
        self.assertIn("weather", weather.normalized.casefold())

        calc = normalize_user_input("calc 25*45")
        self.assertTrue(calc.normalized.casefold().startswith("calculate"))
        self.assertIn("25*45", calc.normalized)

    def test_preserves_urls_filenames_and_code(self):
        url = "see https://example.com/docs?x=1 please"
        self.assertIn("https://example.com/docs?x=1", normalize_user_input(url).normalized)

        path = "read app/services/agent_routing.py please"
        self.assertIn("app/services/agent_routing.py", normalize_user_input(path).normalized)

        code = "fix this: `def add(a, b): return a + b`"
        self.assertIn("`def add(a, b): return a + b`", normalize_user_input(code).normalized)

    def test_mixed_language_telugu_hindi(self):
        te = normalize_user_input("fastapi lo dependency injection enti")
        self.assertIn("what is", te.normalized.casefold())
        self.assertIn("dependency injection", te.normalized.casefold())
        self.assertIn("fastapi", te.normalized.casefold())
        self.assertIn("te-Latn", te.language_hints)

        hi = normalize_user_input("FastAPI kya hai aur kaise kaam karta hai?")
        self.assertIn("what is fastapi", hi.normalized.casefold())
        self.assertIn("hi-Latn", hi.language_hints)

    def test_informal_codegen_stack(self):
        result = normalize_user_input("make login page react")
        self.assertIn("React", result.normalized)
        self.assertIn("login page", result.normalized.casefold())

    def test_empty_and_none(self):
        self.assertEqual(normalize_user_input("").normalized, "")
        self.assertEqual(normalize_user_input(None).normalized, "")
        self.assertEqual(normalize_user_input("   ").normalized, "")


class InputNormalizationRoutingTests(unittest.TestCase):
    def _route(self, message: str):
        return route_request(
            user_message=message,
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
        )

    def test_shortcuts_route_to_current_information(self):
        for msg in ("cm ap", "who cm ts", "whos cm of andhraprdesh", "who is telangana cm currently"):
            with self.subTest(msg=msg):
                decision = self._route(msg)
                self.assertEqual(decision.intent, INTENT_CURRENT)
                names = {
                    ((t.get("function") or {}).get("name"))
                    for t in decision.tool_definitions
                }
                self.assertIn("web_search", names)
                self.assertTrue(decision.normalization_transforms)

    def test_calc_shorthand_routes_to_calculator(self):
        decision = self._route("calc 25*45")
        self.assertEqual(decision.intent, INTENT_CALC)
        names = {
            ((t.get("function") or {}).get("name"))
            for t in decision.tool_definitions
        }
        self.assertEqual(names, {"calculator"})

    def test_mixed_language_routes_technical(self):
        decision = self._route("fastapi lo dependency injection enti")
        self.assertEqual(decision.intent, INTENT_TECHNICAL)
        self.assertFalse(decision.tools_enabled)

    def test_routing_text_helper(self):
        self.assertIn("chief minister", routing_text("cm ap").casefold())


if __name__ == "__main__":
    unittest.main()
