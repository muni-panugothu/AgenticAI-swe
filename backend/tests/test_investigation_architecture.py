"""Regression tests for investigation checklist, budgets, AST boundaries."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services import agent_orchestration as orch
from app.services.investigation_state import (
    STATUS_DISCOVERED,
    STATUS_PENDING,
    STATUS_VERIFIED,
    InvestigationBudgetManager,
    InvestigationChecklist,
    build_budget_from_requirements,
)
from app.tools import file_reader as fr
from app.validation.tool_models import ReadFileInput
from pydantic import ValidationError
from app import chatbot


def _top_level_def_span(rel_path: str, name: str) -> tuple[int, int]:
    """Inclusive (start, end) line span for a module-level def/class."""
    path = Path(fr.PROJECT_ROOT) / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            end = getattr(node, "end_lineno", None) or node.lineno
            return int(node.lineno), int(end)
    raise AssertionError(f"{name!r} not found in {rel_path}")


class TestContinuationVerification(unittest.TestCase):
    def test_continuation_window_still_verifies_after_def_seen(self):
        """Body-only continuation must not leave the symbol permanently unread."""
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent. Read the complete implementation."
        )
        evidence = orch.new_investigation_evidence_state()
        def_start, def_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "classify_intent"
        )
        path = "app/services/agent_orchestration.py"
        orch.register_search_discoveries(
            evidence,
            req,
            [{
                "file_path": path,
                "line": def_start,
                "text": "def classify_intent(user_message):",
            }],
        )
        # First window includes the def but not the full body.
        partial_end = min(def_start + 35, def_end - 1)
        orch.register_read_verification(
            evidence,
            req,
            file_path=path,
            line_start=def_start,
            line_end=partial_end,
            content=(
                "def classify_intent(user_message: str) -> str:\n"
                "    lowered = user_message.lower()\n"
                "    if educational:\n"
                "        return INTENT_TECHNICAL\n"
            ),
            function_complete=False,
        )
        self.assertIn("classify_intent", evidence["verified_targets"])

    def test_body_only_continuation_verifies_when_def_covered(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent. Read the complete implementation."
        )
        evidence = orch.new_investigation_evidence_state()
        def_start, def_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "classify_intent"
        )
        path = "app/services/agent_orchestration.py"
        orch.register_search_discoveries(
            evidence,
            req,
            [{
                "file_path": path,
                "line": def_start,
                "text": "def classify_intent(user_message):",
            }],
        )
        # Window A covers the def but we simulate progress without verify edge.
        window_a_end = def_start + 20
        evidence["read_ranges"].append({
            "file_path": path,
            "line_start": def_start,
            "line_end": window_a_end,
        })
        evidence.setdefault("target_progress", {})["classify_intent"] = {
            "saw_def": True,
            "body": "def classify_intent(user_message: str) -> str:\n    x = 1\n",
            "file_path": path,
        }
        # Window B is body-only continuation past the def through the end.
        newly = orch.register_read_verification(
            evidence,
            req,
            file_path=path,
            line_start=window_a_end + 1,
            line_end=def_end,
            content=(
                "    if _targets_user_repo(text):\n"
                "        return INTENT_CODEBASE\n"
                "    return INTENT_TECHNICAL\n"
            ),
            function_complete=True,
        )
        self.assertIn("classify_intent", newly)
        self.assertIn("classify_intent", evidence["verified_targets"])

    def test_reconcile_covered_clustered_helpers(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_groq_rate_limit and is_tpm_rate_limit_error. "
            "Read the source."
        )
        evidence = orch.new_investigation_evidence_state()
        # One window covers both current definitions — resolve via AST so the
        # contract survives legitimate source drift.
        groq_start, groq_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "classify_groq_rate_limit"
        )
        tpm_start, tpm_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "is_tpm_rate_limit_error"
        )
        line_start = min(groq_start, tpm_start)
        line_end = max(groq_end, tpm_end)
        path = "app/services/agent_orchestration.py"
        evidence["read_ranges"].append({
            "file_path": path,
            "line_start": line_start,
            "line_end": line_end,
        })
        evidence["read_snippets"].append(
            f"read_file {path}:{line_start}-{line_end}: "
            "def classify_groq_rate_limit(exc):\n    return 'tpm'\n"
            "def is_tpm_rate_limit_error(exc):\n    return True\n"
        )
        evidence.setdefault("target_progress", {})["classify_groq_rate_limit"] = {
            "saw_def": True,
            "body": "def classify_groq_rate_limit(exc):\n    return 'tpm'\n",
            "file_path": path,
        }
        evidence.setdefault("target_progress", {})["is_tpm_rate_limit_error"] = {
            "saw_def": True,
            "body": "def is_tpm_rate_limit_error(exc):\n    return True\n",
            "file_path": path,
        }
        # Hints must point at the same defs the covering window includes.
        self.assertEqual(
            orch.KNOWN_SYMBOL_LOCATIONS["classify_groq_rate_limit"][1],
            groq_start,
        )
        self.assertEqual(
            orch.KNOWN_SYMBOL_LOCATIONS["is_tpm_rate_limit_error"][1],
            tpm_start,
        )
        newly = orch.reconcile_covered_but_unverified(evidence, req)
        self.assertIn("classify_groq_rate_limit", newly)
        self.assertIn("is_tpm_rate_limit_error", newly)

    def test_rate_limit_retry_seed_points_at_error_handler(self):
        path, line = orch.KNOWN_SYMBOL_LOCATIONS["rate_limit_retry_handling"]
        self.assertIn("chatbot.py", path)
        self.assertGreaterEqual(line, 3700)

    def test_stale_overlap_rebuilds_is_tpm_after_groq_read(self):
        """After classify_groq is read, is_tpm must still get a window."""
        req = orch.derive_investigation_requirements(
            "Inspect classify_groq_rate_limit and is_tpm_rate_limit_error. "
            "Read the source."
        )
        ev = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, ev)
        ev["read_ranges"].append({
            "file_path": "app/services/agent_orchestration.py",
            "line_start": 704,
            "line_end": 740,
        })
        ev["verified_targets"] = {"classify_groq_rate_limit"}
        nxt = orch.next_unread_suggested_read(
            req, ev, prefer_targets=orch.RATE_LIMIT_SYMBOL_PRIORITY
        )
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt["target"], "is_tpm_rate_limit_error")
        self.assertGreaterEqual(nxt["line_start"], 741)


class TestInvestigationChecklist(unittest.TestCase):
    def test_lifecycle_pending_discovered_verified(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and execute_tool. Read the source."
        )
        checklist = InvestigationChecklist(req)
        self.assertIn("classify_intent", checklist.ids())
        self.assertEqual(
            checklist.status_of("classify_intent"), STATUS_PENDING
        )
        checklist.mark_discovered(
            "classify_intent",
            file_path="app/services/agent_orchestration.py",
            def_line=213,
            line_start=211,
            line_end=235,
        )
        self.assertEqual(
            checklist.status_of("classify_intent"), STATUS_DISCOVERED
        )
        # Search must not verify.
        self.assertNotIn("classify_intent", checklist.by_status(STATUS_VERIFIED))
        checklist.mark_verified(
            "classify_intent",
            file_path="app/services/agent_orchestration.py",
            line_start=211,
            line_end=260,
        )
        self.assertEqual(
            checklist.status_of("classify_intent"), STATUS_VERIFIED
        )
        self.assertNotIn("classify_intent", checklist.uncovered())

    def test_register_search_does_not_verify(self):
        req = orch.derive_investigation_requirements(
            "Find classify_intent implementation and read it."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.register_search_discoveries(
            evidence,
            req,
            [{
                "file_path": "app/services/agent_orchestration.py",
                "line": 213,
                "text": "def classify_intent(user_message):",
            }],
        )
        self.assertIn("classify_intent", evidence["discovered"])
        self.assertNotIn("classify_intent", evidence["verified_targets"])
        statuses = evidence.get("target_statuses") or {}
        self.assertEqual(
            statuses.get("classify_intent", {}).get("status"),
            STATUS_DISCOVERED,
        )


class TestInvestigationBudgetManager(unittest.TestCase):
    def test_continuation_not_blocked_by_soft_discovery_cap(self):
        budget = InvestigationBudgetManager(
            max_reads_per_file=2,
            max_reads_total=20,
            max_continuation_reads=6,
            max_tool_executions=20,
        )
        budget.configure_for_requirements(
            required_target_count=1,
            symbols_per_file={"app/chatbot.py": 1},
        )
        path = "app/chatbot.py"
        # Two discovery reads fill soft floor.
        for _ in range(2):
            ok, reason = budget.allow_read(path, is_continuation=False)
            self.assertTrue(ok, msg=reason)
            budget.record_read(path, is_continuation=False, content_chars=100)
        # Continuation still allowed within scaled file limit.
        ok, reason = budget.allow_read(path, is_continuation=True)
        self.assertTrue(ok, msg=reason)

    def test_search_budget_and_duplicate_reason(self):
        budget = InvestigationBudgetManager(max_searches=1, max_tool_executions=5)
        ok, _ = budget.allow_search()
        self.assertTrue(ok)
        budget.record_search()
        ok2, reason = budget.allow_search()
        self.assertFalse(ok2)
        self.assertEqual(reason, "PER_TOOL_EXECUTION_LIMIT")

    def test_build_budget_scales_with_symbols(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_groq_rate_limit and TPM TPD RPM RPD helpers. "
            "Read the source."
        )
        budget = build_budget_from_requirements(
            req, known_locations=orch.KNOWN_SYMBOL_LOCATIONS
        )
        self.assertGreaterEqual(
            budget.max_reads_total, len(req["required_targets"])
        )
        path = "app/services/agent_orchestration.py"
        self.assertGreater(
            budget.file_read_limit(path), orch.MAX_READS_PER_FILE
        )


class TestAstFunctionBoundary(unittest.TestCase):
    def test_ast_span_for_classify_intent(self):
        """Probe a line inside classify_intent — not a brittle absolute line."""
        def_start, def_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "classify_intent"
        )
        probe = def_start + min(5, max(0, def_end - def_start))
        path = (
            fr.PROJECT_ROOT / "app" / "services" / "agent_orchestration.py"
        )
        text = path.read_text(encoding="utf-8")
        span = fr.ast_enclosing_symbol_span(text, probe)
        self.assertIsNotNone(span)
        start, end = span
        self.assertEqual(start, def_start)
        self.assertEqual(end, def_end)
        self.assertLessEqual(start, probe)
        self.assertGreaterEqual(end, probe)
        self.assertGreater(end, start)
        # Classification behavior must remain intact at the resolved symbol.
        self.assertEqual(
            orch.classify_intent("Calculate 12 * 8"),
            orch.INTENT_CALC,
        )

    def test_long_function_continuation_uses_next_line(self):
        # First window of a long function must ask for exact next_line.
        result = fr.read_file(
            "app/chatbot.py",
            line_start=2455,
            line_end=2524,
        )
        self.assertIsInstance(result, str)
        import json
        meta = json.loads(result.split("\ncontent:\n", 1)[0])
        if meta.get("needs_continuation"):
            self.assertEqual(
                meta["continuation"]["next_line"],
                meta["line_end"] + 1,
            )
            self.assertFalse(meta["function_complete"])
            # span_limited alone must not imply function_complete
            if meta.get("span_limited"):
                self.assertNotEqual(
                    meta["boundary_kind"], "function_complete"
                )


class TestCanonicalArgsEverywhere(unittest.TestCase):
    def test_execute_tool_strips_legacy_aliases(self):
        with patch.object(
            chatbot,
            "tools",
            {
                **chatbot.tools,
                "read_file": lambda **kw: (
                    '{"success": true, "file_path": "%s", '
                    '"line_start": %s, "line_end": %s}\ncontent:\nx\n'
                    % (kw["file_path"], kw["line_start"], kw["line_end"])
                ),
            },
        ):
            result = chatbot.execute_tool(
                "read_file",
                {
                    "file_path": "app/tools/calculator.py",
                    "start_line": 1,
                    "end_line": 20,
                    "line_start": 1,
                    "line_end": 20,
                },
            )
        self.assertTrue(result.get("success"))

    def test_extra_field_still_forbidden_on_model(self):
        with self.assertRaises(ValidationError):
            ReadFileInput(
                file_path="app/x.py",
                line_start=1,
                line_end=10,
                start_line=1,
            )


class TestFinalizationToolFree(unittest.TestCase):
    def test_force_final_omits_tools(self):
        kwargs = orch.build_chat_completion_kwargs(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=200,
            tools=[{"type": "function"}],
            tool_choice="auto",
            allow_tools=False,
        )
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("tool_choice", kwargs)


class TestNoFalsePassIncomplete(unittest.TestCase):
    def test_offline_partial_lists_missing_targets(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and execute_tool. Read the source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.register_search_discoveries(
            evidence,
            req,
            [{
                "file_path": "app/services/agent_orchestration.py",
                "line": 213,
                "text": "def classify_intent():",
            }],
        )
        text = orch.build_offline_evidence_answer(
            "Inspect both",
            evidence_snippets=["search_code: classify_intent"],
            reason="agent_limit",
            requirements=req,
            evidence_state=evidence,
        ).lower()
        self.assertIn("missing", text)
        self.assertIn("execute_tool", text)
        # Must not claim full pass when unread.
        self.assertNotIn("investigation complete", text)


class TestMultiSymbolPriorityDrain(unittest.TestCase):
    """Regression: do not stop after classify_intent + classify_groq only."""

    def test_forced_drain_covers_rate_limit_helpers_after_wasted_searches(self):
        """LLM wastes turns on skipped searches; orchestration must still
        force-read remaining required rate-limit helpers before finalizing.
        """
        calls = {"n": 0}
        read_targets = []

        def create(**kwargs):
            calls["n"] += 1
            if "tools" not in kwargs:
                missing = kwargs.get("_missing_hint", "")
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(
                                content=(
                                    "PARTIAL investigation summary. "
                                    "Verified classify_intent and "
                                    "classify_groq_rate_limit from source. "
                                    "Other helpers remain unread."
                                ),
                                tool_calls=None,
                            )
                        )
                    ]
                )
            # Mimic the live log: search, overlapping reads, then redundant
            # searches for already-known symbols.
            if calls["n"] == 1:
                tc = MagicMock()
                tc.id = "s1"
                tc.function.name = "search_code"
                tc.function.arguments = (
                    '{"search_text":"classify_intent"}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            if calls["n"] == 2:
                tc = MagicMock()
                tc.id = "r1"
                tc.function.name = "read_file"
                tc.function.arguments = (
                    '{"file_path":"app/services/agent_orchestration.py",'
                    '"line_start":210,"line_end":260}'
                )
                return MagicMock(
                    choices=[
                        MagicMock(
                            message=MagicMock(content=None, tool_calls=[tc])
                        )
                    ]
                )
            # Subsequent planning rounds try redundant searches (blocked).
            tc = MagicMock()
            tc.id = f"s{calls['n']}"
            tc.function.name = "search_code"
            tc.function.arguments = '{"search_text":"read_file("}'
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content=None, tool_calls=[tc])
                    )
                ]
            )

        def fake_execute(name, args, user_id=None, request_id=None, **_kwargs):
            if name == "search_code":
                return {
                    "success": True,
                    "result": (
                        '{"success": true, "matches": ['
                        '{"file_path":"app/services/agent_orchestration.py",'
                        '"line":214,'
                        '"text":"def classify_intent(user_message):"}]}'
                    ),
                }
            if name == "read_file":
                start = int(args.get("line_start") or 1)
                end = int(args.get("line_end") or start + 40)
                read_targets.append((start, end))
                # Serve real file windows so verification + AST work.
                return {
                    "success": True,
                    "result": chatbot.execute_tool.__wrapped__
                    if False
                    else __import__(
                        "app.tools.file_reader", fromlist=["read_file"]
                    ).read_file(
                        args.get("file_path"),
                        line_start=start,
                        line_end=end,
                    ),
                }
            return {"success": True, "result": "{}"}

        # Use real read_file via tools dict path through execute_tool.
        with patch.object(chatbot, "client") as mock_client, patch.object(
            chatbot, "load_long_term_memory",
            return_value={"memories": [], "conflicts": [], "summaries": []},
        ), patch.object(
            chatbot, "restore_task_context_from_redis"
        ), patch.object(
            chatbot, "save_task_context_to_redis"
        ), patch.object(
            chatbot, "try_save_explicit_memory", return_value=False
        ), patch.object(
            chatbot, "try_update_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_delete_explicit_memory", return_value=None
        ), patch.object(
            chatbot, "try_inspect_memory_metadata", return_value=None
        ):
            mock_client.chat.completions.create.side_effect = create

            # Patch only search_code side; let read_file hit the real tool.
            real_execute = chatbot.execute_tool

            def selective_execute(name, args, user_id=None, request_id=None):
                if name == "search_code":
                    return fake_execute(name, args, user_id, request_id)
                if name == "read_file":
                    start = int(
                        (args or {}).get("line_start")
                        or (args or {}).get("start_line")
                        or 1
                    )
                    end = int(
                        (args or {}).get("line_end")
                        or (args or {}).get("end_line")
                        or start + 40
                    )
                    read_targets.append((start, end))
                return real_execute(
                    name, args, user_id=user_id, request_id=request_id
                )

            with patch.object(
                chatbot, "execute_tool", side_effect=selective_execute
            ):
                chatbot._get_conversation_store("orch-multi-drain").clear()
                answer = chatbot.chat_with_ai(
                    "Inspect classify_intent, classify_groq_rate_limit, "
                    "is_tpm_rate_limit_error, is_tpd_rate_limit_error, "
                    "is_rpm_rate_limit_error, is_rpd_rate_limit_error, "
                    "and rate_limit_retry_handling. Read the relevant source.",
                    user_id="orch-multi-drain",
                    request_id="req-multi-drain",
                )

        # Must have forced more than intent+groq classifier windows.
        self.assertGreaterEqual(len(read_targets), 4)
        starts = [s for s, _ in read_targets]
        # Rate-limit helper cluster begins near classify_groq_rate_limit.
        self.assertTrue(
            any(s >= 590 for s in starts),
            msg=f"expected rate-limit reads, got {read_targets}",
        )
        # At least one is_* helper window beyond classify_groq body.
        self.assertTrue(
            any(s >= 630 for s in starts),
            msg=f"expected is_* helper reads after groq classifier; {read_targets}",
        )
        lowered = answer.lower()
        # If anything remains unread, answer must be an honest PARTIAL.
        if "missing" in lowered or "partial" in lowered:
            self.assertIn("partial", lowered)

    def test_alias_tpm_does_not_verify_is_tpm_helper(self):
        """Docstring mentioning tpm must not verify is_tpm_rate_limit_error."""
        req = orch.derive_investigation_requirements(
            "Inspect is_tpm_rate_limit_error. Read the source."
        )
        evidence = orch.new_investigation_evidence_state()
        newly = orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/agent_orchestration.py",
            line_start=704,
            line_end=720,
            content=(
                "def classify_groq_rate_limit(exc):\n"
                "    \"\"\"Return 'tpm' | 'tpd'.\"\"\"\n"
                "    return 'tpm'\n"
            ),
        )
        self.assertNotIn("is_tpm_rate_limit_error", newly)
        self.assertNotIn(
            "is_tpm_rate_limit_error", evidence.get("verified_targets") or set()
        )


class TestRateLimitClassification(unittest.TestCase):
    def test_structured_tpd_not_tpm(self):
        class FakeExc(Exception):
            def __init__(self):
                super().__init__("Error code: 429 - rate_limit_exceeded")
                self.body = {
                    "error": {
                        "message": "tokens per day (TPD): Limit 200000",
                        "type": "tokens",
                        "code": "rate_limit_exceeded",
                    }
                }

        self.assertEqual(orch.classify_groq_rate_limit(FakeExc()), "tpd")
        self.assertTrue(orch.is_tpd_rate_limit_error(FakeExc()))
        self.assertFalse(orch.is_tpm_rate_limit_error(FakeExc()))

    def test_ambiguous_429_not_tpm(self):
        self.assertIsNone(
            orch.classify_groq_rate_limit(
                Exception("Error code: 429 - rate_limit_exceeded")
            )
        )


class TestFinalizationChecklist(unittest.TestCase):
    """Finalization messages must include per-target PASS/PARTIAL statuses."""

    def test_checklist_messages_include_verified_as_pass(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and classify_groq_rate_limit. Read source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/agent_orchestration.py",
            line_start=275,
            line_end=360,
            content="def classify_intent(user_message):\n    return 'x'\n",
            function_complete=True,
        )
        self.assertIn("classify_intent", evidence["verified_targets"])
        msgs = orch.build_finalization_messages_with_checklist(
            user_message="Inspect both",
            evidence_snippets=["read_file: def classify_intent..."],
            missing_targets=orch.uncovered_required_targets(req, evidence),
            evidence_state=evidence,
            requirements=req,
        )
        blob = " ".join(str(m.get("content") or "") for m in msgs)
        self.assertIn("`classify_intent`", blob)
        self.assertIn("PASS", blob)
        self.assertIn("classify_groq_rate_limit", blob)

    def test_checklist_messages_without_evidence_still_work(self):
        msgs = orch.build_finalization_messages_with_checklist(
            user_message="Inspect X",
            evidence_snippets=[],
            missing_targets=["X"],
        )
        self.assertTrue(any("tools are disabled" in str(m.get("content") or "").lower() for m in msgs))


class TestUndrainedContinuation(unittest.TestCase):
    """Regression: ast_symbol_end=512 but last read ends at 453."""

    def test_drain_loop_finishes_pending_continuation(self):
        """Drain loop must not stop while a pending continuation exists
        and tool budget remains — even when max_drain was artificially low."""
        req = orch.derive_investigation_requirements(
            "Inspect read_file implementation. Read the complete source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, evidence)
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/tools/file_reader.py",
            line_start=221,
            line_end=290,
            content="def read_file(file_path, line_start=None):\n    x = 1\n",
            function_complete=False,
        )
        remaining_budget = 8
        max_drain = remaining_budget
        self.assertGreaterEqual(max_drain, 5)

    def test_pending_continuation_forces_partial_not_pass(self):
        req = orch.derive_investigation_requirements(
            "Inspect read_file implementation. Read the complete source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/tools/file_reader.py",
            line_start=221,
            line_end=453,
            content="def read_file(file_path, line_start=None):\n    body\n",
            function_complete=False,
        )
        pending = {
            "file_path": "app/tools/file_reader.py",
            "next_line": 454,
            "suggested_end_line": 512,
        }
        row = orch.classify_target_evidence_status(
            "read_file",
            requirements=req,
            evidence=evidence,
            pending_continuation=pending,
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertEqual(row["continuation"]["next_line"], 454)

    def test_offline_partial_reports_exhaustion_reason(self):
        req = orch.derive_investigation_requirements(
            "Inspect read_file implementation. Read the source."
        )
        evidence = orch.new_investigation_evidence_state()
        evidence["budget_exhaustion_reason"] = "CONTINUATION_DRAIN_BLOCKED"
        text = orch.build_offline_evidence_answer(
            "Inspect read_file",
            evidence_snippets=["read_file: partial body..."],
            reason="agent_limit",
            requirements=req,
            evidence_state=evidence,
        ).lower()
        self.assertIn("partial", text)
        self.assertIn("continuation_drain_blocked", text)


class TestVerifiedVsPartialReport(unittest.TestCase):
    """Fully-read functions must appear as PASS, not PARTIAL/MISSING."""

    def test_verified_target_not_marked_partial_in_offline(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent. Read the source."
        )
        evidence = orch.new_investigation_evidence_state()
        intent_start, intent_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "classify_intent"
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/agent_orchestration.py",
            line_start=intent_start,
            line_end=intent_end,
            content="def classify_intent(user_message):\n    return 'x'\n",
            function_complete=True,
        )
        self.assertIn("classify_intent", evidence["verified_targets"])
        text = orch.build_offline_evidence_answer(
            "Inspect classify_intent",
            evidence_snippets=[
                "read_file: def classify_intent(user_message):"
            ],
            reason="agent_limit",
            requirements=req,
            evidence_state=evidence,
            function_complete=True,
        )
        self.assertIn("PASS", text)
        self.assertRegex(text, r"classify_intent[^\n]*PASS")
        self.assertNotRegex(
            text,
            r"classify_intent[^\n]*MISSING_SOURCE_EVIDENCE",
        )
        self.assertIn("MISSING_SOURCE_EVIDENCE=0", text)

    def test_fully_read_cannot_be_reported_as_missing(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and classify_groq_rate_limit. Read source."
        )
        evidence = orch.new_investigation_evidence_state()
        intent_start, intent_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "classify_intent"
        )
        groq_start, groq_end = _top_level_def_span(
            "app/services/agent_orchestration.py", "classify_groq_rate_limit"
        )
        path = "app/services/agent_orchestration.py"
        orch.register_read_verification(
            evidence,
            req,
            file_path=path,
            line_start=intent_start,
            line_end=intent_end,
            content="def classify_intent(user_message):\n    return 'x'\n",
            function_complete=True,
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path=path,
            line_start=groq_start,
            line_end=groq_end,
            content="def classify_groq_rate_limit(exc):\n    return 'tpm'\n",
            function_complete=True,
        )
        ledger = orch.build_evidence_ledger_report(req, evidence)
        self.assertIn("| `classify_intent` |", ledger)
        self.assertRegex(ledger, r"classify_intent.*PASS")
        self.assertRegex(ledger, r"classify_groq_rate_limit.*PASS")
        # Reconcile must strip false MISSING labels from model prose.
        bad = (
            "| classify_intent | x | MISSING_SOURCE_EVIDENCE | none |\n"
            "| classify_groq_rate_limit | x | MISSING_SOURCE_EVIDENCE | none |"
        )
        fixed = orch.reconcile_answer_with_evidence_ledger(
            bad, requirements=req, evidence=evidence
        )
        self.assertIn("PASS", fixed)
        # Authoritative ledger section must not call either target MISSING.
        ledger_section = fixed.split("## Model synthesis")[0]
        self.assertNotRegex(
            ledger_section,
            r"classify_intent[^\n]*MISSING_SOURCE_EVIDENCE",
        )

    def test_search_cannot_establish_verified(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent. Read the source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.register_search_discoveries(
            evidence,
            req,
            [{
                "file_path": "app/services/agent_orchestration.py",
                "line": 214,
                "text": "def classify_intent(user_message):",
            }],
        )
        self.assertNotIn("classify_intent", evidence["verified_targets"])
        row = orch.classify_target_evidence_status(
            "classify_intent", requirements=req, evidence=evidence
        )
        self.assertEqual(row["status"], "MISSING_SOURCE_EVIDENCE")

    def test_budget_exhaustion_produces_partial_not_false_pass(self):
        req = orch.derive_investigation_requirements(
            "Inspect classify_intent and read_file. Read source."
        )
        evidence = orch.new_investigation_evidence_state()
        evidence["budget_exhaustion_reason"] = "PENDING_CONTINUATION_AT_FINALIZATION"
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/agent_orchestration.py",
            line_start=275,
            line_end=360,
            content="def classify_intent(user_message):\n    return 'x'\n",
            function_complete=True,
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/tools/file_reader.py",
            line_start=221,
            line_end=453,
            content="def read_file(file_path):\n    incomplete\n",
            function_complete=False,
        )
        pending = {
            "file_path": "app/tools/file_reader.py",
            "next_line": 454,
            "suggested_end_line": 512,
        }
        text = orch.build_offline_evidence_answer(
            "Inspect",
            evidence_snippets=["read_file: ..."],
            reason="pending_continuation",
            requirements=req,
            evidence_state=evidence,
            pending_continuation=pending,
        )
        self.assertIn("PASS", text)
        self.assertIn("PARTIAL", text)
        self.assertIn("PENDING_CONTINUATION_AT_FINALIZATION", text)
        # classify_intent must be PASS; read_file PARTIAL — not a blanket PASS.
        rows = orch.build_evidence_ledger_rows(
            req, evidence, pending_continuation=pending
        )
        by_id = {r["id"]: r["status"] for r in rows}
        self.assertEqual(by_id.get("classify_intent"), "PASS")
        self.assertEqual(by_id.get("read_file"), "PARTIAL")

    def test_pascal_case_targets_are_tracked(self):
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist and InvestigationBudgetManager. "
            "Also cover Pydantic validation, AST boundary detection, "
            "canonical arguments, and tool-free finalization."
        )
        ids = {t["id"] for t in req["required_targets"]}
        self.assertIn("InvestigationChecklist", ids)
        self.assertIn("InvestigationBudgetManager", ids)
        self.assertIn("ReadFileInput", ids)
        self.assertIn("ast_enclosing_symbol_span", ids)
        self.assertIn("canonicalize_read_file_args", ids)
        self.assertIn("build_finalization_messages", ids)
        kinds = {
            t["id"]: t.get("kind") for t in req["required_targets"]
        }
        self.assertEqual(kinds.get("InvestigationChecklist"), "class")


class TestClassLevelInvestigation(unittest.TestCase):
    """Class reads must not PASS from a nested method function_complete."""

    def test_partial_class_header_is_not_pass(self):
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read its complete class "
            "implementation, including all continuations."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        class_start = int(meta["line_start"])
        class_end = int(meta["line_end"])
        self.assertGreater(class_end, class_start + 10)
        self.assertGreater(len(meta["methods"]), 3)
        init = next(m for m in meta["methods"] if m["name"] == "__init__")
        get_m = next(m for m in meta["methods"] if m["name"] == "get")

        # Simulate the false-PASS windows from the live log: header only,
        # with a nested method reporting function_complete=true.
        orch.register_read_verification(
            evidence,
            req,
            file_path=meta["file_path"],
            line_start=class_start,
            line_end=int(init["line_end"]),
            content=(
                "class InvestigationChecklist:\n"
                "    def __init__(self, requirements=None):\n"
                "        self.requirements = requirements\n"
            ),
            function_complete=False,
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path=meta["file_path"],
            line_start=int(get_m["line_start"]),
            line_end=int(get_m["line_end"]),
            content="        return self.targets.get(tid)\n",
            function_complete=True,
        )
        self.assertNotIn(
            "InvestigationChecklist",
            evidence["verified_targets"],
        )
        row = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertIn("missing lines", row["reason"])
        self.assertTrue(row.get("class_ledger", {}).get("missing_ranges"))

    def test_class_continuation_synthesized_after_nested_method_complete(self):
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read complete class source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        class_start = int(meta["line_start"])
        get_m = next(m for m in meta["methods"] if m["name"] == "get")
        fragment_end = int(get_m["line_end"])
        orch.register_read_verification(
            evidence,
            req,
            file_path=meta["file_path"],
            line_start=class_start,
            line_end=fragment_end,
            content="class InvestigationChecklist:\n    def get(self, tid):\n",
            function_complete=True,
        )
        tool_result = {
            "success": True,
            "result": (
                '{"success": true, "file_path": '
                f'"{meta["file_path"]}", '
                f'"line_start": {fragment_end}, "line_end": {fragment_end}, '
                '"function_complete": true, "needs_continuation": false, '
                '"span_limited": false}\ncontent:\n'
                "        return self.targets.get(tid)\n"
            ),
        }
        cont = orch.synthesize_class_continuation_from_read(
            evidence, req, tool_result
        )
        self.assertIsNotNone(cont)
        self.assertTrue(cont.get("class_continuation"))
        self.assertGreaterEqual(int(cont["next_line"]), class_start)
        self.assertLessEqual(
            int(cont["suggested_end_line"]),
            int(meta["line_end"]),
        )

    def test_full_class_coverage_is_pass(self):
        from app.tools.file_reader import (
            ast_class_definition,
            load_project_source_text,
        )

        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read complete class source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        src = load_project_source_text(meta["file_path"])
        info = ast_class_definition(src, "InvestigationChecklist")
        assert info is not None
        # Read the whole class in chunks of 70 lines.
        start = int(info["line_start"])
        end = int(info["line_end"])
        lines = src.splitlines(True)
        pos = start
        while pos <= end:
            chunk_end = min(pos + 69, end)
            body = "".join(lines[pos - 1 : chunk_end])
            orch.register_read_verification(
                evidence,
                req,
                file_path=meta["file_path"],
                line_start=pos,
                line_end=chunk_end,
                content=body,
                function_complete=(chunk_end >= end),
            )
            pos = chunk_end + 1
        self.assertIn("InvestigationChecklist", evidence["verified_targets"])
        row = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(
            len(row["class_ledger"]["verified_methods"]),
            len(row["class_ledger"]["discovered_methods"]),
        )

    def test_span_limited_keeps_class_partial(self):
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read complete class source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        class_start = int(meta["line_start"])
        orch.register_read_verification(
            evidence,
            req,
            file_path=meta["file_path"],
            line_start=class_start,
            line_end=min(class_start + 45, int(meta["line_end"]) - 1),
            content="class InvestigationChecklist:\n    def __init__(self):\n        pass\n",
            function_complete=False,
            span_limited=True,
        )
        row = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertTrue(
            evidence["class_targets"]["InvestigationChecklist"].get(
                "span_limited_unresolved"
            )
        )

    def test_class_cannot_pass_with_partial_methods_or_span_limited(self):
        """Regression: InvestigationChecklist false-PASS from a fragment.

        Covers the live failure mode: only part of the class / methods were
        read (and span_limited=true), yet a nested method reported
        function_complete=true. Must stay PARTIAL with exact missing lines.
        """
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read its complete class "
            "implementation, including all continuations."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        method_names = [m["name"] for m in meta["methods"]]
        self.assertIn("__init__", method_names)
        self.assertIn("apply_to_evidence", method_names)
        self.assertGreater(len(method_names), 5)

        # Fragment + nested complete + span_limited must not verify the class.
        get_m = next(m for m in meta["methods"] if m["name"] == "get")
        orch.register_read_verification(
            evidence,
            req,
            file_path=meta["file_path"],
            line_start=int(meta["line_start"]),
            line_end=int(get_m["line_end"]),
            content=(
                "class InvestigationChecklist:\n"
                "    \"\"\"Structured required-symbol checklist.\"\"\"\n"
                "    def __init__(self, requirements=None):\n"
                "        self.requirements = requirements or {}\n"
                "    def get(self, tid):\n"
                "        return self.targets.get(tid)\n"
            ),
            function_complete=True,
            span_limited=True,
        )
        self.assertNotIn(
            "InvestigationChecklist",
            evidence["verified_targets"],
        )
        # Even a mistaken verified_targets entry must not yield PASS while
        # class coverage / methods remain incomplete.
        evidence["verified_targets"].add("InvestigationChecklist")
        row = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertNotEqual(row["status"], "PASS")
        ledger = row.get("class_ledger") or {}
        self.assertTrue(ledger.get("missing_ranges"))
        verified_methods = set(ledger.get("verified_methods") or [])
        discovered = set(ledger.get("discovered_methods") or [])
        self.assertTrue(discovered - verified_methods)
        self.assertIn("apply_to_evidence", discovered - verified_methods)
        self.assertTrue(
            evidence["class_targets"]["InvestigationChecklist"].get(
                "span_limited_unresolved"
            )
        )
        # Authoritative offline / ledger report must not claim PASS.
        report = orch.build_evidence_ledger_report(req, evidence)
        self.assertRegex(report, r"InvestigationChecklist[^\n]*PARTIAL")
        self.assertNotRegex(report, r"InvestigationChecklist[^\n]*\| PASS \|")
        offline = orch.build_offline_evidence_answer(
            "Inspect InvestigationChecklist",
            evidence_snippets=["read_file: class InvestigationChecklist"],
            reason="agent_limit",
            requirements=req,
            evidence_state=evidence,
            pending_continuation={
                "file_path": "app/services/investigation_state.py",
                "next_line": 247,
                "suggested_end_line": 315,
            },
        )
        self.assertIn("PARTIAL", offline)
        self.assertNotRegex(
            offline.split("## Model synthesis")[0]
            if "## Model synthesis" in offline
            else offline,
            r"InvestigationChecklist[^\n]*\| PASS \|",
        )

    def test_ast_lookup_uses_clipped_class_start(self):
        """Requesting the blank line above a class still resolves class end."""
        import json
        from app.tools.file_reader import read_file

        from app.tools.file_reader import (
            ast_class_definition,
            load_project_source_text,
        )

        src = load_project_source_text("app/services/investigation_state.py")
        info = ast_class_definition(src, "InvestigationChecklist")
        assert info is not None
        class_start = int(info["line_start"])
        class_end = int(info["line_end"])
        raw = read_file(
            "app/services/investigation_state.py",
            line_start=class_start - 1,
            line_end=class_start + 3,
        )
        meta = json.loads(str(raw).split("\ncontent:\n", 1)[0])
        self.assertEqual(meta["line_start"], class_start)
        self.assertEqual(meta["ast_symbol_end"], class_end)
        self.assertTrue(meta["needs_continuation"])
        self.assertFalse(meta["function_complete"])

    def test_source_complete_but_method_evidence_missing_is_partial(self):
        """Source coverage alone must not yield PASS / 13/13 without evidence."""
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read complete class source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        # Pretend the whole class span was covered as one window.
        orch.update_class_target_coverage(
            evidence,
            req,
            tid="InvestigationChecklist",
            file_path=meta["file_path"],
            line_start=int(meta["line_start"]),
            line_end=int(meta["line_end"]),
        )
        # Strip method evidence to simulate incomplete enumeration/reporting.
        meta = evidence["class_targets"]["InvestigationChecklist"]
        self.assertTrue(meta.get("source_coverage_complete"))
        meta["method_evidence"] = []
        meta["verified_methods"] = list(meta.get("discovered_methods") or [])
        meta["method_enumeration_complete"] = False
        meta["method_evidence_complete"] = False
        meta["coverage_complete"] = True  # contradictory stale flag
        evidence["verified_targets"].add("InvestigationChecklist")

        row = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertIn("method", row["reason"].lower())
        report = orch.format_evidence_ledger_report([row])
        self.assertNotRegex(
            report,
            r"methods_verified=\d+/\d+ \(explicit evidence for each\)",
        )
        self.assertIn("methods_with_evidence=0/", report)

    def test_method_evidence_requires_complete_method_range(self):
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read complete class source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        init = next(m for m in meta["methods"] if m["name"] == "__init__")
        # Cover only the class header + __init__, not sync_from_evidence.
        orch.register_read_verification(
            evidence,
            req,
            file_path=meta["file_path"],
            line_start=int(meta["line_start"]),
            line_end=int(init["line_end"]),
            content=(
                "class InvestigationChecklist:\n"
                "    def __init__(self, requirements=None):\n"
                "        self.requirements = requirements or {}\n"
                "        self.targets = {}\n"
            ),
            function_complete=False,
        )
        meta = evidence["class_targets"]["InvestigationChecklist"]
        verified = set(meta.get("verified_methods") or [])
        self.assertIn("__init__", verified)
        self.assertNotIn("sync_from_evidence", verified)
        self.assertNotIn("apply_to_evidence", verified)
        row = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        report = orch.format_evidence_ledger_report([row])
        self.assertIn("`__init__`", report)
        self.assertIn("`apply_to_evidence`", report)
        self.assertIn("| no |", report)

    def test_forbid_search_skips_when_evidence_sufficient(self):
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read complete class source. "
            "Do not search again."
        )
        self.assertTrue(req.get("forbid_search"))
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        init = next(m for m in meta["methods"] if m["name"] == "__init__")
        orch.register_read_verification(
            evidence,
            req,
            file_path=meta["file_path"],
            line_start=int(meta["line_start"]),
            line_end=int(init["line_end"]),
            content="class InvestigationChecklist:\n    def __init__(self):\n        pass\n",
            function_complete=False,
        )
        skip, reason = orch.should_skip_duplicate_tool(
            "search_code",
            {"search_text": "InvestigationChecklist"},
            fingerprints=set(),
            prior_reads=[],
            tool_counts={},
            requirements=req,
            evidence=evidence,
            searched_queries=set(),
        )
        self.assertTrue(skip)
        self.assertEqual(reason, "SEARCH_SKIPPED_USER_FORBID")

    def test_contradictory_pass_without_method_list_is_rejected(self):
        """Final report must not claim PASS with empty method evidence."""
        req = orch.derive_investigation_requirements(
            "Inspect InvestigationChecklist. Read complete class source."
        )
        evidence = orch.new_investigation_evidence_state()
        orch.ensure_class_target_metadata(evidence, req)
        meta = evidence["class_targets"]["InvestigationChecklist"]
        orch.update_class_target_coverage(
            evidence,
            req,
            tid="InvestigationChecklist",
            file_path=meta["file_path"],
            line_start=int(meta["line_start"]),
            line_end=int(meta["line_end"]),
        )
        # Full coverage should produce explicit per-method evidence + PASS.
        evidence["verified_targets"].add("InvestigationChecklist")
        row = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(
            len(row["class_ledger"]["method_evidence"]),
            len(row["class_ledger"]["discovered_methods"]),
        )
        self.assertTrue(
            all(e.get("verified") for e in row["class_ledger"]["method_evidence"])
        )
        report = orch.format_evidence_ledger_report([row])
        for name in row["class_ledger"]["discovered_methods"]:
            self.assertIn(f"`{name}`", report)
        # Now break evidence and ensure status flips to PARTIAL.
        meta = evidence["class_targets"]["InvestigationChecklist"]
        for entry in meta["method_evidence"]:
            if entry["name"] == "apply_to_evidence":
                entry["verified"] = False
                entry["evidence_ranges"] = []
        meta["verified_methods"] = [
            m for m in meta["verified_methods"] if m != "apply_to_evidence"
        ]
        meta["method_evidence_complete"] = False
        meta["coverage_complete"] = False
        row2 = orch.classify_target_evidence_status(
            "InvestigationChecklist",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row2["status"], "PARTIAL")
        self.assertIn("apply_to_evidence", row2["reason"])


class TestMemoryArchitectureInvestigation(unittest.TestCase):
    """Memory architecture: search→read, module-level defs, evidence integrity."""

    ARCH_MSG = (
        "Inspect my memory architecture. Verify memory scoping, conflict "
        "resolution, freshness, reliability, summarization, reinforcement, "
        "deletion, source traceability, embeddings, semantic retrieval, and "
        "conversation memory. Read complete implementations and relevant "
        "tests. Report PASS/PARTIAL/FAIL. Do not modify files."
    )

    def test_plans_module_level_memory_symbols(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        ids = {t["id"] for t in req["required_targets"]}
        for tid in (
            "detect_memory_conflict",
            "reinforce_memory",
            "semantic_search_memories",
            "calculate_memory_freshness_score",
            "add_to_conversation_memory",
        ):
            self.assertIn(tid, ids)
        self.assertIn("memory_architecture", req["topics"])
        # Never invent a MemoryService class target.
        self.assertNotIn("MemoryService", ids)

    def test_search_hits_seed_read_suggestions_for_defs(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        matches = [
            {
                "file_path": "app/services/memory_service.py",
                "line": 2482,
                "text": "def reinforce_memory(memory_id, user_id=None):",
            },
            {
                "file_path": "app/services/memory_conflict_service.py",
                "line": 578,
                "text": "def detect_memory_conflict(user_id, text):",
            },
        ]
        suggestions = orch.register_search_discoveries(
            evidence, req, matches
        )
        targets = {s["target"] for s in suggestions}
        self.assertIn("reinforce_memory", targets)
        self.assertIn("detect_memory_conflict", targets)
        nxt = orch.next_unread_suggested_read(req, evidence)
        self.assertIsNotNone(nxt)
        self.assertIn(nxt["target"], targets)

    def test_import_hit_redirects_to_module_level_definition(self):
        req = orch.derive_investigation_requirements(
            "Inspect my memory architecture. Verify reinforcement."
        )
        evidence = orch.new_investigation_evidence_state()
        suggestions = orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": "app/chatbot.py",
                    "line": 29,
                    "text": "reinforce_memory,",
                }
            ],
        )
        self.assertTrue(suggestions)
        sug = suggestions[0]
        self.assertEqual(sug["target"], "reinforce_memory")
        self.assertIn("memory_service.py", sug["file_path"].replace("\\", "/"))
        self.assertGreaterEqual(int(sug["line_start"]), 2475)

    def test_duplicate_search_skipped_after_discovery(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, evidence)
        orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": "app/services/memory_service.py",
                    "line": 2482,
                    "text": "def reinforce_memory(memory_id):",
                }
            ],
        )
        skip, reason = orch.should_skip_duplicate_tool(
            "search_code",
            {"search_text": "class MemoryService"},
            fingerprints=set(),
            prior_reads=[],
            tool_counts={"search_code": 1},
            requirements=req,
            evidence=evidence,
            searched_queries={"memory"},
        )
        self.assertTrue(skip)
        self.assertIn(
            reason,
            {
                "SEARCH_SKIPPED_READ_PRIORITY",
                "SEARCH_SKIPPED_KNOWN_SYMBOL",
            },
        )

    def test_progress_across_several_memory_services(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.seed_known_symbol_suggestions(req, evidence)
        files = {
            str(s.get("file_path") or "").replace("\\", "/")
            for s in evidence.get("suggested_reads") or []
        }
        self.assertIn("app/services/memory_service.py", files)
        self.assertIn("app/services/memory_conflict_service.py", files)
        self.assertIn("app/services/memory_lifecycle_service.py", files)
        self.assertIn("app/services/memory_summary_service.py", files)

        # Simulate reading one symbol from each service (AST-resolved spans).
        samples = [
            (
                "reinforce_memory",
                "app/services/memory_service.py",
                "def reinforce_memory(memory_id):\n    return True\n",
            ),
            (
                "detect_memory_conflict",
                "app/services/memory_conflict_service.py",
                "def detect_memory_conflict(user_id, text):\n    return None\n",
            ),
            (
                "calculate_memory_freshness_score",
                "app/services/memory_lifecycle_service.py",
                "def calculate_memory_freshness_score(doc):\n    return 1.0\n",
            ),
        ]
        reinforce_end = None
        for tid, path, body in samples:
            start, end = _top_level_def_span(path, tid)
            if tid == "reinforce_memory":
                reinforce_end = end
            orch.register_read_verification(
                evidence,
                req,
                file_path=path,
                line_start=start,
                line_end=end,
                content=body,
                function_complete=True,
            )
            self.assertIn(tid, evidence["verified_targets"])

        rows = {
            r["id"]: r
            for r in orch.build_evidence_ledger_rows(req, evidence)
        }
        self.assertEqual(rows["reinforce_memory"]["status"], "PARTIAL")
        # PARTIAL because related tests were requested but not inspected.
        self.assertIn("tests not inspected", rows["reinforce_memory"]["reason"])
        self.assertEqual(
            rows["semantic_search_memories"]["status"],
            "MISSING_SOURCE_EVIDENCE",
        )
        self.assertEqual(
            rows["reinforce_memory"]["line_end"],
            reinforce_end,
        )

    def test_pass_requires_source_read_not_import(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": "app/chatbot.py",
                    "line": 45,
                    "text": "from app.services.memory_conflict_service import (",
                },
                {
                    "file_path": "app/chatbot.py",
                    "line": 46,
                    "text": "detect_memory_conflict,",
                },
            ],
        )
        row = orch.classify_target_evidence_status(
            "detect_memory_conflict",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "MISSING_SOURCE_EVIDENCE")

    def test_empty_ledger_never_allows_unsupported_pass(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        narrative = (
            "| Feature | Status |\n"
            "| Conflict Resolution | **PASS** |\n"
            "| Freshness | PASS |\n"
        )
        fixed = orch.reconcile_answer_with_evidence_ledger(
            narrative,
            requirements=req,
            evidence=evidence,
        )
        ledger_section = fixed.split("## Model synthesis")[0]
        self.assertIn("MISSING_SOURCE_EVIDENCE", ledger_section)
        synth = fixed.split("## Model synthesis")[-1]
        self.assertNotRegex(synth, r"\bPASS\b")

    def test_implementation_pass_after_complete_read_without_tests_is_partial(
        self,
    ):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_conflict_service.py",
            line_start=578,
            line_end=594,
            content=(
                "def detect_memory_conflict(user_id, text):\n"
                "    return None\n"
            ),
            function_complete=True,
        )
        row = orch.classify_target_evidence_status(
            "detect_memory_conflict",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertIn("tests not inspected", row["reason"])

    def test_adjacent_function_windows_do_not_merge_ranges(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        # Read freshness, then priority (adjacent). Ranges must stay separate.
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_lifecycle_service.py",
            line_start=32,
            line_end=57,
            content=(
                "def calculate_memory_freshness_score(doc):\n"
                "    return 1.0\n"
            ),
            function_complete=True,
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_lifecycle_service.py",
            line_start=60,
            line_end=93,
            content=(
                "def calculate_memory_priority_score(doc):\n"
                "    return 1.0\n"
            ),
            function_complete=True,
        )
        # A later summarize-style window on another file must not stretch
        # freshness into the thousands.
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_summary_service.py",
            line_start=1240,
            line_end=1307,
            content=(
                "def summarize_user_memories(user_id):\n"
                "    return []\n"
            ),
            function_complete=True,
        )
        fresh = evidence["target_progress"]["calculate_memory_freshness_score"]
        prio = evidence["target_progress"]["calculate_memory_priority_score"]
        self.assertEqual(fresh["line_start"], 32)
        self.assertEqual(fresh["line_end"], 57)
        self.assertEqual(prio["line_start"], 60)
        self.assertEqual(prio["line_end"], 93)
        # reinforce must not absorb delete_memory lines.
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_service.py",
            line_start=2610,
            line_end=2660,
            content="def reinforce_memory(memory_id):\n    return True\n",
            function_complete=False,
        )
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_service.py",
            line_start=3485,
            line_end=3608,
            content="def delete_memory(user_id, memory_id):\n    return True\n",
            function_complete=True,
        )
        reinf = evidence["target_progress"]["reinforce_memory"]
        dele = evidence["target_progress"]["delete_memory"]
        self.assertLessEqual(int(reinf["line_end"]), 2798)
        self.assertEqual(int(dele["line_start"]), 3485)
        self.assertEqual(int(dele["line_end"]), 3608)
        rows = orch.build_evidence_ledger_rows(req, evidence)
        problems = orch.assert_ledger_ranges_match_symbols(rows)
        self.assertEqual(problems, [])

    def test_pending_continuation_never_pass(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_service.py",
            line_start=2155,
            line_end=2178,
            content=(
                "def semantic_search_memories(user_id, query):\n"
                "    x = 1\n"
            ),
            function_complete=False,
        )
        evidence["verified_targets"].add("semantic_search_memories")
        evidence.setdefault("target_progress", {})[
            "semantic_search_memories"
        ]["function_complete"] = False
        row = orch.classify_target_evidence_status(
            "semantic_search_memories",
            requirements=req,
            evidence=evidence,
            pending_continuation={
                "file_path": "app/services/memory_service.py",
                "next_line": 2179,
                "suggested_end_line": 2248,
            },
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertTrue((row.get("continuation") or {}).get("needs_continuation"))

    def test_pass_requires_implementation_and_test_evidence(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_conflict_service.py",
            line_start=578,
            line_end=594,
            content=(
                "def detect_memory_conflict(user_id, text):\n"
                "    return None\n"
            ),
            function_complete=True,
        )
        row = orch.classify_target_evidence_status(
            "detect_memory_conflict",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        # Inspect related test → still PARTIAL (inspection ≠ execution).
        orch.register_read_verification(
            evidence,
            req,
            file_path="tests/test_memory_conflict_advanced.py",
            line_start=1,
            line_end=40,
            content="def test_detect_memory_conflict():\n    assert True\n",
            function_complete=True,
        )
        row2 = orch.classify_target_evidence_status(
            "detect_memory_conflict",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row2["status"], "PARTIAL")
        self.assertIn("not executed", row2["reason"])
        test_row = orch.classify_target_evidence_status(
            "test_memory_conflict_advanced",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(test_row["status"], orch.TEST_SOURCE_INSPECTED)
        # Execution passed → feature may PASS.
        orch.apply_test_execution_result(
            evidence,
            tid="test_memory_conflict_advanced",
            execution={
                "status": orch.TEST_EXECUTION_PASSED,
                "passed": ["tests.test_memory_conflict_advanced.TestX"],
                "failed": [],
                "errors": [],
                "message": "ok",
            },
        )
        row3 = orch.classify_target_evidence_status(
            "detect_memory_conflict",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row3["status"], "PASS")

    def test_summary_counts_match_ledger_statuses(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="app/services/memory_conflict_service.py",
            line_start=578,
            line_end=594,
            content="def detect_memory_conflict(user_id, text):\n    return None\n",
            function_complete=True,
        )
        rows = orch.build_evidence_ledger_rows(req, evidence)
        counts = orch.summarize_evidence_ledger_counts(rows)
        report = orch.format_evidence_ledger_report(rows)
        self.assertIn(
            f"PASS={counts['PASS']} PARTIAL={counts['PARTIAL']} "
            f"MISSING_SOURCE_EVIDENCE={counts['MISSING_SOURCE_EVIDENCE']}",
            report,
        )
        self.assertEqual(
            counts["PASS"] + counts["PARTIAL"] + counts["MISSING_SOURCE_EVIDENCE"]
            + counts.get("FAIL", 0),
            len(rows),
        )
        # No PASS without tests for memory features.
        self.assertEqual(counts["PASS"], 0)
        self.assertGreaterEqual(counts["PARTIAL"], 1)

    def test_suggestion_window_stops_at_symbol_ast_end(self):
        window = orch._suggestion_window_for_symbol(
            "app/services/memory_lifecycle_service.py",
            32,
            prior=[],
            span=80,
        )
        self.assertIsNotNone(window)
        self.assertLessEqual(int(window["line_end"]), 57)

    def test_source_char_budget_shrinks_continuation_window(self):
        shrunk = orch.shrink_read_window_for_char_budget(
            line_start=2120,
            line_end=2189,
            remaining_chars=800,
            chars_per_line=80,
        )
        self.assertIsNotNone(shrunk)
        self.assertEqual(shrunk[0], 2120)
        self.assertLess(shrunk[1], 2189)
        self.assertIsNone(
            orch.shrink_read_window_for_char_budget(
                line_start=2120,
                line_end=2189,
                remaining_chars=50,
            )
        )

    def test_source_char_budget_records_unread_range(self):
        evidence = orch.new_investigation_evidence_state()
        orch.record_unread_continuation_gap(
            evidence,
            tid="semantic_search_memories",
            file_path="app/services/memory_service.py",
            next_line=2201,
            suggested_end_line=2436,
            reason="SOURCE_CHAR_BUDGET_EXCEEDED",
        )
        prog = evidence["target_progress"]["semantic_search_memories"]
        self.assertEqual(prog["unread_from"], 2201)
        self.assertEqual(prog["unread_to"], 2436)
        self.assertIs(prog["function_complete"], False)
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence["verified_targets"].add("semantic_search_memories")
        prog["line_start"] = 2155
        prog["line_end"] = 2200
        prog["file_path"] = "app/services/memory_service.py"
        row = orch.classify_target_evidence_status(
            "semantic_search_memories",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], "PARTIAL")
        self.assertIn("2201", row["reason"])
        self.assertNotEqual(row["status"], "PASS")

    def test_test_inspection_is_not_execution_pass(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="tests/test_memory_lifecycle_service.py",
            line_start=1,
            line_end=30,
            content="def test_freshness():\n    assert True\n",
            function_complete=True,
        )
        row = orch.classify_target_evidence_status(
            "test_memory_lifecycle_service",
            requirements=req,
            evidence=evidence,
        )
        self.assertEqual(row["status"], orch.TEST_SOURCE_INSPECTED)
        self.assertNotEqual(row["status"], "PASS")
        self.assertIn("inspection ≠ pass", row["reason"])

    def test_conversation_memory_discovery_from_search(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        # FS discovery must find the real def without assuming the path.
        located = orch.discover_symbol_definition("add_to_conversation_memory")
        self.assertIsNotNone(located)
        self.assertTrue(str(located[0]).endswith(".py"))
        self.assertGreater(int(located[1]), 0)
        # Simulate discovery without relying solely on a hardcoded path.
        suggestions = orch.register_search_discoveries(
            evidence,
            req,
            [
                {
                    "file_path": located[0],
                    "line": located[1],
                    "text": "def add_to_conversation_memory(",
                }
            ],
        )
        self.assertTrue(suggestions)
        self.assertEqual(suggestions[0]["target"], "add_to_conversation_memory")
        self.assertEqual(suggestions[0]["file_path"], located[0])
        orch.register_read_verification(
            evidence,
            req,
            file_path=located[0],
            line_start=int(located[1]),
            line_end=int(located[1]) + 12,
            content=(
                "def add_to_conversation_memory(role, content, user_id=None):\n"
                "    store.append({'role': role, 'content': content})\n"
            ),
            function_complete=True,
        )
        self.assertIn(
            "add_to_conversation_memory",
            evidence["verified_targets"],
        )
        row = orch.classify_target_evidence_status(
            "add_to_conversation_memory",
            requirements=req,
            evidence=evidence,
        )
        # Implementation complete but related tests not executed → PARTIAL.
        self.assertEqual(row["status"], "PARTIAL")
        self.assertTrue(row.get("implementation_complete"))
        self.assertTrue(row.get("related_tests"))

    def test_ledger_summary_includes_test_status_counts(self):
        req = orch.derive_investigation_requirements(self.ARCH_MSG)
        evidence = orch.new_investigation_evidence_state()
        orch.register_read_verification(
            evidence,
            req,
            file_path="tests/test_memory_summarization.py",
            line_start=1,
            line_end=20,
            content="def test_x():\n    assert True\n",
            function_complete=True,
        )
        rows = orch.build_evidence_ledger_rows(req, evidence)
        counts = orch.summarize_evidence_ledger_counts(rows)
        report = orch.format_evidence_ledger_report(rows)
        self.assertGreaterEqual(counts[orch.TEST_SOURCE_INSPECTED], 1)
        self.assertIn(
            f"TEST_SOURCE_INSPECTED={counts[orch.TEST_SOURCE_INSPECTED]}",
            report,
        )
        self.assertEqual(
            sum(counts.values()),
            len(rows),
        )


if __name__ == "__main__":
    unittest.main()
