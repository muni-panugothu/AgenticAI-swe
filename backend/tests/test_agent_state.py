"""Topic 33 — AgentState unit tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services.agent_orchestration import (
    KNOWN_SYMBOL_LOCATIONS,
    derive_investigation_requirements,
    new_investigation_evidence_state,
)
from app.services.agent_state import AgentState
from app.services.investigation_state import (
    InvestigationBudgetManager,
    InvestigationChecklist,
    build_budget_from_requirements,
)


class TestAgentStateConstruction(unittest.TestCase):
    def test_construction_with_defaults(self):
        state = AgentState()
        self.assertEqual(state.current_step, 0)
        self.assertEqual(state.max_steps, 0)
        self.assertEqual(state.llm_iterations, 0)
        self.assertEqual(state.total_tool_executions, 0)
        self.assertEqual(state.successful_tool_count, 0)
        self.assertEqual(state.successful_searches, 0)
        self.assertEqual(state.successful_reads, 0)
        self.assertEqual(state.blocked_in_a_row, 0)
        self.assertIsNone(state.pending_continuation)
        self.assertFalse(state.function_complete)
        self.assertFalse(state.calculator_used)
        self.assertIsNone(state.investigation_checklist)
        self.assertIsNone(state.budget_manager)

    def test_construction_with_investigation_refs(self):
        req = derive_investigation_requirements(
            "Inspect classify_intent. Read the complete implementation."
        )
        evidence = new_investigation_evidence_state()
        checklist = InvestigationChecklist(req)
        budget = build_budget_from_requirements(
            req, known_locations=KNOWN_SYMBOL_LOCATIONS
        )
        state = AgentState(
            max_steps=8,
            read_budget_total=14,
            tool_budget_total=14,
            investigation_requirements=req,
            investigation_evidence=evidence,
            investigation_checklist=checklist,
            budget_manager=budget,
        )
        self.assertIs(state.investigation_requirements, req)
        self.assertIs(state.investigation_evidence, evidence)
        self.assertIs(state.investigation_checklist, checklist)
        self.assertIs(state.budget_manager, budget)
        self.assertEqual(state.max_steps, 8)
        self.assertEqual(state.read_budget_total, 14)
        self.assertEqual(state.tool_budget_total, 14)


class TestAgentStateDefaultsIndependent(unittest.TestCase):
    def test_mutable_defaults_are_independent(self):
        a = AgentState()
        b = AgentState()

        a.tool_execution_counts["read_file"] = 1
        a.executed_tool_fingerprints.add("fp-a")
        a.failed_tool_fingerprints.add("fail-a")
        a.committed_mutations.add("mut-a")
        a.evidence_snippets.append("snippet-a")
        a.searched_queries.add("query-a")
        a.files_already_read.add("file-a")
        a.terminal_file_paths.add("term-a")
        a.successful_tool_results["k"] = "v"
        a.file_result_by_path["p"] = "body"
        a.read_counts_by_file["p"] = 2
        a.prior_read_args.append({"file_path": "p"})
        a.investigation_requirements["needs_calculator"] = True
        a.investigation_evidence["budget_exhaustion_reason"] = "x"
        a.tool_failure_counts["search_code"] = 3

        self.assertEqual(b.tool_execution_counts, {})
        self.assertEqual(b.executed_tool_fingerprints, set())
        self.assertEqual(b.failed_tool_fingerprints, set())
        self.assertEqual(b.committed_mutations, set())
        self.assertEqual(b.evidence_snippets, [])
        self.assertEqual(b.searched_queries, set())
        self.assertEqual(b.files_already_read, set())
        self.assertEqual(b.terminal_file_paths, set())
        self.assertEqual(b.successful_tool_results, {})
        self.assertEqual(b.file_result_by_path, {})
        self.assertEqual(b.read_counts_by_file, {})
        self.assertEqual(b.prior_read_args, [])
        self.assertEqual(b.investigation_requirements, {})
        self.assertEqual(b.investigation_evidence, {})
        self.assertEqual(b.tool_failure_counts, {})

    def test_budget_manager_default_factory_not_shared(self):
        """Checklist/budget are explicit refs; bare instances stay None."""
        a = AgentState()
        b = AgentState()
        self.assertIsNone(a.budget_manager)
        self.assertIsNone(b.budget_manager)
        a.budget_manager = InvestigationBudgetManager()
        self.assertIsNotNone(a.budget_manager)
        self.assertIsNone(b.budget_manager)


class TestAgentStateUpdates(unittest.TestCase):
    def test_scalar_and_counter_updates(self):
        state = AgentState(max_steps=5)
        state.current_step = 2
        state.llm_iterations = 3
        state.total_tool_executions = 4
        state.successful_searches = 1
        state.successful_reads = 2
        state.successful_tool_count = 3
        state.blocked_in_a_row = 1
        state.pending_continuation = {
            "file_path": "app/chatbot.py",
            "next_line": 10,
        }
        state.function_complete = False
        state.tool_execution_counts["read_file"] = 2
        state.executed_tool_fingerprints.add("read_file:{}")

        self.assertEqual(state.current_step, 2)
        self.assertEqual(state.max_steps, 5)
        self.assertEqual(state.llm_iterations, 3)
        self.assertEqual(state.total_tool_executions, 4)
        self.assertEqual(state.successful_searches, 1)
        self.assertEqual(state.successful_reads, 2)
        self.assertEqual(state.successful_tool_count, 3)
        self.assertEqual(state.blocked_in_a_row, 1)
        self.assertEqual(
            state.pending_continuation["file_path"], "app/chatbot.py"
        )
        self.assertFalse(state.function_complete)
        self.assertEqual(state.tool_execution_counts["read_file"], 2)
        self.assertIn("read_file:{}", state.executed_tool_fingerprints)


class TestAgentStateChatIntegration(unittest.TestCase):
    """Verify chat_with_ai constructs AgentState with live investigation refs."""

    @patch("app.chatbot.client")
    @patch("app.chatbot.save_task_context_to_redis")
    @patch("app.chatbot.restore_task_context_from_redis")
    @patch("app.chatbot.add_to_conversation_memory")
    @patch("app.chatbot.get_conversation_memory", return_value=[])
    @patch("app.chatbot.load_long_term_memory")
    def test_agent_state_initialized_with_existing_refs(
        self,
        mock_ltm,
        _mock_conv,
        _mock_add,
        _mock_restore,
        _mock_save,
        mock_client,
    ):
        mock_ltm.return_value = {
            "memories": [],
            "conflicts": [],
            "summaries": [],
        }

        def _fake_create(**kwargs):
            msg = MagicMock()
            msg.content = "Hello there."
            msg.tool_calls = None
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = MagicMock(total_tokens=10)
            return resp

        mock_client.chat.completions.create.side_effect = _fake_create

        from app.services import agent_state as agent_state_mod
        from app import chatbot

        original_cls = agent_state_mod.AgentState
        instances: list[AgentState] = []

        class TrackingAgentState(original_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instances.append(self)

        # chat_with_ai imports AgentState inside the function body.
        with patch(
            "app.services.agent_state.AgentState",
            TrackingAgentState,
        ):
            # Use a standalone technical question so B2 clarification does not
            # short-circuit before AgentState is constructed.
            answer = chatbot.chat_with_ai(
                "What is FastAPI?",
                user_id="test-user-topic33",
            )

        self.assertIsInstance(answer, str)
        self.assertTrue(instances, "AgentState was not constructed")
        state = instances[0]
        self.assertIsInstance(state.investigation_requirements, dict)
        self.assertIsInstance(state.investigation_evidence, dict)
        self.assertIsInstance(
            state.investigation_checklist, InvestigationChecklist
        )
        self.assertIsInstance(state.budget_manager, InvestigationBudgetManager)
        self.assertGreaterEqual(state.max_steps, 1)
        self.assertGreaterEqual(state.read_budget_total, 1)
        self.assertGreaterEqual(state.tool_budget_total, 1)
        # Loop should have mirrored at least one LLM iteration.
        self.assertGreaterEqual(state.llm_iterations, 1)
        self.assertIsInstance(state.executed_tool_fingerprints, set)
        self.assertIsInstance(state.evidence_snippets, list)
        self.assertIsInstance(state.tool_execution_counts, dict)


if __name__ == "__main__":
    unittest.main()
