"""Topic 34 — AgentContext unit and integration tests."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.services.agent_context import (
    MEMORY_UNTRUSTED_ACK,
    AgentContext,
    build_agent_context,
    build_initial_messages,
)
from app.services.agent_orchestration import GROQ_TPM_BUDGET


class TestAgentContextConstruction(unittest.TestCase):
    def test_defaults(self):
        ctx = AgentContext()
        self.assertEqual(ctx.user_id, "")
        self.assertEqual(ctx.messages, [])
        self.assertEqual(ctx.tool_definitions, [])
        self.assertFalse(ctx.tools_enabled)
        self.assertIsNone(ctx.rag_context)

    def test_mutable_messages_independent(self):
        a = AgentContext()
        b = AgentContext()
        a.messages.append({"role": "user", "content": "hi"})
        a.tool_definitions.append({"name": "calculator"})
        self.assertEqual(b.messages, [])
        self.assertEqual(b.tool_definitions, [])


class TestBuildInitialMessages(unittest.TestCase):
    def test_assembly_order_system_memory_history(self):
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
        ]
        msgs = build_initial_messages(
            system_prompt="SYS",
            memory_context="USER_MEMORY_REFERENCE\nfact",
            rag_context=None,
            conversation_history=history,
            max_history_messages=10,
        )
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "SYS")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("USER_MEMORY_REFERENCE", msgs[1]["content"])
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[2]["content"], MEMORY_UNTRUSTED_ACK)
        # Compacted history follows memory pair.
        roles_after = [m["role"] for m in msgs[3:]]
        self.assertIn("user", roles_after)
        self.assertNotEqual(msgs[1]["role"], "system")

    def test_rag_evidence_before_memory(self):
        def fake_rag(text: str):
            return [{"role": "user", "content": f"RAG:{text}"}]

        msgs = build_initial_messages(
            system_prompt="SYS",
            memory_context="MEM",
            rag_context="doc-evidence",
            conversation_history=[],
            max_history_messages=10,
            rag_evidence_messages_fn=fake_rag,
        )
        self.assertEqual(msgs[0]["content"], "SYS")
        self.assertEqual(msgs[1]["content"], "RAG:doc-evidence")
        self.assertEqual(msgs[2]["content"], "MEM")
        self.assertEqual(msgs[3]["content"], MEMORY_UNTRUSTED_ACK)

    def test_memory_never_injected_as_system(self):
        msgs = build_initial_messages(
            system_prompt="SYS",
            memory_context="secret-memory",
            conversation_history=[],
        )
        system_contents = [
            m["content"] for m in msgs if m.get("role") == "system"
        ]
        self.assertEqual(system_contents, ["SYS"])
        self.assertTrue(
            any(
                m.get("role") == "user" and "secret-memory" in m["content"]
                for m in msgs
            )
        )


class TestBuildAgentContext(unittest.TestCase):
    def test_live_messages_reference(self):
        ctx = build_agent_context(
            user_id="u1",
            request_id="r1",
            project_id="p1",
            tenant_id="t1",
            user_message="Hello",
            intent="general_conversation",
            tools_enabled=False,
            system_prompt="SYS",
            memory_context="",
            rag_context=None,
            tool_definitions=[],
            conversation_history=[],
            max_history_messages=10,
        )
        self.assertIs(ctx.messages, ctx.messages)
        alias = ctx.messages
        ctx.append_system("nudge")
        self.assertIs(alias, ctx.messages)
        self.assertEqual(alias[-1]["content"], "nudge")
        self.assertEqual(ctx.user_id, "u1")
        self.assertEqual(ctx.project_id, "p1")
        self.assertEqual(ctx.tenant_id, "t1")

    def test_prune_keeps_same_list_object(self):
        huge = "x" * 50000
        ctx = AgentContext(
            tools_enabled=False,
            system_prompt="SYS",
            messages=[
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": huge},
                {"role": "assistant", "content": huge},
            ],
        )
        before = ctx.messages
        returned = ctx.prune_for_budget(tool_defs=None, budget=GROQ_TPM_BUDGET)
        self.assertIs(returned, before)
        self.assertIs(ctx.messages, before)


class TestAgentContextChatIntegration(unittest.TestCase):
    @patch("app.chatbot.client")
    @patch("app.chatbot.save_task_context_to_redis")
    @patch("app.chatbot.restore_task_context_from_redis")
    @patch("app.chatbot.add_to_conversation_memory")
    @patch("app.chatbot.get_conversation_memory", return_value=[])
    @patch("app.chatbot.load_long_term_memory")
    def test_chat_builds_one_agent_context(
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
            msg.content = "Hi!"
            msg.tool_calls = None
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            resp.usage = MagicMock(total_tokens=5)
            return resp

        mock_client.chat.completions.create.side_effect = _fake_create

        from app.services import agent_context as ctx_mod
        from app import chatbot

        original = ctx_mod.build_agent_context
        captured: list = []

        def tracking(**kwargs):
            ctx = original(**kwargs)
            captured.append(ctx)
            return ctx

        with patch.object(ctx_mod, "build_agent_context", side_effect=tracking):
            # chat_with_ai imports build_agent_context inside the function.
            with patch(
                "app.services.agent_context.build_agent_context",
                side_effect=tracking,
            ):
                # Standalone technical ask reaches the loop (avoids B2 clarify).
                answer = chatbot.chat_with_ai(
                    "What is FastAPI?",
                    user_id="test-user-topic34",
                )

        self.assertIsInstance(answer, str)
        self.assertEqual(len(captured), 1)
        ctx = captured[0]
        self.assertIsInstance(ctx, AgentContext)
        self.assertEqual(ctx.user_id, "test-user-topic34")
        self.assertTrue(ctx.messages)
        self.assertEqual(ctx.messages[0]["role"], "system")
        # Technical path should not inject memory into the system role.
        self.assertFalse(
            any(
                m.get("role") == "system"
                and "USER_MEMORY_REFERENCE" in str(m.get("content") or "")
                for m in ctx.messages
            )
        )


if __name__ == "__main__":
    unittest.main()
