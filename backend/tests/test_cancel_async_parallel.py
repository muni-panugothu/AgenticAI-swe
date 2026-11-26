"""Topics 42c / 43 / 44 — cancellation, async tools, bounded parallel tools."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("JWT_SECRET", "e2e-test-jwt-secret-key-32chars-min!!")

from fastapi.testclient import TestClient

from app.services.request_cancellation import (
    CancellationToken,
    RequestCancelled,
)
from app.services.tool_execution import (
    PreparedToolCall,
    can_parallelize,
    execute_tool_async,
    execute_tool_sync,
    execute_tools_bounded,
    get_tool_meta,
    is_parallel_safe,
)


def _parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        eid = event_name = data_raw = None
        for line in block.split("\n"):
            if line.startswith("id:"):
                eid = line[3:].strip()
            elif line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_raw = line[5:].strip()
        if data_raw is None:
            continue
        events.append(
            {"id": eid, "event": event_name, "data": json.loads(data_raw)}
        )
    return events


class CancellationTokenTests(unittest.TestCase):
    def test_request_isolation(self):
        a = CancellationToken(request_id="r1", user_id="u1")
        b = CancellationToken(request_id="r2", user_id="u2")
        a.cancel("client_disconnect")
        self.assertTrue(a.is_cancelled())
        self.assertFalse(b.is_cancelled())
        with self.assertRaises(RequestCancelled):
            a.check()
        b.check()  # must not raise


class ToolMetaTests(unittest.TestCase):
    def test_read_only_parallel_safe(self):
        self.assertTrue(is_parallel_safe("read_file"))
        self.assertTrue(is_parallel_safe("search_code"))
        self.assertTrue(is_parallel_safe("calculator"))

    def test_memory_mutations_not_parallel(self):
        for name in (
            "save_memory",
            "forget_memory",
            "reinforce_memory",
            "confirm_memory_update",
        ):
            self.assertFalse(is_parallel_safe(name))
            self.assertEqual(get_tool_meta(name).max_concurrency, 1)

    def test_unknown_defaults_sequential(self):
        self.assertFalse(is_parallel_safe("totally_unknown_tool"))


class AsyncToolBridgeTests(unittest.TestCase):
    def test_sync_tool_via_async_bridge(self):
        def fake_execute(name, args, **kwargs):
            return {"success": True, "result": f"{name}:{args.get('x')}"}

        async def _run():
            return await execute_tool_async(
                "calculator",
                {"x": 1},
                execute_fn=fake_execute,
            )

        result = asyncio.run(_run())
        self.assertTrue(result["success"])
        self.assertIn("calculator", result["result"])

    def test_cancel_before_async_tool(self):
        token = CancellationToken(request_id="c1")
        token.cancel("client_disconnect")

        async def _run():
            await execute_tool_async(
                "calculator",
                {},
                cancellation=token,
                execute_fn=lambda *a, **k: {"success": True},
            )

        with self.assertRaises(RequestCancelled):
            asyncio.run(_run())

    def test_sync_worker_cancel_does_not_kill_running_fn(self):
        """Documented limitation: cancel during sync work is cooperative only."""
        started = threading.Event()
        finished = threading.Event()
        token = CancellationToken(request_id="c2")

        def slow_tool(name, args, **kwargs):
            started.set()
            time.sleep(0.2)
            finished.set()
            return {"success": True, "result": "done"}

        def runner():
            try:
                execute_tool_sync(
                    "read_file",
                    {},
                    cancellation=token,
                    execute_fn=slow_tool,
                )
            except RequestCancelled:
                pass

        t = threading.Thread(target=runner)
        t.start()
        self.assertTrue(started.wait(1.0))
        token.cancel("client_disconnect")
        t.join(1.0)
        # Worker may still finish — we only prove cancel was signaled.
        self.assertTrue(token.is_cancelled())


class ParallelExecutionTests(unittest.TestCase):
    def test_overlap_of_independent_tools(self):
        lock = threading.Lock()
        active = {"n": 0, "max": 0}

        def slow(name, args, **kwargs):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.15)
            with lock:
                active["n"] -= 1
            return {"success": True, "result": name, "id": args.get("id")}

        calls = [
            PreparedToolCall(0, "c0", "read_file", {"id": 0}),
            PreparedToolCall(1, "c1", "search_code", {"id": 1}),
            PreparedToolCall(2, "c2", "calculator", {"id": 2}),
        ]
        self.assertTrue(can_parallelize(calls))
        results = execute_tools_bounded(
            calls, max_concurrency=3, execute_fn=slow
        )
        self.assertEqual(len(results), 3)
        self.assertEqual([r["id"] for r in results], [0, 1, 2])
        self.assertGreaterEqual(active["max"], 2)

    def test_bounded_concurrency(self):
        lock = threading.Lock()
        active = {"n": 0, "max": 0}

        def slow(name, args, **kwargs):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.08)
            with lock:
                active["n"] -= 1
            return {"success": True, "result": "ok"}

        calls = [
            PreparedToolCall(i, f"c{i}", "read_file", {}) for i in range(6)
        ]
        execute_tools_bounded(calls, max_concurrency=2, execute_fn=slow)
        self.assertLessEqual(active["max"], 2)

    def test_memory_tools_force_sequential(self):
        order: list[str] = []

        def track(name, args, **kwargs):
            order.append(name)
            time.sleep(0.02)
            return {"success": True, "result": name}

        calls = [
            PreparedToolCall(0, "a", "save_memory", {}),
            PreparedToolCall(1, "b", "forget_memory", {}),
        ]
        self.assertFalse(can_parallelize(calls))
        execute_tools_bounded(calls, max_concurrency=4, execute_fn=track)
        self.assertEqual(order, ["save_memory", "forget_memory"])

    def test_individual_failure_preserves_siblings(self):
        def flaky(name, args, **kwargs):
            if args.get("boom"):
                return {"success": False, "error": "boom"}
            return {"success": True, "result": "ok"}

        calls = [
            PreparedToolCall(0, "c0", "read_file", {}),
            PreparedToolCall(1, "c1", "search_code", {"boom": True}),
            PreparedToolCall(2, "c2", "calculator", {}),
        ]
        results = execute_tools_bounded(calls, execute_fn=flaky)
        self.assertTrue(results[0]["success"])
        self.assertFalse(results[1]["success"])
        self.assertTrue(results[2]["success"])

    def test_cancel_stops_scheduling(self):
        token = CancellationToken(request_id="p1")
        started = {"n": 0}

        def slow(name, args, **kwargs):
            started["n"] += 1
            time.sleep(0.3)
            return {"success": True}

        calls = [
            PreparedToolCall(i, f"c{i}", "read_file", {}) for i in range(4)
        ]

        def cancel_soon():
            time.sleep(0.05)
            token.cancel("client_disconnect")

        threading.Thread(target=cancel_soon, daemon=True).start()
        with self.assertRaises(RequestCancelled):
            execute_tools_bounded(
                calls, cancellation=token, max_concurrency=2, execute_fn=slow
            )
        self.assertLess(started["n"], 4)


class StreamCancelIntegrationTests(unittest.TestCase):
    def setUp(self):
        from app.core.security import create_access_token
        from app.main import app

        self.client = TestClient(app)
        self.user = {
            "_id": "usr_cancel",
            "email": "c@example.com",
            "status": "active",
            "tenant_id": "ten_c",
            "display_name": "C",
        }
        token = create_access_token(
            user_id="usr_cancel",
            email="c@example.com",
            tenant_id="ten_c",
            session_id="sess_c",
        )
        self.headers = {"Authorization": f"Bearer {token}"}

    def test_cancel_during_agent_emits_cancelled_not_done(self):
        def wrapping_chat(message, **kwargs):
            tok = kwargs.get("cancellation")
            self.assertIsNotNone(tok)

            def boom():
                time.sleep(0.05)
                tok.cancel("client_disconnect")

            threading.Thread(target=boom, daemon=True).start()
            for _ in range(50):
                if tok.is_cancelled():
                    raise RequestCancelled(tok.reason)
                time.sleep(0.02)
            return "should not finish"

        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai", side_effect=wrapping_chat
        ):
            res = self.client.post(
                "/api/chat/stream",
                headers=self.headers,
                json={"message": "long running"},
            )

        self.assertEqual(res.status_code, 200, res.text)
        events = _parse_sse(res.text)
        types = [e["event"] for e in events]
        self.assertEqual(types[0], "meta")
        self.assertIn("error", types)
        self.assertNotIn("done", types)
        err = [e for e in events if e["event"] == "error"][-1]
        self.assertEqual(err["data"]["code"], "CANCELLED")
        self.assertNotIn("Traceback", res.text)

    def test_rest_unchanged_without_cancellation_kw(self):
        with patch("app.api.deps.get_user_by_id", return_value=self.user), patch(
            "app.api.deps.auth_service.get_active_session", return_value=True
        ), patch("app.api.routes.chat.enforce_chat_rate_limit"), patch(
            "app.api.routes.chat.is_document_question", return_value=False
        ), patch(
            "app.api.routes.chat.chat_with_ai", return_value="ok"
        ) as mock_chat:
            res = self.client.post(
                "/api/chat",
                headers=self.headers,
                json={"message": "hi"},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["reply"], "ok")
        self.assertIsNone(mock_chat.call_args.kwargs.get("cancellation"))

    def test_heartbeat_frame_shape(self):
        from app.services.sse_protocol import build_heartbeat_frame

        frame = build_heartbeat_frame(event_id=7)
        events = _parse_sse(frame)
        self.assertEqual(events[0]["event"], "heartbeat")
        self.assertEqual(events[0]["data"]["type"], "heartbeat")


class B6CancelCleanupTests(unittest.TestCase):
    def test_cleanup_interrupts_running_only(self):
        from types import SimpleNamespace

        from app.services.task_orchestration import safe_cleanup_request_tasks
        from app.services.task_progress import (
            build_progress_board_from_decomposition,
            start_task,
        )
        from app.validation.task_models import DecomposedTask, TaskDecomposition

        decomp = TaskDecomposition(
            original_request="Do A",
            trusted_instruction="Do A",
            is_multi_intent=False,
            tasks=[
                DecomposedTask(
                    id="t_run", description="running", category="api_development"
                ),
            ],
            execution_order=["t_run"],
            independent_task_ids=["t_run"],
        )
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t_run")
        self.assertEqual(board.tasks["t_run"].status, "running")
        state = SimpleNamespace(task_progress=board, active_task_id="t_run")
        interrupted = safe_cleanup_request_tasks(
            state, reason="cancelled:client_disconnect"
        )
        self.assertEqual(interrupted, ["t_run"])
        # B5 interrupt_task returns the task to pending (valid transition).
        self.assertEqual(board.tasks["t_run"].status, "pending")
        self.assertIsNone(state.active_task_id)
        # Idempotent second cleanup
        again = safe_cleanup_request_tasks(state, reason="cancelled:again")
        self.assertEqual(again, [])


class NativeAsyncToolTests(unittest.TestCase):
    def test_prepare_tool_invocation_strips_llm_scope_and_injects_auth(self):
        from app.chatbot import prepare_tool_invocation

        err, fn, kwargs = prepare_tool_invocation(
            "search_memory",
            {
                "query": "prefs",
                "user_id": "attacker",
                "project_id": "evil_proj",
                "tenant_id": "evil_ten",
            },
            user_id="real_user",
            request_id="req_auth",
            project_id="proj_ok",
            tenant_id="ten_ok",
        )
        self.assertIsNone(err)
        self.assertIsNotNone(fn)
        assert kwargs is not None
        self.assertEqual(kwargs.get("user_id"), "real_user")
        self.assertEqual(kwargs.get("project_id"), "proj_ok")
        self.assertEqual(kwargs.get("tenant_id"), "ten_ok")
        self.assertNotEqual(kwargs.get("user_id"), "attacker")

    def test_native_async_execute_fn_awaited_directly(self):
        seen = {"thread": None}

        async def async_exec(name, args, **kwargs):
            seen["thread"] = threading.current_thread().name
            await asyncio.sleep(0.01)
            return {"success": True, "result": f"async:{name}"}

        async def _run():
            return await execute_tool_async(
                "calculator",
                {"expression": "1+1"},
                execute_fn=async_exec,
            )

        result = asyncio.run(_run())
        self.assertTrue(result["success"])
        self.assertEqual(result["result"], "async:calculator")
        # Native async runs on the event-loop thread, not a worker pool thread.
        self.assertNotIn("agent-tool", seen["thread"] or "")

    def test_sync_tool_does_not_block_event_loop(self):
        progressed = {"n": 0}

        def slow_sync(name, args, **kwargs):
            time.sleep(0.2)
            return {"success": True, "result": "ok"}

        async def _run():
            async def ticker():
                for _ in range(5):
                    progressed["n"] += 1
                    await asyncio.sleep(0.02)

            tool_task = asyncio.create_task(
                execute_tool_async(
                    "calculator",
                    {},
                    execute_fn=slow_sync,
                )
            )
            tick_task = asyncio.create_task(ticker())
            result, _ = await asyncio.gather(tool_task, tick_task)
            return result

        result = asyncio.run(_run())
        self.assertTrue(result["success"])
        self.assertGreaterEqual(progressed["n"], 3)


class GlobalConcurrencyTests(unittest.TestCase):
    def setUp(self):
        os.environ["AGENT_GLOBAL_TOOL_CONCURRENCY"] = "2"
        from app.services import tool_execution as te

        te.reset_global_tool_pool_for_tests()

    def tearDown(self):
        os.environ.pop("AGENT_GLOBAL_TOOL_CONCURRENCY", None)
        from app.services import tool_execution as te

        te.reset_global_tool_pool_for_tests()

    def test_global_slot_caps_across_parallel_batch(self):
        from app.services.tool_execution import (
            execute_tools_bounded,
            reset_global_tool_pool_for_tests,
        )

        reset_global_tool_pool_for_tests()
        lock = threading.Lock()
        active = {"n": 0, "max": 0}

        def slow(name, args, **kwargs):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.08)
            with lock:
                active["n"] -= 1
            return {"success": True, "result": "ok"}

        calls = [
            PreparedToolCall(i, f"c{i}", "read_file", {}) for i in range(5)
        ]
        execute_tools_bounded(calls, max_concurrency=5, execute_fn=slow)
        self.assertLessEqual(active["max"], 2)


class AgentLoopBoundedDispatchTests(unittest.TestCase):
    """Prove the real agent loop routes multi-tool batches via execute_tools_bounded."""

    def _harness_patches(self):
        from app import chatbot

        return [
            patch.object(chatbot, "client"),
            patch.object(
                chatbot,
                "load_long_term_memory",
                return_value={"memories": [], "conflicts": [], "summaries": []},
            ),
            patch.object(chatbot, "restore_task_context_from_redis"),
            patch.object(chatbot, "save_task_context_to_redis"),
            patch.object(chatbot, "try_save_explicit_memory", return_value=False),
            patch.object(chatbot, "try_update_explicit_memory", return_value=None),
            patch.object(chatbot, "try_delete_explicit_memory", return_value=None),
            patch.object(chatbot, "try_inspect_memory_metadata", return_value=None),
            patch(
                "app.services.memory_scope_service.get_active_project_scope",
                return_value={"project_id": "proj_a", "tenant_id": "ten_a"},
            ),
            patch.object(chatbot, "add_to_conversation_memory"),
            patch.object(chatbot, "get_conversation_memory", return_value=[]),
        ]

    def test_agent_loop_calls_execute_tools_bounded(self):
        from app import chatbot

        patches = self._harness_patches()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            c1 = MagicMock()
            c1.id = "tc_calc"
            c1.function.name = "calculator"
            c1.function.arguments = json.dumps({"expression": "2+2"})

            c2 = MagicMock()
            c2.id = "tc_time"
            c2.function.name = "get_current_time"
            c2.function.arguments = json.dumps({})

            step1 = MagicMock()
            step1.tool_calls = [c1, c2]
            step1.content = None
            final = MagicMock()
            final.tool_calls = None
            final.content = "4 and noon."

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            def fake_execute(name, args, **kwargs):
                return {
                    "success": True,
                    "result": json.dumps({"success": True, "name": name}),
                }

            with patch.dict(
                "app.services.agent_orchestration.INTENT_TOOL_ALLOWLIST",
                {
                    "calculation": frozenset(
                        {"calculator", "get_current_time"}
                    )
                },
            ), patch.object(
                chatbot, "execute_tool", side_effect=fake_execute
            ), patch(
                "app.services.tool_execution.execute_tools_bounded",
                wraps=execute_tools_bounded,
            ) as bounded:
                answer = chatbot.chat_with_ai(
                    "Calculate 2+2 and tell me the time.",
                    user_id="u_bound",
                    request_id="req_bound",
                )

            self.assertTrue(bounded.called)
            call_kw = bounded.call_args
            prepared = call_kw.args[0] if call_kw.args else call_kw.kwargs.get("calls")
            self.assertEqual(len(prepared), 2)
            self.assertEqual(
                [p.tool_call_id for p in prepared],
                ["tc_calc", "tc_time"],
            )
            self.assertIsInstance(answer, str)
            self.assertTrue(len(answer) > 0)
        finally:
            for p in patches:
                p.stop()

    def test_agent_loop_parallel_overlap_for_safe_tools(self):
        from app import chatbot

        patches = self._harness_patches()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            calls_meta = []
            for i, name in enumerate(
                ("calculator", "get_current_time", "calculator")
            ):
                tc = MagicMock()
                tc.id = f"tc{i}"
                tc.function.name = name
                tc.function.arguments = json.dumps(
                    {"expression": f"{i}+{i}"} if name == "calculator" else {}
                )
                calls_meta.append(tc)

            step1 = MagicMock()
            step1.tool_calls = calls_meta
            step1.content = None
            final = MagicMock()
            final.tool_calls = None
            final.content = "done"

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            lock = threading.Lock()
            active = {"n": 0, "max": 0}
            order_ids: list[str] = []

            def slow_execute(name, args, **kwargs):
                with lock:
                    active["n"] += 1
                    active["max"] = max(active["max"], active["n"])
                time.sleep(0.12)
                with lock:
                    active["n"] -= 1
                order_ids.append(name)
                return {
                    "success": True,
                    "result": json.dumps({"success": True, "name": name}),
                }

            # Intent allowlist is now an execution gate: expand calculation
            # tools so the parallel batch under test is actually permitted.
            with patch.dict(
                "app.services.agent_orchestration.INTENT_TOOL_ALLOWLIST",
                {
                    "calculation": frozenset(
                        {"calculator", "get_current_time"}
                    )
                },
            ), patch.object(
                chatbot, "execute_tool", side_effect=slow_execute
            ):
                chatbot.chat_with_ai(
                    "Calculate and get time twice.",
                    user_id="u_overlap",
                    request_id="req_overlap",
                )

            self.assertGreaterEqual(
                active["max"],
                2,
                "agent-loop batch must overlap independent safe tools",
            )
            self.assertEqual(len(order_ids), 3)
        finally:
            for p in patches:
                p.stop()

    def test_agent_loop_unsafe_tools_sequential(self):
        from app import chatbot
        from app.services import agent_orchestration as orch

        patches = self._harness_patches()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            s1 = MagicMock()
            s1.id = "m1"
            s1.function.name = "save_memory"
            s1.function.arguments = json.dumps({"memory": "fact one"})
            s2 = MagicMock()
            s2.id = "m2"
            s2.function.name = "forget_memory"
            s2.function.arguments = json.dumps({"memory_id": "mem_x"})

            step1 = MagicMock()
            step1.tool_calls = [s1, s2]
            step1.content = None
            final = MagicMock()
            final.tool_calls = None
            final.content = "saved and forgotten"

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            lock = threading.Lock()
            active = {"n": 0, "max": 0}
            seen: list[str] = []

            def track(name, args, **kwargs):
                with lock:
                    active["n"] += 1
                    active["max"] = max(active["max"], active["n"])
                time.sleep(0.05)
                with lock:
                    active["n"] -= 1
                seen.append(name)
                return {
                    "success": True,
                    "result": json.dumps({"success": True}),
                    "mutation_committed": name == "save_memory",
                    "raw_success": True,
                }

            with patch.object(
                orch, "classify_intent", return_value=orch.INTENT_MEMORY
            ), patch.object(
                chatbot, "execute_tool", side_effect=track
            ):
                chatbot.chat_with_ai(
                    "Please remember that fact one is true. "
                    "Then forget memory mem_x.",
                    user_id="u_seq",
                    request_id="req_seq",
                )

            self.assertEqual(active["max"], 1)
            self.assertEqual(seen, ["save_memory", "forget_memory"])
        finally:
            for p in patches:
                p.stop()

    def test_agent_loop_cancel_during_parallel_batch(self):
        from app import chatbot
        from app.services.request_cancellation import (
            CancellationToken,
            RequestCancelled,
        )

        patches = self._harness_patches()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]
            cancel = CancellationToken(request_id="req_cancel_batch", user_id="u_c")

            tools = []
            for i in range(4):
                tc = MagicMock()
                tc.id = f"cx{i}"
                tc.function.name = "calculator"
                tc.function.arguments = json.dumps({"expression": f"{i}+1"})
                tools.append(tc)

            step1 = MagicMock()
            step1.tool_calls = tools
            step1.content = None
            final = MagicMock()
            final.tool_calls = None
            final.content = "should not reach"

            responses = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]
            call_n = {"n": 0}
            provider_after_cancel = {"n": 0}

            def counting_create(*a, **k):
                call_n["n"] += 1
                if cancel.is_cancelled():
                    provider_after_cancel["n"] += 1
                return responses[min(call_n["n"] - 1, len(responses) - 1)]

            mock_client.chat.completions.create.side_effect = counting_create

            started = {"n": 0}

            def slow_execute(name, args, **kwargs):
                started["n"] += 1
                if started["n"] == 1:
                    cancel.cancel("client_disconnect")
                time.sleep(0.15)
                return {
                    "success": True,
                    "result": json.dumps({"success": True}),
                }

            with patch.object(
                chatbot, "execute_tool", side_effect=slow_execute
            ):
                with self.assertRaises(RequestCancelled):
                    chatbot.chat_with_ai(
                        "Calculate 1+1, 2+2, 3+3, and 4+4 for me please.",
                        user_id="u_c",
                        request_id="req_cancel_batch",
                        cancellation=cancel,
                    )

            self.assertTrue(cancel.is_cancelled())
            self.assertLess(started["n"], 4)
            # No follow-up provider turn after cancel mid-batch.
            self.assertEqual(provider_after_cancel["n"], 0)
        finally:
            for p in patches:
                p.stop()

    def test_agent_loop_preserves_tool_call_ids_and_partial_failure(self):
        from app import chatbot

        patches = self._harness_patches()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]

            t0 = MagicMock()
            t0.id = "id_ok_a"
            t0.function.name = "calculator"
            t0.function.arguments = json.dumps({"expression": "1+1"})
            t1 = MagicMock()
            t1.id = "id_fail"
            t1.function.name = "get_current_time"
            t1.function.arguments = json.dumps({})
            t2 = MagicMock()
            t2.id = "id_ok_b"
            t2.function.name = "calculator"
            t2.function.arguments = json.dumps({"expression": "3+3"})

            step1 = MagicMock()
            step1.tool_calls = [t0, t1, t2]
            step1.content = None
            final = MagicMock()
            final.tool_calls = None
            final.content = "partial ok"

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            def flaky(name, args, **kwargs):
                if name == "get_current_time":
                    return {"success": False, "error": "boom"}
                return {
                    "success": True,
                    "result": json.dumps({"success": True, "v": args}),
                }

            captured: list = []

            def capturing_bounded(*args, **kwargs):
                results = execute_tools_bounded(*args, **kwargs)
                captured.append(results)
                return results

            with patch.object(
                chatbot, "execute_tool", side_effect=flaky
            ), patch(
                "app.services.tool_execution.execute_tools_bounded",
                side_effect=capturing_bounded,
            ) as bounded:
                answer = chatbot.chat_with_ai(
                    "Calculate 1+1, tell me the current time, and calculate 3+3.",
                    user_id="u_partial",
                    request_id="req_partial",
                )

            self.assertTrue(bounded.called)
            prepared = bounded.call_args.args[0]
            self.assertEqual(
                [p.tool_call_id for p in prepared],
                ["id_ok_a", "id_fail", "id_ok_b"],
            )
            self.assertEqual(len(captured), 1)
            results = captured[0]
            self.assertTrue(results[0]["success"])
            self.assertFalse(results[1]["success"])
            self.assertTrue(results[2]["success"])
            self.assertIsInstance(answer, str)
            self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        finally:
            for p in patches:
                p.stop()

    def test_auth_and_project_scope_reach_execute_tool(self):
        from app import chatbot

        patches = self._harness_patches()
        mocks = [p.start() for p in patches]
        try:
            mock_client = mocks[0]
            tc = MagicMock()
            tc.id = "scope1"
            tc.function.name = "search_memory"
            tc.function.arguments = json.dumps({"query": "hello"})

            step1 = MagicMock()
            step1.tool_calls = [tc]
            step1.content = None
            final = MagicMock()
            final.tool_calls = None
            final.content = "found nothing"

            mock_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=step1)]),
                MagicMock(choices=[MagicMock(message=final)]),
            ]

            seen = {}

            def capture(name, args, **kwargs):
                seen.update(kwargs)
                seen["name"] = name
                return {
                    "success": True,
                    "result": json.dumps({"success": True, "memories": []}),
                }

            with patch.object(chatbot, "execute_tool", side_effect=capture):
                chatbot.chat_with_ai(
                    "search my memories for hello",
                    user_id="u_scope",
                    request_id="req_scope",
                )

            self.assertEqual(seen.get("user_id"), "u_scope")
            self.assertEqual(seen.get("project_id"), "proj_a")
            self.assertEqual(seen.get("tenant_id"), "ten_a")
            self.assertEqual(seen.get("request_id"), "req_scope")
        finally:
            for p in patches:
                p.stop()

    def test_source_agent_loop_uses_bounded_dispatcher(self):
        import inspect
        from app import chatbot

        src = inspect.getsource(chatbot.chat_with_ai)
        self.assertIn("execute_tools_bounded", src)
        self.assertIn("PreparedToolCall", src)


class B5EvidenceCancelTests(unittest.TestCase):
    def test_completed_evidence_survives_cancel_cleanup(self):
        from types import SimpleNamespace

        from app.services.task_orchestration import safe_cleanup_request_tasks
        from app.services.task_progress import (
            build_progress_board_from_decomposition,
            complete_task,
            note_observable_tool_result,
            start_task,
        )
        from app.validation.task_models import DecomposedTask, TaskDecomposition

        decomp = TaskDecomposition(
            original_request="Do A then B",
            trusted_instruction="Do A then B",
            is_multi_intent=True,
            tasks=[
                DecomposedTask(
                    id="t_done",
                    description="done",
                    category="calculation",
                    independent=True,
                ),
                DecomposedTask(
                    id="t_run",
                    description="running",
                    category="calculation",
                    depends_on=["t_done"],
                ),
            ],
            execution_order=["t_done", "t_run"],
            independent_task_ids=["t_done"],
        )
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t_done")
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            fingerprint="fp1",
            result_excerpt="4",
            preferred_task_id="t_done",
        )
        complete_task(board, "t_done")
        start_task(board, "t_run")
        state = SimpleNamespace(task_progress=board, active_task_id="t_run")
        safe_cleanup_request_tasks(state, reason="cancelled:client_disconnect")
        self.assertEqual(board.tasks["t_done"].status, "completed")
        self.assertTrue(board.tasks["t_done"].evidence)
        self.assertEqual(board.tasks["t_run"].status, "pending")
        self.assertIsNone(state.active_task_id)


if __name__ == "__main__":
    unittest.main()
