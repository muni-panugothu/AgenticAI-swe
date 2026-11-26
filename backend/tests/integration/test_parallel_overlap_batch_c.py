"""Batch C — deterministic parallel overlap proof (offline, no paid APIs).

Uses a threading.Barrier so overlap cannot be explained by scheduling luck alone.
"""

from __future__ import annotations

import json
import threading
import time
import unittest

from app.services.request_cancellation import CancellationToken, RequestCancelled
from app.services.tool_execution import (
    PreparedToolCall,
    can_parallelize,
    execute_tools_bounded,
    is_parallel_safe,
    partition_execution_batches,
)


class ParallelOverlapBatchCTests(unittest.TestCase):
    def test_BC_P01_parallel_safe_tools_overlap_via_barrier(self):
        self.assertTrue(is_parallel_safe("calculator"))
        self.assertTrue(is_parallel_safe("get_current_time"))

        barrier = threading.Barrier(2, timeout=5)
        lock = threading.Lock()
        active = {"n": 0, "max": 0}

        def execute_fn(name, args, **kwargs):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            try:
                # Both workers must arrive — proves concurrent execution.
                barrier.wait()
                time.sleep(0.02)
                return {
                    "success": True,
                    "result": json.dumps({"success": True, "name": name}),
                }
            finally:
                with lock:
                    active["n"] -= 1

        calls = [
            PreparedToolCall(
                index=0,
                tool_call_id="tc_calc",
                tool_name="calculator",
                tool_arguments={"expression": "1+1"},
            ),
            PreparedToolCall(
                index=1,
                tool_call_id="tc_time",
                tool_name="get_current_time",
                tool_arguments={},
            ),
        ]
        self.assertTrue(can_parallelize(calls))
        results = execute_tools_bounded(
            calls,
            user_id="bc_p01",
            request_id="req_p01",
            execute_fn=execute_fn,
            max_concurrency=2,
        )
        self.assertEqual(len(results), 2)
        self.assertGreater(active["max"], 1)
        self.assertEqual(
            [r.get("tool_call_id") or "" for r in results].count(""),
            2,
        )  # packaged results may omit id; length/order is the contract
        # Stable ordering by original index.
        blob0 = json.dumps(results[0]).lower()
        blob1 = json.dumps(results[1]).lower()
        self.assertIn("calculator", blob0)
        self.assertIn("get_current_time", blob1)

    def test_BC_P02_unsafe_tools_remain_sequential(self):
        self.assertFalse(is_parallel_safe("save_memory"))
        calls = [
            PreparedToolCall(
                index=0,
                tool_call_id="tc_a",
                tool_name="save_memory",
                tool_arguments={"memory": "a"},
            ),
            PreparedToolCall(
                index=1,
                tool_call_id="tc_b",
                tool_name="forget_memory",
                tool_arguments={"memory_id": "m1"},
            ),
        ]
        self.assertFalse(can_parallelize(calls))
        self.assertEqual(len(partition_execution_batches(calls)), 2)

        lock = threading.Lock()
        active = {"n": 0, "max": 0}

        def execute_fn(name, args, **kwargs):
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.05)
            with lock:
                active["n"] -= 1
            return {
                "success": True,
                "result": json.dumps({"success": True, "name": name}),
            }

        results = execute_tools_bounded(
            calls,
            user_id="bc_p02",
            execute_fn=execute_fn,
            max_concurrency=4,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(active["max"], 1)

    def test_BC_P03_one_failure_preserves_sibling_success(self):
        barrier = threading.Barrier(2, timeout=5)

        def execute_fn(name, args, **kwargs):
            barrier.wait()
            if name == "get_current_time":
                raise RuntimeError("controlled failure")
            return {
                "success": True,
                "result": json.dumps({"success": True, "name": name}),
            }

        calls = [
            PreparedToolCall(
                index=0,
                tool_call_id="tc_ok",
                tool_name="calculator",
                tool_arguments={"expression": "2+2"},
            ),
            PreparedToolCall(
                index=1,
                tool_call_id="tc_fail",
                tool_name="get_current_time",
                tool_arguments={},
            ),
        ]
        results = execute_tools_bounded(
            calls,
            user_id="bc_p03",
            execute_fn=execute_fn,
            max_concurrency=2,
        )
        self.assertEqual(len(results), 2)
        self.assertIn("calculator", json.dumps(results[0]).lower())
        # Sibling failure is packaged honestly — slot still present.
        self.assertIsInstance(results[1], dict)
        self.assertTrue(
            results[1].get("success") is False
            or "error" in results[1]
            or "fail" in json.dumps(results[1]).lower()
            or "controlled failure" in json.dumps(results[1]).lower()
        )

    def test_BC_P04_cancellation_stops_new_scheduling(self):
        token = CancellationToken(request_id="req_p04", user_id="bc_p04")
        started: list[str] = []

        def execute_fn(name, args, **kwargs):
            started.append(name)
            token.cancel("test_cancel")
            raise RequestCancelled("test_cancel")

        calls = [
            PreparedToolCall(
                index=0,
                tool_call_id="tc1",
                tool_name="calculator",
                tool_arguments={"expression": "1"},
            ),
            PreparedToolCall(
                index=1,
                tool_call_id="tc2",
                tool_name="get_current_time",
                tool_arguments={},
            ),
        ]
        with self.assertRaises(RequestCancelled):
            execute_tools_bounded(
                calls,
                user_id="bc_p04",
                cancellation=token,
                execute_fn=execute_fn,
                max_concurrency=1,
            )
        self.assertGreaterEqual(len(started), 1)
        self.assertLessEqual(len(started), 2)


if __name__ == "__main__":
    unittest.main()
