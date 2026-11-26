"""Phase B5 — task progress and dependency tracking tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app import chatbot
from app.services.agent_planning import (
    attach_task_decomposition,
    build_agent_plan,
    plan_next_action,
)
from app.services.agent_routing import route_request
from app.services.agent_state import AgentState
from app.services.task_decomposition import decompose_request
from app.services.task_progress import (
    TaskProgressError,
    build_progress_board_from_decomposition,
    complete_task,
    eligible_task_ids,
    fail_task,
    interrupt_task,
    note_observable_tool_result,
    record_evidence,
    retry_task,
    set_uncertain_dependency_policy,
    start_task,
)
from app.validation.task_models import DecomposedTask, TaskDecomposition
from app.validation.task_progress_models import (
    EvidenceRecord,
    TaskProgressBoard,
    TaskProgressRecord,
)


def _board_two_tasks(
    *,
    kind: str = "required",
    cat1: str = "api_development",
    cat2: str = "testing",
) -> TaskProgressBoard:
    decomp = TaskDecomposition(
        original_request="Build API and write tests",
        trusted_instruction="Build API and write tests",
        is_multi_intent=True,
        tasks=[
            DecomposedTask(
                id="t1",
                description="Build API",
                category=cat1,  # type: ignore[arg-type]
            ),
            DecomposedTask(
                id="t2",
                description="Write tests",
                category=cat2,  # type: ignore[arg-type]
                depends_on=["t1"],
                dependency_kind=kind,  # type: ignore[arg-type]
                expected_outputs=["test suite"],
            ),
        ],
        execution_order=["t1", "t2"],
        independent_task_ids=["t1"],
    )
    board = build_progress_board_from_decomposition(decomp)
    assert board is not None
    return board


class TaskProgressModelTests(unittest.TestCase):
    def test_duplicate_task_ids_and_cycles_rejected(self):
        with self.assertRaises(ValidationError):
            TaskProgressBoard(
                board_id="b1",
                tasks={
                    "t1": TaskProgressRecord(task_id="t2", description="mismatch"),
                },
            )
        with self.assertRaises(ValidationError):
            TaskProgressBoard.model_validate(
                {
                    "board_id": "b1",
                    "tasks": {
                        "t1": {
                            "task_id": "t1",
                            "description": "a",
                            "depends_on": [{"predecessor_id": "t2", "kind": "required"}],
                        },
                        "t2": {
                            "task_id": "t2",
                            "description": "b",
                            "depends_on": [{"predecessor_id": "t1", "kind": "required"}],
                        },
                    },
                }
            )

    def test_missing_dependency_record_rejected(self):
        with self.assertRaises(ValidationError):
            TaskProgressBoard.model_validate(
                {
                    "board_id": "b1",
                    "tasks": {
                        "t1": {
                            "task_id": "t1",
                            "description": "a",
                            "depends_on": [
                                {"predecessor_id": "t9", "kind": "required"}
                            ],
                        }
                    },
                }
            )

    def test_llm_claim_cannot_be_verified(self):
        with self.assertRaises(ValidationError):
            EvidenceRecord(
                kind="llm_claim",
                source="system",
                summary="I finished the task",
                verified=True,
            )


class TaskProgressDependencyTests(unittest.TestCase):
    def test_required_dependency_gating(self):
        board = _board_two_tasks(kind="required")
        self.assertEqual(board.tasks["t2"].status, "blocked")
        self.assertIn("t1", board.tasks["t2"].blocked_by)
        with self.assertRaises(TaskProgressError):
            start_task(board, "t2")
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_code_artifact",
                source="assistant_artifact",
                summary="```python\ndef api():\n    return 1\n```",
            ),
        )
        complete_task(board, "t1")
        self.assertEqual(board.tasks["t2"].status, "pending")
        start_task(board, "t2")
        self.assertEqual(board.tasks["t2"].status, "running")

    def test_optional_dependency_failure_does_not_block(self):
        board = _board_two_tasks(kind="optional")
        # Optional edge: t2 should remain pending at init.
        self.assertEqual(board.tasks["t2"].status, "pending")
        start_task(board, "t1")
        fail_task(board, "t1", failure_code="tool_error", detail="boom")
        # t2 still eligible despite t1 failed (optional).
        self.assertIn("t2", eligible_task_ids(board))
        start_task(board, "t2")
        self.assertEqual(board.tasks["t2"].status, "running")

    def test_failed_required_dependency_blocks(self):
        board = _board_two_tasks(kind="required")
        start_task(board, "t1")
        fail_task(board, "t1", failure_code="impl_failed", detail="x")
        self.assertEqual(board.tasks["t2"].status, "blocked")
        with self.assertRaises(TaskProgressError):
            start_task(board, "t2")
        with self.assertRaises(TaskProgressError):
            # cannot complete blocked/pending without running
            complete_task(board, "t2")

    def test_uncertain_dependency_blocks_until_policy(self):
        board = _board_two_tasks(kind="uncertain")
        self.assertEqual(board.tasks["t2"].status, "blocked")
        with self.assertRaises(TaskProgressError):
            start_task(board, "t2")
        set_uncertain_dependency_policy(
            board, "treat_as_optional", reason="reviewer_override"
        )
        self.assertIn("t2", eligible_task_ids(board))
        start_task(board, "t2")

    def test_no_auto_start_when_eligible(self):
        board = _board_two_tasks(kind="required")
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_code_artifact",
                source="assistant_artifact",
                summary="code artifact",
            ),
        )
        complete_task(board, "t1")
        self.assertEqual(board.tasks["t2"].status, "pending")
        self.assertIn("t2", eligible_task_ids(board))
        # Still pending until explicit start.
        self.assertNotEqual(board.tasks["t2"].status, "running")


class TaskProgressTransitionTests(unittest.TestCase):
    def test_valid_and_invalid_transitions(self):
        board = _board_two_tasks()
        with self.assertRaises(TaskProgressError):
            start_task(board, "t2")  # blocked → running forbidden
        start_task(board, "t1")
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")  # no evidence
        interrupt_task(board, "t1", reason="user_cancelled_step")
        self.assertEqual(board.tasks["t1"].status, "pending")
        start_task(board, "t1")
        fail_task(board, "t1", failure_code="x", detail="y")
        self.assertEqual(board.tasks["t1"].failure_detail, "y")
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")  # failed → completed forbidden
        retry_task(board, "t1", reason="explicit_resume")
        self.assertEqual(board.tasks["t1"].status, "pending")
        self.assertEqual(board.tasks["t1"].attempt_count, 2)

    def test_completed_immutable_and_duplicate_complete(self):
        board = _board_two_tasks()
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_code_artifact",
                source="assistant_artifact",
                summary="artifact",
            ),
        )
        complete_task(board, "t1")
        version = board.board_version
        complete_task(board, "t1")  # idempotent
        self.assertEqual(board.board_version, version)
        with self.assertRaises(TaskProgressError):
            start_task(board, "t1")
        with self.assertRaises(TaskProgressError):
            interrupt_task(board, "t1", reason="nope")

    def test_unknown_task_id(self):
        board = _board_two_tasks()
        with self.assertRaises(TaskProgressError):
            start_task(board, "t9")


class TaskProgressEvidenceTests(unittest.TestCase):
    def test_reject_llm_only_completion(self):
        board = _board_two_tasks()
        start_task(board, "t1")
        with self.assertRaises(ValidationError):
            EvidenceRecord(
                kind="llm_claim",
                source="system",
                summary="done",
                verified=True,
            )
        # Unverified llm_claim must not satisfy completion.
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="llm_claim",
                source="system",
                summary="I completed the API",
                verified=False,
            ),
        )
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")

    def test_unsupported_evidence_kind_for_category(self):
        board = _board_two_tasks()
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="memory_tool_result",
                source="tool_result",
                summary="saved",
                tool_name="save_memory",
            ),
        )
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")

    def test_generated_tests_vs_executed_tests(self):
        decomp = TaskDecomposition(
            trusted_instruction="Write and run tests",
            tasks=[
                DecomposedTask(
                    id="t1",
                    description="Write and run tests",
                    category="testing",
                    expected_outputs=["executed test results"],
                    acceptance_criteria=["tests pass"],
                )
            ],
            execution_order=["t1"],
            independent_task_ids=["t1"],
        )
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_tests_artifact",
                source="assistant_artifact",
                summary="def test_x():\n    assert True",
            ),
        )
        with self.assertRaises(TaskProgressError) as ctx:
            complete_task(board, "t1")
        self.assertIn("executed", str(ctx.exception).casefold())

        # Without execution requirement, generated artifact can complete.
        decomp2 = TaskDecomposition(
            trusted_instruction="Write unit tests",
            tasks=[
                DecomposedTask(
                    id="t1",
                    description="Write unit tests",
                    category="testing",
                    expected_outputs=["test suite"],
                )
            ],
            execution_order=["t1"],
            independent_task_ids=["t1"],
        )
        board2 = build_progress_board_from_decomposition(decomp2)
        assert board2 is not None
        start_task(board2, "t1")
        record_evidence(
            board2,
            "t1",
            EvidenceRecord(
                kind="generated_tests_artifact",
                source="assistant_artifact",
                summary="def test_y():\n    assert 1",
            ),
        )
        complete_task(board2, "t1")
        self.assertEqual(board2.tasks["t1"].status, "completed")

    def test_multiple_evidence_for_multiple_criteria(self):
        decomp = TaskDecomposition(
            trusted_instruction="Implement and document",
            tasks=[
                DecomposedTask(
                    id="t1",
                    description="Implement and document",
                    category="documentation",
                    acceptance_criteria=["code section", "run steps"],
                    expected_outputs=["docs"],
                )
            ],
            execution_order=["t1"],
            independent_task_ids=["t1"],
        )
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="final_explanation_output",
                source="loop_finalize",
                summary="How to run: uvicorn main:app",
            ),
        )
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_document_artifact",
                source="assistant_artifact",
                summary="# Setup\n...",
            ),
        )
        complete_task(board, "t1")

    def test_tool_note_does_not_auto_complete(self):
        board = _board_two_tasks()
        start_task(board, "t1")
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            fingerprint="calculator:{}",
            result_excerpt="42",
        )
        # calculator evidence is wrong category for api_development — still running
        self.assertEqual(board.tasks["t1"].status, "running")
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")

    def test_calculator_cannot_complete_codegen(self):
        decomp = TaskDecomposition(
            trusted_instruction="Create a React login page",
            tasks=[
                DecomposedTask(
                    id="t1",
                    description="Create a React login page",
                    category="code_generation",
                )
            ],
            execution_order=["t1"],
            independent_task_ids=["t1"],
        )
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="calculator_result",
                source="tool_result",
                summary="42",
                tool_name="calculator",
            ),
        )
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")

    def test_read_file_alone_cannot_complete_debugging(self):
        decomp = TaskDecomposition(
            trusted_instruction="Find the auth bug in my code",
            tasks=[
                DecomposedTask(
                    id="t1",
                    description="Find the auth bug",
                    category="debugging",
                )
            ],
            execution_order=["t1"],
            independent_task_ids=["t1"],
        )
        board = build_progress_board_from_decomposition(decomp)
        assert board is not None
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="file_read_result",
                source="tool_result",
                summary="def login(): ...",
                tool_name="read_file",
            ),
        )
        with self.assertRaises(TaskProgressError) as ctx:
            complete_task(board, "t1")
        self.assertIn("debugging", str(ctx.exception).casefold())
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="code_search_result",
                source="tool_result",
                summary="login hits",
                tool_name="search_code",
            ),
        )
        complete_task(board, "t1")

    def test_final_explanation_cannot_complete_implementation(self):
        board = _board_two_tasks(cat1="code_generation", cat2="testing")
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="final_explanation_output",
                source="loop_finalize",
                summary="I built the login page.",
            ),
        )
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")

    def test_failed_tool_not_recorded_and_unassigned_not_proof(self):
        board = _board_two_tasks()
        start_task(board, "t1")
        note_observable_tool_result(
            board,
            tool_name="web_search",
            success=False,
            fingerprint="web_search:fail",
            result_excerpt="error",
        )
        self.assertEqual(board.tasks["t1"].evidence, [])
        self.assertEqual(board.unassigned_evidence, [])
        # Successful but category-mismatched tool → unassigned pool only.
        note_observable_tool_result(
            board,
            tool_name="calculator",
            success=True,
            fingerprint="calculator:1",
            result_excerpt="7",
        )
        self.assertEqual(board.tasks["t1"].evidence, [])
        self.assertEqual(len(board.unassigned_evidence), 1)
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")

    def test_prior_attempt_evidence_does_not_complete_after_retry(self):
        board = _board_two_tasks()
        start_task(board, "t1")
        self.assertEqual(board.tasks["t1"].attempt_count, 1)
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_code_artifact",
                source="assistant_artifact",
                summary="attempt1 code",
            ),
        )
        fail_task(board, "t1", failure_code="bad", detail="broke")
        retry_task(board, "t1", reason="explicit_retry")
        self.assertEqual(board.tasks["t1"].attempt_count, 2)
        self.assertEqual(len(board.tasks["t1"].evidence), 1)  # history kept
        start_task(board, "t1")
        self.assertEqual(board.tasks["t1"].attempt_count, 2)
        with self.assertRaises(TaskProgressError):
            complete_task(board, "t1")  # prior attempt evidence superseded
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_code_artifact",
                source="assistant_artifact",
                summary="attempt2 code",
            ),
        )
        complete_task(board, "t1")
        self.assertEqual(board.tasks["t1"].status, "completed")

    def test_interrupt_preserves_evidence_same_attempt(self):
        board = _board_two_tasks()
        start_task(board, "t1")
        record_evidence(
            board,
            "t1",
            EvidenceRecord(
                kind="generated_code_artifact",
                source="assistant_artifact",
                summary="partial",
            ),
        )
        interrupt_task(board, "t1", reason="pause")
        self.assertEqual(len(board.tasks["t1"].evidence), 1)
        self.assertEqual(board.tasks["t1"].attempt_count, 1)
        start_task(board, "t1")  # resume same attempt
        self.assertEqual(board.tasks["t1"].attempt_count, 1)
        complete_task(board, "t1")


class TaskProgressIntegrationTests(unittest.TestCase):
    def test_agent_state_holds_board(self):
        decomp = decompose_request(
            "Create a React login page and write tests"
        )
        board = build_progress_board_from_decomposition(
            decomp, user_id="user_a", request_id="req1"
        )
        state = AgentState(task_progress=board)
        self.assertIsNotNone(state.task_progress)
        self.assertGreaterEqual(len(state.task_progress.tasks), 2)

    def test_plan_next_action_unchanged_with_board(self):
        before = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            step=0,
            max_steps=8,
            investigation_started=False,
            uncovered_count=0,
            successful_searches=0,
            successful_reads=0,
        )
        decomp = decompose_request(
            "Create a React login page and write tests"
        ).model_dump()
        plan = attach_task_decomposition(
            build_agent_plan({"needs_search": True}), decomp
        )
        board = build_progress_board_from_decomposition(decomp)
        after = plan_next_action(
            phase="pre_llm",
            tools_enabled=True,
            step=0,
            max_steps=8,
            investigation_started=False,
            uncovered_count=0,
            successful_searches=0,
            successful_reads=0,
        )
        self.assertEqual(before.action, after.action)
        self.assertEqual(before.reason, after.reason)
        self.assertIsNotNone(board)
        self.assertTrue(plan.decomposed_task_ids)

    def test_allowlists_unchanged_for_multi_intent(self):
        decision = route_request(
            user_message=(
                "Build a FastAPI login API, connect MongoDB, add JWT authentication, "
                "write tests, and explain how to run it"
            ),
            all_tool_definitions=chatbot.tool_definitions,
            multi_tool_fn=chatbot.is_multi_tool_request,
            user_id="user_a",
        )
        names = {
            ((t.get("function") or {}).get("name"))
            for t in decision.tool_definitions
        }
        self.assertNotIn("search_memory", names)
        self.assertNotIn("forget_memory", names)
        board = build_progress_board_from_decomposition(
            decision.task_decomposition
        )
        self.assertIsNotNone(board)
        # Eligibility alone must not imply tools.
        _ = eligible_task_ids(board)


if __name__ == "__main__":
    unittest.main()
