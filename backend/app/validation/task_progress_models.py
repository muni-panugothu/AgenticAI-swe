"""Pydantic models for Phase B5 execution-aware task progress.

Progress tracking is request-scoped advisory state. It never executes tasks,
never grants tools/ACL, and never treats planning hints or free-form LLM
claims as completion proof.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.validation.task_models import DependencyKind, TaskCategory


TaskStatus = Literal["pending", "running", "blocked", "completed", "failed"]

EvidenceKind = Literal[
    "calculator_result",
    "web_search_result",
    "current_time_result",
    "code_search_result",
    "file_read_result",
    "investigation_verified_target",
    "generated_code_artifact",
    "generated_document_artifact",
    "generated_tests_artifact",
    "executed_tests_result",  # reserved; no runner in this codebase yet
    "rag_retrieval_result",
    "memory_tool_result",
    "final_explanation_output",
    "llm_claim",  # explicitly unsupported for completion
    "other_observable",
]

EvidenceSource = Literal[
    "tool_result",
    "investigation_evidence",
    "rag_context",
    "assistant_artifact",
    "loop_finalize",
    "unassigned_pool",
    "system",
]

UncertainDependencyPolicy = Literal["block", "treat_as_optional"]

TransitionAction = Literal[
    "init",
    "start",
    "block",
    "unblock",
    "complete",
    "fail",
    "interrupt",
    "retry",
    "record_evidence",
    "policy_override",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DependencyEdge(BaseModel):
    """One predecessor edge expanded from B4 task-level dependency_kind."""

    model_config = ConfigDict(extra="forbid")

    predecessor_id: str = Field(min_length=1, max_length=32)
    kind: DependencyKind = "required"


class EvidenceRecord(BaseModel):
    """Typed observable evidence — not a free-form model claim."""

    model_config = ConfigDict(extra="forbid")

    kind: EvidenceKind
    source: EvidenceSource
    summary: str = Field(min_length=1, max_length=2000)
    tool_name: str | None = None
    fingerprint: str | None = None
    task_id: str | None = None
    verified: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _reject_unverified_llm_as_ok(self) -> EvidenceRecord:
        if self.kind == "llm_claim" and self.verified:
            # Callers may store llm_claim only as non-verifying audit noise.
            raise ValueError("llm_claim evidence cannot be marked verified=True")
        return self


class TransitionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: TransitionAction
    from_status: TaskStatus | None = None
    to_status: TaskStatus | None = None
    reason: str = ""
    at: datetime = Field(default_factory=_utc_now)
    meta: dict[str, Any] = Field(default_factory=dict)


class TaskProgressRecord(BaseModel):
    """Mutable progress for one B4 task id."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=32)
    description: str = ""
    category: TaskCategory = "general"
    status: TaskStatus = "pending"
    depends_on: list[DependencyEdge] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    attempt_count: int = Field(default=0, ge=0)
    blocked_by: list[str] = Field(default_factory=list)
    failure_code: str | None = None
    failure_detail: str | None = None
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    transitions: list[TransitionRecord] = Field(default_factory=list)
    # Immutable snapshot fields from B4 (progress service must not rewrite B4).
    decomposition_task_fingerprint: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TaskProgressBoard(BaseModel):
    """Request-scoped progress board for one chat_with_ai run."""

    model_config = ConfigDict(extra="forbid")

    board_id: str = Field(min_length=1)
    request_id: str | None = None
    user_id: str | None = None
    project_id: str | None = None
    tenant_id: str | None = None
    decomposition_fingerprint: str = ""
    trusted_instruction: str = ""
    uncertain_dependency_policy: UncertainDependencyPolicy = "block"
    tasks: dict[str, TaskProgressRecord] = Field(default_factory=dict)
    execution_order: list[str] = Field(default_factory=list)
    unassigned_evidence: list[EvidenceRecord] = Field(default_factory=list)
    board_version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    # Persistence-ready metadata — not wired to Redis/Mongo in B5.
    persistence_ready: bool = True
    persisted: bool = False

    @model_validator(mode="after")
    def _validate_board(self) -> TaskProgressBoard:
        ids = list(self.tasks.keys())
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task IDs on board")
        id_set = set(ids)
        for tid, rec in self.tasks.items():
            if rec.task_id != tid:
                raise ValueError(
                    f"tasks map key {tid!r} does not match record.task_id"
                )
            for edge in rec.depends_on:
                if edge.predecessor_id not in id_set:
                    raise ValueError(
                        f"task {tid!r} missing dependency progress record "
                        f"for {edge.predecessor_id!r}"
                    )
                if edge.predecessor_id == tid:
                    raise ValueError(f"task {tid!r} cannot depend on itself")

        # Cycle detection on required+optional+uncertain edges alike.
        graph = {
            tid: [e.predecessor_id for e in rec.depends_on]
            for tid, rec in self.tasks.items()
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise ValueError(f"dependency cycle involving {node!r}")
            visiting.add(node)
            for pred in graph.get(node, []):
                visit(pred)
            visiting.remove(node)
            visited.add(node)

        for node in ids:
            visit(node)

        if self.execution_order and set(self.execution_order) != id_set:
            raise ValueError("execution_order must list each task id exactly once")
        return self
