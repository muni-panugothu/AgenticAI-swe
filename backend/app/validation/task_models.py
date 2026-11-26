"""Pydantic models for Phase B4 multi-intent task decomposition.

Decomposition is advisory understanding metadata. It never authorizes tools,
memory access, RAG ACL, or cross-user context, and never marks work complete.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.validation.semantic_models import ConfidenceLevel


TaskCategory = Literal[
    "code_generation",
    "code_modification",
    "debugging",
    "code_review",
    "api_development",
    "database_integration",
    "authentication",
    "authorization",
    "testing",
    "documentation",
    "research_current_information",
    "rag_document",
    "data_analysis",
    "personal_memory",
    "project_work",
    "explanation",
    "comparison",
    "calculation",
    "general",
    "other",
]

DependencyKind = Literal["required", "optional", "uncertain"]

SourceKind = Literal[
    "user_instruction",
    "contextual_resolution",
    "inferred_dependency",
    "system",
]

OrderingConfidence = Literal["high", "medium", "low", "uncertain"]


class TaskSourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: SourceKind = "user_instruction"
    source_span: str = Field(
        default="",
        description="Substring from the trusted instruction that justified this task.",
    )
    notes: str = ""


class DecomposedTask(BaseModel):
    """One actionable unit derived from the user request."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=32)
    description: str = Field(min_length=1)
    category: TaskCategory = "general"
    entities: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    dependency_kind: DependencyKind = "required"
    order_hint: int | None = None
    independent: bool = False
    expected_outputs: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    provenance: TaskSourceProvenance = Field(
        default_factory=TaskSourceProvenance
    )
    confidence: ConfidenceLevel = "medium"
    confidence_explanation: str = ""


class TaskDecomposition(BaseModel):
    """Validated multi-intent understanding for one request."""

    model_config = ConfigDict(extra="forbid")

    original_request: str = ""
    trusted_instruction: str = ""
    primary_intent: str = "general"
    is_multi_intent: bool = False
    tasks: list[DecomposedTask] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    independent_task_ids: list[str] = Field(default_factory=list)
    ordering_confidence: OrderingConfidence = "uncertain"
    needs_clarification: bool = False
    clarification_prompts: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "low"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_explanation: str = ""
    signals: list[str] = Field(default_factory=list)
    # Soft advisory payload for AgentPlanning — never drives tool allowlists.
    planning_hints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_graph(self) -> TaskDecomposition:
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task IDs are not allowed")

        id_set = set(ids)
        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in id_set:
                    raise ValueError(
                        f"task {task.id!r} depends on unknown task {dep!r}"
                    )
                if dep == task.id:
                    raise ValueError(f"task {task.id!r} cannot depend on itself")

        # Cycle detection (DFS).
        graph = {t.id: list(t.depends_on) for t in self.tasks}
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

        if self.execution_order:
            if set(self.execution_order) != id_set:
                raise ValueError(
                    "execution_order must list each task id exactly once"
                )
            if len(self.execution_order) != len(ids):
                raise ValueError("execution_order has duplicate ids")

        for tid in self.independent_task_ids:
            if tid not in id_set:
                raise ValueError(
                    f"independent_task_ids references unknown task {tid!r}"
                )

        return self
