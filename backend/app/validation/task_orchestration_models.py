"""Pydantic models for Phase B6 task-aware execution orchestration.

Orchestration decisions are advisory coordination data only. They never grant
tools, widen allowlists, execute work, or mark B5 tasks complete.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


OrchestrationStatus = Literal[
    "no_board",
    "no_eligible",
    "candidate_selected",
    "plan_blocks",
    "clarification_required",
    "unsupported_path",
    "budget_exhausted",
    "partial_finalize",
    "abort",
]


class OrchestrationDecision(BaseModel):
    """One advisory coordination decision for the existing agent loop.

    Does not authorize tools, mutate B5 status by itself, or replace Topic 35.
    """

    model_config = ConfigDict(extra="forbid")

    status: OrchestrationStatus
    selected_task_id: str | None = None
    eligible_task_ids: list[str] = Field(default_factory=list)
    selection_reason: str = ""
    plan_action: str | None = None
    plan_compatible: bool = False
    # True when the existing loop may treat selected_task_id as the active hint.
    # Does NOT mean the task has been started or that a tool may run.
    execution_may_proceed: bool = False
    clarification_required: bool = False
    partial_finalize_recommended: bool = False
    block_reason: str | None = None
    active_task_hint: str | None = None
    unsupported_category: str | None = None
    incomplete_task_ids: list[str] = Field(default_factory=list)
