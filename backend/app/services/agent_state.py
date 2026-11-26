"""Formal execution state for one agent run (Topic 33).

Mirrors the loop-local variables maintained inside ``chat_with_ai`` without
changing orchestration algorithms. Mutable containers use ``default_factory``
so each ``AgentState`` instance owns independent collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.investigation_state import (
    InvestigationBudgetManager,
    InvestigationChecklist,
)


@dataclass
class AgentState:
    """Execution state for a single ``chat_with_ai`` agent loop.

    Fields correspond to state already tracked in ``app/chatbot.py``.
    Investigation objects are optional at construction so unit tests can
    build a bare state, then attach checklist/budget when integrating.
    """

    # --- step / iteration limits ---
    current_step: int = 0
    max_steps: int = 0
    llm_iterations: int = 0

    # --- tool execution counters ---
    total_tool_executions: int = 0
    tool_execution_counts: dict[str, int] = field(default_factory=dict)
    tool_failure_counts: dict[str, int] = field(default_factory=dict)
    successful_tool_count: int = 0
    successful_searches: int = 0
    successful_reads: int = 0
    blocked_in_a_row: int = 0
    forced_target_reads: int = 0
    forced_continuation_count: int = 0

    # --- tool-call fingerprints ---
    executed_tool_fingerprints: set[str] = field(default_factory=set)
    failed_tool_fingerprints: set[str] = field(default_factory=set)
    committed_mutations: set[str] = field(default_factory=set)

    # --- investigation ---
    investigation_requirements: dict[str, Any] = field(default_factory=dict)
    investigation_evidence: dict[str, Any] = field(default_factory=dict)
    investigation_checklist: InvestigationChecklist | None = None
    budget_manager: InvestigationBudgetManager | None = None
    read_budget_total: int = 0
    tool_budget_total: int = 0
    searched_queries: set[str] = field(default_factory=set)

    # --- pending continuations ---
    pending_continuation: dict[str, Any] | None = None
    function_complete: bool = False

    # --- supporting loop state (from chat_with_ai) ---
    evidence_snippets: list[str] = field(default_factory=list)
    prior_read_args: list[dict[str, Any]] = field(default_factory=list)
    read_counts_by_file: dict[str, int] = field(default_factory=dict)
    files_already_read: set[str] = field(default_factory=set)
    terminal_file_paths: set[str] = field(default_factory=set)
    successful_tool_results: dict[str, str] = field(default_factory=dict)
    file_result_by_path: dict[str, str] = field(default_factory=dict)
    calculator_used: bool = False
    session_tokens_used: int = 0
    tpm_retries: int = 0

    # Phase B5 — request-scoped task progress (advisory; does not execute).
    # Typed as Any to avoid circular imports with validation models.
    task_progress: Any = None

    # Phase B6 — request-local active-task hint for evidence attribution.
    # Never shared across requests/users; does not grant tools.
    active_task_id: str | None = None
    # Last advisory OrchestrationDecision for this request (not persisted).
    orchestration_decision: Any = None
