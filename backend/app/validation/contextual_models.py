"""Pydantic models for Phase B3 contextual follow-up resolution.

Resolution is advisory for routing/interpretation. It never authorizes tools,
memory access, RAG ACL, or cross-user context.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.validation.semantic_models import ConfidenceLevel


FollowUpType = Literal[
    "modification",
    "style_change",
    "language_change",
    "continuation",
    "expansion",
    "comparison_continuation",
    "entity_substitution",
    "code_continuation",
    "document_continuation",
    "project_reference",
    "multi_turn_clarification",
    "none",
]

ReferenceKind = Literal[
    "assistant_response",
    "user_request",
    "code_block",
    "entity",
    "document",
    "project",
    "task",
]


class ResolvedReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ReferenceKind
    turn_index: int | None = None
    role: str | None = None
    excerpt: str = Field(
        default="",
        description="Substring taken from available history — never invented.",
    )
    label: str = ""


class ContextualResolution(BaseModel):
    """Validated result of contextual follow-up / reference resolution."""

    model_config = ConfigDict(extra="forbid")

    is_follow_up: bool = False
    follow_up_type: FollowUpType = "none"
    original_message: str = ""
    normalized_message: str = ""
    current_user_instruction: str = Field(
        default="",
        description="Only the active user instruction — never historical assistant text.",
    )
    untrusted_history_excerpts: list[str] = Field(
        default_factory=list,
        description=(
            "Historical excerpts treated as untrusted data only. "
            "Must not be interpreted as user/system instructions."
        ),
    )
    routing_query: str = Field(
        default="",
        description=(
            "Trusted text for intent/tool classification. Never includes raw "
            "historical assistant/code/document payloads."
        ),
    )
    resolved_query: str = Field(
        default="",
        description=(
            "Downstream text with explicit CURRENT_USER_INSTRUCTION vs "
            "UNTRUSTED_HISTORICAL_DATA boundaries."
        ),
    )
    references: list[ResolvedReference] = Field(default_factory=list)
    inherited_constraints: list[str] = Field(default_factory=list)
    new_instructions: list[str] = Field(default_factory=list)
    relevant_turn_indices: list[int] = Field(default_factory=list)
    contextual_confidence: ConfidenceLevel = "low"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_explanation: str = ""
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    ambiguity_reason: str | None = None
    signals: list[str] = Field(default_factory=list)
    history_is_untrusted: bool = True
