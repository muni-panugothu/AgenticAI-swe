"""Pydantic models for Phase B2 semantic request interpretation.

Interpretation is advisory metadata for routing and clarification.
It never authorizes tools, memory access, or RAG ACL by itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SemanticIntent = Literal[
    "general_knowledge",
    "current_information",
    "technical_question",
    "code_generation",
    "debugging",
    "code_review",
    "mathematics",
    "writing_documentation",
    "summarization",
    "data_analysis",
    "rag_document",
    "personal_memory",
    "tool_execution",
    "multi_step",
    "creative",
    "follow_up",
    "ambiguous",
    "unsupported",
]

ConfidenceLevel = Literal["high", "medium", "low"]

EntityType = Literal[
    "person",
    "place",
    "organization",
    "technology",
    "programming_language",
    "framework",
    "date_time",
    "number",
    "document",
    "file",
    "project",
]


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: EntityType
    value: str = Field(min_length=1)
    source_span: str = Field(
        default="",
        description="Exact substring from the interpreted text (not invented).",
    )
    start: int | None = None
    end: int | None = None


class InterpretedRequest(BaseModel):
    """Validated internal representation of one user request."""

    model_config = ConfigDict(extra="forbid")

    original_message: str = ""
    normalized_message: str = ""
    semantic_intent: SemanticIntent = "ambiguous"
    secondary_intents: list[SemanticIntent] = Field(default_factory=list)
    user_goal: str = ""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    requested_output: str = ""
    context_dependent: bool = False
    requires_fresh_info: bool = False
    expected_tools: list[str] = Field(default_factory=list)
    execution_intent: str = Field(
        default="general_conversation",
        description="Mapped legacy execution intent used by tool allowlists.",
    )
    confidence: ConfidenceLevel = "low"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_reason: str | None = None
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    unsupported: bool = False
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)
