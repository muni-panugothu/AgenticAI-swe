"""Trusted server-authored agent turn result models (Topic 41a).

These models describe the outcome of one chat turn for API mapping and
future SSE serialization. They are never filled by the LLM and must not
carry identity, ACL, tool-allowlist, or B5/B6 mutation fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


AgentResponseType = Literal[
    "answer",
    "clarification",
    "partial",
    "abstain",
    "error",
]

CompletionStatus = Literal[
    "complete",
    "partial",
    "clarification_needed",
    "aborted",
    "failed",
]


class CitationRef(BaseModel):
    """Server-trusted citation reference (from RAG retrieval, not the LLM)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    document_id: str = Field(min_length=1, max_length=128)
    source: str | None = None
    section: str | None = None
    chunk_index: int = Field(default=0, ge=0)
    excerpt: str = Field(default="", max_length=4000)
    score: float = 0.0


class AgentTurnResult(BaseModel):
    """Trusted outcome of one agent turn.

    ``reply`` is the user-facing text already produced by the existing agent.
    Classification fields are server-authored metadata only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reply: str = Field(min_length=0, max_length=200_000)
    response_type: AgentResponseType = "answer"
    completion_status: CompletionStatus = "complete"
    uncertain: bool = False
    clarification_required: bool = False
    partial: bool = False
    citations: list[CitationRef] = Field(default_factory=list)
    error_code: str | None = Field(default=None, max_length=64)
    error_message_safe: str | None = Field(default=None, max_length=500)

    @field_validator("error_code", "error_message_safe")
    @classmethod
    def _empty_str_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, value: Any) -> Any:
        if value is None:
            return []
        return value
