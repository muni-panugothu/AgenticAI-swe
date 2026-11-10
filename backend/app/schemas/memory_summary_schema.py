"""Stage 19 — Memory summarization document schema.

Summaries live in a dedicated collection and never replace original memories.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SummaryType = Literal["user", "project", "preferences", "general"]
SummaryStatus = Literal["active", "outdated", "archived"]


class MemorySummaryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    summary_type: SummaryType
    # Topic/project isolation key, e.g. "listifys", "primary", "preferred_database".
    scope_key: str = Field(default="_global", min_length=1)
    summary_text: str = Field(min_length=1)
    source_memory_ids: list[str] = Field(default_factory=list)
    source_memory_count: int = Field(default=0, ge=0)
    source_hash: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    created_at: datetime
    updated_at: datetime
    status: SummaryStatus = "active"
    conflict_references: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Idempotency / lineage
    operation_id: str | None = None
    previous_summary_id: str | None = None
