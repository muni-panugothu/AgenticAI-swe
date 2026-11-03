from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class MemoryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    memory: str = Field(min_length=1)
    embedding: list[float] | None = None

    importance: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    created_at: datetime
    updated_at: datetime

    # Stage 17 metadata
    status: Literal["active", "needs_review", "superseded", "resolved"] = "active"
    fact_key: str | None = None
    conflict_type: Literal["none", "possible_contradiction", "confirmed_update"] = "none"
    conflict_group_id: str | None = None
    conflict_with_memory_id: str | None = None
    supersedes_memory_id: str | None = None
    superseded_by_memory_id: str | None = None
    version: int = Field(default=1, ge=1)