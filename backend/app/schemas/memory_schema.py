from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


MemoryStatus = Literal[
    "active",
    "needs_review",
    "superseded",
    "resolved",
    "forgotten",
    "deduplicated",
]

ConflictType = Literal[
    "none",
    "possible_contradiction",
    "confirmed_update",
    "exact_duplicate",
    "paraphrase_duplicate",
]

# Topic 20 — memory scope kinds.
MemoryScope = Literal["user", "project", "shared"]

# Topic 21 — permission levels.
# private:       owner only
# project_read:  same project can read; owner writes
# project_write: same project can read/write
# public_read:   same tenant can read; owner writes
PermissionLevel = Literal[
    "private",
    "project_read",
    "project_write",
    "public_read",
]


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
    status: MemoryStatus = "active"
    fact_key: str | None = None
    normalized_value: str | None = None
    conflict_type: ConflictType = "none"
    conflict_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    conflict_group_id: str | None = None
    conflict_with_memory_id: str | None = None
    supersedes_memory_id: str | None = None
    superseded_by_memory_id: str | None = None
    version: int = Field(default=1, ge=1)

    # Stage 18 lifecycle metadata
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    last_reinforced_at: datetime | None = None
    reinforcement_count: int = Field(default=0, ge=0)
    is_stale: bool = False
    stale_reason: str | None = None

    # Production reliability
    operation_id: str | None = None
    # Request-scoped reinforce operation IDs applied to this document.
    # Backward-compatible optional field used for concurrent idempotency.
    last_reinforce_operation_id: str | None = None
    reinforce_operation_ids: list[str] = Field(default_factory=list)

    # Topic 20 — scoping (optional; missing = legacy user-scoped record)
    project_id: str | None = None
    tenant_id: str | None = None
    scope: MemoryScope = "user"

    # Topic 21 — permissions (defaults preserve legacy private-owner behavior)
    owner_id: str | None = None
    shared_with: list[str] = Field(default_factory=list)
    permission_level: PermissionLevel = "private"


class MemoryAuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    memory_id: str | None = None
    event_type: str = Field(min_length=1)
    operation_id: str | None = None
    details: dict = Field(default_factory=dict)
    created_at: datetime
