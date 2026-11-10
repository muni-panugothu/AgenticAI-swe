
# Stage 18: memory reinforcement and freshness-aware retrieval.
# Freshness affects ranking and stale identification only.
# This module never deletes or archives memories automatically.

from datetime import datetime, timezone
import math

HALF_LIFE_DAYS_BY_IMPORTANCE = {1: 7, 2: 30, 3: 90, 4: 180, 5: 365}
DEFAULT_FRESHNESS_SCORE = 0.5

# A memory is flagged stale when freshness falls at or below this threshold.
STALE_FRESHNESS_THRESHOLD = 0.15

SEMANTIC_WEIGHT = 0.70
IMPORTANCE_WEIGHT = 0.15
CONFIDENCE_WEIGHT = 0.10
FRESHNESS_WEIGHT = 0.05


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _as_utc(value: datetime) -> datetime:
    # PyMongo may return naive UTC datetimes depending on client settings.
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def calculate_memory_freshness_score(
    memory: dict,
    now: datetime | None = None,
) -> float:
    # Explicit reaffirmation time takes precedence over original creation time.
    # Merely retrieving a memory does not refresh its factual freshness.
    if not isinstance(memory, dict):
        return DEFAULT_FRESHNESS_SCORE

    reference_time = memory.get("last_reinforced_at") or memory.get("created_at")
    if not isinstance(reference_time, datetime):
        return DEFAULT_FRESHNESS_SCORE

    current_time = _as_utc(now or datetime.now(timezone.utc))
    reference_time = _as_utc(reference_time)
    age_days = max(0.0, (current_time - reference_time).total_seconds() / 86400)

    try:
        importance = int(memory.get("importance", 3))
    except (TypeError, ValueError):
        importance = 3

    importance = max(1, min(5, importance))
    half_life_days = HALF_LIFE_DAYS_BY_IMPORTANCE[importance]
    score = math.pow(0.5, age_days / half_life_days)
    return _clamp(score, 0.0, 1.0)


def calculate_memory_priority_score(
    memory: dict,
    now: datetime | None = None,
) -> float:
    if not isinstance(memory, dict):
        return 0.0

    try:
        semantic = float(memory.get("score", 0.0))
    except (TypeError, ValueError):
        semantic = 0.0

    try:
        importance = float(memory.get("importance", 3)) / 5.0
    except (TypeError, ValueError):
        importance = 3.0 / 5.0

    try:
        confidence = float(memory.get("confidence", 0.75))
    except (TypeError, ValueError):
        confidence = 0.75

    semantic = _clamp(semantic, 0.0, 1.0)
    importance = _clamp(importance, 0.0, 1.0)
    confidence = _clamp(confidence, 0.0, 1.0)
    freshness = calculate_memory_freshness_score(memory, now=now)

    priority = (
        semantic * SEMANTIC_WEIGHT
        + importance * IMPORTANCE_WEIGHT
        + confidence * CONFIDENCE_WEIGHT
        + freshness * FRESHNESS_WEIGHT
    )
    return _clamp(priority, 0.0, 1.0)


def identify_stale_memory(
    memory: dict,
    now: datetime | None = None,
    *,
    threshold: float = STALE_FRESHNESS_THRESHOLD,
) -> dict:
    """Flag stale facts for review without deleting them."""
    freshness = calculate_memory_freshness_score(memory, now=now)
    is_stale = freshness <= threshold
    reason = None
    if is_stale:
        reason = (
            f"Freshness score {freshness:.4f} is at or below "
            f"threshold {threshold:.4f}"
        )

    return {
        "memory_id": memory.get("memory_id") or memory.get("_id"),
        "is_stale": is_stale,
        "freshness_score": freshness,
        "stale_reason": reason,
        "priority_score": calculate_memory_priority_score(memory, now=now),
    }


def annotate_memories_with_lifecycle(
    memories: list[dict],
    now: datetime | None = None,
) -> list[dict]:
    annotated = []
    for item in memories:
        if not isinstance(item, dict):
            continue
        enriched = dict(item)
        stale = identify_stale_memory(enriched, now=now)
        enriched["freshness_score"] = stale["freshness_score"]
        enriched["priority_score"] = stale["priority_score"]
        enriched["is_stale"] = stale["is_stale"]
        enriched["stale_reason"] = stale["stale_reason"]
        annotated.append(enriched)
    return annotated
