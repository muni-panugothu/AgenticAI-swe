
from datetime import datetime
from pathlib import Path
import shutil

ROOT = Path.cwd()
SCHEMA_PATH = ROOT / "app" / "schemas" / "memory_schema.py"
SERVICE_PATH = ROOT / "app" / "services" / "memory_service.py"
CHATBOT_PATH = ROOT / "app" / "chatbot.py"
TESTS_DIR = ROOT / "tests"
LIFECYCLE_PATH = ROOT / "app" / "services" / "memory_lifecycle_service.py"
UNIT_TEST_PATH = TESTS_DIR / "test_memory_lifecycle_service.py"
INTEGRATION_TEST_PATH = TESTS_DIR / "test_memory_lifecycle_integration.py"

LIFECYCLE_SERVICE = r"""
# Stage 18: memory reinforcement and freshness-aware retrieval.
# Freshness affects ranking only. This module never deletes or archives memories.

from datetime import datetime, timezone
import math

HALF_LIFE_DAYS_BY_IMPORTANCE = {1: 7, 2: 30, 3: 90, 4: 180, 5: 365}
DEFAULT_FRESHNESS_SCORE = 0.5

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
"""

UNIT_TEST = r"""
import unittest
from datetime import datetime, timedelta, timezone

from app.services.memory_lifecycle_service import (
    calculate_memory_freshness_score,
    calculate_memory_priority_score,
)


class TestMemoryLifecycleScoring(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)

    def test_recent_memory_has_higher_freshness_than_old_memory(self):
        recent = {"importance": 3, "created_at": self.now - timedelta(days=2)}
        old = {"importance": 3, "created_at": self.now - timedelta(days=180)}
        self.assertGreater(
            calculate_memory_freshness_score(recent, now=self.now),
            calculate_memory_freshness_score(old, now=self.now),
        )

    def test_high_importance_memory_decays_more_slowly(self):
        age = timedelta(days=90)
        high = {"importance": 5, "created_at": self.now - age}
        low = {"importance": 1, "created_at": self.now - age}
        self.assertGreater(
            calculate_memory_freshness_score(high, now=self.now),
            calculate_memory_freshness_score(low, now=self.now),
        )

    def test_explicit_reinforcement_refreshes_freshness(self):
        old = {"importance": 3, "created_at": self.now - timedelta(days=180)}
        reaffirmed = {**old, "last_reinforced_at": self.now - timedelta(days=1)}
        self.assertGreater(
            calculate_memory_freshness_score(reaffirmed, now=self.now),
            calculate_memory_freshness_score(old, now=self.now),
        )

    def test_access_does_not_refresh_factual_freshness(self):
        old = {"importance": 3, "created_at": self.now - timedelta(days=180)}
        accessed = {**old, "last_accessed_at": self.now, "access_count": 100}
        self.assertEqual(
            calculate_memory_freshness_score(old, now=self.now),
            calculate_memory_freshness_score(accessed, now=self.now),
        )

    def test_future_timestamp_is_clamped_to_full_freshness(self):
        future = {"importance": 3, "created_at": self.now + timedelta(days=2)}
        self.assertEqual(
            calculate_memory_freshness_score(future, now=self.now), 1.0
        )

    def test_missing_timestamp_uses_neutral_default(self):
        self.assertEqual(
            calculate_memory_freshness_score({"importance": 3}, now=self.now),
            0.5,
        )

    def test_priority_score_is_bounded_between_zero_and_one(self):
        memory = {
            "score": 1.5,
            "importance": 99,
            "confidence": -5,
            "created_at": self.now,
        }
        score = calculate_memory_priority_score(memory, now=self.now)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
"""

INTEGRATION_TEST = r"""
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from bson import ObjectId

from app.services import memory_service


class TestMemoryLifecycleIntegration(unittest.TestCase):
    def test_record_memory_access_increments_counter_and_sets_timestamp(self):
        collection = Mock()
        memory_id = ObjectId()

        with patch.object(memory_service, "memory_collection", collection):
            memory_service._record_memory_access(
                "stage18-test-user", [{"memory_id": str(memory_id)}]
            )

        collection.update_many.assert_called_once()
        query, update = collection.update_many.call_args.args
        self.assertEqual(query["user_id"], "stage18-test-user")
        self.assertEqual(query["_id"]["$in"], [memory_id])
        self.assertEqual(update["$inc"], {"access_count": 1})
        self.assertIsInstance(update["$set"]["last_accessed_at"], datetime)

    def test_reinforce_memory_updates_target_users_active_memory(self):
        collection = Mock()
        collection.update_one.return_value.matched_count = 1
        memory_id = ObjectId()

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.reinforce_memory(
                "stage18-test-user", str(memory_id)
            )

        self.assertTrue(result["success"])
        query, update = collection.update_one.call_args.args
        self.assertEqual(query["_id"], memory_id)
        self.assertEqual(query["user_id"], "stage18-test-user")
        self.assertEqual(update["$inc"], {"reinforcement_count": 1})
        self.assertIsInstance(update["$set"]["last_reinforced_at"], datetime)

    def test_reinforcement_is_scoped_to_user(self):
        collection = Mock()
        collection.update_one.return_value.matched_count = 0
        memory_id = ObjectId()

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.reinforce_memory(
                "wrong-user", str(memory_id)
            )

        self.assertFalse(result["success"])
        self.assertEqual(
            result["error"], "Active memory not found for this user"
        )


if __name__ == "__main__":
    unittest.main()
"""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match for {label}; found {count}. "
            "No project files have been changed."
        )
    return source.replace(old, new, 1)


def main() -> int:
    required = [SCHEMA_PATH, SERVICE_PATH, CHATBOT_PATH, TESTS_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("Run this script from your backend folder. Missing:")
        for item in missing:
            print(f" - {item}")
        return 1

    new_files = [LIFECYCLE_PATH, UNIT_TEST_PATH, INTEGRATION_TEST_PATH]
    existing = [str(path) for path in new_files if path.exists()]
    if existing:
        print("Patch stopped: Stage 18 files already exist:")
        for item in existing:
            print(f" - {item}")
        print("No files were changed. Rename or move them before retrying.")
        return 1

    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    service = SERVICE_PATH.read_text(encoding="utf-8")
    chatbot = CHATBOT_PATH.read_text(encoding="utf-8")

    schema = replace_once(
        schema,
        '    version: int = Field(default=1, ge=1)\n',
        (
            '    version: int = Field(default=1, ge=1)\n'
            '\n'
            '    # Stage 18 lifecycle metadata\n'
            '    last_accessed_at: datetime | None = None\n'
            '    access_count: int = Field(default=0, ge=0)\n'
            '    last_reinforced_at: datetime | None = None\n'
            '    reinforcement_count: int = Field(default=0, ge=0)\n'
        ),
        "MemoryDocument.version",
    )

    service = replace_once(
        service,
        'from uuid import uuid4\n',
        (
            'from uuid import uuid4\n'
            '\n'
            'from app.services.memory_lifecycle_service import (\n'
            '    calculate_memory_freshness_score,\n'
            '    calculate_memory_priority_score,\n'
            ')\n'
        ),
        "memory_service imports",
    )

    service = replace_once(
        service,
        (
            '    superseded_by_memory_id=None,\n'
            '    version=1,\n'
            '):\n'
            '    try:\n'
            '        current_time = datetime.now(timezone.utc)\n'
        ),
        (
            '    superseded_by_memory_id=None,\n'
            '    version=1,\n'
            '    last_accessed_at=None,\n'
            '    access_count=0,\n'
            '    last_reinforced_at=None,\n'
            '    reinforcement_count=None,\n'
            '):\n'
            '    try:\n'
            '        current_time = datetime.now(timezone.utc)\n'
            '        if reinforcement_count is None:\n'
            '            reinforcement_count = 1 if status == "active" else 0\n'
            '        if last_reinforced_at is None and status == "active":\n'
            '            last_reinforced_at = current_time\n'
        ),
        "save_memory signature/defaults",
    )

    service = replace_once(
        service,
        (
            '            superseded_by_memory_id=superseded_by_memory_id,\n'
            '            version=version,\n'
            '        )\n'
        ),
        (
            '            superseded_by_memory_id=superseded_by_memory_id,\n'
            '            version=version,\n'
            '            last_accessed_at=last_accessed_at,\n'
            '            access_count=access_count,\n'
            '            last_reinforced_at=last_reinforced_at,\n'
            '            reinforcement_count=reinforcement_count,\n'
            '        )\n'
        ),
        "save_memory Pydantic mapping",
    )

    service = replace_once(
        service,
        '                "fact_key": 1,\n            },\n',
        (
            '                "fact_key": 1,\n'
            '                "last_accessed_at": 1,\n'
            '                "access_count": 1,\n'
            '                "last_reinforced_at": 1,\n'
            '                "reinforcement_count": 1,\n'
            '            },\n'
        ),
        "get_memories projection",
    )
    service = replace_once(
        service,
        (
            '                "fact_key": item.get("fact_key"),\n'
            '\n'
            '            })\n'
        ),
        (
            '                "fact_key": item.get("fact_key"),\n'
            '                "last_accessed_at": item.get("last_accessed_at"),\n'
            '                "access_count": item.get("access_count", 0),\n'
            '                "last_reinforced_at": item.get("last_reinforced_at"),\n'
            '                "reinforcement_count": item.get("reinforcement_count", 0),\n'
            '\n'
            '            })\n'
        ),
        "get_memories lifecycle fields",
    )

    semantic_marker = (
        '# ============================================================\n'
        '# SEMANTIC SEARCH MEMORIES\n'
        '# ============================================================\n'
    )
    access_helper = """def _record_memory_access(user_id, memories):
    # Track retrieval usage without changing factual freshness.
    try:
        from bson import ObjectId

        memory_ids = []
        for item in memories:
            try:
                memory_ids.append(ObjectId(item["memory_id"]))
            except (KeyError, TypeError, ValueError):
                continue

        if not memory_ids:
            return

        memory_collection.update_many(
            {
                "_id": {"$in": memory_ids},
                "user_id": user_id,
                "$or": [
                    {"status": "active"},
                    {"status": {"$exists": False}},
                ],
            },
            {
                "$inc": {"access_count": 1},
                "$set": {"last_accessed_at": datetime.now(timezone.utc)},
            },
        )
    except Exception as e:
        # Do not fail retrieval if telemetry tracking is unavailable.
        print(f"[Memory access tracking failed]: {e}")


"""
    service = replace_once(
        service,
        semantic_marker,
        access_helper + semantic_marker,
        "semantic search insertion point",
    )

    service = replace_once(
        service,
        (
            '                    "version": 1,\n'
            '                    "score": {\n'
        ),
        (
            '                    "version": 1,\n'
            '                    "last_accessed_at": 1,\n'
            '                    "access_count": 1,\n'
            '                    "last_reinforced_at": 1,\n'
            '                    "reinforcement_count": 1,\n'
            '                    "score": {\n'
        ),
        "semantic search projection",
    )
    service = replace_once(
        service,
        (
            '                "version": item.get(\n'
            '                    "version",\n'
            '                    1\n'
            '                ),\n'
            '            })\n'
        ),
        (
            '                "version": item.get(\n'
            '                    "version",\n'
            '                    1\n'
            '                ),\n'
            '                "last_accessed_at": item.get("last_accessed_at"),\n'
            '                "access_count": item.get("access_count", 0),\n'
            '                "last_reinforced_at": item.get("last_reinforced_at"),\n'
            '                "reinforcement_count": item.get("reinforcement_count", 0),\n'
            '            })\n'
        ),
        "semantic result lifecycle fields",
    )

    service = replace_once(
        service,
        (
            '        memories.sort(\n'
            '            key=lambda item: (\n'
            '                item["score"] * 0.75\n'
            '                + (item.get("importance", 3) / 5) * 0.15\n'
            '                + item.get("confidence", 0.75) * 0.10\n'
            '            ),\n'
            '            reverse=True,\n'
            '        )\n'
        ),
        (
            '        memories.sort(\n'
            '            key=calculate_memory_priority_score,\n'
            '            reverse=True,\n'
            '        )\n'
        ),
        "semantic ranking",
    )

    old_log = (
        '        memories = memories[:limit]\n'
        '        print("[Memory retrieval ranking]")\n'
        '        for item in memories:\n'
        '            weighted_score = (\n'
        '                item["score"] * 0.75\n'
        '                + (item.get(\'importance\', 3) / 5) * 0.15\n'
        '                + item.get(\'confidence\', 0.75) * 0.10\n'
        '            )\n'
        '            print(\n'
        '                f"Importance: {item.get(\'importance\', 3)}/5 | "\n'
        '                f"Confidence: {item.get(\'confidence\', 0.75):.2f} | "\n'
        '                f"Similarity: {item[\'score\']:.4f} | "\n'
        '                f"Weighted: {weighted_score:.4f} | "\n'
        '                f"Memory: {item[\'memory\']}"\n'
        '            )\n'
        '\n'
        '        return {"success": True, "memories": memories}\n'
    )
    new_log = (
        '        memories = memories[:limit]\n'
        '        _record_memory_access(user_id, memories)\n'
        '        print("[Memory retrieval ranking]")\n'
        '        for item in memories:\n'
        '            weighted_score = calculate_memory_priority_score(item)\n'
        '            freshness_score = calculate_memory_freshness_score(item)\n'
        '            print(\n'
        '                f"Importance: {item.get(\'importance\', 3)}/5 | "\n'
        '                f"Confidence: {item.get(\'confidence\', 0.75):.2f} | "\n'
        '                f"Similarity: {item[\'score\']:.4f} | "\n'
        '                f"Freshness: {freshness_score:.4f} | "\n'
        '                f"Weighted: {weighted_score:.4f} | "\n'
        '                f"Memory: {item[\'memory\']}"\n'
        '            )\n'
        '\n'
        '        return {"success": True, "memories": memories}\n'
    )
    service = replace_once(service, old_log, new_log, "semantic ranking log/access update")

    reinforce_function = """def reinforce_memory(user_id, memory_id):
    # Update only a matching active (or legacy active) memory.
    from bson import ObjectId

    try:
        current_time = datetime.now(timezone.utc)
        result = memory_collection.update_one(
            {
                "_id": ObjectId(memory_id),
                "user_id": user_id,
                "$or": [
                    {"status": "active"},
                    {"status": {"$exists": False}},
                ],
            },
            {
                "$inc": {"reinforcement_count": 1},
                "$set": {"last_reinforced_at": current_time},
            },
        )

        if result.matched_count == 0:
            return {
                "success": False,
                "error": "Active memory not found for this user",
            }

        return {
            "success": True,
            "memory_id": memory_id,
            "last_reinforced_at": current_time,
        }
    except Exception as e:
        print(f"[Memory reinforcement error]: {e}")
        return {"success": False, "error": str(e)}


"""
    importance_marker = (
        '# ============================================================\n'
        '# UPDATE MEMORY\n'
        '# ============================================================\n'
    )
    service = replace_once(
        service,
        importance_marker,
        reinforce_function + importance_marker,
        "reinforce_memory insertion point",
    )

    service = replace_once(
        service,
        (
            '                    "version": 1,\n'
            '                    "score": {"$meta": "vectorSearchScore"},\n'
        ),
        (
            '                    "version": 1,\n'
            '                    "last_accessed_at": 1,\n'
            '                    "access_count": 1,\n'
            '                    "last_reinforced_at": 1,\n'
            '                    "reinforcement_count": 1,\n'
            '                    "score": {"$meta": "vectorSearchScore"},\n'
        ),
        "duplicate candidate projection",
    )
    service = replace_once(
        service,
        (
            '                "version": item.get("version", 1),\n'
            '                "score": float(item.get("score", 0.0)),\n'
        ),
        (
            '                "version": item.get("version", 1),\n'
            '                "last_accessed_at": item.get("last_accessed_at"),\n'
            '                "access_count": item.get("access_count", 0),\n'
            '                "last_reinforced_at": item.get("last_reinforced_at"),\n'
            '                "reinforcement_count": item.get("reinforcement_count", 0),\n'
            '                "score": float(item.get("score", 0.0)),\n'
        ),
        "duplicate candidate lifecycle fields",
    )

    service = replace_once(
        service,
        (
            '                    supersedes_memory_id=str(old_object_id),\n'
            '                    version=next_version,\n'
            '                ).model_dump()\n'
        ),
        (
            '                    supersedes_memory_id=str(old_object_id),\n'
            '                    version=next_version,\n'
            '                    last_accessed_at=None,\n'
            '                    access_count=0,\n'
            '                    last_reinforced_at=current_time,\n'
            '                    reinforcement_count=int(existing.get("reinforcement_count", 0)) + 1,\n'
            '                ).model_dump()\n'
        ),
        "supersede_memory lifecycle fields",
    )

    chatbot = replace_once(
        chatbot,
        (
            '    update_memory_importance,\n'
            '    delete_memory,\n'
        ),
        (
            '    update_memory_importance,\n'
            '    reinforce_memory,\n'
            '    delete_memory,\n'
        ),
        "chatbot imports",
    )
    chatbot = replace_once(
        chatbot,
        (
            '        print("[Skipping duplicate memory save]")\n'
            '        return True\n'
        ),
        (
            '        reinforcement_result = reinforce_memory(\n'
            '            user_id=user_id,\n'
            '            memory_id=duplicate_result["memory_id"],\n'
            '        )\n'
            '        if reinforcement_result.get("success"):\n'
            '            print("[Memory explicitly reaffirmed]")\n'
            '        else:\n'
            '            print(\n'
            '                "[Memory reinforcement skipped]: "\n'
            '                f"{reinforcement_result.get(\'error\', \'Unknown error\')}"\n'
            '            )\n'
            '\n'
            '        print("[Skipping duplicate memory save]")\n'
            '        return True\n'
        ),
        "chatbot duplicate reinforcement",
    )

    compile(LIFECYCLE_SERVICE, str(LIFECYCLE_PATH), "exec")
    compile(UNIT_TEST, str(UNIT_TEST_PATH), "exec")
    compile(INTEGRATION_TEST, str(INTEGRATION_TEST_PATH), "exec")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    updates = {
        SCHEMA_PATH: schema,
        SERVICE_PATH: service,
        CHATBOT_PATH: chatbot,
    }

    # Keep rollback copies outside the Git repository so they are not staged.
    backup_dir = ROOT.parent.parent / f"{ROOT.parent.name}_stage18_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backups = []
    for path in updates:
        safe_name = path.relative_to(ROOT).as_posix().replace("/", "_")
        backup = backup_dir / f"{safe_name}__{timestamp}.bak"
        shutil.copy2(path, backup)
        backups.append(backup)

    for path, content in updates.items():
        path.write_text(content, encoding="utf-8")

    LIFECYCLE_PATH.write_text(LIFECYCLE_SERVICE, encoding="utf-8")
    UNIT_TEST_PATH.write_text(UNIT_TEST, encoding="utf-8")
    INTEGRATION_TEST_PATH.write_text(INTEGRATION_TEST, encoding="utf-8")

    print("Stage 18 patch applied successfully.")
    print("Backups:")
    for backup in backups:
        print(f" - {backup.relative_to(ROOT)}")
    print("New files:")
    for path in [LIFECYCLE_PATH, UNIT_TEST_PATH, INTEGRATION_TEST_PATH]:
        print(f" - {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Stage 18 patch aborted safely: {exc}")
        raise SystemExit(1)
