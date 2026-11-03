import unittest

from app.services.memory_conflict_service import (
    extract_exclusive_fact,
    find_conflicting_memories,
)


class TestMemoryConflictService(unittest.TestCase):
    def test_same_role_different_value_is_conflict(self):
        candidates = [
            {"memory_id": "1", "memory": "My role is Junior Developer"}
        ]
        conflicts = find_conflicting_memories(
            "My role is Software Engineer",
            candidates,
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["matched_fact_key"], "role")

    def test_same_fact_same_value_is_not_conflict(self):
        candidates = [
            {"memory_id": "1", "memory": "My preferred language is Python"}
        ]
        conflicts = find_conflicting_memories(
            "My preferred language is Python.",
            candidates,
        )
        self.assertEqual(conflicts, [])

    def test_learning_multiple_languages_is_not_conflict(self):
        self.assertIsNone(extract_exclusive_fact("I am learning JavaScript"))
        candidates = [
            {"memory_id": "1", "memory": "I am learning JavaScript"}
        ]
        self.assertEqual(
            find_conflicting_memories("I am learning Python", candidates),
            [],
        )

    def test_current_role_alias_matches_role(self):
        candidates = [
            {"memory_id": "1", "memory": "My role is Junior Developer"}
        ]
        conflicts = find_conflicting_memories(
            "My current role is Software Engineer",
            candidates,
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["matched_fact_key"], "role")


if __name__ == "__main__":
    unittest.main()
