import unittest

from app.services.memory_conflict_service import (
    detect_memory_conflict,
    extract_exclusive_fact,
    find_conflicting_memories,
    score_contradiction,
)


class TestAdvancedMemoryConflictIntelligence(unittest.TestCase):
    def test_paraphrased_role_is_exclusive(self):
        self.assertEqual(
            extract_exclusive_fact("I am a Software Engineer"),
            ("role", "software engineer"),
        )

    def test_company_paraphrase_is_exclusive(self):
        self.assertEqual(
            extract_exclusive_fact("I work at Acme Corp"),
            ("company", "acme corp"),
        )

    def test_conflict_includes_confidence_score(self):
        conflicts = find_conflicting_memories(
            "My role is Software Engineer",
            [{
                "memory_id": "1",
                "memory": "My role is Junior Developer",
                "score": 0.93,
            }],
        )
        self.assertEqual(len(conflicts), 1)
        self.assertGreaterEqual(conflicts[0]["conflict_confidence"], 0.82)

    def test_detect_memory_conflict_requires_confirmation(self):
        result = detect_memory_conflict(
            "I work as Backend Engineer",
            [{"memory_id": "1", "memory": "I work as Junior Developer"}],
        )
        self.assertTrue(result["has_conflict"])
        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["fact_key"], "role")

    def test_same_value_scores_zero(self):
        self.assertEqual(
            score_contradiction("role", "engineer", "role", "engineer"),
            0.0,
        )

    def test_learning_is_still_non_exclusive(self):
        self.assertIsNone(extract_exclusive_fact("I am learning Rust"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
