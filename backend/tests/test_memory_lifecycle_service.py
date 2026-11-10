import unittest
from datetime import datetime, timedelta, timezone

from app.services.memory_lifecycle_service import (
    calculate_memory_freshness_score,
    calculate_memory_priority_score,
    identify_stale_memory,
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

    def test_stale_identification_does_not_delete(self):
        old = {
            "memory_id": "m1",
            "importance": 1,
            "created_at": self.now - timedelta(days=365),
        }
        result = identify_stale_memory(old, now=self.now)
        self.assertTrue(result["is_stale"])
        self.assertIn("Freshness score", result["stale_reason"])
        # Identification only — no mutation of the source document.
        self.assertNotIn("status", old)


if __name__ == "__main__":
    unittest.main()
