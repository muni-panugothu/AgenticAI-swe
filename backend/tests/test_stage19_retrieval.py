"""Stage 19 — Memory retrieval, relevance ranking, and deduplication tests.

Covers:
- Lexical relevance boost.
- Duplicate suppression in results.
- Priority score ranking.
- Safe deduplication.
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from app.services import memory_service
from app.services.memory_lifecycle_service import (
    calculate_memory_freshness_score,
    calculate_memory_priority_score,
)


class TestLexicalRelevanceBoost(unittest.TestCase):
    """Stage 19: Memories with zero lexical overlap should be penalized."""

    def test_full_overlap_returns_high_score(self):
        score = memory_service._lexical_relevance_boost(
            "my project", "my project is Alpha"
        )
        self.assertGreater(score, 0.0)

    def test_no_overlap_returns_zero(self):
        score = memory_service._lexical_relevance_boost(
            "my project", "my goal is to finish a marathon in under four hours"
        )
        self.assertEqual(score, 0.0)

    def test_partial_overlap(self):
        score = memory_service._lexical_relevance_boost(
            "project Alpha", "my project is Alpha"
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_query_returns_half(self):
        score = memory_service._lexical_relevance_boost(
            "the", "my project is Alpha"
        )
        # All words are stop words, so returns 0.5
        self.assertEqual(score, 0.5)


class TestDeduplicateResults(unittest.TestCase):
    """Stage 19: Textual duplicates in retrieval results should be suppressed."""

    def test_duplicate_suppression(self):
        now = datetime.now(timezone.utc)
        memories = [
            {
                "memory_id": "1",
                "memory": "my project is Alpha",
                "score": 0.95,
                "importance": 3,
                "confidence": 0.75,
                "created_at": now,
            },
            {
                "memory_id": "2",
                "memory": "My project is Alpha",
                "score": 0.93,
                "importance": 3,
                "confidence": 0.75,
                "created_at": now,
            },
        ]
        deduped = memory_service._deduplicate_results(memories)
        self.assertEqual(len(deduped), 1)

    def test_different_content_preserved(self):
        now = datetime.now(timezone.utc)
        memories = [
            {
                "memory_id": "1",
                "memory": "my project is Alpha",
                "score": 0.95,
                "importance": 3,
                "confidence": 0.75,
                "created_at": now,
            },
            {
                "memory_id": "2",
                "memory": "I prefer Python",
                "score": 0.90,
                "importance": 4,
                "confidence": 0.95,
                "created_at": now,
            },
        ]
        deduped = memory_service._deduplicate_results(memories)
        self.assertEqual(len(deduped), 2)


class TestPriorityRanking(unittest.TestCase):
    """Priority score should give reasonable ordering."""

    def setUp(self):
        self.now = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)

    def test_higher_importance_ranks_higher(self):
        high = {
            "score": 0.90,
            "importance": 5,
            "confidence": 0.95,
            "created_at": self.now,
        }
        low = {
            "score": 0.90,
            "importance": 1,
            "confidence": 0.95,
            "created_at": self.now,
        }
        self.assertGreater(
            calculate_memory_priority_score(high, now=self.now),
            calculate_memory_priority_score(low, now=self.now),
        )

    def test_higher_similarity_ranks_higher(self):
        high_sim = {
            "score": 0.99,
            "importance": 3,
            "confidence": 0.75,
            "created_at": self.now,
        }
        low_sim = {
            "score": 0.81,
            "importance": 3,
            "confidence": 0.75,
            "created_at": self.now,
        }
        self.assertGreater(
            calculate_memory_priority_score(high_sim, now=self.now),
            calculate_memory_priority_score(low_sim, now=self.now),
        )

    def test_stale_memory_has_lower_freshness(self):
        recent = {
            "importance": 3,
            "created_at": self.now - timedelta(days=1),
        }
        old = {
            "importance": 3,
            "created_at": self.now - timedelta(days=365),
        }
        self.assertGreater(
            calculate_memory_freshness_score(recent, now=self.now),
            calculate_memory_freshness_score(old, now=self.now),
        )


class TestSafeDeduplication(unittest.TestCase):
    """Stage 19: safe_deduplicate_memories must identify and optionally resolve."""

    def test_dry_run_identifies_duplicates(self):
        collection = MagicMock()
        now = datetime.now(timezone.utc)
        collection.find.return_value.sort.return_value.limit.return_value = iter([
            {
                "_id": "id1",
                "user_id": "test-user",
                "owner_id": "test-user",
                "permission_level": "private",
                "shared_with": [],
                "memory": "my project is Alpha",
                "status": "active",
                "updated_at": now,
                "created_at": now,
                "importance": 3,
                "confidence": 0.75,
            },
            {
                "_id": "id2",
                "user_id": "test-user",
                "owner_id": "test-user",
                "permission_level": "private",
                "shared_with": [],
                "memory": "My project is Alpha",
                "status": "active",
                "updated_at": now,
                "created_at": now,
                "importance": 3,
                "confidence": 0.75,
            },
        ])

        with patch.object(memory_service, "memory_collection", collection):
            result = memory_service.safe_deduplicate_memories(
                "test-user", dry_run=True
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["duplicate_groups"], 1)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["resolved"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
