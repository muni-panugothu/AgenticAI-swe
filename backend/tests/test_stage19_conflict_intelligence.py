"""Stage 19 — Advanced memory conflict intelligence tests.

Covers: fact extraction, normalization, contradiction detection,
paraphrase handling, project/goal/preference conflicts, multi-valued
vs exclusive classification, and deduplication.
"""

import unittest

from app.services.memory_conflict_service import (
    classify_duplicate,
    detect_memory_conflict,
    extract_exclusive_fact,
    find_conflicting_memories,
    find_duplicate_groups,
    normalize_fact_key,
    normalize_memory_text,
    score_contradiction,
)


class TestFactExtraction(unittest.TestCase):
    """Verify that all supported fact categories are correctly extracted."""

    # --- Role ---
    def test_role_direct(self):
        self.assertEqual(
            extract_exclusive_fact("My role is Software Engineer"),
            ("role", "software engineer"),
        )

    def test_role_current(self):
        self.assertEqual(
            extract_exclusive_fact("My current role is Tech Lead"),
            ("role", "tech lead"),
        )

    def test_role_job_title(self):
        self.assertEqual(
            extract_exclusive_fact("My job title is Senior Developer"),
            ("role", "senior developer"),
        )

    def test_role_paraphrase_work_as(self):
        self.assertEqual(
            extract_exclusive_fact("I work as a Backend Engineer"),
            ("role", "backend engineer"),
        )

    def test_role_paraphrase_i_am_a(self):
        self.assertEqual(
            extract_exclusive_fact("I am a Software Engineer"),
            ("role", "software engineer"),
        )

    # --- Company ---
    def test_company_direct(self):
        self.assertEqual(
            extract_exclusive_fact("My company is Google"),
            ("company", "google"),
        )

    def test_company_work_at(self):
        self.assertEqual(
            extract_exclusive_fact("I work at Acme Corp"),
            ("company", "acme corp"),
        )

    def test_company_employer_alias(self):
        self.assertEqual(
            extract_exclusive_fact("My employer is Microsoft"),
            ("company", "microsoft"),
        )

    # --- Project (Stage 19) ---
    def test_project_direct(self):
        self.assertEqual(
            extract_exclusive_fact("My project is Alpha"),
            ("primary_project", "alpha"),
        )

    def test_project_current(self):
        self.assertEqual(
            extract_exclusive_fact("My current project is Beta"),
            ("primary_project", "beta"),
        )

    def test_project_main(self):
        self.assertEqual(
            extract_exclusive_fact("My main project is Gamma"),
            ("primary_project", "gamma"),
        )

    def test_project_working_on(self):
        # Building/working-on statements are descriptions, not exclusive primaries.
        from app.services.memory_conflict_service import classify_project_fact
        result = classify_project_fact("I am working on Alpha")
        self.assertEqual(result, ("project_description", "alpha"))
        self.assertIsNone(extract_exclusive_fact("I am working on Alpha"))

    def test_project_building(self):
        from app.services.memory_conflict_service import classify_project_fact
        result = classify_project_fact("I am building a project called Phoenix")
        self.assertEqual(result[0], "project_name")
        self.assertIsNone(
            extract_exclusive_fact("I am building a project called Phoenix")
        )

    # --- Goal (Stage 19) ---
    def test_goal_direct(self):
        self.assertEqual(
            extract_exclusive_fact("My goal is to become a tech lead"),
            ("goal", "to become a tech lead"),
        )

    def test_goal_career(self):
        self.assertEqual(
            extract_exclusive_fact("My career goal is to become CTO"),
            ("goal", "to become cto"),
        )

    def test_goal_want_to_become(self):
        result = extract_exclusive_fact("I want to become a senior engineer")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "goal")

    # --- Preferred Language ---
    def test_preferred_language(self):
        self.assertEqual(
            extract_exclusive_fact("My preferred programming language is Python"),
            ("preferred_language", "python"),
        )

    def test_preferred_language_i_prefer(self):
        result = extract_exclusive_fact("I prefer Python")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "preferred_language")
        self.assertEqual(result[1], "python")

    def test_prefer_generic_non_language(self):
        result = extract_exclusive_fact("I prefer dark mode")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "preference")
        self.assertEqual(result[1], "dark mode")

    # --- Location ---
    def test_location_direct(self):
        self.assertEqual(
            extract_exclusive_fact("My location is San Francisco"),
            ("location", "san francisco"),
        )

    def test_location_live_in(self):
        self.assertEqual(
            extract_exclusive_fact("I live in New York"),
            ("location", "new york"),
        )

    # --- Non-exclusive ---
    def test_learning_is_non_exclusive(self):
        self.assertIsNone(extract_exclusive_fact("I am learning JavaScript"))
        self.assertIsNone(extract_exclusive_fact("I am learning Rust"))

    def test_hobby_is_non_exclusive(self):
        self.assertIsNone(extract_exclusive_fact("I enjoy hiking"))

    def test_empty_string(self):
        self.assertIsNone(extract_exclusive_fact(""))
        self.assertIsNone(extract_exclusive_fact("   "))

    def test_unrecognized_phrase(self):
        self.assertIsNone(extract_exclusive_fact("The weather is nice today"))


class TestProjectConflictDetection(unittest.TestCase):
    """Critical: Alpha vs Beta project conflict must be detected."""

    def test_alpha_vs_beta_project_conflict(self):
        candidates = [
            {"memory_id": "1", "memory": "My project is Alpha", "score": 0.90}
        ]
        conflicts = find_conflicting_memories(
            "My project is Beta",
            candidates,
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["matched_fact_key"], "primary_project")
        self.assertEqual(conflicts[0]["incoming_value"], "beta")
        self.assertEqual(conflicts[0]["existing_value"], "alpha")

    def test_same_project_is_not_conflict(self):
        candidates = [
            {"memory_id": "1", "memory": "My project is Alpha"}
        ]
        conflicts = find_conflicting_memories(
            "My project is Alpha.",
            candidates,
        )
        self.assertEqual(conflicts, [])

    def test_project_conflict_with_fact_key_metadata(self):
        candidates = [
            {
                "memory_id": "1",
                "memory": "My project is Alpha",
                "fact_key": "project",
                "normalized_value": "alpha",
                "score": 0.92,
            }
        ]
        result = detect_memory_conflict("My project is Beta", candidates)
        self.assertTrue(result["has_conflict"])
        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["fact_key"], "primary_project")

    def test_project_conflict_with_working_on_paraphrase(self):
        # Descriptions do not exclusive-conflict with a primary project name.
        candidates = [
            {"memory_id": "1", "memory": "My project is Alpha", "score": 0.88}
        ]
        conflicts = find_conflicting_memories(
            "I am working on Beta",
            candidates,
        )
        self.assertEqual(len(conflicts), 0)


class TestRoleChangeConflict(unittest.TestCase):
    def test_role_change_detected(self):
        candidates = [
            {"memory_id": "1", "memory": "My role is Junior Developer", "score": 0.93}
        ]
        conflicts = find_conflicting_memories(
            "My role is Software Engineer",
            candidates,
        )
        self.assertEqual(len(conflicts), 1)
        self.assertGreaterEqual(conflicts[0]["conflict_confidence"], 0.82)

    def test_company_change_detected(self):
        candidates = [
            {"memory_id": "1", "memory": "My company is Acme", "score": 0.91}
        ]
        conflicts = find_conflicting_memories(
            "My company is Google",
            candidates,
        )
        self.assertEqual(len(conflicts), 1)

    def test_preferred_language_change(self):
        candidates = [
            {"memory_id": "1", "memory": "My preferred language is Java"}
        ]
        conflicts = find_conflicting_memories(
            "My preferred language is Python",
            candidates,
        )
        self.assertEqual(len(conflicts), 1)


class TestContradictionScoring(unittest.TestCase):
    def test_different_keys_score_zero(self):
        self.assertEqual(
            score_contradiction("role", "engineer", "company", "google"), 0.0
        )

    def test_same_value_scores_zero(self):
        self.assertEqual(
            score_contradiction("role", "engineer", "role", "engineer"), 0.0
        )

    def test_different_values_same_key_scores_high(self):
        score = score_contradiction(
            "project", "alpha", "project", "beta", vector_score=0.90
        )
        self.assertGreaterEqual(score, 0.82)
        self.assertLessEqual(score, 1.0)

    def test_high_vector_score_boosts_confidence(self):
        low = score_contradiction("role", "a", "role", "b", vector_score=0.5)
        high = score_contradiction("role", "a", "role", "b", vector_score=0.95)
        self.assertGreater(high, low)


class TestNormalization(unittest.TestCase):
    def test_normalize_fact_key_aliases(self):
        self.assertEqual(normalize_fact_key("current role"), "role")
        self.assertEqual(normalize_fact_key("job title"), "role")
        self.assertEqual(normalize_fact_key("current project"), "primary_project")
        self.assertEqual(normalize_fact_key("main project"), "primary_project")
        self.assertEqual(normalize_fact_key("career goal"), "goal")

    def test_normalize_memory_text(self):
        self.assertEqual(
            normalize_memory_text("My project is Alpha."),
            "my project is alpha",
        )
        self.assertEqual(
            normalize_memory_text("  My Project Is Alpha!  "),
            "my project is alpha",
        )
        self.assertEqual(
            normalize_memory_text("My project is Alpha"),
            normalize_memory_text("my project is Alpha"),
        )


class TestDuplicateClassification(unittest.TestCase):
    def test_exact_duplicate(self):
        result = classify_duplicate(
            "My project is Alpha",
            "my project is Alpha.",
        )
        self.assertEqual(result["classification"], "exact_duplicate")
        self.assertEqual(result["confidence"], 1.0)

    def test_paraphrase_duplicate_same_fact(self):
        result = classify_duplicate(
            "My project is Alpha",
            "My current project is Alpha",
        )
        self.assertEqual(result["classification"], "paraphrase_duplicate")
        self.assertGreaterEqual(result["confidence"], 0.90)

    def test_contradiction_classified(self):
        result = classify_duplicate(
            "My project is Alpha",
            "My project is Beta",
            vector_score=0.90,
        )
        self.assertEqual(result["classification"], "possible_contradiction")
        self.assertGreater(result["confidence"], 0.0)

    def test_independent_facts(self):
        result = classify_duplicate(
            "My project is Alpha",
            "I am learning JavaScript",
            vector_score=0.3,
        )
        self.assertEqual(result["classification"], "independent")


class TestDuplicateGroups(unittest.TestCase):
    def test_finds_duplicate_group(self):
        memories = [
            {"memory_id": "1", "memory": "my project is Alpha"},
            {"memory_id": "2", "memory": "My project is Alpha"},
            {"memory_id": "3", "memory": "I am learning Python"},
        ]
        groups = find_duplicate_groups(memories)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)

    def test_no_duplicates_returns_empty(self):
        memories = [
            {"memory_id": "1", "memory": "My project is Alpha"},
            {"memory_id": "2", "memory": "I am learning Python"},
        ]
        groups = find_duplicate_groups(memories)
        self.assertEqual(len(groups), 0)


class TestLearningFacts(unittest.TestCase):
    def test_multiple_learning_no_conflict(self):
        candidates = [
            {"memory_id": "1", "memory": "I am learning JavaScript"}
        ]
        conflicts = find_conflicting_memories(
            "I am learning Python", candidates
        )
        self.assertEqual(len(conflicts), 0)

    def test_learning_does_not_conflict_with_project(self):
        candidates = [
            {"memory_id": "1", "memory": "My project is Alpha"}
        ]
        conflicts = find_conflicting_memories(
            "I am learning Python", candidates
        )
        self.assertEqual(len(conflicts), 0)


class TestAmbiguousFactExtraction(unittest.TestCase):
    def test_unstructured_sentence_not_extracted(self):
        self.assertIsNone(
            extract_exclusive_fact("I had a great day today")
        )

    def test_question_not_extracted(self):
        self.assertIsNone(
            extract_exclusive_fact("What is my project?")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
