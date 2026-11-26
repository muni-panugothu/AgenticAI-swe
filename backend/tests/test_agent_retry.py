"""Topic 38 — typed agent retry policy (RetryDecision)."""

from __future__ import annotations

import random
import unittest

from app.services import agent_orchestration as orch
from app.services.agent_retry import (
    ACTION_ABORT,
    ACTION_DEGRADE,
    ACTION_EXECUTE,
    ACTION_RETRY,
    ACTION_SKIP,
    DEFAULT_JITTER_RATIO,
    ERROR_COMMITTED_MUTATION,
    ERROR_RATE_AMBIGUOUS,
    ERROR_REPEATED_FAILURE,
    ERROR_RPD,
    ERROR_RPM,
    ERROR_TOKEN_LIMIT,
    ERROR_TOOL_CHOICE,
    ERROR_TPM,
    ERROR_TPD,
    PHASE_FINALIZATION,
    PHASE_LOOP,
    backoff_with_jitter,
    classify_provider_error,
    decide_llm_retry,
    decide_tool_execution,
    honored_retry_after_wait,
    parse_retry_after_hint_seconds,
)


def _tpm_exc(wait: str = "4.5525s") -> Exception:
    return Exception(
        "Error code: 429 - Rate limit reached for model on tokens per "
        f"minute (TPM): Limit 8000, Used 3433, Requested 5174. "
        f"Please try again in {wait}."
    )


def _tpd_exc() -> Exception:
    return Exception(
        "Error code: 429 - Rate limit reached for model on tokens per "
        "day (TPD): Limit 200000, Used 198569, Requested 4963. "
        "Please try again in 25m25.823999999s."
    )


def _rpd_exc() -> Exception:
    return Exception(
        "Error code: 429 - Rate limit reached for model on requests per "
        "day (RPD): Limit 14400, Used 14400, Requested 1. "
        "Please try again in 12h."
    )


def _ambiguous_429() -> Exception:
    return Exception(
        "Error code: 429 - {'error': {'message': 'Rate limit exceeded', "
        "'type': 'rate_limit_exceeded'}}"
    )


class TestClassifyProviderError(unittest.TestCase):
    def test_tpm_tpd_rpd_rpm(self):
        self.assertEqual(classify_provider_error(_tpm_exc()), ERROR_TPM)
        self.assertEqual(classify_provider_error(_tpd_exc()), ERROR_TPD)
        self.assertEqual(classify_provider_error(_rpd_exc()), ERROR_RPD)
        rpm = Exception(
            "Error code: 429 - requests per minute (RPM): try again in 1.2s."
        )
        self.assertEqual(classify_provider_error(rpm), ERROR_RPM)

    def test_ambiguous_429_not_tpm(self):
        self.assertIsNone(orch.classify_groq_rate_limit(_ambiguous_429()))
        self.assertEqual(
            classify_provider_error(_ambiguous_429()),
            ERROR_RATE_AMBIGUOUS,
        )
        decision = decide_llm_retry(
            exc=_ambiguous_429(),
            tpm_retries=0,
            apply_jitter=False,
        )
        self.assertNotEqual(decision.error_class, ERROR_TPM)
        self.assertNotEqual(decision.action, ACTION_RETRY)


class TestDecideLlmRetry(unittest.TestCase):
    def test_tpm_retries_while_budget_remains(self):
        decision = decide_llm_retry(
            exc=_tpm_exc("0.01s"),
            tpm_retries=0,
            tpm_retry_limit=2,
            apply_jitter=False,
            max_wait=30.0,
        )
        self.assertEqual(decision.action, ACTION_RETRY)
        self.assertEqual(decision.error_class, ERROR_TPM)
        expected = honored_retry_after_wait(0.01, max_wait=30.0)
        self.assertIsNotNone(expected)
        self.assertAlmostEqual(decision.wait_seconds, expected, places=4)
        self.assertLessEqual(decision.wait_seconds, 30.0)

    def test_tpm_exhausted_degrades_with_evidence(self):
        decision = decide_llm_retry(
            exc=_tpm_exc(),
            tpm_retries=2,
            tpm_retry_limit=2,
            tools_enabled=True,
            has_evidence=True,
            apply_jitter=False,
        )
        self.assertEqual(decision.action, ACTION_DEGRADE)
        self.assertEqual(decision.error_class, ERROR_TPM)

    def test_tpm_exhausted_aborts_without_evidence(self):
        decision = decide_llm_retry(
            exc=_tpm_exc(),
            tpm_retries=2,
            tpm_retry_limit=2,
            tools_enabled=False,
            has_evidence=False,
            apply_jitter=False,
        )
        self.assertEqual(decision.action, ACTION_ABORT)

    def test_tpd_and_rpd_never_retry(self):
        for exc, klass in ((_tpd_exc(), ERROR_TPD), (_rpd_exc(), ERROR_RPD)):
            with self.subTest(klass=klass):
                decision = decide_llm_retry(
                    exc=exc,
                    tpm_retries=0,
                    tools_enabled=True,
                    has_evidence=True,
                    apply_jitter=False,
                )
                self.assertEqual(decision.action, ACTION_DEGRADE)
                self.assertEqual(decision.error_class, klass)
                self.assertEqual(decision.wait_seconds, 0.0)

    def test_finalization_never_retries_tpm(self):
        decision = decide_llm_retry(
            exc=_tpm_exc(),
            tpm_retries=0,
            phase=PHASE_FINALIZATION,
            apply_jitter=False,
        )
        self.assertEqual(decision.action, ACTION_DEGRADE)
        self.assertEqual(decision.error_class, ERROR_TPM)

    def test_token_limit_degrades(self):
        exc = Exception("Error code: 413 - Request too large for context")
        decision = decide_llm_retry(
            exc=exc,
            tpm_retries=0,
            phase=PHASE_LOOP,
            apply_jitter=False,
        )
        self.assertEqual(decision.error_class, ERROR_TOKEN_LIMIT)
        self.assertEqual(decision.action, ACTION_DEGRADE)

    def test_tool_choice_clean_retry_when_tools_disabled(self):
        exc = Exception("Error code: 400 - tool choice is none but tool_use")
        decision = decide_llm_retry(
            exc=exc,
            tpm_retries=0,
            tools_enabled=False,
            phase=PHASE_LOOP,
            apply_jitter=False,
        )
        self.assertEqual(decision.action, ACTION_RETRY)
        self.assertEqual(decision.error_class, ERROR_TOOL_CHOICE)

    def test_tool_choice_finalize_when_tools_enabled(self):
        exc = Exception("Error code: 400 - tool choice is none but tool_use")
        decision = decide_llm_retry(
            exc=exc,
            tpm_retries=0,
            tools_enabled=True,
            has_evidence=True,
            phase=PHASE_LOOP,
            apply_jitter=False,
        )
        self.assertEqual(decision.action, ACTION_DEGRADE)
        self.assertEqual(decision.error_class, ERROR_TOOL_CHOICE)

    def test_retry_after_exactly_30_honors_and_retries(self):
        exc = _tpm_exc("30s")
        self.assertEqual(parse_retry_after_hint_seconds(exc), 30.0)
        floor = honored_retry_after_wait(30.0, max_wait=30.0)
        self.assertEqual(floor, 30.0)
        decision = decide_llm_retry(
            exc=exc,
            tpm_retries=0,
            max_wait=30.0,
            apply_jitter=True,
            jitter_ratio=0.5,
            rng=random.Random(1),
        )
        self.assertEqual(decision.action, ACTION_RETRY)
        self.assertEqual(decision.error_class, ERROR_TPM)
        self.assertEqual(decision.wait_seconds, 30.0)

    def test_retry_after_31_does_not_retry_early(self):
        exc = _tpm_exc("31s")
        self.assertEqual(parse_retry_after_hint_seconds(exc), 31.0)
        self.assertIsNone(honored_retry_after_wait(31.0, max_wait=30.0))
        for tools, evidence, expected in (
            (True, True, ACTION_DEGRADE),
            (False, False, ACTION_ABORT),
        ):
            with self.subTest(tools=tools, evidence=evidence):
                decision = decide_llm_retry(
                    exc=exc,
                    tpm_retries=0,
                    tpm_retry_limit=2,
                    tools_enabled=tools,
                    has_evidence=evidence,
                    max_wait=30.0,
                    apply_jitter=True,
                )
                self.assertEqual(decision.action, expected)
                self.assertEqual(decision.error_class, ERROR_TPM)
                self.assertEqual(decision.wait_seconds, 0.0)
                self.assertEqual(
                    decision.reason, "tpm_retry_after_exceeds_max_wait"
                )

    def test_retry_after_60_does_not_retry_early(self):
        exc = _tpm_exc("60s")
        self.assertEqual(parse_retry_after_hint_seconds(exc), 60.0)
        decision = decide_llm_retry(
            exc=exc,
            tpm_retries=0,
            tpm_retry_limit=2,
            tools_enabled=True,
            has_evidence=True,
            max_wait=30.0,
            apply_jitter=True,
        )
        self.assertEqual(decision.action, ACTION_DEGRADE)
        self.assertEqual(decision.wait_seconds, 0.0)
        self.assertNotEqual(decision.action, ACTION_RETRY)

    def test_jittered_tpm_wait_never_below_retry_after(self):
        exc = _tpm_exc("4.5525s")
        floor = honored_retry_after_wait(4.5525, max_wait=30.0)
        self.assertIsNotNone(floor)
        rng = random.Random(99)
        for _ in range(40):
            decision = decide_llm_retry(
                exc=exc,
                tpm_retries=0,
                tpm_retry_limit=2,
                max_wait=30.0,
                apply_jitter=True,
                jitter_ratio=DEFAULT_JITTER_RATIO,
                rng=rng,
            )
            self.assertEqual(decision.action, ACTION_RETRY)
            self.assertEqual(decision.error_class, ERROR_TPM)
            self.assertGreaterEqual(decision.wait_seconds, floor - 1e-9)
            self.assertGreaterEqual(decision.wait_seconds, 4.5525 - 1e-9)
            self.assertLessEqual(decision.wait_seconds, 30.0)


class TestDecideToolExecution(unittest.TestCase):
    def test_committed_mutation_skips(self):
        decision = decide_tool_execution(
            tool_name="save_memory",
            fingerprint="save_memory:abc",
            committed_mutations={"save_memory:abc"},
            failed_fingerprints=set(),
            mutating_tools={"save_memory"},
        )
        self.assertEqual(decision.action, ACTION_SKIP)
        self.assertEqual(decision.error_class, ERROR_COMMITTED_MUTATION)

    def test_repeated_failure_skips(self):
        decision = decide_tool_execution(
            tool_name="read_file",
            fingerprint="read_file:xyz",
            committed_mutations=set(),
            failed_fingerprints={"read_file:xyz"},
            mutating_tools={"save_memory"},
        )
        self.assertEqual(decision.action, ACTION_SKIP)
        self.assertEqual(decision.error_class, ERROR_REPEATED_FAILURE)

    def test_ok_to_execute(self):
        decision = decide_tool_execution(
            tool_name="search_code",
            fingerprint="search_code:q",
            committed_mutations=set(),
            failed_fingerprints=set(),
            mutating_tools={"save_memory"},
        )
        self.assertEqual(decision.action, ACTION_EXECUTE)

    def test_repeated_failure_precedence_over_committed(self):
        fp = "save_memory:dup"
        decision = decide_tool_execution(
            tool_name="save_memory",
            fingerprint=fp,
            committed_mutations={fp},
            failed_fingerprints={fp},
            mutating_tools={"save_memory"},
        )
        self.assertEqual(decision.error_class, ERROR_REPEATED_FAILURE)


class TestBackoffJitter(unittest.TestCase):
    def test_positive_jitter_never_below_retry_after_floor(self):
        base = 10.0
        ratio = DEFAULT_JITTER_RATIO
        rng = random.Random(42)
        samples = [
            backoff_with_jitter(
                base,
                attempt=0,
                max_wait=30.0,
                jitter_ratio=ratio,
                rng=rng,
            )
            for _ in range(50)
        ]
        high = base * (1.0 + ratio)
        for wait in samples:
            self.assertGreaterEqual(wait, base - 1e-9)
            self.assertLessEqual(wait, min(30.0, high) + 1e-9)

    def test_jitter_never_exceeds_max_wait(self):
        rng = random.Random(7)
        for _ in range(30):
            wait = backoff_with_jitter(
                28.0,
                max_wait=30.0,
                jitter_ratio=0.2,
                rng=rng,
            )
            self.assertGreaterEqual(wait, 28.0 - 1e-9)
            self.assertLessEqual(wait, 30.0 + 1e-9)

    def test_max_wait_produces_no_extra_delay(self):
        rng = random.Random(3)
        for _ in range(20):
            wait = backoff_with_jitter(
                30.0,
                max_wait=30.0,
                jitter_ratio=0.5,
                rng=rng,
            )
            self.assertEqual(wait, 30.0)
        # Pre-capped oversize base also stays at the maximum.
        self.assertEqual(
            backoff_with_jitter(90.0, max_wait=30.0, jitter_ratio=0.5),
            30.0,
        )

    def test_zero_jitter_returns_base(self):
        self.assertEqual(
            backoff_with_jitter(5.5, jitter_ratio=0.0, max_wait=30.0),
            5.5,
        )


if __name__ == "__main__":
    unittest.main()
