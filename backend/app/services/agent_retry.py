"""Typed LLM / tool retry policy (Topic 38).

Thin wrappers around existing Groq classifiers and loop protections.
Does not invent new tool-execution retries — tool policy only encodes
skip/execute decisions already enforced in ``chat_with_ai``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Collection

from app.services.agent_orchestration import (
    TPM_RETRY_LIMIT,
    classify_groq_rate_limit,
    is_daily_quota_exhausted,
    is_rate_limit_error,
    is_token_limit_error,
    is_tool_choice_error,
)

# Small pad added to honored Retry-After waits (same as extract_retry_after_seconds).
_RETRY_AFTER_PAD_SECONDS = 0.35


# --- actions ---
ACTION_RETRY = "retry"
ACTION_ABORT = "abort"
ACTION_DEGRADE = "degrade_offline"
ACTION_SKIP = "skip_duplicate"
ACTION_EXECUTE = "execute"

# --- error / decision classes (deterministic labels) ---
ERROR_TPM = "tpm"
ERROR_TPD = "tpd"
ERROR_RPD = "rpd"
ERROR_RPM = "rpm"
ERROR_RATE_AMBIGUOUS = "rate_limit_ambiguous"
ERROR_TOKEN_LIMIT = "token_limit"
ERROR_TOOL_CHOICE = "tool_choice"
ERROR_UNKNOWN = "unknown"
ERROR_COMMITTED_MUTATION = "committed_mutation"
ERROR_REPEATED_FAILURE = "repeated_failed_tool"
ERROR_NONE = "none"

PHASE_LOOP = "loop"
PHASE_FINALIZATION = "finalization"

DEFAULT_JITTER_RATIO = 0.1
DEFAULT_MAX_WAIT = 30.0


@dataclass
class RetryDecision:
    """Policy outcome for one provider or tool-execution decision."""

    action: str
    error_class: str
    wait_seconds: float = 0.0
    reason: str = ""


def classify_provider_error(exc: BaseException) -> str:
    """Map an exception to a stable error class using existing classifiers."""
    kind = classify_groq_rate_limit(exc)
    if kind in {"tpm", "tpd", "rpm", "rpd"}:
        return str(kind)
    if is_daily_quota_exhausted(exc):
        # Defensive: classifiers already cover tpd/rpd.
        return ERROR_TPD
    if is_token_limit_error(exc):
        return ERROR_TOKEN_LIMIT
    if is_tool_choice_error(exc):
        return ERROR_TOOL_CHOICE
    if is_rate_limit_error(exc):
        # Ambiguous 429 — never treat as TPM.
        return ERROR_RATE_AMBIGUOUS
    return ERROR_UNKNOWN


def parse_retry_after_hint_seconds(
    exc: BaseException,
    default: float = 5.0,
) -> float:
    """Return the server's Retry-After hint in seconds (uncapped, unpadded).

    Parses the same Groq ``try again in …`` forms as
    ``extract_retry_after_seconds``, but does **not** apply the 30s sleep cap
    or the +0.35 pad — callers compare this to ``max_wait`` to avoid early
    retries.
    """
    text = str(exc)
    match_m = re.search(
        r"try again in\s+([0-9]+)\s*m\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        text,
        flags=re.IGNORECASE,
    )
    if match_m:
        try:
            return int(match_m.group(1)) * 60 + float(match_m.group(2))
        except ValueError:
            pass
    match = re.search(
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return float(default)
    try:
        return float(match.group(1))
    except ValueError:
        return float(default)


def honored_retry_after_wait(
    hint_seconds: float,
    *,
    max_wait: float = DEFAULT_MAX_WAIT,
) -> float | None:
    """Return the sleep floor for a Retry-After hint, or ``None`` if too long.

    - hint <= max_wait: at least the requested duration, plus pad when it
      still fits under ``max_wait`` (never above the cap).
    - hint > max_wait: ``None`` — caller must not retry early.
    """
    cap = max(0.0, float(max_wait))
    hint = max(0.0, float(hint_seconds))
    if hint > cap:
        return None
    padded = hint + _RETRY_AFTER_PAD_SECONDS
    return min(cap, padded)


def backoff_with_jitter(
    base_seconds: float,
    *,
    attempt: int = 0,
    max_wait: float = DEFAULT_MAX_WAIT,
    jitter_ratio: float = DEFAULT_JITTER_RATIO,
    rng: random.Random | None = None,
) -> float:
    """Add optional **positive-only** jitter above a Retry-After minimum.

    ``base_seconds`` is the effective Retry-After floor (already honored and
    within ``max_wait``). This helper:

    - never returns less than ``min(base_seconds, max_wait)``;
    - never returns more than ``max_wait`` (default 30s);
    - adds no jitter when the floor already equals ``max_wait``.

    ``attempt`` is accepted for API stability; wait is not grown
    exponentially — matching today's TPM Retry-After policy.
    """
    del attempt  # reserved; base wait already encodes Retry-After
    cap = max(0.0, float(max_wait))
    floor = min(cap, max(0.0, float(base_seconds)))
    ratio = float(jitter_ratio)
    # Already at the maximum: do not add extra delay.
    if floor >= cap or ratio <= 0.0 or floor == 0.0:
        return floor
    headroom = cap - floor
    span = min(floor * ratio, headroom)
    if span <= 0.0:
        return floor
    picker = rng if rng is not None else random
    return floor + picker.uniform(0.0, span)


def decide_llm_retry(
    *,
    exc: BaseException,
    tpm_retries: int,
    tpm_retry_limit: int | None = None,
    tools_enabled: bool = False,
    has_evidence: bool = False,
    phase: str = PHASE_LOOP,
    max_wait: float = DEFAULT_MAX_WAIT,
    apply_jitter: bool = True,
    jitter_ratio: float = DEFAULT_JITTER_RATIO,
    rng: random.Random | None = None,
) -> RetryDecision:
    """Decide retry / abort / offline degrade for a Groq LLM exception.

    Preserves existing policy:
    - TPD/RPD / daily quota → degrade (never short-retry).
    - TPM → retry while ``tpm_retries < limit`` **and** Retry-After <=
      ``max_wait``; longer Retry-After → degrade/abort (never sleep-cap early).
    - Ambiguous 429 → not TPM; degrade if evidence else abort.
    - Token limit → degrade/finalize path (no TPM sleep).
    - Tool-choice 400 → one clean tool-free retry only when tools disabled
      in the loop phase; otherwise degrade/finalize.
    - Finalization phase never retries.
    """
    limit = TPM_RETRY_LIMIT if tpm_retry_limit is None else int(tpm_retry_limit)
    error_class = classify_provider_error(exc)
    offline_ok = bool(tools_enabled and has_evidence)
    in_finalization = phase == PHASE_FINALIZATION

    if error_class in {ERROR_TPD, ERROR_RPD} or is_daily_quota_exhausted(exc):
        return RetryDecision(
            action=ACTION_DEGRADE,
            error_class=error_class if error_class in {ERROR_TPD, ERROR_RPD} else ERROR_TPD,
            reason="daily_quota_exhausted_no_retry",
        )

    if error_class == ERROR_TPM:
        if in_finalization:
            return RetryDecision(
                action=ACTION_DEGRADE,
                error_class=ERROR_TPM,
                reason="finalization_tpm_no_retry",
            )
        if tpm_retries < limit:
            hint = parse_retry_after_hint_seconds(exc)
            base_wait = honored_retry_after_wait(hint, max_wait=max_wait)
            if base_wait is None:
                # Server asked for longer than we will sleep — do not retry early.
                if offline_ok:
                    return RetryDecision(
                        action=ACTION_DEGRADE,
                        error_class=ERROR_TPM,
                        reason="tpm_retry_after_exceeds_max_wait",
                    )
                return RetryDecision(
                    action=ACTION_ABORT,
                    error_class=ERROR_TPM,
                    reason="tpm_retry_after_exceeds_max_wait",
                )
            wait = (
                backoff_with_jitter(
                    base_wait,
                    attempt=tpm_retries,
                    max_wait=max_wait,
                    jitter_ratio=jitter_ratio,
                    rng=rng,
                )
                if apply_jitter
                else base_wait
            )
            return RetryDecision(
                action=ACTION_RETRY,
                error_class=ERROR_TPM,
                wait_seconds=wait,
                reason=f"tpm_retry_{tpm_retries + 1}_of_{limit}",
            )
        if offline_ok:
            return RetryDecision(
                action=ACTION_DEGRADE,
                error_class=ERROR_TPM,
                reason="tpm_retries_exhausted_with_evidence",
            )
        return RetryDecision(
            action=ACTION_ABORT,
            error_class=ERROR_TPM,
            reason="tpm_retries_exhausted_no_evidence",
        )

    if error_class == ERROR_TOKEN_LIMIT:
        return RetryDecision(
            action=ACTION_DEGRADE,
            error_class=ERROR_TOKEN_LIMIT,
            reason=(
                "finalization_token_limit"
                if in_finalization
                else "token_limit_finalize"
            ),
        )

    if error_class == ERROR_TOOL_CHOICE:
        if in_finalization:
            return RetryDecision(
                action=ACTION_DEGRADE,
                error_class=ERROR_TOOL_CHOICE,
                reason="finalization_tool_choice_no_retry",
            )
        if not tools_enabled:
            return RetryDecision(
                action=ACTION_RETRY,
                error_class=ERROR_TOOL_CHOICE,
                reason="tool_choice_clean_direct_retry",
            )
        return RetryDecision(
            action=ACTION_DEGRADE,
            error_class=ERROR_TOOL_CHOICE,
            reason="tool_choice_finalize_no_retry",
        )

    if error_class in {ERROR_RPM, ERROR_RATE_AMBIGUOUS}:
        if in_finalization:
            return RetryDecision(
                action=ACTION_DEGRADE,
                error_class=error_class,
                reason="finalization_rate_limit_non_tpm",
            )
        if offline_ok:
            return RetryDecision(
                action=ACTION_DEGRADE,
                error_class=error_class,
                reason="non_tpm_rate_limit_with_evidence",
            )
        return RetryDecision(
            action=ACTION_ABORT,
            error_class=error_class,
            reason="non_tpm_rate_limit_no_evidence",
        )

    if in_finalization:
        return RetryDecision(
            action=ACTION_DEGRADE,
            error_class=error_class,
            reason="finalization_generic_fallback",
        )

    return RetryDecision(
        action=ACTION_ABORT,
        error_class=error_class,
        reason="unhandled_llm_error",
    )


def decide_tool_execution(
    *,
    tool_name: str,
    fingerprint: str,
    committed_mutations: Collection[str] | None = None,
    failed_fingerprints: Collection[str] | None = None,
    mutating_tools: Collection[str] | None = None,
) -> RetryDecision:
    """Decide whether a tool call should execute or be skipped.

    Does **not** schedule tool retries. Encodes existing protections:
    committed mutating-memory fingerprints and repeated-failure fingerprints.
    Duplicate/overlap budget skips remain in ``should_skip_duplicate_tool``.
    """
    committed = committed_mutations or ()
    failed = failed_fingerprints or ()
    mutating = mutating_tools or ()

    # Order matches chat_with_ai: repeated-failure skip before committed-mutation.
    if fingerprint in failed:
        return RetryDecision(
            action=ACTION_SKIP,
            error_class=ERROR_REPEATED_FAILURE,
            reason="repeated_failed_tool_call",
        )
    if tool_name in mutating and fingerprint in committed:
        return RetryDecision(
            action=ACTION_SKIP,
            error_class=ERROR_COMMITTED_MUTATION,
            reason="mutation_already_committed",
        )
    return RetryDecision(
        action=ACTION_EXECUTE,
        error_class=ERROR_NONE,
        reason="ok_to_execute",
    )
