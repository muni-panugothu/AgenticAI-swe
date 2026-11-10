"""Centralized investigation checklist and budget management.

Tracks required symbols through explicit lifecycle states and enforces
separate, configurable budgets for LLM steps, tools, searches, reads,
continuations, and source evidence size — without disabling safety limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# Explicit target lifecycle (search ≠ verified).
STATUS_PENDING = "pending"
STATUS_DISCOVERED = "discovered"
STATUS_READING = "reading"
STATUS_VERIFIED = "verified"
STATUS_FAILED = "failed"

VALID_STATUSES = frozenset({
    STATUS_PENDING,
    STATUS_DISCOVERED,
    STATUS_READING,
    STATUS_VERIFIED,
    STATUS_FAILED,
})


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class TargetRecord:
    """One required investigation symbol."""

    id: str
    status: str = STATUS_PENDING
    aliases: list[str] = field(default_factory=list)
    kind: str = "symbol"
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    def_line: int | None = None
    failure_reason: str | None = None
    need_read: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "aliases": list(self.aliases),
            "kind": self.kind,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "def_line": self.def_line,
            "failure_reason": self.failure_reason,
            "need_read": self.need_read,
        }


@dataclass
class InvestigationBudgetManager:
    """Separate configurable budgets for investigation execution.

    Continuations are tracked separately from discovery reads so finishing
    a long function is not blocked by the same soft per-file discovery cap.
    """

    max_llm_iterations: int = field(
        default_factory=lambda: _env_int("MAX_AGENT_LLM_ITERATIONS", 8)
    )
    max_tool_executions: int = field(
        default_factory=lambda: _env_int("MAX_TOOL_EXECUTIONS_PER_REQUEST", 14)
    )
    max_reads_total: int = field(
        default_factory=lambda: _env_int("MAX_READ_FILE_TOTAL", 14)
    )
    max_reads_per_file: int = field(
        default_factory=lambda: _env_int("MAX_READS_PER_FILE", 4)
    )
    max_continuation_reads: int = field(
        default_factory=lambda: _env_int("MAX_CONTINUATION_READS", 8)
    )
    max_searches: int = field(
        default_factory=lambda: _env_int("MAX_SEARCH_CODE_EXECUTIONS", 4)
    )
    max_lines_per_read: int = field(
        default_factory=lambda: _env_int("MAX_LINE_SPAN", 70)
    )
    max_source_chars: int = field(
        default_factory=lambda: _env_int("MAX_INVESTIGATION_SOURCE_CHARS", 48000)
    )
    finalization_reserve: int = field(
        default_factory=lambda: _env_int("FINALIZATION_STEP_RESERVE", 1)
    )

    tools_used: int = 0
    searches_used: int = 0
    reads_used: int = 0
    continuation_reads_used: int = 0
    source_chars_used: int = 0
    reads_by_file: dict[str, int] = field(default_factory=dict)
    discovery_reads_by_file: dict[str, int] = field(default_factory=dict)
    # Extra per-file headroom for files that host many required symbols.
    per_file_bonus: dict[str, int] = field(default_factory=dict)

    def configure_for_requirements(
        self,
        *,
        required_target_count: int,
        symbols_per_file: dict[str, int] | None = None,
    ) -> None:
        """Scale read budgets from the investigation checklist size."""
        n = max(0, int(required_target_count))
        # Each symbol may need ~1–2 windows; long bodies need more.
        # Keep headroom for continuations + related test file reads.
        self.max_reads_total = max(
            self.max_reads_total,
            n * 2 + max(2, n // 2) + 4,
        )
        self.max_continuation_reads = max(
            self.max_continuation_reads,
            max(8, n + 4),
        )
        # Large multi-service checklists (e.g. memory architecture) need
        # enough total tool slots for reads, continuations, and searches.
        self.max_tool_executions = max(
            self.max_tool_executions,
            self.max_reads_total + self.max_searches + 4,
        )
        # Long multi-window bodies (e.g. semantic_search_memories) consume
        # many source characters — scale the char budget with checklist size.
        self.max_source_chars = max(
            self.max_source_chars,
            48000 + n * 4000,
        )
        # Reserve LLM turns for reading, not only discovery.
        if n >= 6:
            self.max_llm_iterations = max(
                self.max_llm_iterations,
                min(14, 4 + n // 2),
            )
        for path, count in (symbols_per_file or {}).items():
            key = str(path or "").replace("\\", "/").strip().lower()
            if key and count > 0:
                # Continuations + one window per symbol on that file.
                self.per_file_bonus[key] = max(
                    self.per_file_bonus.get(key, 0),
                    int(count) + 2,
                )

    def remaining_source_chars(self) -> int:
        return max(0, int(self.max_source_chars) - int(self.source_chars_used))

    def can_fit_continuation_chars(self, *, min_chars: int = 400) -> bool:
        """True when at least a small continuation window still fits."""
        return self.remaining_source_chars() >= max(1, int(min_chars))

    def file_read_limit(self, file_path: str) -> int:
        key = str(file_path or "").replace("\\", "/").strip().lower()
        bonus = int(self.per_file_bonus.get(key, 0))
        return max(self.max_reads_per_file, bonus, min(self.max_reads_per_file + bonus, 12))

    def allow_tool(self) -> tuple[bool, str | None]:
        if self.tools_used >= self.max_tool_executions:
            return False, "TOOL_EXECUTION_BUDGET_EXCEEDED"
        return True, None

    def allow_search(self) -> tuple[bool, str | None]:
        ok, reason = self.allow_tool()
        if not ok:
            return ok, reason
        if self.searches_used >= self.max_searches:
            return False, "PER_TOOL_EXECUTION_LIMIT"
        return True, None

    def allow_read(
        self,
        file_path: str,
        *,
        is_continuation: bool = False,
    ) -> tuple[bool, str | None]:
        ok, reason = self.allow_tool()
        if not ok:
            return ok, reason
        if self.reads_used >= self.max_reads_total:
            return False, "PER_TOOL_EXECUTION_LIMIT"
        # Hard stop only when no meaningful continuation can fit.
        if self.remaining_source_chars() <= 0:
            return False, "SOURCE_CHAR_BUDGET_EXCEEDED"
        if (
            is_continuation
            and not self.can_fit_continuation_chars(min_chars=200)
        ):
            return False, "SOURCE_CHAR_BUDGET_EXCEEDED"
        key = str(file_path or "").replace("\\", "/").strip().lower()
        file_count = int(self.reads_by_file.get(key, 0))
        limit = self.file_read_limit(key)
        if is_continuation:
            if self.continuation_reads_used >= self.max_continuation_reads:
                return False, "CONTINUATION_BUDGET_EXCEEDED"
            # Continuations may exceed the soft discovery per-file floor but
            # still respect the hard file_read_limit (scaled for multi-symbol).
            if file_count >= limit:
                return False, "PER_FILE_READ_LIMIT"
            return True, None
        # Discovery / forced-target reads use the soft discovery counter too.
        discovery = int(self.discovery_reads_by_file.get(key, 0))
        soft = max(self.max_reads_per_file, self.per_file_bonus.get(key, 0))
        if discovery >= soft and file_count >= limit:
            return False, "PER_FILE_READ_LIMIT"
        if file_count >= limit:
            return False, "PER_FILE_READ_LIMIT"
        return True, None

    def record_search(self) -> None:
        self.searches_used += 1
        self.tools_used += 1

    def record_read(
        self,
        file_path: str,
        *,
        is_continuation: bool = False,
        content_chars: int = 0,
    ) -> None:
        key = str(file_path or "").replace("\\", "/").strip().lower()
        self.reads_used += 1
        self.tools_used += 1
        self.reads_by_file[key] = int(self.reads_by_file.get(key, 0)) + 1
        if is_continuation:
            self.continuation_reads_used += 1
        else:
            self.discovery_reads_by_file[key] = (
                int(self.discovery_reads_by_file.get(key, 0)) + 1
            )
        if content_chars > 0:
            self.source_chars_used += int(content_chars)

    def can_keep_investigating(self, step: int) -> bool:
        reserve = max(1, int(self.finalization_reserve))
        steps_left = self.max_llm_iterations - 1 - int(step)
        return steps_left > reserve

    def exhausted_reason(self) -> str | None:
        if self.tools_used >= self.max_tool_executions:
            return "TOOL_EXECUTION_BUDGET_EXCEEDED"
        if self.reads_used >= self.max_reads_total:
            return "PER_TOOL_EXECUTION_LIMIT"
        if self.source_chars_used >= self.max_source_chars:
            return "SOURCE_CHAR_BUDGET_EXCEEDED"
        return None


class InvestigationChecklist:
    """Structured required-symbol checklist with explicit statuses."""

    def __init__(self, requirements: dict | None = None):
        self.requirements = requirements or {}
        self.targets: dict[str, TargetRecord] = {}
        for item in self.requirements.get("required_targets") or []:
            tid = str(item.get("id") or "").strip()
            if not tid:
                continue
            aliases = [str(a) for a in (item.get("aliases") or [tid]) if a]
            self.targets[tid] = TargetRecord(
                id=tid,
                status=STATUS_PENDING,
                aliases=aliases,
                kind=str(item.get("kind") or "symbol"),
                need_read=bool(item.get("need_read", True)),
            )

    def ids(self) -> list[str]:
        return list(self.targets.keys())

    def get(self, tid: str) -> TargetRecord | None:
        return self.targets.get(tid)

    def status_of(self, tid: str) -> str:
        rec = self.targets.get(tid)
        return rec.status if rec else STATUS_PENDING

    def by_status(self, status: str) -> list[str]:
        return [t for t, r in self.targets.items() if r.status == status]

    def uncovered(self) -> list[str]:
        """Targets that still need a verified source read."""
        out = []
        for tid, rec in self.targets.items():
            if not rec.need_read:
                continue
            if rec.status not in {STATUS_VERIFIED}:
                out.append(tid)
        return out

    def mark_discovered(
        self,
        tid: str,
        *,
        file_path: str,
        def_line: int,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> None:
        rec = self.targets.get(tid)
        if not rec:
            return
        if rec.status in {STATUS_VERIFIED, STATUS_FAILED}:
            return
        rec.status = STATUS_DISCOVERED
        rec.file_path = str(file_path or "").replace("\\", "/")
        rec.def_line = int(def_line)
        if line_start is not None:
            rec.line_start = int(line_start)
        if line_end is not None:
            rec.line_end = int(line_end)

    def mark_reading(
        self,
        tid: str,
        *,
        file_path: str,
        line_start: int,
        line_end: int,
    ) -> None:
        rec = self.targets.get(tid)
        if not rec:
            return
        if rec.status == STATUS_VERIFIED:
            return
        rec.status = STATUS_READING
        rec.file_path = str(file_path or "").replace("\\", "/")
        rec.line_start = int(line_start)
        rec.line_end = int(line_end)

    def mark_verified(
        self,
        tid: str,
        *,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> None:
        rec = self.targets.get(tid)
        if not rec:
            return
        rec.status = STATUS_VERIFIED
        if file_path:
            rec.file_path = str(file_path).replace("\\", "/")
        if line_start is not None:
            rec.line_start = int(line_start)
        if line_end is not None:
            rec.line_end = int(line_end)
        rec.failure_reason = None

    def mark_failed(self, tid: str, reason: str) -> None:
        rec = self.targets.get(tid)
        if not rec:
            return
        if rec.status == STATUS_VERIFIED:
            return
        rec.status = STATUS_FAILED
        rec.failure_reason = str(reason or "failed")

    def snapshot(self) -> dict[str, Any]:
        return {
            tid: rec.to_dict() for tid, rec in self.targets.items()
        }

    def sync_from_evidence(self, evidence: dict) -> None:
        """Best-effort sync from legacy evidence dict fields."""
        discovered = evidence.get("discovered") or {}
        for tid, hits in discovered.items():
            if tid not in self.targets:
                continue
            if self.targets[tid].status == STATUS_PENDING and hits:
                hit = hits[0]
                self.mark_discovered(
                    tid,
                    file_path=str(hit.get("file_path") or ""),
                    def_line=int(hit.get("line") or 1),
                )
        progress = evidence.get("target_progress") or {}
        prior_statuses = evidence.get("target_statuses") or {}
        for tid in evidence.get("verified_targets") or set():
            if tid not in self.targets:
                continue
            prog = progress.get(tid) or {}
            meta = prior_statuses.get(tid) or {}
            self.mark_verified(
                tid,
                file_path=(
                    prog.get("file_path")
                    or meta.get("file_path")
                    or self.targets[tid].file_path
                ),
                line_start=(
                    prog.get("line_start")
                    or meta.get("line_start")
                    or self.targets[tid].line_start
                ),
                line_end=(
                    prog.get("line_end")
                    or meta.get("line_end")
                    or self.targets[tid].line_end
                ),
            )

    def apply_to_evidence(self, evidence: dict) -> None:
        """Write checklist statuses into the mutable evidence state."""
        evidence["target_statuses"] = self.snapshot()
        evidence["verified_targets"] = set(self.by_status(STATUS_VERIFIED))


def build_budget_from_requirements(
    requirements: dict | None,
    *,
    known_locations: dict[str, tuple[str, int]] | None = None,
) -> InvestigationBudgetManager:
    """Create a budget manager sized for the required-symbol checklist."""
    req = requirements or {}
    targets = list(req.get("required_targets") or [])
    symbols_per_file: dict[str, int] = {}
    known = known_locations or {}
    for item in targets:
        tid = str(item.get("id") or "")
        loc = known.get(tid)
        if loc:
            path = str(loc[0]).replace("\\", "/")
            symbols_per_file[path] = symbols_per_file.get(path, 0) + 1
    budget = InvestigationBudgetManager()
    budget.configure_for_requirements(
        required_target_count=len(targets),
        symbols_per_file=symbols_per_file,
    )
    return budget
