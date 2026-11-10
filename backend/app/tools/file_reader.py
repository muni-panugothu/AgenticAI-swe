"""Secure project-scoped file reader for the agent tool loop."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path


MAX_READ_CHARS = 3200
MAX_LINE_SPAN = int(os.getenv("MAX_LINE_SPAN", "70"))

# Backend package root (…/backend). Reads are confined to this tree.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TOP_DEF_RE = re.compile(r"^(async\s+)?def\s+\w+|^class\s+\w+")


def _error(error_code: str, message: str, **extra) -> dict:
    payload = {
        "success": False,
        "error_code": error_code,
        "message": message,
    }
    payload.update(extra)
    return payload


def _resolve_safe_path(file_path: str) -> Path | dict:
    """Resolve a user-supplied path inside PROJECT_ROOT or return an error dict."""
    if not file_path or not str(file_path).strip():
        return _error("INVALID_PATH", "file_path is required.")

    raw = str(file_path).strip().replace("\\", "/")
    # Reject absolute paths and explicit traversal tokens early.
    if os.path.isabs(raw) or raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
        return _error(
            "PATH_TRAVERSAL_DENIED",
            "Absolute paths are not allowed. Use a project-relative path.",
        )
    if ".." in Path(raw).parts:
        return _error(
            "PATH_TRAVERSAL_DENIED",
            "Path traversal is not allowed.",
        )

    candidate = (PROJECT_ROOT / raw).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return _error(
            "PATH_TRAVERSAL_DENIED",
            "The requested path is outside the allowed project root.",
        )
    return candidate


def _top_level_def_indices(selected_lines: list[str]) -> list[int]:
    idxs: list[int] = []
    for i, line in enumerate(selected_lines):
        stripped = line.strip()
        if not stripped or (line and line[0].isspace()):
            continue
        if _TOP_DEF_RE.match(stripped):
            idxs.append(i)
    return idxs


def _still_indented(lines: list[str]) -> bool:
    for raw in reversed(lines):
        if not raw.strip():
            continue
        return raw.startswith((" ", "\t"))
    return False


def ast_enclosing_symbol_span(
    source_text: str,
    line: int,
    *,
    prefer_class: bool = False,
) -> tuple[int, int] | None:
    """Return (start_line, end_line) of the tightest def/class covering ``line``.

    Uses Python's AST ``end_lineno`` when available. Returns None on parse
    failure or when no enclosing symbol exists (fallback to heuristic clip).

    When ``prefer_class`` is True, prefer the enclosing ``ClassDef`` span over
    a nested method so class-level investigations are not stopped early.
    """
    if line < 1 or not source_text:
        return None
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    best: tuple[int, int] | None = None
    best_class: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start <= line <= end:
            span = (start, end)
            if isinstance(node, ast.ClassDef):
                if best_class is None or (end - start) < (
                    best_class[1] - best_class[0]
                ):
                    best_class = span
            if best is None or (end - start) < (best[1] - best[0]):
                best = span
    if prefer_class and best_class is not None:
        return best_class
    return best


def ast_class_definition(
    source_text: str,
    class_name: str,
) -> dict | None:
    """Return AST metadata for a top-level or nested class by name.

    Shape:
      {
        "name": str,
        "line_start": int,
        "line_end": int,
        "methods": [{"name", "line_start", "line_end"}, ...],
      }
    """
    name = str(class_name or "").strip()
    if not name or not source_text:
        return None
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    target: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            target = node
            break
    if target is None:
        return None
    start = getattr(target, "lineno", None)
    end = getattr(target, "end_lineno", None)
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    methods: list[dict] = []
    for child in target.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            m_start = getattr(child, "lineno", None)
            m_end = getattr(child, "end_lineno", None)
            if isinstance(m_start, int) and isinstance(m_end, int):
                doc = ast.get_docstring(child) or ""
                purpose = (
                    doc.strip().splitlines()[0].strip()
                    if doc.strip()
                    else f"{'async def' if isinstance(child, ast.AsyncFunctionDef) else 'def'} {child.name}(...)"
                )
                methods.append(
                    {
                        "name": child.name,
                        "line_start": m_start,
                        "line_end": m_end,
                        "purpose": purpose[:200],
                    }
                )
    return {
        "name": name,
        "line_start": start,
        "line_end": end,
        "methods": methods,
    }


def load_project_source_text(file_path: str) -> str | None:
    """Read a project-relative source file for AST inspection, or None."""
    resolved = _resolve_safe_path(file_path)
    if isinstance(resolved, dict):
        return None
    try:
        if not resolved.is_file():
            return None
        return resolved.read_text(encoding="utf-8")
    except OSError:
        return None


def clip_to_function_boundary(
    selected_lines: list[str],
    *,
    start_line: int,
) -> tuple[list[str], int, int, bool, bool]:
    """Clip a window to one function body.

    Returns
    (clipped_lines, abs_start, abs_end, function_complete, still_inside).

    Rules:
    - Continuation windows (start indented): stop before the next top-level
      ``def``/``class`` or ``# ====`` section (do not read the next function).
    - Short leftover (<12 lines) of a previous function before a new ``def``:
      switch target to the *last* ``def`` in the window (search-guided reads).
    - ``span_limited`` alone does not imply ``needs_continuation``.
    """
    if not selected_lines:
        return [], start_line, start_line - 1, False, False

    def_idxs = _top_level_def_indices(selected_lines)
    starts_indented = False
    for line in selected_lines:
        if line.strip():
            starts_indented = line[0].isspace()
            break

    def _slice(a: int, b: int) -> tuple[list[str], int, int, bool, bool]:
        clipped = selected_lines[a:b]
        abs_start = start_line + a
        end_line = start_line + b - 1 if b > a else abs_start - 1
        while clipped and not clipped[-1].strip():
            clipped = clipped[:-1]
            end_line -= 1
        still = _still_indented(clipped)
        complete = (b < len(selected_lines)) or (bool(clipped) and not still)
        return clipped, abs_start, end_line, complete, still

    def _section_after(start_idx: int) -> int | None:
        for i in range(start_idx, len(selected_lines)):
            line = selected_lines[i]
            stripped = line.strip()
            if not stripped or (line and line[0].isspace()):
                continue
            if stripped.startswith("#") and "====" in stripped:
                return i
        return None

    if starts_indented:
        first_def = def_idxs[0] if def_idxs else None
        # Indented windows are either:
        # (a) a short leftover (≤12 lines) of a prior function before the
        #     real target def (search often starts a few lines early)
        #     → switch to that def;
        # (b) a mid-function continuation — keep reading the open body and
        #     stop before the *next* top-level def (do not jump to it).
        if first_def is not None and first_def <= 12:
            primary = first_def
            stop = len(selected_lines)
            sec = _section_after(primary + 1)
            next_defs = [d for d in def_idxs if d > primary]
            if next_defs:
                stop = min(stop, next_defs[0])
            if sec is not None:
                stop = min(stop, sec)
            return _slice(primary, stop)

        if first_def is not None:
            # True continuation of the current function: stop before the
            # next symbol so the caller can continue from next_line.
            stop = first_def
            sec = _section_after(0)
            hit_section = sec is not None and sec < stop
            if hit_section:
                stop = sec
            clipped, abs_start, end_line, _, still = _slice(0, stop)
            # A top-level `# ====…` section ends the prior function even
            # when the last body line is still indented (e.g. `        }`).
            if hit_section:
                return clipped, abs_start, end_line, True, False
            # Still inside the open body — request next_line continuation
            # instead of jumping to the upcoming def.
            complete = not still
            return clipped, abs_start, end_line, complete, still

        sec = _section_after(0)
        if sec is not None:
            clipped, abs_start, end_line, _, _ = _slice(0, sec)
            return clipped, abs_start, end_line, True, False

        clipped, abs_start, end_line, complete, still = _slice(
            0, len(selected_lines)
        )
        return clipped, abs_start, end_line, (not still), still

    if def_idxs:
        # Prefer the first top-level def in a non-indented window so we do
        # not skip the requested symbol when helpers follow in the span.
        primary = def_idxs[0]
        stop = len(selected_lines)
        next_defs = [d for d in def_idxs if d > primary]
        if next_defs:
            stop = next_defs[0]
        sec = _section_after(primary + 1)
        if sec is not None:
            stop = min(stop, sec)
        return _slice(primary, stop)

    return _slice(0, len(selected_lines))


def read_file(file_path, line_start=None, line_end=None):
    """Read a small line window from a project file.

    Returns a JSON string with a trailing content block on success, or a
    structured error dict/JSON on failure. Missing files never report success.
    """
    resolved = _resolve_safe_path(file_path)
    if isinstance(resolved, dict):
        return resolved

    if not resolved.exists():
        return _error(
            "FILE_NOT_FOUND",
            "The requested file does not exist.",
            file_path=str(file_path),
        )

    if resolved.is_dir():
        return _error(
            "IS_DIRECTORY",
            "The requested path is a directory, not a file.",
            file_path=str(file_path),
        )

    if not resolved.is_file():
        return _error(
            "NOT_A_FILE",
            "The requested path is not a readable file.",
            file_path=str(file_path),
        )

    if (
        not isinstance(line_start, int)
        or not isinstance(line_end, int)
        or line_start < 1
        or line_end < line_start
    ):
        return _error(
            "INVALID_LINE_RANGE",
            (
                "Provide line_start and line_end for one function or a small "
                "line range. Do not request the entire file."
            ),
            file_path=str(file_path),
        )

    span_limited = False
    if line_end - line_start + 1 > MAX_LINE_SPAN:
        line_end = line_start + MAX_LINE_SPAN - 1
        span_limited = True

    try:
        with open(resolved, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        return _error(
            "FILE_NOT_FOUND",
            "The requested file does not exist.",
            file_path=str(file_path),
        )
    except OSError as e:
        return _error(
            "READ_ERROR",
            f"Error reading file: {e}",
            file_path=str(file_path),
        )

    if len(lines) == 0:
        return {
            "success": True,
            "file_path": str(file_path),
            "empty": True,
            "line_start": 1,
            "line_end": 0,
            "truncated": False,
            "span_limited": False,
            "needs_continuation": False,
            "message": (
                "The file exists but is empty (0 bytes). "
                "There is nothing to read. Do not search for this path again."
            ),
        }

    if line_start > len(lines):
        return _error(
            "INVALID_LINE_RANGE",
            (
                f"line_start ({line_start}) is beyond the end of the file "
                f"({len(lines)} lines)."
            ),
            file_path=str(file_path),
            file_line_count=len(lines),
        )

    start = line_start - 1
    end = min(line_end, len(lines))
    selected_lines = lines[start:end]

    full_text = "".join(lines)

    # Stop at the next top-level function/class — do not keep reading
    # unrelated code merely because the 70-line span was capped.
    clipped, abs_start, boundary_end, function_complete, still_inside = (
        clip_to_function_boundary(selected_lines, start_line=line_start)
    )
    if not clipped:
        clipped = selected_lines
        abs_start = line_start
        boundary_end = line_start + len(selected_lines) - 1

    # Prefer AST span from the *actual* clipped start. Requesting a blank
    # line just above ``class Foo`` must still resolve the class span
    # (line_start=222 → None, abs_start=223 → class end).
    ast_span = ast_enclosing_symbol_span(full_text, abs_start)
    if ast_span is None and int(line_start) != int(abs_start):
        ast_span = ast_enclosing_symbol_span(full_text, int(line_start))
    ast_end: int | None = ast_span[1] if ast_span else None

    # If AST knows the symbol end, never emit past it in this window.
    if ast_end is not None and boundary_end > ast_end:
        keep = max(0, ast_end - abs_start + 1)
        clipped = clipped[:keep]
        boundary_end = min(boundary_end, ast_end)
        still_inside = abs_start <= ast_end and (
            not clipped or _still_indented(clipped)
        )
        function_complete = boundary_end >= ast_end

    content_parts = []
    used = 0
    line_was_cut = False
    for line in clipped:
        if content_parts and used + len(line) > MAX_READ_CHARS:
            line_was_cut = True
            break
        if not content_parts and len(line) > MAX_READ_CHARS:
            content_parts.append(line[:MAX_READ_CHARS])
            line_was_cut = True
            break
        content_parts.append(line)
        used += len(line)

    included = len(content_parts)
    truncated = line_was_cut
    content = "".join(content_parts)

    actual_start = abs_start
    actual_end = (
        actual_start + included - 1 if included else boundary_end
    )
    if function_complete and boundary_end >= actual_start:
        actual_end = min(actual_end, boundary_end)
        expected_count = actual_end - actual_start + 1
        if expected_count < included:
            content_parts = content_parts[:expected_count]
            included = len(content_parts)
            content = "".join(content_parts)
            actual_end = actual_start + included - 1

    # AST end reached → function complete even if heuristic still_inside.
    if ast_end is not None and actual_end >= ast_end:
        function_complete = True
        still_inside = False
        actual_end = min(actual_end, ast_end)
        expected_count = actual_end - actual_start + 1
        if expected_count < included:
            content_parts = content_parts[:expected_count]
            included = len(content_parts)
            content = "".join(content_parts)

    next_unread = actual_end + 1
    eof_reached = actual_end >= len(lines)

    # needs_continuation is independent of span_limited: only continue when
    # we are still inside the target function body.
    needs_continuation = (
        (not eof_reached)
        and (not function_complete)
        and still_inside
        and next_unread <= len(lines)
    )
    # AST says more body remains beyond this window.
    if (
        ast_end is not None
        and actual_end < ast_end
        and not eof_reached
        and next_unread <= len(lines)
    ):
        needs_continuation = True
        function_complete = False
        still_inside = True
    # Char truncation mid-body also requires continuation.
    if truncated and still_inside and not eof_reached:
        needs_continuation = True
        function_complete = False

    function_done = bool(function_complete) and not needs_continuation
    # Distinguish a finished function from a hard span/char limit.
    if function_done:
        boundary_kind = "function_complete"
    elif needs_continuation and span_limited:
        boundary_kind = "span_limited_open"
    elif needs_continuation:
        boundary_kind = "function_open"
    elif span_limited:
        boundary_kind = "span_limited_closed"
    else:
        boundary_kind = "window_closed"

    content_line_count = content.count("\n") + (
        1 if content and not content.endswith("\n") else 0
    )
    # Empty content → 0 lines; trailing newline alone does not inflate count.
    if not content.strip():
        content_line_count = 0
    elif content.endswith("\n"):
        content_line_count = content.count("\n")
    else:
        content_line_count = content.count("\n") + 1

    metadata = {
        "success": True,
        "file_path": str(file_path),
        "line_start": actual_start,
        "line_end": actual_end,
        "lines_read": included,
        "content_line_count": content_line_count,
        "metadata_matches_content": (
            included == content_line_count
            if content_line_count or included
            else included == 0
        ),
        "next_line": next_unread,
        "truncated": truncated,
        "span_limited": span_limited,
        "empty": False,
        "needs_continuation": needs_continuation,
        "function_complete": function_done,
        "boundary_kind": boundary_kind,
        "eof_reached": eof_reached,
        "ast_symbol_end": ast_end,
    }

    if needs_continuation:
        suggested_end = min(next_unread + MAX_LINE_SPAN - 1, len(lines))
        metadata["continuation"] = {
            "file_path": str(file_path),
            "next_line": next_unread,
            "suggested_end_line": suggested_end,
            "message": (
                f"Function body is still open. Continue with read_file "
                f"starting at line {next_unread} (not {actual_end}). "
                f"Do not re-read lines {actual_start}–{actual_end}."
            ),
        }

    if truncated and needs_continuation:
        metadata["message"] = (
            "Only part of this small range was returned. "
            f"Continue from line {next_unread} for at most "
            f"{MAX_LINE_SPAN} lines. Do not request the entire file."
        )
    elif function_complete and not needs_continuation:
        metadata["message"] = (
            f"Read lines {actual_start}–{actual_end}. "
            "Reached the end of the current function/class "
            "(next top-level definition or section). "
            "No further continuation is required for this symbol."
        )
    elif span_limited and needs_continuation:
        metadata["message"] = (
            f"Requested range limited to {MAX_LINE_SPAN} lines and the "
            f"function body is still open. Continue from line {next_unread}."
        )
    elif span_limited:
        metadata["message"] = (
            f"Requested range limited to {MAX_LINE_SPAN} lines. "
            f"Read lines {actual_start}–{actual_end}. "
            "No further continuation is required for this window."
        )
    elif needs_continuation:
        metadata["message"] = (
            f"Read lines {actual_start}–{actual_end}, but the block still "
            f"looks open. Continue from line {next_unread}."
        )
    else:
        metadata["message"] = (
            f"Read lines {actual_start}–{actual_end}. "
            "No further continuation is required for this window."
        )

    return (
        json.dumps(metadata, ensure_ascii=False, indent=2)
        + "\ncontent:\n"
        + content
    )
