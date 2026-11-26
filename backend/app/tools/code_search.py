"""Project-scoped code search for the agent tool loop."""

from __future__ import annotations

import os
from pathlib import Path

from app.tools.path_sandbox import (
    ERROR_PATH_TRAVERSAL,
    PROJECT_ROOT,
    is_sensitive_basename,
    open_text_under_root,
    path_error,
    path_still_inside_root,
    relative_to_root,
    resolve_under_root,
    safe_walk_filter,
)


# Kept for callers/tests that import the historical name.
IGNORE_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
}


SOURCE_EXTENSIONS = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".cs",
    ".go",
    ".html",
    ".css",
)


MAX_SEARCH_MATCHES = 20

MAX_LINE_CHARS = 180


def search_code(
    search_text,
    folder_path="app",
):
    matches = []
    truncated = False

    try:
        resolved = resolve_under_root(
            folder_path if folder_path is not None else "app",
            root=PROJECT_ROOT,
            expect_dir=True,
        )
        if isinstance(resolved, dict):
            code = resolved.get("error_code") or ERROR_PATH_TRAVERSAL
            message = resolved.get("message") or "Access to this path is denied."
            # Preserve historical ``error`` key while adding error_code.
            return {
                "success": False,
                "error_code": code,
                "error": message,
                "message": message,
            }

        if not resolved.exists():
            return {
                "success": False,
                "error_code": "FILE_NOT_FOUND",
                "error": "Folder not found.",
                "message": "Folder not found.",
            }

        if not resolved.is_dir():
            return {
                "success": False,
                "error_code": "NOT_A_DIRECTORY",
                "error": "The requested path is not a directory.",
                "message": "The requested path is not a directory.",
            }

        # Confirm the walk root did not escape via a late symlink swap.
        if not path_still_inside_root(resolved, root=PROJECT_ROOT):
            return path_error(ERROR_PATH_TRAVERSAL)

        needle = (search_text or "").lower()
        if not needle:
            return {
                "success": False,
                "error_code": "INVALID_QUERY",
                "error": "search_text is required.",
                "message": "search_text is required.",
            }

        for root, directories, files in os.walk(resolved, followlinks=False):
            root_path = Path(root)

            # Never follow / enter paths that left the sandbox (nested symlink).
            if not path_still_inside_root(root_path, root=PROJECT_ROOT):
                directories[:] = []
                continue

            safe_walk_filter(directories)

            # Drop directories whose resolved target escapes the root.
            kept: list[str] = []
            for directory in directories:
                child = root_path / directory
                try:
                    if child.is_symlink():
                        # Resolve target; skip if outside root.
                        if not path_still_inside_root(child, root=PROJECT_ROOT):
                            continue
                    elif not path_still_inside_root(child, root=PROJECT_ROOT):
                        continue
                except OSError:
                    continue
                kept.append(directory)
            directories[:] = kept

            for file_name in files:
                if not file_name.endswith(SOURCE_EXTENSIONS):
                    continue
                if is_sensitive_basename(file_name):
                    continue

                file_path = root_path / file_name
                try:
                    if file_path.is_symlink():
                        if not path_still_inside_root(file_path, root=PROJECT_ROOT):
                            continue
                    if not path_still_inside_root(file_path, root=PROJECT_ROOT):
                        continue
                    if is_sensitive_basename(file_path.name):
                        continue
                except OSError:
                    continue

                try:
                    # Open only after a final confinement check (same root).
                    with open_text_under_root(
                        file_path,
                        root=PROJECT_ROOT,
                        encoding="utf-8",
                        errors="ignore",
                    ) as file:
                        lines = file.readlines()
                except (OSError, PermissionError):
                    continue

                try:
                    rel_display = relative_to_root(file_path, root=PROJECT_ROOT)
                except ValueError:
                    continue

                for line_number, line in enumerate(lines, start=1):
                    if needle not in line.lower():
                        continue

                    if len(matches) >= MAX_SEARCH_MATCHES:
                        truncated = True
                        break

                    text = line.strip()
                    if len(text) > MAX_LINE_CHARS:
                        text = text[:MAX_LINE_CHARS] + "..."

                    matches.append({
                        "file_path": rel_display,
                        "line": line_number,
                        "text": text,
                    })

                if truncated:
                    break

            if truncated:
                break

        if not matches:
            return {
                "success": True,
                "query": search_text,
                "matches": [],
                "message": "No matching lines found.",
            }

        result = {
            "success": True,
            "query": search_text,
            "match_count": len(matches),
            "matches": matches,
        }

        if truncated:
            result["truncated"] = True
            result["message"] = (
                "Showing the first matching lines only. "
                "Read those line numbers with read_file. "
                "Use a more specific search if these lines "
                "are not the code you need."
            )
        else:
            result["message"] = (
                "These matches already show the definition and "
                "every use. Call read_file once on this path for "
                "that function only: a few lines before it, and "
                "about 70 lines at most. Do not read the entire "
                "file. Then answer. Do not search again for this "
                "symbol, its expire calls, or the file name."
            )

        return result

    except Exception:
        # Never leak exception paths or absolute filesystem details.
        return {
            "success": False,
            "error_code": "SEARCH_ERROR",
            "error": "Code search failed.",
            "message": "Code search failed.",
        }
