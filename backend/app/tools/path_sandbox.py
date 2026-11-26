"""Shared filesystem sandbox for agent tools (Phase A).

Canonical-path confinement under the backend project root plus explicit
sensitive-file / directory rules. Tools must not trust raw client paths.

Platform notes
--------------
* Symlink escape checks use ``Path.resolve()`` then ``relative_to(root)``.
* Where ``os.O_NOFOLLOW`` exists (POSIX), ``open_text_under_root`` opens the
  final path without following a replaced leaf symlink (TOCTOU mitigation).
* On Windows, ``O_NOFOLLOW`` is typically unavailable; we re-resolve and
  re-check confinement immediately before open, and refuse paths whose
  final component is still a symlink. Junction/reparse-point races remain a
  residual platform limitation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, TextIO


# Backend package root (…/backend). Matches historical file_reader.PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ERROR_PATH_TRAVERSAL = "PATH_TRAVERSAL_DENIED"
ERROR_PATH_DENIED = "PATH_DENIED"

SAFE_DENY_MESSAGE = "Access to this path is denied."

# Exact basenames (case-insensitive).
_SENSITIVE_BASENAMES = frozenset({
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.staging",
    ".env.test",
    ".env.example",
    ".flaskenv",
    "credentials.json",
    "credential.json",
    "secrets.json",
    "secret.json",
    "serviceaccount.json",
    "service-account.json",
    "google-credentials.json",
    "gcp-credentials.json",
    "aws_credentials",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa.pub",
    "authorized_keys",
    "known_hosts",
    "htpasswd",
    "netrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "pgpass",
    ".pgpass",
    "docker-compose.override.yml",
})

# Exact stems denied even with source-like extensions (credentials.py, etc.).
_SENSITIVE_STEMS = frozenset({
    "credentials",
    "credential",
    "secrets",
    "secret",
    "private_key",
    "privatekey",
    "serviceaccount",
    "service_account",
    "service-account",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "htpasswd",
    "netrc",
    "pgpass",
    "aws_credentials",
})

_SENSITIVE_PREFIXES = (
    ".env.",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
)

_SENSITIVE_SUFFIXES = frozenset({
    ".pem",
    ".p12",
    ".pfx",
    ".key",
    ".keystore",
    ".jks",
    ".kdbx",
    ".env",
})

# Substring heuristics for non-source-like names only.
_SENSITIVE_NAME_SUBSTRINGS = (
    "secret",
    "credential",
    "private_key",
    "private-key",
    "serviceaccount",
    "service_account",
    "service-account",
)

_SOURCE_LIKE_EXTENSIONS = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
    ".c", ".cc", ".cpp", ".cs", ".rs", ".rb", ".php", ".swift",
    ".kt", ".html", ".css", ".scss", ".vue", ".svelte",
})

SENSITIVE_DIRECTORY_NAMES = frozenset({
    ".git",
    ".svn",
    ".hg",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".aws",
    ".ssh",
    ".gnupg",
    ".kube",
    "secrets",
    "private",
    "credentials",
})

_UNC_RE = re.compile(r"^[/\\]{2}")
_NULL_RE = re.compile(r"[\x00]")


def path_error(error_code: str, message: str | None = None, **extra) -> dict:
    """Structured sandbox denial — never include resolved absolute paths."""
    payload = {
        "success": False,
        "error_code": error_code,
        "message": message or SAFE_DENY_MESSAGE,
        "error": message or SAFE_DENY_MESSAGE,
    }
    # Drop accidental absolute-path extras.
    for key in ("file_path", "path", "resolved", "abs_path"):
        extra.pop(key, None)
    payload.update(extra)
    return payload


def _looks_absolute_or_drive(raw: str) -> bool:
    text = raw.strip()
    if not text:
        return False
    if _UNC_RE.match(text):
        return True
    if text.startswith("/"):
        return True
    # Windows drive absolute or drive-relative (C:\... or C:foo).
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return True
    if os.path.isabs(text):
        return True
    if PureWindowsPath(text).is_absolute():
        return True
    if PurePosixPath(text.replace("\\", "/")).is_absolute():
        return True
    return False


def _has_traversal(raw: str) -> bool:
    normalized = raw.replace("\\", "/")
    # Collapse empty segments but keep ``..`` visible.
    parts = PurePosixPath(normalized).parts
    return ".." in parts


def is_sensitive_basename(name: str) -> bool:
    """True if a file basename matches secret / credential policy."""
    base = Path(str(name)).name
    lower = base.lower()
    if lower in _SENSITIVE_BASENAMES:
        return True
    if lower.startswith(_SENSITIVE_PREFIXES):
        return True
    if lower.startswith(".env"):
        return True

    # Multi-suffix: secrets.json.py / archive.tar.key
    suffixes = [s.lower() for s in Path(lower).suffixes]
    if any(suf in _SENSITIVE_SUFFIXES for suf in suffixes):
        return True
    if lower.endswith(tuple(_SENSITIVE_SUFFIXES)):
        return True

    # Stem after stripping all suffixes (credentials.py, secrets.json.py).
    stem = lower
    for suf in suffixes:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    if stem in _SENSITIVE_STEMS:
        return True

    final_suffix = Path(lower).suffix
    if final_suffix not in _SOURCE_LIKE_EXTENSIONS:
        for token in _SENSITIVE_NAME_SUBSTRINGS:
            if token in lower:
                return True
    return False


def is_sensitive_directory_name(name: str) -> bool:
    return str(name).lower() in {n.lower() for n in SENSITIVE_DIRECTORY_NAMES}


def _norm(path: Path) -> str:
    """Case-normalized absolute string for confinement comparisons."""
    return os.path.normcase(str(path))


def _is_under_root(candidate: Path, root: Path) -> bool:
    try:
        # normcase helps on case-insensitive volumes (Windows).
        Path(_norm(candidate)).relative_to(Path(_norm(root)))
        return True
    except ValueError:
        return False


def _reject_unsafe_raw(raw: str) -> dict | None:
    if _NULL_RE.search(raw):
        return path_error(ERROR_PATH_TRAVERSAL, "Invalid path.")
    # Reject NT path prefixes / device paths that can bypass naive joins.
    lowered = raw.replace("\\", "/").lower()
    if lowered.startswith("//./") or lowered.startswith("//?/") or lowered.startswith("\\\\.\\") or lowered.startswith("\\\\?\\"):
        return path_error(ERROR_PATH_TRAVERSAL)
    if _looks_absolute_or_drive(raw):
        return path_error(
            ERROR_PATH_TRAVERSAL,
            "Absolute paths are not allowed. Use a project-relative path.",
        )
    if _has_traversal(raw):
        return path_error(
            ERROR_PATH_TRAVERSAL,
            "Path traversal is not allowed.",
        )
    return None


def _sensitive_rel_parts(rel_parts: tuple[str, ...], *, expect_dir: bool | None) -> bool:
    if not rel_parts:
        return False
    parents = rel_parts if expect_dir is True else rel_parts[:-1]
    for part in parents:
        if is_sensitive_directory_name(part):
            return True
    if expect_dir is True and is_sensitive_directory_name(rel_parts[-1]):
        return True
    if expect_dir is not True and is_sensitive_basename(rel_parts[-1]):
        return True
    return False


def resolve_under_root(
    user_path: str | None,
    *,
    root: Path | None = None,
    must_exist: bool = False,
    expect_dir: bool | None = None,
) -> Path | dict:
    """Resolve ``user_path`` under ``root`` or return a structured error dict.

    Rejects absolute/UNC/drive paths, ``..`` segments, null bytes, symlink
    escapes, and sensitive basenames/directories. Never echoes absolute paths.
    """
    project_root = (root or PROJECT_ROOT).resolve()

    if user_path is None or not str(user_path).strip():
        return path_error(
            "INVALID_PATH",
            "A project-relative path is required.",
        )

    raw = str(user_path).strip()
    bad = _reject_unsafe_raw(raw)
    if bad is not None:
        return bad

    display = raw.replace("\\", "/")

    if display in {".", "./", ""}:
        if expect_dir is False:
            return path_error("INVALID_PATH", "A project-relative file path is required.")
        candidate = project_root
    else:
        rel = display[2:] if display.startswith("./") else display
        # Reject empty segments that collapse oddly (app//foo still OK via Path).
        candidate = project_root.joinpath(*PurePosixPath(rel).parts)

    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        return path_error(ERROR_PATH_TRAVERSAL)

    if not _is_under_root(resolved, project_root):
        return path_error(ERROR_PATH_TRAVERSAL)

    try:
        rel_parts = Path(_norm(resolved)).relative_to(Path(_norm(project_root))).parts
    except ValueError:
        return path_error(ERROR_PATH_TRAVERSAL)

    # Also deny based on the *requested* lexical segments (before symlink resolve)
    # so nested sensitive dirs are blocked even if renamed via link tricks.
    lexical = PurePosixPath(display).parts if display not in {".", "./", ""} else ()
    if any(is_sensitive_directory_name(p) for p in lexical[:-1]):
        return path_error(ERROR_PATH_DENIED)
    if expect_dir is True and lexical and is_sensitive_directory_name(lexical[-1]):
        return path_error(ERROR_PATH_DENIED)
    if lexical and is_sensitive_basename(lexical[-1]):
        return path_error(ERROR_PATH_DENIED)

    if _sensitive_rel_parts(rel_parts, expect_dir=expect_dir):
        return path_error(ERROR_PATH_DENIED)

    if must_exist and not resolved.exists():
        return path_error(
            "FILE_NOT_FOUND",
            "The requested path does not exist.",
        )

    if expect_dir is True:
        if resolved.exists() and not resolved.is_dir():
            return path_error(
                "NOT_A_DIRECTORY",
                "The requested path is not a directory.",
            )
    elif expect_dir is False:
        if resolved.exists() and resolved.is_dir():
            return path_error(
                "IS_DIRECTORY",
                "The requested path is a directory, not a file.",
            )

    try:
        final = resolved.resolve(strict=False)
    except (OSError, RuntimeError):
        return path_error(ERROR_PATH_TRAVERSAL)
    if not _is_under_root(final, project_root):
        return path_error(ERROR_PATH_TRAVERSAL)
    if final != project_root and is_sensitive_basename(final.name):
        return path_error(ERROR_PATH_DENIED)

    return final


def relative_to_root(path: Path, *, root: Path | None = None) -> str:
    """POSIX-style path relative to project root for safe tool outputs."""
    project_root = (root or PROJECT_ROOT).resolve()
    rel = path.resolve().relative_to(project_root)
    return rel.as_posix()


def safe_walk_filter(dirnames: list[str]) -> None:
    """In-place filter for ``os.walk`` directory lists."""
    dirnames[:] = [
        d for d in dirnames
        if not is_sensitive_directory_name(d)
    ]


def path_still_inside_root(path: Path, *, root: Path | None = None) -> bool:
    """True if resolved ``path`` is still under ``root`` (symlink-safe check)."""
    project_root = (root or PROJECT_ROOT).resolve()
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return _is_under_root(resolved, project_root)


def open_text_under_root(
    path: Path,
    *,
    root: Path | None = None,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> TextIO:
    """Open a text file after a final confinement check.

    Minimizes TOCTOU by re-resolving immediately before open. On platforms
    with ``os.O_NOFOLLOW``, the leaf is opened without following a symlink.
    Raises ``OSError`` / ``PermissionError`` on failure — callers map to
    safe tool errors without echoing absolute paths.
    """
    project_root = (root or PROJECT_ROOT).resolve()
    try:
        final = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise PermissionError("path denied") from exc

    if not _is_under_root(final, project_root):
        raise PermissionError("path denied")
    if is_sensitive_basename(final.name):
        raise PermissionError("path denied")

    # Refuse opening through a still-symlinked leaf when we can detect it.
    # (Resolved paths are usually non-symlinks; this catches edge cases.)
    try:
        if final.is_symlink():
            raise PermissionError("path denied")
    except OSError as exc:
        raise PermissionError("path denied") from exc

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    # POSIX: refuse to follow a leaf symlink swapped in after resolve.
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow

    try:
        fd = os.open(final, flags)
    except OSError:
        if not nofollow:
            raise
        # Rare platforms/paths reject O_NOFOLLOW — re-validate then open plain.
        final2 = path.resolve(strict=False)
        if (
            not _is_under_root(final2, project_root)
            or is_sensitive_basename(final2.name)
        ):
            raise PermissionError("path denied")
        try:
            if final2.is_symlink():
                raise PermissionError("path denied")
        except OSError as exc:
            raise PermissionError("path denied") from exc
        plain = os.O_RDONLY | (os.O_BINARY if hasattr(os, "O_BINARY") else 0)
        fd = os.open(final2, plain)

    return os.fdopen(fd, "r", encoding=encoding, errors=errors)
