"""Sandbox boundary for high-risk / filesystem tools (Topic 53).

Honest guarantees for this Windows development environment:

* Application-level path confinement via ``path_sandbox`` (resolve + relative_to).
* Sensitive basename / stem deny lists.
* Output size and wall-clock limits at the tool-reliability layer.
* **No** OS container, Job Object, or network namespace isolation is claimed.
* **No** arbitrary shell / process execution tool is offered (fail closed).
* POSIX ``O_NOFOLLOW`` is typically unavailable on Windows; junction races remain.

Tools that would require true OS isolation (shell, arbitrary code) are refused.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.agent_security_audit import record_security_event
from app.services.agent_security_policy import get_tool_risk_meta
from app.tools import path_sandbox

# Hard deny — never introduce arbitrary execution just to demo sandboxing.
FORBIDDEN_EXECUTION_TOOLS = frozenset({
    "shell",
    "bash",
    "cmd",
    "powershell",
    "exec",
    "run_command",
    "subprocess",
    "eval",
    "python_exec",
    "code_interpreter",
})

DEFAULT_MAX_OUTPUT_BYTES = int(os.getenv("AGENT_SANDBOX_MAX_OUTPUT_BYTES", str(512 * 1024)))
DEFAULT_MAX_FILE_BYTES = int(os.getenv("AGENT_SANDBOX_MAX_FILE_BYTES", str(2 * 1024 * 1024)))
SANDBOX_ROOT_ENV = os.getenv("AGENT_SANDBOX_ROOT", "").strip()


@dataclass(frozen=True)
class SandboxCapabilities:
    path_confinement: bool = True
    sensitive_path_deny: bool = True
    output_size_limit: bool = True
    wall_clock_timeout: bool = True  # via tool_reliability
    process_allowlist: bool = True  # empty allowlist = no process exec
    os_container: bool = False
    job_object: bool = False
    network_namespace: bool = False
    o_nofollow: bool = False  # Windows typical
    claim_escape_proof: bool = False


def get_sandbox_capabilities() -> SandboxCapabilities:
    o_nofollow = hasattr(os, "O_NOFOLLOW")
    return SandboxCapabilities(
        o_nofollow=bool(o_nofollow),
        # Still do not claim escape-proof even on POSIX without containers.
        claim_escape_proof=False,
        os_container=False,
        job_object=False,
        network_namespace=False,
    )


def sandbox_root() -> Path:
    if SANDBOX_ROOT_ENV:
        return Path(SANDBOX_ROOT_ENV).resolve()
    return path_sandbox.PROJECT_ROOT


def refuse_forbidden_execution_tool(tool_name: str) -> dict[str, Any] | None:
    if tool_name.lower() in FORBIDDEN_EXECUTION_TOOLS:
        record_security_event(
            "sandbox_violation",
            tool_name=tool_name,
            detail={"reason": "process_execution_not_offered"},
        )
        return {
            "success": False,
            "error": "Process execution tools are not available",
            "error_code": "SANDBOX_PROCESS_DENIED",
            "mutation_committed": False,
        }
    return None


def enforce_sandbox_for_tool(
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    user_id: str | None = None,
) -> dict[str, Any] | None:
    """Pre-execution sandbox gate. Returns error dict or None if OK."""
    blocked = refuse_forbidden_execution_tool(tool_name)
    if blocked is not None:
        return blocked

    meta = get_tool_risk_meta(tool_name)
    args = arguments or {}

    # Reject client-supplied sandbox overrides.
    if "sandbox" in (arguments or {}) or "sandbox_config" in (arguments or {}):
        record_security_event(
            "sandbox_violation",
            user_id=user_id,
            tool_name=tool_name,
            detail={"reason": "client_sandbox_override"},
        )
        return {
            "success": False,
            "error": "Sandbox configuration cannot be supplied by the model",
            "error_code": "SANDBOX_OVERRIDE_REJECTED",
            "mutation_committed": False,
        }

    if not meta.requires_sandbox and tool_name not in {"read_file", "search_code"}:
        return None

    # Filesystem tools: validate path through path_sandbox when a path is present.
    # Include search_code's ``folder_path`` so the gate matches the tool schema
    # (body still re-checks via path_sandbox; this fails closed earlier).
    path_keys = ("file_path", "path", "directory", "folder_path")
    for key in path_keys:
        if key not in args or args[key] in (None, ""):
            continue
        raw = str(args[key])
        # Directory-shaped keys must deny sensitive directory basenames (e.g. .git).
        expect_dir = True if key in {"directory", "folder_path"} else None
        resolved = path_sandbox.resolve_under_root(
            raw,
            root=sandbox_root(),
            expect_dir=expect_dir,
        )
        if isinstance(resolved, dict):
            record_security_event(
                "sandbox_violation",
                user_id=user_id,
                tool_name=tool_name,
                detail={
                    "reason": resolved.get("error_code", "PATH_DENIED"),
                    "key": key,
                },
            )
            return {
                "success": False,
                "error": resolved.get("error") or path_sandbox.SAFE_DENY_MESSAGE,
                "error_code": resolved.get("error_code", "PATH_DENIED"),
                "mutation_committed": False,
            }
        # Size gate when reading existing files.
        try:
            if resolved.is_file():
                size = resolved.stat().st_size
                if size > DEFAULT_MAX_FILE_BYTES:
                    record_security_event(
                        "resource_limit_violation",
                        user_id=user_id,
                        tool_name=tool_name,
                        detail={"reason": "file_too_large", "size": size},
                    )
                    return {
                        "success": False,
                        "error": "File exceeds sandbox size limit",
                        "error_code": "SANDBOX_FILE_TOO_LARGE",
                        "mutation_committed": False,
                    }
        except OSError:
            pass
    return None


def create_request_workspace(request_id: str | None = None) -> Path:
    """Per-request temp directory under system temp (cleaned by caller)."""
    base = Path(tempfile.gettempdir()) / "agentic_sandbox"
    base.mkdir(parents=True, exist_ok=True)
    name = (request_id or "anon")[:64].replace("..", "_")
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def cleanup_request_workspace(path: Path) -> None:
    try:
        if not path.exists():
            return
        # Only delete under our temp base.
        base = Path(tempfile.gettempdir()) / "agentic_sandbox"
        path.resolve().relative_to(base.resolve())
        for child in sorted(path.rglob("*"), reverse=True):
            try:
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    child.rmdir()
            except OSError:
                pass
        try:
            path.rmdir()
        except OSError:
            pass
    except (ValueError, OSError):
        record_security_event(
            "sandbox_violation",
            detail={"reason": "cleanup_refused_or_failed", "path": str(path)[:200]},
        )


def network_access_allowed_for_tool(tool_name: str) -> bool:
    """Default deny network for sandbox-class tools; web_search is explicit."""
    return tool_name == "web_search"


def summarize_isolation_guarantees() -> dict[str, Any]:
    caps = get_sandbox_capabilities()
    return {
        "active": {
            "path_confinement": caps.path_confinement,
            "sensitive_path_deny": caps.sensitive_path_deny,
            "output_size_limit": caps.output_size_limit,
            "wall_clock_timeout": caps.wall_clock_timeout,
            "process_execution": False,
            "process_allowlist_empty": True,
        },
        "not_enforced_in_this_environment": {
            "os_container": True,
            "windows_job_object": True,
            "network_namespace": True,
            "escape_proof": True,
        },
        "platform_notes": {
            "o_nofollow": caps.o_nofollow,
            "junction_reparse_residual_risk": True,
        },
    }
