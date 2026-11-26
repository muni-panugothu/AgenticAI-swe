"""BD-F — Sandbox and filesystem boundary attacks (Windows-aware)."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-32chars-min!!")

from app.chatbot import execute_tool  # noqa: E402
from app.services.sandbox_service import (  # noqa: E402
    enforce_sandbox_for_tool,
    refuse_forbidden_execution_tool,
    summarize_isolation_guarantees,
)
from app.tools import path_sandbox  # noqa: E402
from tests.security._fixtures import SecurityCase  # noqa: E402


class TestSandboxAdversarial(SecurityCase):
    def test_BD_F01_dotdot_traversal(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": "../Secrets/passwords.txt"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)
        self.assertIn(err.get("error_code"), {"PATH_TRAVERSAL_DENIED", "PATH_DENIED"})

    def test_BD_F02_absolute_posix(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": "/etc/passwd"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)

    def test_BD_F03_windows_drive(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": r"C:\Windows\System32\config\SAM"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)

    def test_BD_F04_unc_path(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": r"\\evil-host\share\secrets.txt"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)

    def test_BD_F05_mixed_slashes_traversal(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": r"app\..\..\..\Windows\win.ini"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)

    def test_BD_F06_null_byte(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": "app/main.py\x00.env"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.get("error_code"), "PATH_TRAVERSAL_DENIED")

    def test_BD_F07_encoded_traversal_literal(self):
        # Percent-encoding is not decoded into ``..``; treat as opaque name.
        resolved = path_sandbox.resolve_under_root("%2e%2e/%2e%2e/.env")
        # Either denied as sensitive/traversal or fails confinement / not found policy.
        if isinstance(resolved, dict):
            self.assertIn(
                resolved.get("error_code"),
                {
                    "PATH_TRAVERSAL_DENIED",
                    "PATH_DENIED",
                    "FILE_NOT_FOUND",
                    "INVALID_PATH",
                },
            )
        else:
            # If somehow resolved under root, basename policy should still block .env
            self.assertNotIn(".env", str(resolved).lower())

    def test_BD_F08_sensitive_env(self):
        result = execute_tool(
            "read_file",
            {"file_path": ".env", "line_start": 1, "line_end": 3},
            user_id="user-a",
        )
        self.assertFalse(result.get("success"))
        # Never echo secrets or absolute paths in denial.
        blob = str(result)
        self.assertNotIn("JWT_SECRET", blob)
        self.assertNotRegex(blob, r"[A-Za-z]:\\Users\\")

    def test_BD_F09_ssh_private_key_basename(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": "deploy/id_rsa"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)
        self.assertEqual(err.get("error_code"), "PATH_DENIED")

    def test_BD_F10_git_directory(self):
        err = enforce_sandbox_for_tool(
            "search_code",
            {"search_text": "secret", "folder_path": ".git"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)

    def test_BD_F11_nt_device_path(self):
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": r"\\.\C:\Windows\System32\drivers\etc\hosts"},
            user_id="user-a",
        )
        self.assertIsNotNone(err)

    def test_BD_F12_process_tools_refused(self):
        for name in ("shell", "powershell", "cmd", "eval", "run_command"):
            err = refuse_forbidden_execution_tool(name)
            self.assertIsNotNone(err)
            self.assertEqual(err["error_code"], "SANDBOX_PROCESS_DENIED")

    def test_BD_F13_honest_capabilities_no_os_isolation_claim(self):
        caps = summarize_isolation_guarantees()
        self.assertTrue(caps["active"]["path_confinement"])
        self.assertFalse(caps["active"]["process_execution"])
        self.assertTrue(caps["not_enforced_in_this_environment"]["os_container"])
        self.assertTrue(caps["not_enforced_in_this_environment"]["escape_proof"])

    def test_BD_F14_symlink_escape_if_supported(self):
        """Symlink escape: SKIPPED when privileges unavailable."""
        root = Path(tempfile.mkdtemp(prefix="bd_sandbox_"))
        outside = Path(tempfile.mkdtemp(prefix="bd_outside_"))
        try:
            secret = outside / "secret.txt"
            secret.write_text("TOP_SECRET", encoding="utf-8")
            link = root / "escape.txt"
            try:
                link.symlink_to(secret)
            except OSError as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            resolved = path_sandbox.resolve_under_root(
                "escape.txt",
                root=root,
            )
            self.assertIsInstance(resolved, dict)
            self.assertEqual(resolved.get("error_code"), "PATH_TRAVERSAL_DENIED")
        finally:
            for path in (link if "link" in dir() else None, secret, outside, root):
                if path is None:
                    continue
                try:
                    if path.is_symlink() or path.is_file():
                        path.unlink(missing_ok=True)
                    elif path.is_dir():
                        path.rmdir()
                except OSError:
                    pass

    def test_BD_F15_legitimate_relative_read_still_allowed(self):
        # Use a known project file under backend root.
        sample = "requirements.txt"
        if not (path_sandbox.PROJECT_ROOT / sample).exists():
            self.skipTest("requirements.txt missing under project root")
        err = enforce_sandbox_for_tool(
            "read_file",
            {"file_path": sample},
            user_id="user-a",
        )
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
