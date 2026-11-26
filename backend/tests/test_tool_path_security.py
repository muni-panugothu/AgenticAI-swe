"""Phase A — filesystem sandbox security regression tests.

Uses temporary directories and fake secrets only. Never reads the real
backend ``.env`` or live credentials.

Symlink tests: a module-level probe records whether this OS/user can create
symlinks. When unsupported, those cases are skipped with an explicit reason
(reported in the suite summary — not treated as silent passes).
"""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.chatbot import execute_tool
from app.tools import code_search as code_search_mod
from app.tools import file_reader as file_reader_mod
from app.tools import path_sandbox as sandbox
from app.tools.code_search import search_code
from app.tools.file_reader import read_file
from app.tools.path_sandbox import (
    ERROR_PATH_DENIED,
    ERROR_PATH_TRAVERSAL,
    is_sensitive_basename,
    resolve_under_root,
)


def _write(path: Path, text: str = "print('ok')\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _probe_symlink_support() -> tuple[bool, str]:
    base = Path(tempfile.mkdtemp())
    try:
        target = base / "t.txt"
        target.write_text("x", encoding="utf-8")
        link = base / "l"
        try:
            link.symlink_to(target)
        except OSError as exc:
            return False, f"symlink create failed: {exc}"
        return True, "symlinks available"
    finally:
        try:
            if (base / "l").exists() or (base / "l").is_symlink():
                (base / "l").unlink()
        except OSError:
            pass
        try:
            (base / "t.txt").unlink()
            base.rmdir()
        except OSError:
            pass


SYMLINKS_SUPPORTED, SYMLINK_PROBE_REASON = _probe_symlink_support()


class TestSensitiveBasenamePolicy(unittest.TestCase):
    def test_env_variants(self):
        for name in (".env", ".env.local", ".env.production", ".ENV", "foo.env"):
            self.assertTrue(is_sensitive_basename(name), msg=name)

    def test_keys_and_pem(self):
        for name in ("id_rsa", "server.pem", "app.key", "store.p12", "x.pem.py"):
            self.assertTrue(is_sensitive_basename(name), msg=name)

    def test_source_helper_names_allowed(self):
        self.assertFalse(is_sensitive_basename("secret_scanner.py"))
        self.assertFalse(is_sensitive_basename("app_config.py"))

    def test_misleading_source_secret_stems_denied(self):
        for name in ("credentials.py", "secrets.py", "private_key.py", "Secrets.PY"):
            self.assertTrue(is_sensitive_basename(name), msg=name)

    def test_non_source_secret_names(self):
        self.assertTrue(is_sensitive_basename("secrets.txt"))
        self.assertTrue(is_sensitive_basename("credentials.json"))
        self.assertTrue(is_sensitive_basename("my_secret_config.yaml"))


class TestResolveUnderRoot(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name).resolve()
        _write(self.root / "app" / "hello.py", "def hello():\n    return 1\n")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_relative_ok(self):
        resolved = resolve_under_root("app/hello.py", root=self.root)
        self.assertIsInstance(resolved, Path)

    def test_mixed_separators_ok(self):
        resolved = resolve_under_root(r"app\hello.py", root=self.root)
        self.assertIsInstance(resolved, Path)

    def test_traversal_variants_denied(self):
        for attack in (
            "../outside.py",
            r"..\outside.py",
            "app/../../outside.py",
            r"app\..\..\outside.py",
            "./../outside.py",
        ):
            err = resolve_under_root(attack, root=self.root)
            self.assertEqual(err["error_code"], ERROR_PATH_TRAVERSAL, msg=attack)
            self.assertNotIn(str(self.root), err.get("message", ""))

    def test_null_byte_denied(self):
        err = resolve_under_root("app/hello.py\x00.env", root=self.root)
        self.assertEqual(err["error_code"], ERROR_PATH_TRAVERSAL)

    def test_absolute_denied(self):
        outside = Path(tempfile.gettempdir()).resolve() / "nope.py"
        err = resolve_under_root(str(outside), root=self.root)
        self.assertEqual(err["error_code"], ERROR_PATH_TRAVERSAL)

    def test_windows_drive_and_relative_drive_denied(self):
        for attack in (
            r"C:\Windows\System32\drivers\etc\hosts",
            "C:/Windows/win.ini",
            "C:foo.txt",
            r"D:\secrets\passwords.txt",
        ):
            err = resolve_under_root(attack, root=self.root)
            self.assertEqual(err["error_code"], ERROR_PATH_TRAVERSAL, msg=attack)

    def test_unc_denied(self):
        for attack in (r"\\server\share\secret", "//server/share/secret"):
            err = resolve_under_root(attack, root=self.root)
            self.assertEqual(err["error_code"], ERROR_PATH_TRAVERSAL, msg=attack)

    def test_env_denied(self):
        _write(self.root / ".env", "SECRET=fake-not-real\n")
        err = resolve_under_root(".env", root=self.root)
        self.assertEqual(err["error_code"], ERROR_PATH_DENIED)

    def test_nested_sensitive_directory_denied(self):
        _write(self.root / "app" / ".ssh" / "config", "Host x\n")
        err = resolve_under_root("app/.ssh/config", root=self.root)
        self.assertEqual(err["error_code"], ERROR_PATH_DENIED)

    @unittest.skipUnless(SYMLINKS_SUPPORTED, SYMLINK_PROBE_REASON)
    def test_symlink_escape_denied(self):
        outside_dir = Path(tempfile.mkdtemp())
        try:
            outside_file = outside_dir / "leak.txt"
            outside_file.write_text("LEAKED", encoding="utf-8")
            link = self.root / "app" / "escape_link"
            link.symlink_to(outside_file, target_is_directory=False)
            err = resolve_under_root("app/escape_link", root=self.root)
            self.assertEqual(err["error_code"], ERROR_PATH_TRAVERSAL)
            self.assertNotIn("LEAKED", str(err))
        finally:
            try:
                (self.root / "app" / "escape_link").unlink()
            except OSError:
                pass
            try:
                outside_file.unlink()
                outside_dir.rmdir()
            except OSError:
                pass


class _SandboxCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name).resolve()
        _write(self.root / "app" / "sample.py", "def alpha():\n    return 42\n")
        self._patches = [
            mock.patch.object(file_reader_mod, "PROJECT_ROOT", self.root),
            mock.patch.object(code_search_mod, "PROJECT_ROOT", self.root),
            mock.patch.object(sandbox, "PROJECT_ROOT", self.root),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmpdir.cleanup()


class TestReadFileSandbox(_SandboxCase):
    def test_env_denied(self):
        _write(self.root / ".env", "API_KEY=fake-test-key-not-real\n")
        result = read_file(".env", line_start=1, line_end=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], ERROR_PATH_DENIED)
        self.assertNotIn("fake-test-key", str(result))

    def test_env_local_denied(self):
        _write(self.root / ".env.local", "TOKEN=fake\n")
        result = read_file(".env.local", line_start=1, line_end=5)
        self.assertEqual(result["error_code"], ERROR_PATH_DENIED)

    def test_private_key_and_pem_denied(self):
        _write(self.root / "id_rsa", "-----BEGIN FAKE PRIVATE KEY-----\n")
        self.assertEqual(
            read_file("id_rsa", line_start=1, line_end=5)["error_code"],
            ERROR_PATH_DENIED,
        )
        _write(self.root / "certs" / "server.pem", "FAKEPEM\n")
        self.assertEqual(
            read_file("certs/server.pem", line_start=1, line_end=5)["error_code"],
            ERROR_PATH_DENIED,
        )

    def test_misleading_credentials_py_denied(self):
        _write(self.root / "app" / "credentials.py", "KEY='fake'\n")
        result = read_file("app/credentials.py", line_start=1, line_end=5)
        self.assertEqual(result["error_code"], ERROR_PATH_DENIED)

    def test_nested_sensitive_dir_denied(self):
        _write(self.root / "app" / "secrets" / "note.py", "x=1\n")
        result = read_file("app/secrets/note.py", line_start=1, line_end=5)
        self.assertEqual(result["error_code"], ERROR_PATH_DENIED)

    def test_traversal_and_absolute_denied(self):
        self.assertEqual(
            read_file("../outside.txt", line_start=1, line_end=5)["error_code"],
            ERROR_PATH_TRAVERSAL,
        )
        self.assertEqual(
            read_file(r"app\..\outside.txt", line_start=1, line_end=5)["error_code"],
            ERROR_PATH_TRAVERSAL,
        )
        outside = Path(tempfile.gettempdir()) / "phase_a_outside.py"
        outside.write_text("x=1\n", encoding="utf-8")
        try:
            self.assertEqual(
                read_file(str(outside.resolve()), line_start=1, line_end=5)["error_code"],
                ERROR_PATH_TRAVERSAL,
            )
        finally:
            outside.unlink(missing_ok=True)
        self.assertEqual(
            read_file(r"D:\secrets\passwords.txt", line_start=1, line_end=5)["error_code"],
            ERROR_PATH_TRAVERSAL,
        )

    @unittest.skipUnless(SYMLINKS_SUPPORTED, SYMLINK_PROBE_REASON)
    def test_symlink_escape_denied(self):
        outside_dir = Path(tempfile.mkdtemp())
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("LEAK", encoding="utf-8")
        link = self.root / "app" / "link_out"
        try:
            link.symlink_to(outside_file)
            result = read_file("app/link_out", line_start=1, line_end=5)
            self.assertEqual(result["error_code"], ERROR_PATH_TRAVERSAL)
            self.assertNotIn("LEAK", str(result))
        finally:
            try:
                link.unlink()
            except OSError:
                pass
            outside_file.unlink(missing_ok=True)
            try:
                outside_dir.rmdir()
            except OSError:
                pass

    def test_normal_source_readable(self):
        result = read_file("app/sample.py", line_start=1, line_end=10)
        self.assertIsInstance(result, str)
        self.assertIn("def alpha", result)

    def test_denial_message_has_no_absolute_root(self):
        result = read_file(".env", line_start=1, line_end=2)
        self.assertNotIn(str(self.root), str(result))


class TestSearchCodeSandbox(_SandboxCase):
    def setUp(self):
        super().setUp()
        _write(self.root / "app" / "greeter.py", "def greet():\n    return 'hi'\n")
        _write(self.root / ".env", "SHOULD_NOT_MATCH=1\nSECRET_MARKER=zzz\n")
        _write(self.root / "app" / "id_rsa", "PRIVATE_KEY_MARKER\n")
        _write(self.root / "app" / "secrets" / "hidden.py", "HIDDEN_TOKEN=1\n")

    def test_normal_search_works(self):
        result = search_code("greet", folder_path="app")
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result.get("match_count", 0), 1)
        for m in result["matches"]:
            self.assertFalse(os.path.isabs(m["file_path"]))
            self.assertNotIn(str(self.root), m["file_path"])

    def test_traversal_and_absolute_denied(self):
        self.assertEqual(search_code("x", folder_path="..")["error_code"], ERROR_PATH_TRAVERSAL)
        self.assertEqual(
            search_code("x", folder_path=str(Path(tempfile.gettempdir()).resolve()))[
                "error_code"
            ],
            ERROR_PATH_TRAVERSAL,
        )
        self.assertEqual(
            search_code("x", folder_path=r"C:\Windows\System32")["error_code"],
            ERROR_PATH_TRAVERSAL,
        )

    def test_secret_files_and_dirs_excluded(self):
        result = search_code("SECRET_MARKER", folder_path=".")
        self.assertTrue(result["success"])
        self.assertEqual(result.get("matches"), [])

        result2 = search_code("PRIVATE_KEY_MARKER", folder_path="app")
        self.assertEqual(result2.get("matches"), [])

        result3 = search_code("HIDDEN_TOKEN", folder_path="app")
        self.assertEqual(result3.get("matches"), [])

        # Direct search into sensitive dir denied.
        denied = search_code("HIDDEN_TOKEN", folder_path="app/secrets")
        self.assertFalse(denied["success"])
        self.assertEqual(denied["error_code"], ERROR_PATH_DENIED)

    @unittest.skipUnless(SYMLINKS_SUPPORTED, SYMLINK_PROBE_REASON)
    def test_symlink_escape_denied_for_folder(self):
        outside = Path(tempfile.mkdtemp())
        _write(outside / "leak.py", "LEAK_TOKEN = 1\n")
        link = self.root / "app" / "out_dir"
        try:
            link.symlink_to(outside, target_is_directory=True)
            result = search_code("LEAK_TOKEN", folder_path="app/out_dir")
            self.assertFalse(result["success"])
            self.assertIn(
                result["error_code"],
                {ERROR_PATH_TRAVERSAL, ERROR_PATH_DENIED},
            )
            self.assertNotIn("LEAK_TOKEN", str(result.get("matches")))
        finally:
            try:
                if link.exists() or link.is_symlink():
                    link.unlink()
            except OSError:
                pass
            try:
                (outside / "leak.py").unlink()
                outside.rmdir()
            except OSError:
                pass

    def test_inaccessible_files_do_not_crash(self):
        locked = self.root / "app" / "locked.py"
        _write(locked, "findme = 1\n")
        try:
            os.chmod(locked, 0)
        except OSError:
            pass
        try:
            result = search_code("findme", folder_path="app")
            self.assertIn("success", result)
        finally:
            try:
                os.chmod(locked, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass


def _unwrap_execute_tool(result):
    """execute_tool wraps tool payloads; recover the inner tool response."""
    import json

    if isinstance(result, dict) and "result" in result:
        inner = result["result"]
        if isinstance(inner, str):
            try:
                return json.loads(inner.split("\ncontent:\n", 1)[0])
            except json.JSONDecodeError:
                return inner
        return inner
    return result


class TestExecuteToolIntegration(_SandboxCase):
    """Actual tool invocation via execute_tool (validation + sandbox)."""

    def test_execute_tool_read_env_denied(self):
        _write(self.root / ".env", "API_KEY=fake-execute-tool\n")
        result = execute_tool(
            "read_file",
            {"file_path": ".env", "line_start": 1, "line_end": 5},
            user_id="phase-a-test",
        )
        inner = _unwrap_execute_tool(result)
        self.assertIsInstance(inner, dict)
        self.assertFalse(inner.get("success", True))
        self.assertEqual(inner.get("error_code"), ERROR_PATH_DENIED)
        self.assertNotIn("fake-execute-tool", str(result))

    def test_execute_tool_read_source_ok(self):
        result = execute_tool(
            "read_file",
            {"file_path": "app/sample.py", "line_start": 1, "line_end": 10},
            user_id="phase-a-test",
        )
        # Wrapper carries the successful read payload as a string with content.
        payload = result.get("result") if isinstance(result, dict) else result
        self.assertIsInstance(payload, str)
        self.assertIn("def alpha", payload)

    def test_execute_tool_search_traversal_denied(self):
        result = execute_tool(
            "search_code",
            {"search_text": "alpha", "folder_path": ".."},
            user_id="phase-a-test",
        )
        inner = _unwrap_execute_tool(result)
        self.assertIsInstance(inner, dict)
        self.assertFalse(inner.get("success", True))
        self.assertEqual(inner.get("error_code"), ERROR_PATH_TRAVERSAL)


class TestChatbotDenialHandling(unittest.TestCase):
    def test_prompt_mentions_path_denied(self):
        from app import chatbot

        self.assertIn("PATH_DENIED", chatbot.SYSTEM_PROMPT)

    def test_terminal_path_codes_include_path_denied(self):
        # Source-level contract: denied reads mark terminal paths.
        import inspect
        from app import chatbot

        src = inspect.getsource(chatbot.chat_with_ai)
        self.assertIn("PATH_DENIED", src)
        self.assertIn("PATH_TRAVERSAL_DENIED", src)


class TestLiveProjectRootRegression(unittest.TestCase):
    def test_real_source_readable(self):
        result = read_file("app/tools/calculator.py", line_start=1, line_end=20)
        self.assertIsInstance(result, str)

    def test_real_env_path_denied_without_reading(self):
        result = read_file(".env", line_start=1, line_end=5)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], ERROR_PATH_DENIED)
        joined = str(result)
        self.assertNotIn("GROQ_API_KEY", joined)
        self.assertNotIn("JWT_SECRET", joined)

    def test_real_search_app_works(self):
        result = search_code("def calculator", folder_path="app/tools")
        self.assertTrue(result["success"])
        self.assertGreaterEqual(
            result.get("match_count", len(result.get("matches", []))),
            1,
        )


class TestSymlinkSupportReport(unittest.TestCase):
    def test_symlink_support_is_reported(self):
        # Always runs — documents environment capability for CI readers.
        self.assertIsInstance(SYMLINKS_SUPPORTED, bool)
        self.assertTrue(SYMLINK_PROBE_REASON)


if __name__ == "__main__":
    unittest.main()
