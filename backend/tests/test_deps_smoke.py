"""Phase C — required vs optional dependency smoke checks.

Required imports must succeed. Optional packages (e.g. sentence-transformers)
are reported separately and must not fail the suite when absent.
"""

from __future__ import annotations

import importlib
import unittest
from importlib import metadata


REQUIRED_MODULES: dict[str, str] = {
    # module import path -> distribution name (for version reporting)
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "python_multipart": "python-multipart",
    "dotenv": "python-dotenv",
    "groq": "groq",
    "pymongo": "pymongo",
    "pydantic": "pydantic",
    "jwt": "PyJWT",
    "argon2": "argon2-cffi",
    "slowapi": "slowapi",
    "upstash_redis": "upstash-redis",
    "pypdf": "pypdf",
    "docx": "python-docx",
    "lxml": "lxml",
    "bs4": "beautifulsoup4",
    "tiktoken": "tiktoken",
    "keyring": "keyring",
    "langgraph": "langgraph",
    "langchain_core": "langchain-core",
    "google.genai": "google-genai",
    "requests": "requests",
    "email_validator": "email-validator",
}

OPTIONAL_MODULES: dict[str, str] = {
    "sentence_transformers": "sentence-transformers",
}


def _dist_version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "unknown"


class TestRequiredDependencies(unittest.TestCase):
    def test_required_modules_import(self):
        failures: list[str] = []
        for module_name, dist_name in REQUIRED_MODULES.items():
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 — report all import failures
                failures.append(f"{module_name} ({dist_name}): {type(exc).__name__}: {exc}")
        self.assertEqual(
            failures,
            [],
            "Required dependencies failed to import:\n" + "\n".join(failures),
        )

    def test_rag_parsers_are_required_not_optional(self):
        for module_name in ("pypdf", "docx", "lxml", "bs4"):
            self.assertIn(module_name, REQUIRED_MODULES)
            self.assertNotIn(module_name, OPTIONAL_MODULES)
            importlib.import_module(module_name)

    def test_groq_phase_b_client_api(self):
        """Preserve Phase B timeout / max_retries configuration surface."""
        import inspect

        from groq import Groq

        version = _dist_version("groq")
        self.assertTrue(version.startswith("1."), f"unexpected groq major: {version}")
        params = inspect.signature(Groq.__init__).parameters
        self.assertIn("timeout", params)
        self.assertIn("max_retries", params)
        # Default SDK retries remain 2; Phase B must pass max_retries=0 explicitly.
        self.assertEqual(params["max_retries"].default, 2)

    def test_pypdf_and_docx_usable(self):
        from io import BytesIO

        from docx import Document
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = BytesIO()
        writer.write(buf)
        reader = PdfReader(BytesIO(buf.getvalue()))
        self.assertGreaterEqual(len(reader.pages), 1)

        doc = Document()
        doc.add_paragraph("phase-c-smoke")
        out = BytesIO()
        doc.save(out)
        self.assertGreater(out.tell(), 0)

    def test_lxml_etree_available(self):
        from lxml import etree

        root = etree.fromstring(b"<a><b>1</b></a>")
        self.assertEqual(root.findtext("b"), "1")


class TestOptionalDependencies(unittest.TestCase):
    def test_optional_modules_reported_without_failing(self):
        status: dict[str, str] = {}
        for module_name, dist_name in OPTIONAL_MODULES.items():
            try:
                importlib.import_module(module_name)
                status[module_name] = f"present ({_dist_version(dist_name)})"
            except Exception as exc:  # noqa: BLE001
                status[module_name] = f"absent ({type(exc).__name__})"
        # Optional packages must never fail this test.
        self.assertEqual(set(status), set(OPTIONAL_MODULES))
        # sentence-transformers is intentionally optional for CI size.
        self.assertIn("sentence_transformers", status)


if __name__ == "__main__":
    unittest.main()
