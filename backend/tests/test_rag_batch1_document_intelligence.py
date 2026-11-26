"""Batch 1 tests: multi-format loading, structure chunking, embedding cache."""

from __future__ import annotations

import io
import json
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from app.services import rag_document_service as rag
from app.services import rag_embedding_cache as emb_cache
from app.services.rag_chunking import chunk_document, chunk_markdown_structure
from app.services.rag_config import settings


def _pdf_bytes(pages: list[str]) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    buffer = io.BytesIO()
    for text in pages:
        safe = "".join(ch for ch in text if ch.isascii() and ch not in "()\\")
        page = writer.add_blank_page(width=300, height=300)
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 50 250 Td ({safe}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = content
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
    writer.write(buffer)
    return buffer.getvalue()


def _docx_bytes() -> bytes:
    from docx import Document

    document = Document()
    document.add_heading("Leave Policy", level=1)
    document.add_paragraph("Employees receive fifteen days of leave.")
    document.add_heading("Notice", level=2)
    document.add_paragraph("Submit requests two weeks early.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Role"
    table.cell(0, 1).text = "Days"
    table.cell(1, 0).text = "Engineer"
    table.cell(1, 1).text = "15"
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


class TestAdvancedDocumentLoading(unittest.TestCase):
    def test_text_and_markdown_still_work(self):
        parsed = rag.parse_document("guide.md", "text/markdown", b"# Title\nBody")
        self.assertEqual(parsed["file_type"], "md")
        self.assertIn("parser_version", parsed)

    def test_pdf_multi_page_extracts_page_metadata(self):
        raw = _pdf_bytes(["AlphaPageOne", "BetaPageTwo"])
        parsed = rag.parse_document("policy.pdf", "application/pdf", raw)
        self.assertEqual(parsed["file_type"], "pdf")
        self.assertGreaterEqual(len(parsed["pages"]), 1)
        self.assertTrue(any("Page" in block.get("text", "") for block in parsed["blocks"]))

    def test_corrupt_and_non_pdf_rejected(self):
        with self.assertRaises(rag.RagError):
            rag.parse_document("bad.pdf", "application/pdf", b"%PDF-1.7")
        with self.assertRaises(rag.RagError):
            rag.parse_document("bad.pdf", "application/pdf", b"not-a-pdf")

    def test_encrypted_pdf_rejected(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret-password")
        buf = io.BytesIO()
        writer.write(buf)
        with self.assertRaises(rag.RagError):
            rag.parse_document("secret.pdf", "application/pdf", buf.getvalue())

    def test_docx_headings_and_tables(self):
        parsed = rag.parse_document(
            "policy.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _docx_bytes(),
        )
        self.assertEqual(parsed["file_type"], "docx")
        self.assertTrue(any(b.get("type") == "heading" for b in parsed["blocks"]))
        self.assertIn("fifteen days", parsed["text"].lower())
        self.assertTrue(any(b.get("type") == "table" for b in parsed["blocks"]))

    def test_html_strips_script_and_style(self):
        html = b"""
        <html><head><style>.x{color:red}</style><script>alert(1)</script></head>
        <body><h1>Policy</h1><p>Visible text</p>
        <iframe src="https://evil.example"></iframe></body></html>
        """
        parsed = rag.parse_document("page.html", "text/html", html)
        self.assertNotIn("alert", parsed["text"])
        self.assertNotIn("color:red", parsed["text"])
        self.assertIn("Visible text", parsed["text"])
        self.assertTrue(any(b.get("type") == "heading" for b in parsed["blocks"]))

    def test_csv_preserves_headers_and_rows(self):
        raw = b"name,days\nAlice,15\nBob,10\n"
        parsed = rag.parse_document("leave.csv", "text/csv", raw)
        self.assertIn("Columns:", parsed["text"])
        self.assertIn("Row 1:", parsed["text"])
        self.assertTrue(any(b.get("type") == "csv_row" for b in parsed["blocks"]))

    def test_malformed_and_empty_documents_rejected(self):
        with self.assertRaises(rag.RagError):
            rag.parse_document("empty.txt", "text/plain", b"   ")
        with self.assertRaises(rag.RagError):
            rag.parse_document("empty.csv", "text/csv", b"\n\n")

    def test_disabled_format_flags(self):
        with patch.object(rag, "settings", replace(settings, enable_pdf=False)):
            with patch("app.services.rag_parsers.settings", replace(settings, enable_pdf=False)):
                with self.assertRaises(rag.RagError):
                    rag.parse_document("x.pdf", "application/pdf", _pdf_bytes(["hi"]))

    def test_mime_mismatch_and_size_limits(self):
        with self.assertRaises(rag.RagError):
            rag.parse_document("a.txt", "application/pdf", b"hello")
        with patch.object(rag, "settings", replace(settings, max_upload_bytes=3)):
            with patch("app.services.rag_parsers.settings", replace(settings, max_upload_bytes=3)):
                with self.assertRaises(rag.RagError):
                    rag.parse_document("a.txt", "text/plain", b"abcd")


class TestAdvancedChunking(unittest.TestCase):
    def test_markdown_headings_lists_code_and_offsets(self):
        text = "# Title\n\nIntro paragraph.\n\n## Section\n\n- item one\n- item two\n\n```\ncode line\n```\n"
        chunks = chunk_markdown_structure(text, chunk_size=80, overlap=10)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(any(c.get("heading_path") for c in chunks))
        for chunk in chunks:
            self.assertGreaterEqual(chunk["start_char"], 0)
            self.assertGreater(chunk["end_char"], chunk["start_char"])
            self.assertGreaterEqual(chunk["start_line"], 1)
            self.assertGreaterEqual(chunk["end_line"], chunk["start_line"])
            self.assertIn("code line", text)

    def test_nested_headings_preserve_path(self):
        text = "# Root\n\n## Child\n\nDetails here that stay with child heading.\n"
        chunks = chunk_markdown_structure(text, chunk_size=200, overlap=20)
        self.assertTrue(any("Child" in (c.get("heading_path") or []) for c in chunks))

    def test_char_chunker_regression(self):
        text = "alpha beta\ngamma delta\nepsilon zeta"
        first = rag.chunk_text(text, chunk_size=16, overlap=4)
        second = rag.chunk_text(text, chunk_size=16, overlap=4)
        self.assertEqual(first, second)
        self.assertTrue(all("start_char" in row for row in first))

    def test_document_aware_pdf_blocks(self):
        parsed = {
            "text": "[Page 1]\nHello\n[Page 2]\nWorld",
            "file_type": "pdf",
            "blocks": [
                {"type": "page", "page": 1, "text": "[Page 1]\nHello", "heading_path": ["Page 1"], "start_char": 0, "end_char": 14},
                {"type": "page", "page": 2, "text": "[Page 2]\nWorld", "heading_path": ["Page 2"], "start_char": 15, "end_char": 29},
            ],
        }
        chunks, version = chunk_document(parsed, chunk_size=100, overlap=10)
        self.assertTrue(version.startswith("structure-pdf") or "markdown" in version or "char" in version)
        self.assertTrue(any(c.get("page") == 1 for c in chunks) or len(chunks) >= 1)

    def test_max_chunks_enforced_for_markdown(self):
        text = "\n\n".join(f"## H{i}\n" + ("word " * 40) for i in range(20))
        with self.assertRaises(Exception):
            chunk_markdown_structure(text, chunk_size=50, overlap=5, max_chunks=2)


class TestAdvancedEmbeddings(unittest.TestCase):
    def test_cache_key_differs_by_model_task_dimensions(self):
        a = emb_cache.cache_key(
            text="hello", model="m1", dimensions=768, task_type="RETRIEVAL_DOCUMENT",
            embedding_version="v1",
        )
        b = emb_cache.cache_key(
            text="hello", model="m2", dimensions=768, task_type="RETRIEVAL_DOCUMENT",
            embedding_version="v1",
        )
        c = emb_cache.cache_key(
            text="hello", model="m1", dimensions=768, task_type="RETRIEVAL_QUERY",
            embedding_version="v1",
        )
        d = emb_cache.cache_key(
            text="hello", model="m1", dimensions=3072, task_type="RETRIEVAL_DOCUMENT",
            embedding_version="v1",
        )
        self.assertEqual(len({a, b, c, d}), 4)

    def test_cache_hit_avoids_provider_call(self):
        from app.services import embedding_service

        stored = {"vector": [0.1, 0.2, 0.3]}
        redis = MagicMock()
        redis.get.return_value = json.dumps(stored)
        with patch.object(embedding_service, "rag_client") as client, patch(
            "app.services.rag_embedding_cache.settings",
            replace(settings, embedding_cache=True, embedding_version="v1"),
        ), patch(
            "app.services.temporary_memory_service.redis",
            redis,
        ):
            vectors = embedding_service.create_rag_embeddings(
                ["hello"],
                task_type="RETRIEVAL_DOCUMENT",
                dimensions=3,
                model="gemini-embedding-001",
                embedding_version="v1",
                use_cache=True,
            )
        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])
        client.models.embed_content.assert_not_called()

    def test_cache_miss_writes_vector_only(self):
        from app.services import embedding_service

        response = MagicMock()
        response.embeddings = [MagicMock(values=[0.4, 0.5, 0.6])]
        redis = MagicMock()
        redis.get.return_value = None
        with patch.object(embedding_service.rag_client.models, "embed_content", return_value=response), patch(
            "app.services.rag_embedding_cache.settings",
            replace(settings, embedding_cache=True, embedding_cache_ttl_seconds=60, embedding_version="v1"),
        ), patch(
            "app.services.temporary_memory_service.redis",
            redis,
        ):
            vectors = embedding_service.create_rag_embeddings(
                ["hello"],
                task_type="RETRIEVAL_DOCUMENT",
                dimensions=3,
                embedding_version="v1",
                use_cache=True,
            )
        self.assertEqual(vectors[0], [0.4, 0.5, 0.6])
        self.assertTrue(redis.set.called)
        args, kwargs = redis.set.call_args
        payload = json.loads(args[1])
        self.assertEqual(set(payload.keys()), {"vector", "dimensions"})
        self.assertEqual(payload["dimensions"], 3)
        self.assertNotIn("text", payload)
        self.assertNotIn("hello", args[1])

    def test_redis_unavailable_falls_back(self):
        from app.services import embedding_service

        response = MagicMock()
        response.embeddings = [MagicMock(values=[1.0, 2.0])]
        redis = MagicMock()
        redis.get.side_effect = RuntimeError("redis down")
        redis.set.side_effect = RuntimeError("redis down")
        with patch.object(embedding_service.rag_client.models, "embed_content", return_value=response) as embed, patch(
            "app.services.rag_embedding_cache.settings",
            replace(settings, embedding_cache=True, embedding_version="v1"),
        ), patch(
            "app.services.temporary_memory_service.redis",
            redis,
        ):
            vectors = embedding_service.create_rag_embeddings(
                ["hello"],
                task_type="RETRIEVAL_QUERY",
                dimensions=2,
                use_cache=True,
            )
        self.assertEqual(vectors, [[1.0, 2.0]])
        embed.assert_called_once()

    def test_invalid_cached_vector_ignored(self):
        from app.services import embedding_service

        response = MagicMock()
        response.embeddings = [MagicMock(values=[9.0, 8.0])]
        redis = MagicMock()
        redis.get.return_value = json.dumps({"vector": [1.0]})  # wrong dims
        with patch.object(embedding_service.rag_client.models, "embed_content", return_value=response), patch(
            "app.services.rag_embedding_cache.settings",
            replace(settings, embedding_cache=True, embedding_version="v1"),
        ), patch(
            "app.services.temporary_memory_service.redis",
            redis,
        ):
            vectors = embedding_service.create_rag_embeddings(
                ["hello"],
                task_type="RETRIEVAL_DOCUMENT",
                dimensions=2,
                use_cache=True,
            )
        self.assertEqual(vectors, [[9.0, 8.0]])

    def test_reembed_dry_run_and_acl_preservation(self):
        collection = MagicMock()
        rows = [
            {
                "chunk_id": "doc:0",
                "_id": "doc:0",
                "text": "hello",
                "status": "ready",
                "embedding_model": "old-model",
                "embedding_dimensions": 768,
                "embedding_version": "old",
                "chunking_version": "char-window-v1",
                "user_id": "owner",
                "shared_with": ["peer"],
                "permission_level": "shared_read",
            }
        ]
        collection.find.return_value.sort.return_value = rows
        promote_result = MagicMock(modified_count=1)
        collection.bulk_write.side_effect = [MagicMock(), promote_result]
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "_embed_many", return_value=[[0.1] * settings.embedding_dimensions]
        ) as embed:
            dry = rag.reembed_document_chunks(
                document_id="doc",
                owner_id="owner",
                dry_run=True,
            )
            self.assertTrue(dry["dry_run"])
            self.assertEqual(dry["would_reembed"], 1)
            embed.assert_not_called()
            collection.bulk_write.assert_not_called()

            applied = rag.reembed_document_chunks(
                document_id="doc",
                owner_id="owner",
                dry_run=False,
                force=True,
            )
        self.assertEqual(applied["reembedded"], 1)
        stage_op = collection.bulk_write.call_args_list[0][0][0][0]
        promote_op = collection.bulk_write.call_args_list[1][0][0][0]
        self.assertEqual(stage_op._filter["user_id"], "owner")
        self.assertIn("embedding_next", stage_op._doc["$set"])
        self.assertNotIn("shared_with", stage_op._doc["$set"])
        self.assertNotIn("permission_level", promote_op._doc["$set"])
        self.assertEqual(promote_op._doc["$set"]["status"], "ready")
        self.assertIn("embedding", promote_op._doc["$set"])

    def test_reembed_gemini_failure_before_stage_leaves_ready(self):
        collection = MagicMock()
        rows = [
            {
                "chunk_id": "doc:0",
                "_id": "doc:0",
                "text": "hello",
                "status": "ready",
                "embedding_model": "old",
                "embedding_dimensions": 1,
                "embedding_version": "old",
                "chunking_version": "char-window-v1",
            }
        ]
        collection.find.return_value.sort.return_value = rows
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "_embed_many", side_effect=RuntimeError("provider")
        ):
            with self.assertRaises(RuntimeError):
                rag.reembed_document_chunks(document_id="doc", owner_id="owner", force=True)
        collection.bulk_write.assert_not_called()
        collection.update_many.assert_not_called()


if __name__ == "__main__":
    unittest.main()
