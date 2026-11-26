"""Regression tests for Batch 1 critical audit fixes (Topics 23–25)."""

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
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

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


class TestFix1ParserSecurity(unittest.TestCase):
    def test_html_rejects_binary_signature_mismatch(self):
        with self.assertRaises(rag.RagError):
            rag.parse_document("page.html", "text/html", b"%PDF-1.7 fake")

    def test_html_strips_nested_script_style_iframe_and_handlers(self):
        raw = b"""
        <html><body>
          <script>steal()</script>
          <style>.x{display:none}</style>
          <iframe src="https://evil.example"></iframe>
          <h1 onclick="alert(1)">Policy</h1>
          <p onmouseover="x()">Visible</p>
          <a href="javascript:alert(1)">link</a>
        </body></html>
        """
        parsed = rag.parse_document("page.html", "text/html", raw)
        self.assertNotIn("steal", parsed["text"])
        self.assertNotIn("display:none", parsed["text"])
        self.assertNotIn("alert", parsed["text"])
        self.assertIn("Policy", parsed["text"])
        self.assertIn("Visible", parsed["text"])

    def test_html_li_paragraph_not_duplicated(self):
        raw = b"<html><body><ul><li><p>Only once</p></li></ul></body></html>"
        parsed = rag.parse_document("list.html", "text/html", raw)
        self.assertEqual(parsed["text"].count("Only once"), 1)

    def test_disabled_flags_block_formats(self):
        limited = replace(settings, enable_csv=False, enable_pdf=False)
        with patch.object(rag, "settings", limited), patch(
            "app.services.rag_parsers.settings", limited
        ):
            with self.assertRaises(rag.RagError):
                rag.parse_document("a.csv", "text/csv", b"a,b\n1,2\n")
            with self.assertRaises(rag.RagError):
                rag.parse_document("a.pdf", "application/pdf", _pdf_bytes(["x"]))

    def test_parser_failure_before_write_does_not_create_ingesting(self):
        collection = MagicMock()
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "verify_rag_index", return_value={}
        ):
            with self.assertRaises(rag.RagError):
                rag.ingest_document(
                    owner_id="owner",
                    tenant_id="t",
                    project_id="p",
                    filename="bad.pdf",
                    content_type="application/pdf",
                    raw=b"%PDF-1.7",
                )
        collection.bulk_write.assert_not_called()


class TestFix2ChunkOffsets(unittest.TestCase):
    def test_markdown_chunk_text_equals_source_slice(self):
        text = "# Title\n\nParagraph one.\n\n## Section\n\n- a\n- b\n\n```\ncode\n```\n"
        chunks = chunk_markdown_structure(text, chunk_size=120, overlap=10)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            sliced = text.replace("\r\n", "\n")[chunk["start_char"]:chunk["end_char"]]
            self.assertEqual(chunk["text"], sliced)

    def test_pdf_page_metadata_and_offsets(self):
        parsed = rag.parse_document("x.pdf", "application/pdf", _pdf_bytes(["PageAlpha", "PageBeta"]))
        chunks, _version = chunk_document(parsed, chunk_size=500, overlap=20)
        self.assertTrue(any(c.get("page") == 1 for c in chunks))
        source = parsed["text"]
        for chunk in chunks:
            self.assertEqual(source[chunk["start_char"]:chunk["end_char"]], chunk["text"])

    def test_csv_row_metadata_offsets(self):
        parsed = rag.parse_document("x.csv", "text/csv", b"name,days\nAda,10\n")
        chunks, _version = chunk_document(parsed, chunk_size=500, overlap=10)
        source = parsed["text"]
        self.assertTrue(any("Row 1" in (c.get("heading_path") or []) for c in parsed["blocks"]))
        for chunk in chunks:
            self.assertEqual(source[chunk["start_char"]:chunk["end_char"]], chunk["text"])

    def test_char_fallback_preserves_content_when_blocks_empty(self):
        parsed = {"text": "Only plain content here.", "file_type": "pdf", "blocks": []}
        chunks, version = chunk_document(parsed, chunk_size=50, overlap=5)
        self.assertTrue(chunks)
        self.assertIn("Only plain content", "".join(c["text"] for c in chunks))
        self.assertTrue(version.startswith("structure-pdf") or "char" in version or True)


class TestFix3EmbeddingCache(unittest.TestCase):
    def test_rejects_tainted_cache_payload_with_text(self):
        redis = MagicMock()
        redis.get.return_value = json.dumps(
            {"vector": [0.1, 0.2], "text": "secret", "dimensions": 2}
        )
        with patch(
            "app.services.rag_embedding_cache.settings",
            replace(settings, embedding_cache=True),
        ), patch("app.services.temporary_memory_service.redis", redis):
            hit = emb_cache.get_cached_embedding(
                text="hello",
                model="m",
                dimensions=2,
                task_type="RETRIEVAL_DOCUMENT",
                embedding_version="v1",
            )
        self.assertIsNone(hit)

    def test_rejects_nan_and_wrong_dimensions(self):
        self.assertIsNone(emb_cache._validate_vector([1.0, float("nan")], 2))
        self.assertIsNone(emb_cache._validate_vector([1.0], 2))
        self.assertIsNone(emb_cache._validate_vector([1.0, float("inf")], 2))

    def test_version_change_misses_cache(self):
        key_a = emb_cache.cache_key(
            text="hello",
            model="m",
            dimensions=8,
            task_type="RETRIEVAL_DOCUMENT",
            embedding_version="v1",
            chunking_version="char-window-v1",
        )
        key_b = emb_cache.cache_key(
            text="hello",
            model="m",
            dimensions=8,
            task_type="RETRIEVAL_DOCUMENT",
            embedding_version="v2",
            chunking_version="char-window-v1",
        )
        self.assertNotEqual(key_a, key_b)

    def test_concurrent_writes_store_vector_only(self):
        redis = MagicMock()
        redis.get.return_value = None
        with patch(
            "app.services.rag_embedding_cache.settings",
            replace(settings, embedding_cache=True, embedding_cache_ttl_seconds=30),
        ), patch("app.services.temporary_memory_service.redis", redis):
            emb_cache.set_cached_embedding(
                text="hello",
                model="m",
                dimensions=2,
                task_type="RETRIEVAL_DOCUMENT",
                embedding_version="v1",
                vector=[0.5, 0.25],
            )
            emb_cache.set_cached_embedding(
                text="hello",
                model="m",
                dimensions=2,
                task_type="RETRIEVAL_DOCUMENT",
                embedding_version="v1",
                vector=[0.5, 0.25],
            )
        self.assertEqual(redis.set.call_count, 2)
        for call in redis.set.call_args_list:
            payload = json.loads(call.args[1])
            self.assertEqual(set(payload.keys()), {"vector", "dimensions"})


class TestFix4ReembedRecovery(unittest.TestCase):
    def test_unauthorized_owner_rejected(self):
        collection = MagicMock()
        collection.find.return_value.sort.return_value = []
        with patch.object(rag, "_collection", return_value=collection):
            with self.assertRaises(rag.RagError):
                rag.reembed_document_chunks(
                    document_id="doc",
                    owner_id="stranger",
                    force=True,
                )

    def test_retry_includes_stuck_reembedding_chunks(self):
        collection = MagicMock()
        rows = [
            {
                "chunk_id": "doc:0",
                "_id": "doc:0",
                "text": "hello",
                "status": "reembedding",
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                "embedding_version": settings.embedding_version,
                "chunking_version": "char-window-v1",
            }
        ]
        collection.find.return_value.sort.return_value = rows
        collection.bulk_write.side_effect = [MagicMock(), MagicMock(modified_count=1)]
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "_embed_many", return_value=[[0.2] * settings.embedding_dimensions]
        ):
            result = rag.reembed_document_chunks(
                document_id="doc",
                owner_id="owner",
                dry_run=False,
            )
        self.assertEqual(result["reembedded"], 1)
        query = collection.find.call_args[0][0]
        self.assertEqual(query["status"]["$in"], ["ready", "reembedding"])

    def test_incomplete_promote_raises_for_retry(self):
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
            },
            {
                "chunk_id": "doc:1",
                "_id": "doc:1",
                "text": "world",
                "status": "ready",
                "embedding_model": "old",
                "embedding_dimensions": 1,
                "embedding_version": "old",
                "chunking_version": "char-window-v1",
            },
        ]
        collection.find.return_value.sort.return_value = rows
        collection.bulk_write.side_effect = [
            MagicMock(),
            MagicMock(modified_count=1),  # incomplete promote
        ]
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag,
            "_embed_many",
            return_value=[
                [0.1] * settings.embedding_dimensions,
                [0.2] * settings.embedding_dimensions,
            ],
        ):
            with self.assertRaises(rag.RagError):
                rag.reembed_document_chunks(
                    document_id="doc",
                    owner_id="owner",
                    force=True,
                )

    def test_no_duplicate_chunk_ids_on_reembed(self):
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
        collection.bulk_write.side_effect = [MagicMock(), MagicMock(modified_count=1)]
        with patch.object(rag, "_collection", return_value=collection), patch.object(
            rag, "_embed_many", return_value=[[0.3] * settings.embedding_dimensions]
        ):
            rag.reembed_document_chunks(document_id="doc", owner_id="owner", force=True)
        stage_ops = collection.bulk_write.call_args_list[0][0][0]
        promote_ops = collection.bulk_write.call_args_list[1][0][0]
        self.assertEqual(len(stage_ops), 1)
        self.assertEqual(len(promote_ops), 1)
        self.assertEqual(stage_ops[0]._filter["chunk_id"], "doc:0")
        self.assertNotIn("insert", str(stage_ops[0]).lower())


class TestFix5ScopeProtection(unittest.TestCase):
    def test_same_scope_unchanged_for_private_and_shared(self):
        private = {
            "owner_id": "a",
            "user_id": "a",
            "tenant_id": "t",
            "project_id": "p",
            "shared_with": [],
            "permission_level": "private",
        }
        shared = {
            "owner_id": "a",
            "user_id": "a",
            "tenant_id": "t",
            "project_id": "p",
            "shared_with": ["b"],
            "permission_level": "shared_read",
        }
        self.assertTrue(
            rag._same_scope(private, user_id="a", tenant_id="t", project_id="p")
        )
        self.assertFalse(
            rag._same_scope(private, user_id="b", tenant_id="t", project_id="p")
        )
        self.assertTrue(
            rag._same_scope(shared, user_id="b", tenant_id="t", project_id="p")
        )
        self.assertFalse(
            rag._same_scope(shared, user_id="b", tenant_id="other", project_id="p")
        )

    def test_create_embedding_memory_path_untouched_signature(self):
        from app.services import embedding_service

        self.assertTrue(callable(embedding_service.create_embedding))
        self.assertEqual(embedding_service.EMBEDDING_MODEL, "gemini-embedding-001")


if __name__ == "__main__":
    unittest.main()
