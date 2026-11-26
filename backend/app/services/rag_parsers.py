"""Bounded multi-format document parsers for RAG ingestion.

Parsers never execute macros, never fetch remote resources, and never treat
document content as instructions. OCR is intentionally out of scope.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
from pathlib import PurePosixPath
from typing import Any

from app.services.rag_config import settings

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class ParserError(ValueError):
    """Safe parser rejection; mapped to RagError by the document service."""


def normalize_filename(filename: str | None) -> str:
    raw = (filename or "document.txt").replace("\\", "/")
    base = PurePosixPath(raw).name
    clean = "".join(ch for ch in base if ch.isprintable() and ch not in "\x00\r\n")
    clean = clean.strip()
    if not clean or len(clean) > 180:
        raise ParserError("Invalid filename")
    return clean


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_empty(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if not cleaned.strip():
        raise ParserError("The uploaded document contains no extractable text")
    return cleaned


def _check_control_ratio(text: str) -> None:
    disallowed = sum(
        1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\t"
    )
    if disallowed / max(1, len(text)) > 0.01:
        raise ParserError("Document contains unsupported control characters")


def _base_result(
    *,
    filename: str,
    file_type: str,
    text: str,
    raw: bytes,
    parser_version: str,
    pages: list[dict[str, Any]] | None = None,
    blocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = _reject_empty(text)
    _check_control_ratio(text)
    return {
        "filename": filename,
        "file_type": file_type,
        "text": text,
        "line_count": len(text.splitlines()),
        "content_sha256": _sha256(raw),
        "parser_version": parser_version,
        "pages": pages or [],
        "blocks": blocks or [],
    }


def parse_plain_text(
    filename: str,
    content_type: str | None,
    raw: bytes,
    *,
    file_type: str,
) -> dict[str, Any]:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    accepted = {"", "application/octet-stream", "text/plain"}
    if file_type in {"md", "markdown"}:
        accepted |= {"text/markdown", "text/x-markdown"}
    if mime not in accepted:
        raise ParserError("File type does not match the supported text formats")
    if raw.startswith((b"%PDF-", b"PK\x03\x04", b"\x89PNG")) or b"\x00" in raw:
        raise ParserError("The file content is binary or does not match its extension")
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ParserError("Document must be valid UTF-8 text") from exc
    return _base_result(
        filename=filename,
        file_type=file_type,
        text=text,
        raw=raw,
        parser_version=settings.parser_version_plain,
    )


def parse_pdf(filename: str, content_type: str | None, raw: bytes) -> dict[str, Any]:
    if not settings.enable_pdf:
        raise ParserError("PDF uploads are disabled")
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in {"", "application/octet-stream", "application/pdf"}:
        raise ParserError("File type does not match the supported PDF format")
    if not raw.startswith(b"%PDF-"):
        raise ParserError("The file content is not a valid PDF")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ParserError("PDF support is unavailable (pypdf is not installed)") from exc
    try:
        reader = PdfReader(io.BytesIO(raw), strict=False)
    except Exception as exc:
        raise ParserError(f"PDF could not be read ({type(exc).__name__})") from exc
    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
            if not unlocked:
                raise ParserError("Encrypted PDFs are not supported")
        except ParserError:
            raise
        except Exception as exc:
            raise ParserError("Encrypted PDFs are not supported") from exc
    pages: list[dict[str, Any]] = []
    try:
        total = len(reader.pages)
    except Exception as exc:
        raise ParserError(f"Corrupt or unsupported PDF ({type(exc).__name__})") from exc
    if total < 1:
        raise ParserError("PDF contains no pages")
    if total > 500:
        raise ParserError("PDF exceeds the maximum page limit")
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise ParserError(f"PDF page extraction failed ({type(exc).__name__})") from exc
        page_text = page_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if page_text:
            pages.append({"page": index, "text": page_text})
    if not pages:
        raise ParserError(
            "PDF contains no extractable text (scanned or image-only PDFs are not supported without OCR)"
        )
    text_parts: list[str] = []
    rebuilt_pages: list[dict[str, Any]] = []
    offset = 0
    for page in pages:
        segment = f"[Page {page['page']}]\n{page['text']}"
        start_char = offset
        end_char = offset + len(segment)
        rebuilt_pages.append(
            {
                "page": page["page"],
                "text": page["text"],
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        text_parts.append(segment)
        offset = end_char + 1
    text = "\n".join(text_parts)
    blocks = [
        {
            "type": "page",
            "page": page["page"],
            "text": f"[Page {page['page']}]\n{page['text']}",
            "heading_path": [f"Page {page['page']}"],
            "start_char": page["start_char"],
            "end_char": page["end_char"],
        }
        for page in rebuilt_pages
    ]
    return _base_result(
        filename=filename,
        file_type="pdf",
        text=text,
        raw=raw,
        parser_version=settings.parser_version_pdf,
        pages=rebuilt_pages,
        blocks=blocks,
    )


def parse_docx(filename: str, content_type: str | None, raw: bytes) -> dict[str, Any]:
    if not settings.enable_docx:
        raise ParserError("DOCX uploads are disabled")
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in {
        "",
        "application/octet-stream",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        raise ParserError("File type does not match the supported DOCX format")
    if not raw.startswith(b"PK\x03\x04"):
        raise ParserError("The file content is not a valid DOCX archive")
    try:
        from docx import Document
    except ImportError as exc:
        raise ParserError("DOCX support is unavailable (python-docx is not installed)") from exc
    try:
        document = Document(io.BytesIO(raw))
    except Exception as exc:
        raise ParserError(f"DOCX could not be read ({type(exc).__name__})") from exc

    heading_path: list[str] = []
    blocks: list[dict[str, Any]] = []
    parts: list[str] = []
    offset = 0

    def _append(block_type: str, text: str, *, level: int | None = None) -> None:
        nonlocal offset
        text = text.strip()
        if not text:
            return
        start = offset
        end = offset + len(text)
        blocks.append(
            {
                "type": block_type,
                "text": text,
                "level": level,
                "heading_path": list(heading_path),
                "start_char": start,
                "end_char": end,
            }
        )
        parts.append(text)
        offset = end + 1

    for paragraph in document.paragraphs:
        style_name = (paragraph.style.name if paragraph.style is not None else "") or ""
        text = paragraph.text or ""
        if style_name.lower().startswith("heading"):
            match = re.search(r"(\d+)", style_name)
            level = int(match.group(1)) if match else 1
            level = max(1, min(level, 6))
            heading_path = heading_path[: level - 1] + [text.strip()]
            _append("heading", text, level=level)
        else:
            _append("paragraph", text)

    for table in document.tables:
        rows: list[str] = []
        for row in table.rows:
            cells = [
                " ".join((cell.text or "").split())
                for cell in row.cells
                if " ".join((cell.text or "").split())
            ]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            _append("table", "\n".join(rows))

    text = "\n".join(parts)
    return _base_result(
        filename=filename,
        file_type="docx",
        text=text,
        raw=raw,
        parser_version=settings.parser_version_docx,
        blocks=blocks,
    )


def parse_html(filename: str, content_type: str | None, raw: bytes) -> dict[str, Any]:
    if not settings.enable_html:
        raise ParserError("HTML uploads are disabled")
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in {"", "application/octet-stream", "text/html", "application/xhtml+xml"}:
        raise ParserError("File type does not match the supported HTML format")
    # Reject binary/misleading payloads even when the extension is .html.
    if raw.startswith((b"%PDF-", b"PK\x03\x04", b"\x89PNG")) or b"\x00" in raw:
        raise ParserError("The file content is binary or does not match its extension")
    try:
        html = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ParserError("HTML document must be valid UTF-8 text") from exc
    try:
        from bs4 import BeautifulSoup
        from bs4.element import Tag
    except ImportError as exc:
        raise ParserError("HTML support is unavailable (beautifulsoup4 is not installed)") from exc

    # html.parser never fetches URLs; remove executable / remote containers first.
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "iframe", "object", "embed", "noscript", "template", "link", "meta"]
    ):
        tag.decompose()
    for tag in soup.find_all(True):
        for key in list(tag.attrs):
            lowered = str(key).lower()
            raw_value = tag.attrs.get(key, "")
            if isinstance(raw_value, list):
                value = " ".join(str(item) for item in raw_value).lower()
            else:
                value = str(raw_value).lower()
            if (
                lowered.startswith("on")
                or lowered in {"href", "src", "xlink:href", "action", "formaction", "poster"}
                or "javascript:" in value
                or "data:" in value
            ):
                del tag.attrs[key]

    heading_path: list[str] = []
    blocks: list[dict[str, Any]] = []
    parts: list[str] = []
    offset = 0
    # Skip nested duplicates (e.g. <li><p>…</p></li>) by only emitting leaf content tags.
    content_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"}

    def _append(block_type: str, text: str, *, level: int | None = None) -> None:
        nonlocal offset
        text = text.strip()
        if block_type != "code":
            text = " ".join(text.split())
        if not text:
            return
        start = offset
        end = offset + len(text)
        blocks.append(
            {
                "type": block_type,
                "text": text,
                "level": level,
                "heading_path": list(heading_path),
                "start_char": start,
                "end_char": end,
            }
        )
        parts.append(text)
        offset = end + 1

    body = soup.body or soup
    for element in body.find_all(list(content_tags)):
        if not isinstance(element, Tag):
            continue
        name = element.name.lower()
        # Prefer the outer list item when it wraps a paragraph.
        if name == "p" and element.find_parent("li") is not None:
            continue
        if name == "pre":
            _append("code", element.get_text("\n", strip=False))
            continue
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            text = element.get_text(" ", strip=True)
            heading_path = heading_path[: level - 1] + [text]
            _append("heading", text, level=level)
            continue
        if name == "li":
            _append("list", f"- {element.get_text(' ', strip=True)}")
            continue
        _append("paragraph", element.get_text(" ", strip=True))

    if not parts:
        fallback = soup.get_text("\n", strip=True)
        return _base_result(
            filename=filename,
            file_type="html",
            text=fallback,
            raw=raw,
            parser_version=settings.parser_version_html,
            blocks=[],
        )
    text = "\n".join(parts)
    return _base_result(
        filename=filename,
        file_type="html",
        text=text,
        raw=raw,
        parser_version=settings.parser_version_html,
        blocks=blocks,
    )


def parse_csv(filename: str, content_type: str | None, raw: bytes) -> dict[str, Any]:
    if not settings.enable_csv:
        raise ParserError("CSV uploads are disabled")
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime not in {"", "application/octet-stream", "text/csv", "application/csv", "text/plain"}:
        raise ParserError("File type does not match the supported CSV format")
    if raw.startswith((b"%PDF-", b"PK\x03\x04", b"\x89PNG")) or b"\x00" in raw:
        raise ParserError("The file content is binary or does not match its extension")
    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ParserError("CSV document must be valid UTF-8 text") from exc
    decoded = _CONTROL_RE.sub("", decoded)
    try:
        dialect = csv.Sniffer().sniff(decoded[:4096], delimiters=",\t|;")
    except csv.Error:
        dialect = csv.excel
    rows: list[list[str]] = []
    try:
        reader = csv.reader(io.StringIO(decoded), dialect)
        for index, row in enumerate(reader):
            if index >= 5000:
                raise ParserError("CSV exceeds the maximum row limit")
            rows.append([str(cell).strip() for cell in row])
    except ParserError:
        raise
    except csv.Error as exc:
        raise ParserError(f"Malformed CSV ({type(exc).__name__})") from exc
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        raise ParserError("CSV contains no data rows")
    headers = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    blocks: list[dict[str, Any]] = []
    parts: list[str] = []
    header_line = "Columns: " + " | ".join(headers)
    blocks.append(
        {
            "type": "csv_header",
            "text": header_line,
            "heading_path": ["CSV"],
            "start_char": 0,
            "end_char": len(header_line),
        }
    )
    parts.append(header_line)
    offset = len(header_line) + 1
    for row_index, row in enumerate(data_rows, start=1):
        paired = []
        for col_index, header in enumerate(headers):
            value = row[col_index] if col_index < len(row) else ""
            if header or value:
                paired.append(f"{header or f'column_{col_index + 1}'}: {value}")
        line = f"Row {row_index}: " + "; ".join(paired)
        start = offset
        end = offset + len(line)
        blocks.append(
            {
                "type": "csv_row",
                "text": line,
                "heading_path": ["CSV", f"Row {row_index}"],
                "start_char": start,
                "end_char": end,
                "row_index": row_index,
            }
        )
        parts.append(line)
        offset = end + 1
    text = "\n".join(parts)
    return _base_result(
        filename=filename,
        file_type="csv",
        text=text,
        raw=raw,
        parser_version=settings.parser_version_csv,
        blocks=blocks,
    )


def parse_document(filename: str, content_type: str | None, raw: bytes) -> dict[str, Any]:
    """Dispatch to a format parser with extension + content validation."""
    safe_name = normalize_filename(filename)
    suffix = PurePosixPath(safe_name.lower()).suffix
    if not raw:
        raise ParserError("The uploaded document is empty")
    if len(raw) > settings.max_upload_bytes:
        raise ParserError("Document exceeds the upload size limit")

    if suffix == ".txt":
        return parse_plain_text(safe_name, content_type, raw, file_type="txt")
    if suffix in {".md", ".markdown"}:
        return parse_plain_text(
            safe_name,
            content_type,
            raw,
            file_type=suffix.lstrip("."),
        )
    if suffix == ".pdf":
        return parse_pdf(safe_name, content_type, raw)
    if suffix == ".docx":
        return parse_docx(safe_name, content_type, raw)
    if suffix in {".html", ".htm"}:
        return parse_html(safe_name, content_type, raw)
    if suffix == ".csv":
        return parse_csv(safe_name, content_type, raw)
    raise ParserError(
        "Unsupported file type. Supported types are .txt, .md, .markdown, .pdf, .docx, .html, .htm, and .csv"
    )
