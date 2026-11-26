"""Character and structure-aware chunking for RAG documents.

Parent-child chunk storage is deferred: expanding child hits into parent text
requires Batch-2/3 retrieval wiring and ACL re-checks that are not part of
Batch 1. Structure metadata (heading_path, page) is stored on flat chunks.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.rag_config import settings

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^```")


class ChunkingError(ValueError):
    """Raised when chunk limits or configuration are invalid."""


def _line_number_at(text: str, index: int) -> int:
    return 1 + text.count("\n", 0, max(0, index))


def chunk_text_char_window(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_chunks: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministic character-window chunking with accurate source locations."""
    size = settings.chunk_size if chunk_size is None else int(chunk_size)
    shared = settings.chunk_overlap if overlap is None else int(overlap)

    if size < 1 or shared < 0 or shared >= size:
        raise ValueError(
            "chunk_size must be positive and overlap smaller than chunk_size"
        )
    if max_chunks is not None and max_chunks < 1:
        raise ValueError("max_chunks must be positive")
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    result: list[dict[str, Any]] = []
    start = 0
    line_number = 1
    line_scan_pos = 0

    while start < len(normalized):
        while line_scan_pos < start:
            if normalized[line_scan_pos] == "\n":
                line_number += 1
            line_scan_pos += 1

        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = max(
                normalized.rfind("\n", start, end),
                normalized.rfind(" ", start, end),
            )
            if boundary > start + size // 2:
                end = boundary

        raw_content = normalized[start:end]
        leading_whitespace = len(raw_content) - len(raw_content.lstrip())
        trailing_whitespace = len(raw_content) - len(raw_content.rstrip())
        content_start = start + leading_whitespace
        content_end = end - trailing_whitespace
        content = normalized[content_start:content_end]

        if content:
            if max_chunks is not None and len(result) >= max_chunks:
                raise ChunkingError(
                    "Document creates to many chunks for one ingestion request"
                )
            start_line = line_number + normalized.count("\n", start, content_start)
            end_line = start_line + normalized.count("\n", content_start, content_end)
            result.append(
                {
                    "text": content,
                    "chunk_index": len(result),
                    "start_char": content_start,
                    "end_char": content_end,
                    "start_line": start_line,
                    "end_line": end_line,
                    "heading_path": [],
                    "page": None,
                }
            )

        if end >= len(normalized):
            break
        start = max(start + 1, end - shared)

    return result


def _parse_markdown_blocks(text: str) -> list[dict[str, Any]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    blocks: list[dict[str, Any]] = []
    heading_path: list[str] = []
    in_fence = False
    fence_lines: list[str] = []
    fence_start_line = 1
    list_lines: list[str] = []
    list_start_line = 1
    para_lines: list[str] = []
    para_start_line = 1
    offset = 0
    line_start_offsets: list[int] = []
    cursor = 0
    for line in lines:
        line_start_offsets.append(cursor)
        cursor += len(line) + 1

    def flush_paragraph() -> None:
        nonlocal para_lines
        if not para_lines:
            return
        body = "\n".join(para_lines).strip()
        if body:
            start_line = para_start_line
            start_char = line_start_offsets[start_line - 1]
            end_char = line_start_offsets[start_line - 1 + len(para_lines) - 1] + len(
                lines[start_line - 1 + len(para_lines) - 1]
            )
            blocks.append(
                {
                    "type": "paragraph",
                    "text": body,
                    "heading_path": list(heading_path),
                    "start_char": start_char,
                    "end_char": end_char,
                    "start_line": start_line,
                    "end_line": start_line + len(para_lines) - 1,
                }
            )
        para_lines = []

    def flush_list() -> None:
        nonlocal list_lines
        if not list_lines:
            return
        body = "\n".join(list_lines).strip()
        if body:
            start_line = list_start_line
            start_char = line_start_offsets[start_line - 1]
            end_line = start_line + len(list_lines) - 1
            end_char = line_start_offsets[end_line - 1] + len(lines[end_line - 1])
            blocks.append(
                {
                    "type": "list",
                    "text": body,
                    "heading_path": list(heading_path),
                    "start_char": start_char,
                    "end_char": end_char,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            )
        list_lines = []

    for index, line in enumerate(lines, start=1):
        if _FENCE_RE.match(line.strip()):
            flush_paragraph()
            flush_list()
            if not in_fence:
                in_fence = True
                fence_lines = [line]
                fence_start_line = index
            else:
                fence_lines.append(line)
                body = "\n".join(fence_lines)
                start_char = line_start_offsets[fence_start_line - 1]
                end_char = line_start_offsets[index - 1] + len(line)
                blocks.append(
                    {
                        "type": "code",
                        "text": body,
                        "heading_path": list(heading_path),
                        "start_char": start_char,
                        "end_char": end_char,
                        "start_line": fence_start_line,
                        "end_line": index,
                    }
                )
                in_fence = False
                fence_lines = []
            continue
        if in_fence:
            fence_lines.append(line)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_path = heading_path[: level - 1] + [title]
            start_char = line_start_offsets[index - 1]
            end_char = start_char + len(line)
            blocks.append(
                {
                    "type": "heading",
                    "text": line.strip(),
                    "level": level,
                    "heading_path": list(heading_path),
                    "start_char": start_char,
                    "end_char": end_char,
                    "start_line": index,
                    "end_line": index,
                }
            )
            continue

        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            flush_paragraph()
            if not list_lines:
                list_start_line = index
            list_lines.append(line.rstrip())
            continue

        if not line.strip():
            flush_paragraph()
            flush_list()
            continue

        flush_list()
        if not para_lines:
            para_start_line = index
        para_lines.append(line.rstrip())

    flush_paragraph()
    flush_list()
    if in_fence and fence_lines:
        body = "\n".join(fence_lines)
        start_char = line_start_offsets[fence_start_line - 1]
        end_line = fence_start_line + len(fence_lines) - 1
        end_char = line_start_offsets[end_line - 1] + len(lines[end_line - 1])
        blocks.append(
            {
                "type": "code",
                "text": body,
                "heading_path": list(heading_path),
                "start_char": start_char,
                "end_char": end_char,
                "start_line": fence_start_line,
                "end_line": end_line,
            }
        )
    return blocks


def _pack_blocks(
    blocks: list[dict[str, Any]],
    *,
    source_text: str,
    chunk_size: int,
    overlap: int,
    max_chunks: int | None,
) -> list[dict[str, Any]]:
    """Pack blocks so stored chunk text equals source_text[start_char:end_char]."""
    if not blocks:
        return []
    normalized_source = source_text.replace("\r\n", "\n").replace("\r", "\n")
    result: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_len = 0

    def emit(group: list[dict[str, Any]]) -> None:
        if not group:
            return
        if max_chunks is not None and len(result) >= max_chunks:
            raise ChunkingError(
                "Document creates to many chunks for one ingestion request"
            )
        start_char = min(int(item["start_char"]) for item in group)
        end_char = max(int(item["end_char"]) for item in group)
        start_char = max(0, min(start_char, len(normalized_source)))
        end_char = max(start_char, min(end_char, len(normalized_source)))
        # Exact source span — offsets identify the stored chunk text.
        text = normalized_source[start_char:end_char]
        leading = len(text) - len(text.lstrip())
        trailing = len(text) - len(text.rstrip())
        content_start = start_char + leading
        content_end = end_char - trailing
        content = normalized_source[content_start:content_end]
        if not content.strip():
            return
        start_line = _line_number_at(normalized_source, content_start)
        end_line = _line_number_at(normalized_source, max(content_start, content_end - 1))
        heading: list[str] = []
        for item in group:
            path = list(item.get("heading_path") or [])
            if item.get("type") == "heading":
                continue
            if len(path) >= len(heading):
                heading = path
        if not heading:
            for item in group:
                path = list(item.get("heading_path") or [])
                if len(path) >= len(heading):
                    heading = path
        page = None
        for item in group:
            if item.get("page") is not None:
                page = item.get("page")
                break
        result.append(
            {
                "text": content,
                "chunk_index": len(result),
                "start_char": content_start,
                "end_char": content_end,
                "start_line": start_line,
                "end_line": end_line,
                "heading_path": heading,
                "page": page,
            }
        )

    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        block_type = block.get("type")
        atomic = block_type in {"code", "table", "page", "csv_header", "csv_row"}
        piece_len = len(text)
        if atomic and piece_len > chunk_size:
            emit(current)
            current = []
            current_len = 0
            block_start = int(block["start_char"])
            block_end = int(block["end_char"])
            block_slice = normalized_source[block_start:block_end]
            sub = chunk_text_char_window(
                block_slice,
                chunk_size=chunk_size,
                overlap=overlap,
                max_chunks=None,
            )
            for row in sub:
                if max_chunks is not None and len(result) >= max_chunks:
                    raise ChunkingError(
                        "Document creates to many chunks for one ingestion request"
                    )
                abs_start = block_start + int(row["start_char"])
                abs_end = block_start + int(row["end_char"])
                result.append(
                    {
                        "text": normalized_source[abs_start:abs_end],
                        "chunk_index": len(result),
                        "heading_path": list(block.get("heading_path") or []),
                        "page": block.get("page"),
                        "start_char": abs_start,
                        "end_char": abs_end,
                        "start_line": _line_number_at(normalized_source, abs_start),
                        "end_line": _line_number_at(
                            normalized_source, max(abs_start, abs_end - 1)
                        ),
                    }
                )
            continue

        separator = 2 if current else 0
        if current and current_len + separator + piece_len > chunk_size:
            emit(current)
            if overlap > 0 and current:
                tail = current[-1]
                current = [tail]
                current_len = len(str(tail.get("text") or ""))
            else:
                current = []
                current_len = 0
            separator = 2 if current else 0
            if current_len + separator + piece_len > chunk_size:
                emit(current)
                current = []
                current_len = 0
                separator = 0

        current.append(block)
        current_len += separator + piece_len

    emit(current)
    # Safety: never silently drop source content when packing yields nothing.
    if not result and normalized_source.strip():
        return chunk_text_char_window(
            normalized_source,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks,
        )
    return result


def chunk_markdown_structure(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_chunks: int | None = None,
) -> list[dict[str, Any]]:
    size = settings.chunk_size if chunk_size is None else int(chunk_size)
    shared = settings.chunk_overlap if overlap is None else int(overlap)
    if size < 1 or shared < 0 or shared >= size:
        raise ValueError(
            "chunk_size must be positive and overlap smaller than chunk_size"
        )
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = _parse_markdown_blocks(normalized)
    if not blocks:
        return chunk_text_char_window(
            normalized, chunk_size=size, overlap=shared, max_chunks=max_chunks
        )
    return _pack_blocks(
        blocks,
        source_text=normalized,
        chunk_size=size,
        overlap=shared,
        max_chunks=max_chunks,
    )


def chunk_structured_blocks(
    blocks: list[dict[str, Any]],
    *,
    full_text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_chunks: int | None = None,
) -> list[dict[str, Any]]:
    """Pack parser-produced blocks (PDF pages, DOCX/HTML/CSV sections)."""
    size = settings.chunk_size if chunk_size is None else int(chunk_size)
    shared = settings.chunk_overlap if overlap is None else int(overlap)
    normalized = full_text.replace("\r\n", "\n").replace("\r", "\n")
    if not blocks:
        return chunk_text_char_window(
            normalized, chunk_size=size, overlap=shared, max_chunks=max_chunks
        )
    normalized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        start_char = int(block.get("start_char") or 0)
        end_char = int(block.get("end_char") or (start_char + len(text)))
        start_line = int(block.get("start_line") or _line_number_at(normalized, start_char))
        end_line = int(
            block.get("end_line") or _line_number_at(normalized, max(0, end_char - 1))
        )
        normalized_blocks.append(
            {
                "type": block.get("type") or "paragraph",
                "text": text,
                "heading_path": list(block.get("heading_path") or []),
                "page": block.get("page"),
                "start_char": start_char,
                "end_char": end_char,
                "start_line": start_line,
                "end_line": end_line,
            }
        )
    return _pack_blocks(
        normalized_blocks,
        source_text=normalized,
        chunk_size=size,
        overlap=shared,
        max_chunks=max_chunks,
    )


def select_chunking_version(strategy: str, file_type: str) -> str:
    if strategy == "markdown" or file_type in {"md", "markdown"}:
        return settings.chunking_version_markdown
    return settings.chunking_version_char


def chunk_document(
    parsed: dict[str, Any],
    *,
    strategy: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    max_chunks: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Choose a chunker from strategy + parser metadata; return chunks and version."""
    text = str(parsed.get("text") or "")
    file_type = str(parsed.get("file_type") or "")
    chosen = (strategy or settings.chunking_strategy or "char").strip().lower()
    blocks = list(parsed.get("blocks") or [])

    if blocks and file_type in {"pdf", "docx", "html", "htm", "csv"}:
        chunks = chunk_structured_blocks(
            blocks,
            full_text=text,
            chunk_size=chunk_size,
            overlap=overlap,
            max_chunks=max_chunks,
        )
        version = (
            settings.chunking_version_markdown
            if chosen == "markdown"
            else f"structure-{file_type}-v1"
        )
        return chunks, version

    if chosen == "markdown" or file_type in {"md", "markdown"}:
        chunks = chunk_markdown_structure(
            text, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks
        )
        return chunks, settings.chunking_version_markdown

    chunks = chunk_text_char_window(
        text, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks
    )
    return chunks, settings.chunking_version_char
