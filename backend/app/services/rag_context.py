"""Token-aware context selection and optional LLM compression."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.rag_config import settings
from app.services.rag_tokens import count_tokens

logger = logging.getLogger("agentic.rag")
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def evidence_block(row: dict[str, Any], *, text_override: str | None = None) -> str:
    text = (text_override if text_override is not None else row.get("text") or "").strip()
    return json.dumps(
        {
            "source_id": row["chunk_id"],
            "filename": row.get("filename") or "",
            "start_line": row.get("start_line"),
            "end_line": row.get("end_line"),
            "evidence": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def evidence_wrapper_overhead_tokens(prefix: str, assistant_ack: str) -> int:
    """Exact token overhead of evidence messages excluding chunk JSON body."""
    return count_tokens(prefix) + count_tokens(assistant_ack)


def _token_set(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1}


def is_near_duplicate(a: str, b: str, threshold: float | None = None) -> bool:
    thr = settings.near_duplicate_jaccard if threshold is None else threshold
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return a.strip().casefold() == b.strip().casefold()
    return (len(ta & tb) / len(ta | tb)) >= thr


def select_context_token_budget(
    candidates: list[dict[str, Any]],
    *,
    budget_tokens: int | None = None,
    max_chunks: int | None = None,
    wrapper_overhead: int = 0,
    drop_near_duplicates: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Pack whole evidence JSON blocks under a token budget; never truncate text."""
    limit = settings.context_token_budget if budget_tokens is None else int(budget_tokens)
    chunk_cap = settings.context_max_chunks if max_chunks is None else int(max_chunks)
    if wrapper_overhead > limit:
        return "", []
    pack_limit = limit - wrapper_overhead
    separator = "\n\n"
    sep_tokens = count_tokens(separator)
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    blocks: list[str] = []
    used = 0
    # Rank by score descending when present.
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row.get("score") or 0.0), str(row.get("chunk_id") or "")),
    )
    for row in ordered:
        if len(selected) >= chunk_cap:
            break
        cid = str(row.get("chunk_id") or "")
        if not cid or cid in seen:
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        if drop_near_duplicates and any(
            is_near_duplicate(text, str(prev.get("text") or "")) for prev in selected
        ):
            continue
        block = evidence_block(row)
        block_tokens = count_tokens(block)
        extra = sep_tokens if blocks else 0
        if used + extra + block_tokens > pack_limit:
            continue
        seen.add(cid)
        selected.append(row)
        blocks.append(block)
        used += extra + block_tokens
    return separator.join(blocks), selected


def compress_evidence_llm(
    query: str,
    selected: list[dict[str, Any]],
    *,
    max_output_tokens: int | None = None,
    enabled: bool | None = None,
) -> list[dict[str, Any]] | None:
    """Optional Groq compression; preserves chunk IDs. Returns None on failure."""
    active = settings.enable_semantic_compression if enabled is None else enabled
    if not active or not selected:
        return None
    cap = max_output_tokens or settings.compression_max_output_tokens
    try:
        from app import chatbot

        client = getattr(chatbot, "client", None)
        if client is None:
            return None
        payload = [
            {
                "source_id": row["chunk_id"],
                "evidence": row.get("text") or "",
            }
            for row in selected
        ]
        prompt = (
            "Compress each evidence item for answering the question. "
            "Keep the same source_id values. Do not invent facts. "
            "Return JSON array of {source_id, evidence}. "
            f"Question: {query}\nEvidence: {json.dumps(payload, ensure_ascii=False)}"
        )
        response = client.chat.completions.create(
            model=getattr(chatbot, "MODEL", None) or "openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=cap,
        )
        content = (response.choices[0].message.content or "").strip()
        start, end = content.find("["), content.rfind("]")
        if start < 0 or end <= start:
            return None
        parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, list):
            return None
        by_id = {str(row["chunk_id"]): row for row in selected}
        compressed: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                return None
            sid = str(item.get("source_id") or "")
            if sid not in by_id:
                return None
            evidence = str(item.get("evidence") or "").strip()
            if not evidence:
                return None
            updated = dict(by_id[sid])
            updated["text"] = evidence
            updated["compressed"] = True
            compressed.append(updated)
        # Preserve original order / membership.
        if {row["chunk_id"] for row in compressed} != set(by_id):
            # Allow subset only if all IDs were authorized originals.
            if not set(row["chunk_id"] for row in compressed).issubset(set(by_id)):
                return None
        return compressed
    except Exception as exc:
        logger.warning("rag_compression_failed error=%s", type(exc).__name__)
        return None
