"""Controlled query rewriting: multi-query, conversation clarify, optional HyDE.

Never changes document ACL. Hypothetical text is retrieval-only and must never
be cited or stored as evidence.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from app.services.rag_config import settings

logger = logging.getLogger("agentic.rag")

_SPLIT_RE = re.compile(r"\s+(?:and|also|additionally|plus)\s+|[;?]\s*", re.I)
_PRONOUN_RE = re.compile(r"(?i)\b(it|its|that|this|they|them|those|these)\b")
_TOPIC_FROM_ABOUT_RE = re.compile(
    r"(?i)\b(?:about|regarding|concerning)\s+"
    r"([A-Za-z0-9][\w'-]*(?:\s+[A-Za-z0-9][\w'-]*){0,4})"
)
_TOPIC_DOCUMENT_NOUN_RE = re.compile(
    r"(?i)\b(the\s+(?:uploaded\s+)?"
    r"(?:[\w'-]+\s+){0,2}"
    r"(?:policy|document|file|report|manual|guide|source|contract|handbook))\b"
)
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1]


def _near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return a.casefold() == b.casefold()
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) >= threshold if union else False


def _history_user_texts(conversation_history: Iterable[Any] | None) -> list[str]:
    if not conversation_history:
        return []
    texts: list[str] = []
    for item in conversation_history:
        if isinstance(item, dict):
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
        else:
            role = str(getattr(item, "role", "") or "").strip().lower()
            content = str(getattr(item, "content", "") or "").strip()
        if role == "user" and content:
            texts.append(content)
    return texts


def _extract_topic_from_prior(prior: str) -> str | None:
    for pattern in (_TOPIC_DOCUMENT_NOUN_RE, _TOPIC_FROM_ABOUT_RE):
        match = pattern.search(prior)
        if not match:
            continue
        topic = match.group(1).strip(" \t\n,.;:?!\"'")
        if len(topic) < 3:
            continue
        if topic.casefold() not in prior.casefold():
            continue
        return topic
    return None


def clarify_referential_query(
    query: str,
    conversation_history: Iterable[Any] | None,
) -> str | None:
    if not settings.enable_conversation_rewrite:
        return None
    original = query.strip()
    if not original:
        return None
    pronoun_match = _PRONOUN_RE.search(original)
    if not pronoun_match:
        return None
    if len(_tokens(original)) > 24:
        return None
    topic: str | None = None
    for prior in reversed(_history_user_texts(conversation_history)):
        if prior.casefold() == original.casefold():
            continue
        topic = _extract_topic_from_prior(prior)
        if topic:
            break
    if not topic:
        return None
    clarified = original[: pronoun_match.start()] + topic + original[pronoun_match.end() :]
    clarified = re.sub(r"\s+", " ", clarified).strip()
    if not clarified or clarified.casefold() == original.casefold():
        return None
    if topic.casefold() not in clarified.casefold():
        return None
    return clarified


def _llm_multi_queries(query: str) -> list[str]:
    """Optional Groq multi-query generation; empty on failure."""
    try:
        from app import chatbot
        client = getattr(chatbot, "client", None)
        if client is None:
            return []
        prompt = (
            "Generate up to 2 short alternative search queries for document retrieval. "
            "Return JSON array of strings only. Do not invent entity IDs. Query: "
            + query
        )
        response = client.chat.completions.create(
            model=getattr(chatbot, "MODEL", None) or "openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        content = (response.choices[0].message.content or "").strip()
        # Extract JSON array if wrapped in prose.
        start = content.find("[")
        end = content.rfind("]")
        if start < 0 or end <= start:
            return []
        parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, list):
            return []
        out: list[str] = []
        for item in parsed:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    except Exception as exc:
        logger.warning("rag_multi_query_failed error=%s", type(exc).__name__)
        return []


def generate_hyde_passage(query: str, *, enabled: bool | None = None) -> str | None:
    """Hypothetical answer passage for retrieval only — never cite or store."""
    if not (settings.enable_hyde if enabled is None else enabled):
        return None
    try:
        from app import chatbot
        client = getattr(chatbot, "client", None)
        if client is None:
            return None
        prompt = (
            "Write a short hypothetical document passage that would answer this "
            "question. Do not claim it is real evidence. Question: " + query
        )
        response = client.chat.completions.create(
            model=getattr(chatbot, "MODEL", None) or "openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception as exc:
        logger.warning("rag_hyde_failed error=%s", type(exc).__name__)
        return None


def rewrite_queries(
    query: str,
    conversation_history: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Build retrieval queries; original always first. Caps and dedupes."""
    original = query.strip()
    if not original:
        return {
            "queries": [],
            "original_query": "",
            "hyde_passage": None,
            "hyde_used": False,
            "multi_query_used": False,
        }
    queries: list[str] = [original]
    if settings.enable_query_decomposition:
        clauses = [part.strip(" \t\n,;?") for part in _SPLIT_RE.split(original)]
        clauses = [part for part in clauses if len(part) >= 8]
        if 2 <= len(clauses) <= 4:
            queries.extend(clauses)
    clarified = clarify_referential_query(original, conversation_history)
    if clarified:
        queries.append(clarified)

    multi_used = False
    if settings.enable_multi_query:
        multi_used = True
        for alt in _llm_multi_queries(original):
            queries.append(alt)

    # Deduplicate exact and near-duplicates; keep original first.
    deduped: list[str] = []
    for item in queries:
        text = item.strip()
        if not text:
            continue
        if any(_near_duplicate(text, existing) for existing in deduped):
            continue
        deduped.append(text)

    max_q = max(1, settings.max_rewritten_queries)
    # Cap total queries but always keep the original.
    if len(deduped) > max_q:
        deduped = [deduped[0]] + deduped[1:max_q]

    hyde = generate_hyde_passage(original) if settings.enable_hyde else None
    return {
        "queries": deduped,
        "original_query": original,
        "hyde_passage": hyde,
        "hyde_used": bool(hyde),
        "multi_query_used": multi_used,
    }
