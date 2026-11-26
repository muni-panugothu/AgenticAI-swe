"""Multi-statement memory fact extraction.

Splits an explicit memory message into independent factual statements,
then extracts canonical fact keys and values for each statement.

Does not treat quoted examples, test instructions, or unrelated
prose as confirmed memories.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.services.memory_conflict_service import (
    EXCLUSIVE_FACT_KEYS,
    classify_project_fact,
    extract_exclusive_fact,
    normalize_fact_key,
    normalize_memory_text,
    _normalize_value,
)


class ExtractedFact(BaseModel):
    statement: str = Field(min_length=1)
    fact_key: str | None = None
    value: str | None = None
    normalized_value: str | None = None
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    exclusive: bool = False


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)
    source_message: str = ""
    uncertain: bool = False


# Sentence split that avoids common abbreviations.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "vs", "etc", "e.g", "i.e",
    "u.s", "u.k", "inc", "ltd", "jr", "sr",
}

_REMEMBER_PREFIX = re.compile(
    r"^\s*(?:please\s+)?remember\s+that\s+",
    re.IGNORECASE,
)

# Patterns that indicate instructional / test text, not a user fact.
_INSTRUCTIONAL = re.compile(
    r"(?i)\b("
    r"acceptance criteria|expected:|should be|test case|"
    r"reinforcement_count|operation_id|operation ids|operation_ids|"
    r"distinct operation|do not|verify that|"
    r"unit test|integration test"
    r")\b"
)

# User questions accidentally appended to memory statements.
_QUESTION_STATEMENT = re.compile(
    r"(?i)^\s*(what|which|who|where|when|why|how|do|does|did|is|are|can|could|would)\b.+\?\s*$"
)

# Digests that must never be re-ingested as ordinary memories.
_SUMMARY_ARTIFACT = re.compile(
    r"(?i)^\s*(summary\s*:|confirmed preferences\s*:|project facts\s*:|"
    r"user profile\s*:|notes\s*:|memory_summaries|"
    r"the user's .+ is unresolved\b|"
    r"source memory id)"
)


def is_summary_artifact_text(text: str) -> bool:
    """True when text looks like a generated summary digest, not a user fact."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _SUMMARY_ARTIFACT.search(cleaned):
        return True
    if "source memory id" in cleaned.lower():
        return True
    return False


def strip_trailing_question(statement: str) -> str:
    """Remove an appended interrogative clause from a fact statement."""
    text = (statement or "").strip()
    if not text:
        return text
    # Split off trailing "What/Which...?" segments.
    parts = re.split(
        r"(?i)(?=(?:what|which|who|where|when|why|how)\b[^?]*\?\s*$)",
        text,
        maxsplit=1,
    )
    head = parts[0].strip().rstrip(".!?")
    return head or text


def is_question_only(statement: str) -> bool:
    text = (statement or "").strip()
    if not text:
        return False
    if _QUESTION_STATEMENT.match(text):
        return True
    # Interrogatives without a terminal '?' still must not become memories.
    if re.match(
        r"(?i)^\s*(what|which|who|where|when|why|how)\b.+\b(am|is|are|do|does|did|can|should)\b",
        text,
    ):
        return True
    return False

_EDITOR_PATTERN = re.compile(
    r"^(?:my\s+(?:preferred\s+)?(?:code\s+)?editor\s+is\s+|"
    r"i\s+(?:now\s+)?prefer\s+(.+?)\s+as\s+(?:my\s+)?(?:code\s+)?editor|"
    r"i\s+(?:now\s+)?prefer\s+(.+?)\s+(?:as\s+)?(?:my\s+)?editor)\s*[.!?]*$",
    re.IGNORECASE,
)

_OS_PATTERN = re.compile(
    r"^(?:my\s+(?:primary\s+)?(?:operating\s+system|os)\s+is\s+|"
    r"i\s+(?:primarily\s+)?(?:use|run)\s+)(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

_PROJECT_QUALIFIER = re.compile(
    r"(?i)\b(current|primary|main|secondary|completed|planned|historical)\b"
)


_SENTENCE_STARTER = re.compile(
    r"^(?:i|my|we|you|please|remember|the|this|that|it|he|she|they)\b",
    re.IGNORECASE,
)


def _is_decimal_or_version_dot(buffer: list[str], next_ch: str) -> bool:
    """True for numeric decimals / versions like 11.0 — not sentence boundaries."""
    if not next_ch.isdigit():
        return False
    # Walk back past the dot we just appended.
    j = len(buffer) - 2
    while j >= 0 and buffer[j].isdigit():
        j -= 1
    # Digit immediately before the dot ⇒ decimal/version.
    return j < len(buffer) - 2 and (j < 0 or not buffer[j].isalnum())


def _should_split_at_punct(buffer: list[str], cleaned: str, punct_index: int) -> bool:
    """Decide whether punctuation at punct_index ends a statement."""
    lookahead = cleaned[punct_index + 1: punct_index + 2]
    rest = cleaned[punct_index + 1:]

    token = "".join(buffer).rstrip(".!?;").split()
    last = token[-1].lower().strip("\"'") if token else ""
    if last in _ABBREVIATIONS:
        return False

    # URLs / emails: keep together.
    joined = "".join(buffer).lower()
    if "http://" in joined or "https://" in joined or "www." in joined:
        if lookahead and lookahead[0].isalnum():
            return False

    # Decimal / version numbers: 11.0, 3.14
    if lookahead and _is_decimal_or_version_dot(buffer, lookahead[0]):
        return False

    # No lookahead ⇒ end of text; split.
    if not lookahead:
        return True

    # Whitespace after punct ⇒ normal sentence boundary.
    if lookahead[0].isspace():
        return True

    # Period/question/exclaim immediately followed by a new sentence starter
    # (with or without space): "VS Code.I prefer" / "Orion-QA.Remember that"
    if lookahead[0].isupper() or _SENTENCE_STARTER.match(rest):
        return True

    # Semicolon often separates clauses with facts.
    if buffer and buffer[-1] == ";":
        return True

    # Lowercase continuation without space (file.py, e.g.something) — keep.
    return False


def split_statements(text: str) -> list[str]:
    """Split text into candidate statements without naive period splitting.

    Handles full stops with or without following whitespace, newlines,
    semicolons, and repeated \"Remember that\" clauses. Preserves
    abbreviations, decimals, URLs, and simple filenames.
    """
    if not text or not str(text).strip():
        return []

    cleaned = str(text).strip()
    # Normalize line breaks into sentence boundaries.
    cleaned = re.sub(r"[\r\n]+", ". ", cleaned)
    # Normalize spaced/unspaced sentence joins around \"Remember that\".
    cleaned = re.sub(
        r"(?i)([.!?])\s*(remember\s+that\b)",
        r"\1 \2",
        cleaned,
    )

    parts: list[str] = []
    buffer: list[str] = []
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        buffer.append(ch)
        if ch in ".!?;":
            if _should_split_at_punct(buffer, cleaned, i):
                statement = "".join(buffer).strip()
                if statement:
                    parts.append(statement)
                buffer = []
        i += 1

    trailing = "".join(buffer).strip()
    if trailing:
        parts.append(trailing)

    # Also split on repeated "Remember that" without relying only on punctuation.
    expanded: list[str] = []
    for part in parts:
        chunks = re.split(r"(?i)(?=\bremember\s+that\b)", part)
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                expanded.append(chunk)

    return expanded or ([cleaned] if cleaned else [])


def _strip_remember_prefix(statement: str) -> str:
    return _REMEMBER_PREFIX.sub("", statement).strip().rstrip(".!?")


def _split_compound_fact_statement(statement: str) -> list[str]:
    """Separate explicitly coordinated profile facts before key extraction.

    Legacy text such as "my preferred language is JavaScript and my project
    is FreshCart" was being stored as one preferred-language value. Only split
    at a conjunction followed by a recognizable fact declaration, so ordinary
    lists like "I like Python and JavaScript" remain intact.
    """
    text = (statement or "").strip()
    preserved_project = re.split(
        r"(?i)\s*,?\s+but\s+keep\s+my\s+(?:current\s+)?project\s+as\s+",
        text,
        maxsplit=1,
    )
    if len(preserved_project) == 2:
        language_clause, project_value = preserved_project
        if language_clause.strip() and project_value.strip():
            return [
                language_clause.strip(),
                f"My project is {project_value.strip()}",
            ]

    fact_start = (
        r"(?:my\s+(?:(?:preferred|primary|current|main|favorite|favourite)\s+)*"
        r"(?:programming\s+)?(?:language|(?:ai\s+)?project|database|editor|goal|role|"
        r"company|job|name|city|location)\b|"
        r"i\s+(?:prefer|work\s+at|am\s+learning|am\s+working\s+on)\b)"
    )
    parts = re.split(
        rf"(?i)\s+(?:and|but)\s+(?={fact_start})",
        text,
        maxsplit=3,
    )
    cleaned = [part.strip(" ,;") for part in parts if part.strip(" ,;")]
    return cleaned if len(cleaned) > 1 else [text]


def _extract_editor(statement: str) -> tuple[str, str] | None:
    text = statement.strip().rstrip(".!?")
    match = re.match(
        r"^my\s+(?:preferred\s+)?(?:code\s+)?editor\s+is\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if match:
        return ("editor", _normalize_value(match.group(1)))

    match = re.match(
        r"^i\s+(?:now\s+)?prefer\s+(.+?)\s+as\s+(?:my\s+)?(?:code\s+)?editor$",
        text,
        re.IGNORECASE,
    )
    if match:
        return ("editor", _normalize_value(match.group(1)))

    match = re.match(
        r"^i\s+(?:now\s+)?prefer\s+(.+?)\s+(?:as\s+)?(?:my\s+)?editor$",
        text,
        re.IGNORECASE,
    )
    if match:
        return ("editor", _normalize_value(match.group(1)))
    return None


def _extract_os(statement: str) -> tuple[str, str] | None:
    text = statement.strip().rstrip(".!?")
    match = re.match(
        r"^my\s+(?:primary\s+)?(?:operating\s+system|os)\s+is\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if match:
        return ("os", _normalize_value(match.group(1)))

    match = re.match(
        r"^i\s+(?:primarily\s+)?(?:use|run)\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if match:
        value = _normalize_value(match.group(1))
        # Avoid classifying "I use Python" as OS.
        os_hints = {
            "windows", "windows 11", "windows 10", "ubuntu", "ubuntu linux",
            "linux", "macos", "mac os", "mac", "debian", "fedora", "arch",
        }
        if value in os_hints or any(h in value for h in ("windows", "linux", "ubuntu", "macos", "mac os")):
            return ("os", value)
    return None


def _extract_project_role(statement: str) -> tuple[str, str, bool] | None:
    """Return (fact_key, normalized_value, exclusive) for project statements."""
    text = statement.strip().rstrip(".!?")

    # An AI project is a named project, not a replacement for the user's
    # exclusive primary-project preference. Keeping it multi-valued prevents
    # statements like "my AI project is X" from conflicting with "my project
    # is Alpha".
    ai_project = re.match(
        r"^my\s+ai\s+project\s+is\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if ai_project:
        value = _normalize_value(ai_project.group(1))
        return ("project_name", value, False) if value else None

    # Explicit secondary / completed / planned / historical remain multi-valued.
    match = re.match(
        r"^my\s+(secondary|completed|planned|historical)\s+project\s+is\s+(.+)$",
        text,
        re.IGNORECASE,
    )
    if match:
        qualifier = match.group(1).strip().lower()
        value = _normalize_value(match.group(2))
        return (f"project_{qualifier}", value, False)

    classified = classify_project_fact(text)
    if not classified:
        return None
    fact_key, value = classified
    return (fact_key, value, fact_key in EXCLUSIVE_FACT_KEYS)


def extract_fact_from_statement(statement: str) -> ExtractedFact | None:
    """Extract one structured fact from a single statement."""
    cleaned = _strip_remember_prefix(statement)
    cleaned = strip_trailing_question(cleaned)
    if not cleaned:
        return None

    if is_question_only(cleaned):
        return None

    if is_summary_artifact_text(cleaned):
        return None

    if _INSTRUCTIONAL.search(cleaned):
        return None

    # Skip quoted instruction blocks that look like test scripts.
    if cleaned.count("\n") > 2 and "expected" in cleaned.lower():
        return None

    editor = _extract_editor(cleaned)
    if editor:
        return ExtractedFact(
            statement=cleaned,
            fact_key=editor[0],
            value=cleaned,
            normalized_value=editor[1],
            confidence=0.95,
            exclusive=True,
        )

    os_fact = _extract_os(cleaned)
    if os_fact:
        return ExtractedFact(
            statement=cleaned,
            fact_key=os_fact[0],
            value=cleaned,
            normalized_value=os_fact[1],
            confidence=0.92,
            exclusive=True,
        )

    project = _extract_project_role(cleaned)
    if project:
        return ExtractedFact(
            statement=cleaned,
            fact_key=project[0],
            value=cleaned,
            normalized_value=project[1],
            confidence=0.95 if project[2] else 0.85,
            exclusive=project[2],
        )

    exclusive = extract_exclusive_fact(cleaned)
    if exclusive:
        return ExtractedFact(
            statement=cleaned,
            fact_key=exclusive[0],
            value=cleaned,
            normalized_value=exclusive[1],
            confidence=0.95,
            exclusive=True,
        )

    # Non-exclusive learning / general remember content.
    # Reject bare instructional fragments that survived splitting.
    if _INSTRUCTIONAL.search(cleaned):
        return None
    if cleaned.lower().startswith("i am learning") or len(cleaned.split()) >= 3:
        return ExtractedFact(
            statement=cleaned,
            fact_key=None,
            value=cleaned,
            normalized_value=normalize_memory_text(cleaned),
            confidence=0.70,
            exclusive=False,
        )

    return None


def extract_facts_from_message(user_message: str) -> ExtractionResult:
    """Extract zero or more independent facts from an explicit memory message."""
    text = (user_message or "").strip()
    if not text:
        return ExtractionResult(source_message=text)

    # Only process as explicit memory when the message starts with remember.
    if not _REMEMBER_PREFIX.match(text) and "remember that" not in text.lower():
        return ExtractionResult(source_message=text)

    body = text
    if _REMEMBER_PREFIX.match(text):
        # Keep full text for multi-statement splitting; strip only for single body.
        body = text

    statements = split_statements(body)
    facts: list[ExtractedFact] = []
    for statement in statements:
        # Drop pure "Remember that" leftovers.
        if re.fullmatch(r"(?i)remember\s+that\.?", statement.strip()):
            continue
        for component in _split_compound_fact_statement(statement):
            fact = extract_fact_from_statement(component)
            if fact is not None:
                facts.append(fact)

    # De-dupe identical normalized facts within the same message.
    unique: list[ExtractedFact] = []
    seen: set[tuple[str | None, str | None]] = set()
    for fact in facts:
        key = (fact.fact_key, fact.normalized_value)
        if key in seen and fact.normalized_value is not None:
            continue
        seen.add(key)
        unique.append(fact)

    uncertain = len(unique) == 0 and bool(text.strip())
    return ExtractionResult(
        facts=unique,
        source_message=text,
        uncertain=uncertain,
    )


def facts_to_dict(result: ExtractionResult) -> dict[str, Any]:
    return result.model_dump()
