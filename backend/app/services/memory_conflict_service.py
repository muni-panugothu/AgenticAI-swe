"""Stage 19 — advanced fact extraction, normalization, conflict scoring,
and deduplication intelligence for the memory system.

Upgrades over Stage 17:
- Project, goal, preference, and hobby fact patterns.
- Multi-valued vs exclusive fact classification.
- Paraphrase-aware extraction.
- Canonical text normalization for deduplication.
- Conflict grouping with confidence scoring.
"""

from __future__ import annotations

import re
from typing import Any


# ============================================================
# EXCLUSIVE FACT PATTERNS
# ============================================================
# Each pattern extracts a (raw_key, raw_value) pair.  The key is
# then normalized through _KEY_ALIASES to produce a canonical
# fact_key.

_FACT_PATTERN = re.compile(
    r"^my\s+(current role|role|job title|preferred programming language|"
    r"primary programming language|preferred language|current company|company|"
    r"highest qualification|degree|location|city|country|salary|name|"
    r"email|timezone|employer|"
    # Stage 19 additions:
    r"project|current project|main project|primary project|"
    r"current primary project|"
    r"goal|primary goal|main goal|career goal|long-term career goal|"
    r"preferred database|favorite database|favourite database|database preference|"
    r"preference|favorite language|favourite language|"
    r"preferred code editor|favorite editor|favourite editor|code editor|editor|ide|"
    r"primary operating system|operating system|os|"
    r"age|birthday|date of birth)\s+"
    r"(?:is|are|=|:)\s+(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

# "I prefer MongoDB as my database" / "I prefer PostgreSQL as my preferred database"
_DATABASE_PREFERENCE_PATTERN = re.compile(
    r"^(?:i\s+(?:still\s+)?prefer\s+(.+?)\s+as\s+(?:my\s+)?(?:preferred\s+)?"
    r"database|"
    r"my\s+database\s+(?:preference|of\s+choice)\s+is\s+(.+?))\s*[.!?]*$",
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r"^i\s+(?:work as|am working as)\s+(?:an?\s+)?(.+?)\s*[.!?]*$|"
    r"^i\s+(?:am a|am an|'m a|'m an)\s+(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

_COMPANY_PATTERN = re.compile(
    r"^i\s+(?:work at|work for|am employed at|am employed by)\s+(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

_LOCATION_PATTERN = re.compile(
    r"^i\s+(?:live in|am based in|am located in)\s+(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

_LANGUAGE_PATTERN = re.compile(
    r"^(?:i\s+)?(?:prefer|primarily use|mostly use)\s+(.+?)\s+"
    r"(?:as\s+)?(?:my\s+)?(?:preferred\s+)?(?:programming\s+)?language\s*[.!?]*$",
    re.IGNORECASE,
)

# Stage 19: explicit primary-project assignment ("My project is X").
_PRIMARY_PROJECT_PATTERN = re.compile(
    r"^my\s+(?:current\s+primary|current|primary|main)?\s*project\s+is\s+(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

# Named project: "Freshcart is my grocery marketplace project"
_NAMED_PROJECT_PATTERN = re.compile(
    r"^(.+?)\s+is\s+my\s+(?:(.+?)\s+)?project\s*[.!?]*$",
    re.IGNORECASE,
)

# Descriptive building/working statements — not exclusive primaries.
_PROJECT_DESCRIPTION_PATTERN = re.compile(
    r"^i\s+am\s+(?:currently\s+)?(?:working on|building|developing)\s+(?:a\s+|an\s+)?(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

# Stage 19: goal paraphrase patterns.
_GOAL_PATTERN = re.compile(
    r"^(?:my\s+(?:main\s+|primary\s+|career\s+)?goal\s+is\s+|"
    r"i\s+want\s+to\s+(?:become|achieve|reach|finish|complete)\s+)(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

# Stage 19: preference paraphrase patterns.
_PREFERENCE_PATTERN = re.compile(
    r"^i\s+(?:still\s+)?prefer\s+(.+?)(?:\s+as\s+my\s+.+)?\s*[.!?]*$",
    re.IGNORECASE,
)


# ============================================================
# KEY NORMALIZATION AND ALIASES
# ============================================================

_KEY_ALIASES = {
    "current role": "role",
    "job title": "role",
    "current company": "company",
    "employer": "company",
    "highest qualification": "degree",
    "preferred programming language": "preferred_language",
    "primary programming language": "preferred_language",
    "preferred language": "preferred_language",
    "favorite language": "preferred_language",
    "favourite language": "preferred_language",
    # Stage 19 project/goal/editor aliases.
    # Generic "project" maps to primary_project for backward compatibility.
    "project": "primary_project",
    "current project": "primary_project",
    "main project": "primary_project",
    "primary project": "primary_project",
    "current primary project": "primary_project",
    "project name": "project_name",
    "project description": "project_description",
    "project category": "project_category",
    "project status": "project_status",
    "project goal": "project_goal",
    "primary goal": "goal",
    "main goal": "goal",
    "career goal": "goal",
    "long-term career goal": "goal",
    "favorite editor": "editor",
    "favourite editor": "editor",
    "preferred code editor": "editor",
    "code editor": "editor",
    "ide": "editor",
    "preferred database": "preferred_database",
    "favorite database": "preferred_database",
    "favourite database": "preferred_database",
    "database preference": "preferred_database",
    "operating system": "os",
    "primary operating system": "os",
    "date of birth": "birthday",
}

# Facts that are inherently single-valued for a user.
# Only primary_project is exclusive among project-related keys —
# users may legitimately have multiple named projects / descriptions.
EXCLUSIVE_FACT_KEYS = frozenset({
    "role", "company", "degree", "preferred_language", "location",
    "city", "country", "salary", "name", "email", "timezone",
    "primary_project", "goal", "editor", "os", "age", "birthday",
    "preferred_database",
    "preference",
})

# Non-exclusive project metadata keys (multiple may coexist).
PROJECT_METADATA_KEYS = frozenset({
    "project_name",
    "project_description",
    "project_category",
    "project_status",
    "project_goal",
    "project_secondary",
    "project_completed",
    "project_planned",
    "project_historical",
})

# Learning statements are intentionally non-exclusive.
_NON_EXCLUSIVE_PATTERN = re.compile(
    r"^i\s+am\s+learning\s+.+$",
    re.IGNORECASE,
)

# Hobby / interest patterns are non-exclusive.
_HOBBY_PATTERN = re.compile(
    r"^(?:i\s+(?:enjoy|like|love)\s+|my\s+hobbies?\s+(?:is|are|include)\s+).+$",
    re.IGNORECASE,
)


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def _normalize_value(value: str) -> str:
    """Lowercase, collapse whitespace, strip non-alphanumeric chars."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def normalize_fact_key(key: str) -> str:
    """Produce a canonical fact key from a human-readable label."""
    cleaned = re.sub(r"\s+", " ", key.lower()).strip()
    return _KEY_ALIASES.get(cleaned, cleaned.replace(" ", "_"))


def classify_project_fact(memory: str) -> tuple[str, str] | None:
    """Classify a project-related statement into a specific fact key.

    Returns (fact_key, normalized_value) or None.

    - primary_project (exclusive): \"My (primary/current) project is X\"
    - project_name (multi): \"Freshcart is my grocery marketplace project\"
    - project_description (multi): \"I am building an Agentic AI bug resolver\"
    - project_category stored as suffix in named-project value when present
      is represented via project_name; category words stay in the value text
      for retrieval rather than forcing a second exclusive key.
    """
    if not isinstance(memory, str) or not memory.strip():
        return None

    text = memory.strip().rstrip(".!?")

    primary = _PRIMARY_PROJECT_PATTERN.fullmatch(text)
    if primary:
        value = _normalize_value(primary.group(1))
        return ("primary_project", value) if value else None

    named = _NAMED_PROJECT_PATTERN.fullmatch(text)
    if named:
        name = _normalize_value(named.group(1))
        category = _normalize_value(named.group(2) or "")
        if not name:
            return None
        # Avoid treating \"My project is X\" (already handled) or vague \"this\".
        if name in {"my", "this", "that", "our"}:
            return None
        if category:
            # Keep category signal in the normalized value for retrieval.
            return ("project_name", f"{name} ({category})")
        return ("project_name", name)

    description = _PROJECT_DESCRIPTION_PATTERN.fullmatch(text)
    if description:
        raw = description.group(1).strip()
        # \"project called Orion\" → treat as named primary-ish name, not exclusive.
        called = re.match(
            r"^(?:a\s+|an\s+)?project\s+(?:called|named)\s+(.+)$",
            raw,
            re.IGNORECASE,
        )
        if called:
            value = _normalize_value(called.group(1))
            return ("project_name", value) if value else None
        value = _normalize_value(raw)
        # Strip trailing \"project\" word for cleaner values.
        value = re.sub(r"\s+project$", "", value).strip()
        return ("project_description", value) if value else None

    return None


def canonicalize_stored_fact_key(fact_key: str | None) -> str | None:
    """Map legacy stored keys (e.g. project) onto current canonical keys."""
    if not fact_key:
        return None
    return normalize_fact_key(str(fact_key).replace("_", " "))


def normalize_memory_text(text: str) -> str:
    """Stage 19: Produce a canonical text for deduplication comparison.

    Lowercased, whitespace-collapsed, punctuation-stripped.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[.!?]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ============================================================
# FACT EXTRACTION
# ============================================================

def extract_exclusive_fact(memory: str) -> tuple[str, str] | None:
    """Return (fact_key, normalized_value) for supported exclusive facts.

    Returns None for non-exclusive statements (learning, hobbies) and
    for text that does not match any supported pattern.
    """
    if not isinstance(memory, str) or not memory.strip():
        return None

    text = memory.strip()

    # Non-exclusive patterns are checked first.
    if _NON_EXCLUSIVE_PATTERN.fullmatch(text):
        return None
    if _HOBBY_PATTERN.fullmatch(text):
        return None

    # Direct "my X is Y" pattern (broadened in Stage 19).
    match = _FACT_PATTERN.fullmatch(text)
    if match:
        key = normalize_fact_key(match.group(1))
        value = _normalize_value(match.group(2))
        return (key, value) if value else None

    # "I work as ..." / "I am a ..." → role
    role_match = _ROLE_PATTERN.fullmatch(text)
    if role_match:
        # Two alternation groups — use whichever matched.
        raw = role_match.group(1) or role_match.group(2)
        value = _normalize_value(raw)
        return ("role", value) if value else None

    # "I work at ..." → company
    company_match = _COMPANY_PATTERN.fullmatch(text)
    if company_match:
        value = _normalize_value(company_match.group(1))
        return ("company", value) if value else None

    # "I live in ..." → location
    location_match = _LOCATION_PATTERN.fullmatch(text)
    if location_match:
        value = _normalize_value(location_match.group(1))
        return ("location", value) if value else None

    # "I prefer/primarily use X as language" → preferred_language
    language_match = _LANGUAGE_PATTERN.fullmatch(text)
    if language_match:
        value = _normalize_value(language_match.group(1))
        return ("preferred_language", value) if value else None

    # Project statements: only exclusive primary_project is returned here.
    # Named projects / descriptions are non-exclusive (see classify_project_fact).
    project_fact = classify_project_fact(text)
    if project_fact and project_fact[0] in EXCLUSIVE_FACT_KEYS:
        return (project_fact[0], project_fact[1])

    # Stage 19: "My goal is / I want to become ..." → goal
    goal_match = _GOAL_PATTERN.fullmatch(text)
    if goal_match:
        value = _normalize_value(goal_match.group(1))
        return ("goal", value) if value else None

    # Editor paraphrases: "I prefer X as my code editor"
    editor_match = re.fullmatch(
        r"i\s+(?:now\s+)?prefer\s+(.+?)\s+as\s+(?:my\s+)?(?:code\s+)?editor\s*[.!?]*$",
        text,
        re.IGNORECASE,
    )
    if editor_match:
        value = _normalize_value(editor_match.group(1))
        return ("editor", value) if value else None

    # Database preference: "I prefer MongoDB as my database"
    database_match = _DATABASE_PREFERENCE_PATTERN.fullmatch(text)
    if database_match:
        raw = database_match.group(1) or database_match.group(2)
        value = _normalize_value(raw)
        return ("preferred_database", value) if value else None

    # OS paraphrase: "I primarily use Ubuntu Linux"
    os_match = re.fullmatch(
        r"i\s+(?:primarily\s+)?(?:use|run)\s+(.+?)\s*[.!?]*$",
        text,
        re.IGNORECASE,
    )
    if os_match:
        value = _normalize_value(os_match.group(1))
        if any(h in value for h in ("windows", "linux", "ubuntu", "macos", "mac os", "debian")):
            return ("os", value)

    # Language paraphrase: "I prefer TypeScript for programming"
    lang_for_match = re.fullmatch(
        r"i\s+prefer\s+(.+?)\s+for\s+programming\s*[.!?]*$",
        text,
        re.IGNORECASE,
    )
    if lang_for_match:
        value = _normalize_value(lang_for_match.group(1))
        return ("preferred_language", value) if value else None

    # Company paraphrase: "I currently/now work at X"
    company_now = re.fullmatch(
        r"i\s+(?:currently\s+|now\s+)?work\s+at\s+(.+?)\s*[.!?]*$",
        text,
        re.IGNORECASE,
    )
    if company_now:
        value = _normalize_value(company_now.group(1))
        return ("company", value) if value else None

    # Stage 19: "I prefer X" → preference (generic)
    preference_match = _PREFERENCE_PATTERN.fullmatch(text)
    if preference_match:
        value = _normalize_value(preference_match.group(1))
        known_languages = {
            "python", "javascript", "typescript", "java", "go",
            "rust", "c", "c++", "c#", "ruby", "php", "swift",
            "kotlin", "scala", "elixir", "haskell", "r",
        }
        if value in known_languages:
            return ("preferred_language", value)
        return ("preference", value) if value else None

    return None


# ============================================================
# CANDIDATE FACT EXTRACTION
# ============================================================

def _candidate_fact(candidate: dict) -> tuple[str, str] | None:
    """Extract a fact from an existing memory candidate document."""
    if candidate.get("fact_key") and candidate.get("normalized_value"):
        return (
            normalize_fact_key(str(candidate["fact_key"])),
            _normalize_value(str(candidate["normalized_value"])),
        )

    if candidate.get("fact_key") and candidate.get("memory"):
        extracted = extract_exclusive_fact(str(candidate["memory"]))
        if extracted and extracted[0] == normalize_fact_key(str(candidate["fact_key"])):
            return extracted

    return extract_exclusive_fact(str(candidate.get("memory", "")))


# ============================================================
# CONTRADICTION SCORING
# ============================================================

def score_contradiction(
    incoming_key: str,
    incoming_value: str,
    existing_key: str,
    existing_value: str,
    *,
    vector_score: float | None = None,
) -> float:
    """Confidence that two exclusive facts contradict each other.

    Returns 0.0 when keys differ, when values are identical, or when
    the fact key is not in the exclusive set.
    """
    incoming_key = normalize_fact_key(str(incoming_key).replace("_", " "))
    existing_key = normalize_fact_key(str(existing_key).replace("_", " "))
    if incoming_key != existing_key:
        return 0.0
    if incoming_value == existing_value:
        return 0.0
    if incoming_key not in EXCLUSIVE_FACT_KEYS:
        return 0.0

    confidence = 0.82
    if vector_score is not None:
        confidence = max(confidence, min(0.99, 0.70 + float(vector_score) * 0.29))
    return round(confidence, 4)


# ============================================================
# DEDUPLICATION CLASSIFICATION (Stage 19)
# ============================================================

def classify_duplicate(
    incoming_text: str,
    existing_text: str,
    vector_score: float | None = None,
) -> dict:
    """Classify the relationship between two memory texts.

    Returns a dict with:
    - classification: one of exact_duplicate, paraphrase_duplicate,
      possible_contradiction, independent, ambiguous
    - confidence: float 0-1
    """
    norm_incoming = normalize_memory_text(incoming_text)
    norm_existing = normalize_memory_text(existing_text)

    # Exact textual duplicate after normalization.
    if norm_incoming == norm_existing:
        return {
            "classification": "exact_duplicate",
            "confidence": 1.0,
        }

    incoming_fact = extract_exclusive_fact(incoming_text)
    existing_fact = extract_exclusive_fact(existing_text)

    # Both express the same fact key with the same value.
    if incoming_fact and existing_fact:
        if incoming_fact[0] == existing_fact[0] and incoming_fact[1] == existing_fact[1]:
            return {
                "classification": "paraphrase_duplicate",
                "confidence": 0.95,
            }
        if incoming_fact[0] == existing_fact[0] and incoming_fact[1] != existing_fact[1]:
            conf = score_contradiction(
                incoming_fact[0], incoming_fact[1],
                existing_fact[0], existing_fact[1],
                vector_score=vector_score,
            )
            return {
                "classification": "possible_contradiction",
                "confidence": conf if conf > 0 else 0.80,
            }

    # High vector similarity alone must NOT classify exclusive preference
    # contradictions as duplicates. Structured fact comparison above is
    # authoritative when both sides extract the same exclusive key.
    if vector_score is not None and vector_score >= 0.95:
        return {
            "classification": "paraphrase_duplicate",
            "confidence": float(vector_score),
        }

    if vector_score is not None and vector_score >= 0.85:
        return {
            "classification": "ambiguous",
            "confidence": float(vector_score),
        }

    return {
        "classification": "independent",
        "confidence": 1.0 - (float(vector_score) if vector_score else 0.5),
    }


# ============================================================
# CONFLICTING MEMORY SEARCH
# ============================================================

def find_conflicting_memories(
    incoming_memory: str,
    candidates: list[dict],
) -> list[dict]:
    """Find candidate memories with the same exclusive key and a different value."""
    incoming_fact = extract_exclusive_fact(incoming_memory)
    if incoming_fact is None:
        return []

    incoming_key, incoming_value = incoming_fact
    conflicts: list[dict[str, Any]] = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        existing_fact = _candidate_fact(candidate)
        if existing_fact is None:
            continue

        existing_key, existing_value = existing_fact
        confidence = score_contradiction(
            incoming_key,
            incoming_value,
            existing_key,
            existing_value,
            vector_score=candidate.get("score"),
        )
        if confidence <= 0.0:
            continue

        conflict = dict(candidate)
        conflict["matched_fact_key"] = incoming_key
        conflict["existing_value"] = existing_value
        conflict["incoming_value"] = incoming_value
        conflict["conflict_confidence"] = confidence
        conflicts.append(conflict)

    conflicts.sort(key=lambda item: item.get("conflict_confidence", 0.0), reverse=True)
    return conflicts


# ============================================================
# STRUCTURED CONFLICT DETECTION
# ============================================================

def detect_memory_conflict(
    incoming_memory: str,
    candidates: list[dict],
) -> dict:
    """Structured conflict detection result used by tools and the chatbot."""
    incoming_fact = extract_exclusive_fact(incoming_memory)
    conflicts = find_conflicting_memories(incoming_memory, candidates)

    return {
        "success": True,
        "has_conflict": bool(conflicts),
        "fact_key": incoming_fact[0] if incoming_fact else None,
        "normalized_value": incoming_fact[1] if incoming_fact else None,
        "conflicts": conflicts,
        "top_conflict": conflicts[0] if conflicts else None,
        "requires_confirmation": bool(conflicts),
    }


# ============================================================
# FIND DUPLICATE GROUPS (Stage 19)
# ============================================================

def find_duplicate_groups(
    memories: list[dict],
) -> list[list[dict]]:
    """Group memories that are textual duplicates after normalization.

    Returns a list of groups, where each group contains 2+ memories
    that are textual duplicates of each other.
    """
    groups: dict[str, list[dict]] = {}
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        norm = normalize_memory_text(mem.get("memory", ""))
        if not norm:
            continue
        groups.setdefault(norm, []).append(mem)

    return [group for group in groups.values() if len(group) >= 2]
