import re


_FACT_PATTERN = re.compile(
    r"^my\s+(current role|role|job title|preferred programming language|"
    r"primary programming language|preferred language|current company|company|"
    r"highest qualification|degree|location|city|country|salary)\s+"
    r"(?:is|are)\s+(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r"^i\s+(?:work as|am working as)\s+(.+?)\s*[.!?]*$",
    re.IGNORECASE,
)

_KEY_ALIASES = {
    "current role": "role",
    "job title": "role",
    "current company": "company",
    "highest qualification": "degree",
}


def _normalize_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def extract_exclusive_fact(memory: str) -> tuple[str, str] | None:
    """Return (fact_key, normalized_value) for supported exclusive facts."""
    if not isinstance(memory, str) or not memory.strip():
        return None

    text = memory.strip()
    match = _FACT_PATTERN.fullmatch(text)

    if match:
        key = re.sub(r"\s+", " ", match.group(1).lower()).strip()
        key = _KEY_ALIASES.get(key, key)
        value = _normalize_value(match.group(2))
        return (key, value) if value else None

    role_match = _ROLE_PATTERN.fullmatch(text)
    if role_match:
        value = _normalize_value(role_match.group(1))
        return ("role", value) if value else None

    # Deliberately do not classify statements such as
    # "I am learning JavaScript" as exclusive facts.
    return None


def find_conflicting_memories(
    incoming_memory: str,
    candidates: list[dict],
) -> list[dict]:
    """Find active candidate memories with the same exclusive key, different value."""
    incoming_fact = extract_exclusive_fact(incoming_memory)
    if incoming_fact is None:
        return []

    incoming_key, incoming_value = incoming_fact
    conflicts = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        existing_fact = extract_exclusive_fact(candidate.get("memory", ""))
        if existing_fact is None:
            continue

        existing_key, existing_value = existing_fact
        if existing_key == incoming_key and existing_value != incoming_value:
            conflict = dict(candidate)
            conflict["matched_fact_key"] = incoming_key
            conflict["existing_value"] = existing_value
            conflict["incoming_value"] = incoming_value
            conflicts.append(conflict)

    return conflicts