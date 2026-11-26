"""Deterministic user-input normalization for routing (Phase B1).

Preserves the original message. Produces a normalized form used by intent
classification and tool routing only. Does not execute tools, call providers,
or authorize actions.

Transformations are rule-based and extensible via alias tables — not a fixed
dictionary of full questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Extensible alias tables (not exhaustive question lists)
# ---------------------------------------------------------------------------

# Unambiguous place aliases → canonical English name.
PLACE_ALIASES: dict[str, str] = {
    "ap": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "andhrapradesh": "Andhra Pradesh",
    "andhraprdesh": "Andhra Pradesh",
    "andhra pradesh": "Andhra Pradesh",
    "ts": "Telangana",
    "tg": "Telangana",
    "telangana": "Telangana",
    "telengana": "Telangana",
    "kn": "Karnataka",
    "ka": "Karnataka",
    "karnataka": "Karnataka",
    "mh": "Maharashtra",
    "maharashtra": "Maharashtra",
    "tn": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "tamil nadu": "Tamil Nadu",
    "dl": "Delhi",
    "delhi": "Delhi",
    "up": "Uttar Pradesh",  # only expanded in place/CM context, never alone
    "uk": "Uttarakhand",
}

CITY_ALIASES: dict[str, str] = {
    "hyd": "Hyderabad",
    "blr": "Bengaluru",
    "bangalore": "Bengaluru",
    "chennai": "Chennai",
    "madras": "Chennai",
    "mumbai": "Mumbai",
    "bom": "Mumbai",
}

# Safe, high-confidence spelling / informal repairs (word-level).
SAFE_TYPO_ALIASES: dict[str, str] = {
    "whos": "who is",
    "whats": "what is",
    "wheres": "where is",
    "hows": "how is",
    "dont": "don't",
    "doesnt": "doesn't",
    "cant": "can't",
    "wont": "won't",
    "im": "i'm",
    "pls": "please",
    "plz": "please",
    "expln": "explain",
    "explaination": "explanation",
    "wrks": "works",
    "implmnt": "implement",
    "implemnt": "implement",
    "dependancy": "dependency",
    "dependancyinjection": "dependency injection",
    "authn": "authentication",
    "authz": "authorization",
}

# Mixed-language romanization cues (Telugu / Hindi informal).
MIXED_LANG_PHRASES: list[tuple[re.Pattern[str], str, str]] = [
    # "X lo Y enti" / "X lo Y ante" → "what is Y in X"
    (
        re.compile(
            r"(?i)^\s*(?P<head>[\w.+#-]+)\s+lo\s+(?P<body>.+?)\s+"
            r"(enti|ante|antedi|entidi)\s*[?!.]*\s*$"
        ),
        "what is {body} in {head}",
        "telugu_lo_enti",
    ),
    # trailing enti/ante on a phrase
    (
        re.compile(
            r"(?i)^\s*(?P<body>.+?)\s+(enti|ante|antedi|entidi)\s*[?!.]*\s*$"
        ),
        "what is {body}",
        "telugu_enti",
    ),
    # "X kya hai" / "X kya hota hai"
    (
        re.compile(
            r"(?i)^\s*(?P<body>.+?)\s+kya\s+(hai|hota\s+hai|hoti\s+hai)"
            r"(?:\s+aur\s+(?P<rest>.+))?\s*[?!.]*\s*$"
        ),
        "what is {body}",
        "hindi_kya_hai",
    ),
]

# Tokens that must never be rewritten as place/CM aliases when alone.
AMBIGUOUS_SINGLE_TOKENS = frozenset(
    {
        "cm",  # centimeters vs chief minister
        "up",  # direction vs Uttar Pradesh
        "uk",  # United Kingdom vs Uttarakhand
        "ca",  # California vs Canada etc.
        "in",  # India vs preposition — never expand alone
        "or",
        "me",
        "ai",
        "ml",
        "ui",
        "ux",
        "db",
        "os",
    }
)


@dataclass(frozen=True)
class NormalizedInput:
    """Result of deterministic input normalization."""

    original: str
    normalized: str
    transforms: tuple[str, ...] = ()
    language_hints: tuple[str, ...] = ()
    protected_spans: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "transforms": list(self.transforms),
            "language_hints": list(self.language_hints),
            "protected_spans": self.protected_spans,
            # Avoid logging full message content in production logs by default.
            "changed": self.original != self.normalized,
        }


@dataclass
class _ProtectState:
    spans: list[str] = field(default_factory=list)

    def stash(self, match: re.Match[str]) -> str:
        idx = len(self.spans)
        self.spans.append(match.group(0))
        return f" __PROT{idx}__ "

    def restore(self, text: str) -> str:
        out = text
        for idx, span in enumerate(self.spans):
            out = out.replace(f"__PROT{idx}__", span)
        return out


_PROTECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"```[\s\S]*?```"),
    re.compile(r"`[^`]+`"),
    re.compile(r"https?://[^\s]+", re.IGNORECASE),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b"),
    re.compile(r"\b(?:app|backend|frontend|src|tests?)/[\w./\\-]+"),
    re.compile(r"\b[\w.-]+\.(?:py|ts|tsx|js|jsx|md|json|ya?ml|toml|css|html)\b"),
    # Keep arithmetic / numeric expressions intact.
    re.compile(r"\b\d+(?:\.\d+)?(?:\s*[\+\-\*/x×÷]\s*\d+(?:\.\d+)?)+"),
)


def _protect(text: str) -> tuple[str, _ProtectState]:
    state = _ProtectState()
    out = text
    for pattern in _PROTECT_PATTERNS:
        out = pattern.sub(state.stash, out)
    return out, state


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_punctuation(text: str) -> str:
    # Smart quotes → ASCII; keep meaning.
    table = str.maketrans({
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    })
    text = text.translate(table)
    # Drop redundant repeated punctuation except ellipsis-like.
    text = re.sub(r"([?!]){2,}", r"\1", text)
    return text


def _apply_safe_typos(text: str) -> tuple[str, list[str]]:
    transforms: list[str] = []
    parts = re.split(r"(\s+)", text)
    out: list[str] = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        key = part.casefold()
        # Strip trailing punctuation for lookup, then reattach.
        m = re.match(r"^([\w']+)([^\w']*)$", part)
        if not m:
            out.append(part)
            continue
        word, trail = m.group(1), m.group(2)
        repl = SAFE_TYPO_ALIASES.get(word.casefold())
        if repl and word.casefold() not in AMBIGUOUS_SINGLE_TOKENS:
            transforms.append(f"typo:{word.casefold()}->{repl}")
            # Preserve crude capitalization of first letter when original was Title.
            if word[:1].isupper() and repl:
                repl = repl[:1].upper() + repl[1:]
            out.append(repl + trail)
        else:
            out.append(part)
    return "".join(out), transforms


def _canonical_place(token: str) -> str | None:
    key = re.sub(r"\s+", " ", (token or "").casefold().strip())
    key_compact = key.replace(" ", "")
    if key in PLACE_ALIASES:
        return PLACE_ALIASES[key]
    if key_compact in PLACE_ALIASES:
        return PLACE_ALIASES[key_compact]
    return None


def _expand_place_token(token: str) -> str:
    return _canonical_place(token) or token


def _place_alternation() -> str:
    """Regex alternation of known place aliases (longest first)."""
    keys = sorted(PLACE_ALIASES.keys(), key=len, reverse=True)
    parts = [re.escape(k).replace(r"\ ", r"\s+") for k in keys]
    return "|".join(parts)


def _expand_cm_and_places(text: str) -> tuple[str, list[str]]:
    """Contextual chief-minister / place expansions only.

    Never expands bare ``cm`` (ambiguous with centimeters).
    Place tokens are limited to known aliases to avoid eating unrelated words.
    """
    transforms: list[str] = []
    out = text
    place = _place_alternation()

    # who[s] cm [of] <place>
    pattern_who_cm = re.compile(
        rf"(?i)\b(who(?:\s+is)?|who's)\s+cm\s+(?:of\s+)?({place})"
        r"(?=\s+(?:currently|now|right\s+now|today)\b|[?!.]|$|\s)"
    )

    def who_cm_sub(match: re.Match[str]) -> str:
        resolved = _expand_place_token(match.group(2).strip())
        transforms.append("cm_context:who_cm")
        return f"who is the current chief minister of {resolved}"

    out = pattern_who_cm.sub(who_cm_sub, out)

    # <place> cm (currently|now|...)
    pattern_place_cm = re.compile(
        rf"(?i)\b({place})\s+cm\s+"
        r"(?=(?:currently|now|right\s+now|today)\b)"
    )

    def place_cm_sub(match: re.Match[str]) -> str:
        resolved = _expand_place_token(match.group(1).strip())
        transforms.append("cm_context:place_cm")
        return f"current chief minister of {resolved} "

    out = pattern_place_cm.sub(place_cm_sub, out)

    # cm of <place>
    pattern_cm_of = re.compile(
        rf"(?i)\bcm\s+of\s+({place})(?=\s*(?:currently|now|today)?\s*[?!.]|$)"
    )

    def cm_of_sub(match: re.Match[str]) -> str:
        resolved = _expand_place_token(match.group(1).strip())
        transforms.append("cm_context:cm_of")
        return f"current chief minister of {resolved}"

    out = pattern_cm_of.sub(cm_of_sub, out)

    # Short forms: "cm ap" / "cm ts" / "cm telangana"
    pattern_cm_short = re.compile(
        rf"(?i)(?<!\w)cm\s+({place})(?=\s*[?!.]|$)"
    )

    def cm_short_sub(match: re.Match[str]) -> str:
        resolved = _expand_place_token(match.group(1).strip())
        transforms.append("cm_context:cm_short")
        return f"current chief minister of {resolved}"

    out = pattern_cm_short.sub(cm_short_sub, out)

    # Expand known place misspellings anywhere (unambiguous compact forms).
    def place_spelling(match: re.Match[str]) -> str:
        raw = match.group(0)
        canon = _canonical_place(raw)
        if canon and canon.casefold() != raw.casefold():
            transforms.append(f"place:{raw.casefold()}->{canon}")
            return canon
        return raw

    out = re.sub(
        r"(?i)\b(andhraprdesh|andhrapradesh|telengana|tamilnadu)\b",
        place_spelling,
        out,
    )
    return out, transforms


def _expand_city_weather(text: str) -> tuple[str, list[str]]:
    transforms: list[str] = []
    out = text
    token_count = len(re.findall(r"\w+", out))

    def city_weather_sub(match: re.Match[str]) -> str:
        raw = match.group(1)
        city = CITY_ALIASES.get(raw.casefold())
        if not city:
            return match.group(0)
        transforms.append(f"city:{raw.casefold()}->{city}")
        weather = match.group(2)
        if weather:
            return f"{city} {weather}"
        # Bare city alias only for very short queries (e.g. "hyd").
        if token_count <= 3:
            return city
        return match.group(0)

    out = re.sub(
        r"(?i)\b(hyd|blr|bom)\b(?:\s+(weather|forecast|temperature|rain))?",
        city_weather_sub,
        out,
    )
    # "weather hyd" ordering
    def weather_city_sub(match: re.Match[str]) -> str:
        alias = match.group(2).casefold()
        city = CITY_ALIASES.get(alias)
        if not city:
            return match.group(0)
        transforms.append(f"city:{alias}->{city}")
        return f"{match.group(1)} {city}"

    out = re.sub(
        r"(?i)\b(weather|forecast)\s+(hyd|blr|bom)\b",
        weather_city_sub,
        out,
    )
    return out, transforms


def _expand_calc_shorthand(text: str) -> tuple[str, list[str]]:
    transforms: list[str] = []

    def calc_sub(_match: re.Match[str]) -> str:
        transforms.append("shorthand:calc")
        return "calculate "

    # Digits or protected math placeholders may follow.
    out = re.sub(
        r"(?i)(?<!\w)calc(?:ulate)?\s+(?=\d|__PROT)",
        calc_sub,
        text,
        count=1,
    )
    out = re.sub(
        r"(?i)^\s*calc\s*:\s*",
        calc_sub,
        out,
        count=1,
    )
    return out, transforms


def _expand_codegen_informal(text: str) -> tuple[str, list[str]]:
    """Light informal codegen phrasing — does not invent frameworks."""
    transforms: list[str] = []
    out = text
    m = re.match(
        r"(?i)^\s*(make|build|create)\s+(?:a\s+)?(?P<what>.+?)\s+"
        r"(?P<stack>react|vue|angular|fastapi|django|flask|next\.?js|node)\s*$",
        out,
    )
    if m:
        what = m.group("what").strip()
        stack = m.group("stack").strip()
        if stack.casefold() == "nextjs":
            stack = "Next.js"
        elif stack.casefold() == "react":
            stack = "React"
        transforms.append("informal:codegen_stack")
        out = f"Create a {what} using {stack}"
        return out, transforms

    m2 = re.match(
        r"(?i)^\s*(make|build)\s+(?:a\s+)?(?P<what>.+?)\s+"
        r"(?P<stack>react|vue|angular)\b",
        out,
    )
    if m2:
        transforms.append("informal:codegen_stack")
        stack = m2.group("stack")
        stack_name = "React" if stack.casefold() == "react" else stack
        out = f"Create a {m2.group('what').strip()} using {stack_name}"
    return out, transforms


def _expand_mixed_language(text: str) -> tuple[str, list[str], list[str]]:
    transforms: list[str] = []
    hints: list[str] = []
    out = text.strip()
    for pattern, template, tag in MIXED_LANG_PHRASES:
        m = pattern.match(out)
        if not m:
            continue
        body = (m.groupdict().get("body") or "").strip()
        head = (m.groupdict().get("head") or "").strip()
        # Keep technical tokens; only restructure order.
        if tag == "hindi_kya_hai":
            rest = (m.groupdict().get("rest") or "").strip()
            # "FastAPI kya hai aur kaise kaam karta hai" → what is FastAPI and how does it work
            if rest:
                rest_norm = rest
                rest_norm = re.sub(
                    r"(?i)\bkaise\s+kaam\s+karta\s+hai\b",
                    "how does it work",
                    rest_norm,
                )
                rest_norm = re.sub(r"(?i)\bkaise\b", "how", rest_norm)
                out = f"what is {body} and {rest_norm}"
            else:
                out = f"what is {body}"
            transforms.append(f"mixed_lang:{tag}")
            hints.append("hi-Latn")
            return out, transforms, hints
        if head and body:
            out = template.format(head=head, body=body)
        else:
            out = template.format(body=body, head=head)
        transforms.append(f"mixed_lang:{tag}")
        if "telugu" in tag:
            hints.append("te-Latn")
        return out, transforms, hints

    # Lightweight cue detection without rewrite.
    if re.search(r"(?i)\b(enti|ante|lo)\b", out):
        hints.append("te-Latn")
    if re.search(r"(?i)\b(kya|kaise|hai)\b", out):
        hints.append("hi-Latn")
    return out, transforms, hints


def detect_language_hints(text: str) -> tuple[str, ...]:
    hints: list[str] = []
    if re.search(r"(?i)\b(enti|ante|antedi|entidi|\blo\b)\b", text or ""):
        hints.append("te-Latn")
    if re.search(r"(?i)\b(kya|kaise|hai|hota)\b", text or ""):
        hints.append("hi-Latn")
    if re.search(r"[\u0C00-\u0C7F]", text or ""):
        hints.append("te")
    if re.search(r"[\u0900-\u097F]", text or ""):
        hints.append("hi")
    if not hints:
        hints.append("en")
    return tuple(dict.fromkeys(hints))


def normalize_user_input(text: str | None) -> NormalizedInput:
    """Normalize user text for routing without destroying meaning.

    Safe for empty / None input. Idempotent for already-normalized text in
    common cases.
    """
    original = "" if text is None else str(text)
    if not original.strip():
        return NormalizedInput(
            original=original,
            normalized=original.strip(),
            transforms=(),
            language_hints=("en",),
            protected_spans=0,
        )

    transforms: list[str] = []
    language_hints: list[str] = []

    protected_text, state = _protect(original)
    working = _collapse_whitespace(protected_text)
    if working != _collapse_whitespace(original) and state.spans == []:
        transforms.append("whitespace")
    working = _normalize_punctuation(working)

    working, t = _apply_safe_typos(working)
    transforms.extend(t)

    working, t, hints = _expand_mixed_language(working)
    transforms.extend(t)
    language_hints.extend(hints)

    working, t = _expand_calc_shorthand(working)
    transforms.extend(t)

    working, t = _expand_cm_and_places(working)
    transforms.extend(t)

    working, t = _expand_city_weather(working)
    transforms.extend(t)

    working, t = _expand_codegen_informal(working)
    transforms.extend(t)

    working = _collapse_whitespace(working)
    working = state.restore(working)
    working = _collapse_whitespace(working)

    if not language_hints:
        language_hints.extend(detect_language_hints(original))

    # Deduplicate transforms while preserving order.
    transforms = list(dict.fromkeys(transforms))
    language_hints = list(dict.fromkeys(language_hints))

    return NormalizedInput(
        original=original,
        normalized=working,
        transforms=tuple(transforms),
        language_hints=tuple(language_hints),
        protected_spans=len(state.spans),
    )


def routing_text(text: str | None) -> str:
    """Convenience: normalized text for classify_intent / route_request."""
    return normalize_user_input(text).normalized
