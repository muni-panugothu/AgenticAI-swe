"""Agent orchestration: intent routing, token budgeting, history compaction.

Keeps Groq requests under TPM limits and prevents wasteful repeated tools
without removing existing tool capabilities.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------
# Config (overridable via env)
# ---------------------------------------------------------------

# Hard ceiling near Groq openai/gpt-oss-20b free-tier TPM (~8000).
GROQ_TPM_BUDGET = int(os.getenv("GROQ_TPM_BUDGET", "7800"))
GROQ_COMPLETION_RESERVE = int(os.getenv("GROQ_COMPLETION_RESERVE", "900"))
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "900"))
GROQ_CHARS_PER_TOKEN = float(os.getenv("GROQ_CHARS_PER_TOKEN", "3.5"))

MAX_AGENT_LLM_ITERATIONS = int(os.getenv("MAX_AGENT_LLM_ITERATIONS", "8"))
MAX_TOOL_EXECUTIONS_PER_REQUEST = int(os.getenv("MAX_TOOL_EXECUTIONS_PER_REQUEST", "14"))
# Global cap for non-read tools (calculator, etc.).
MAX_SAME_TOOL_EXECUTIONS = int(os.getenv("MAX_SAME_TOOL_EXECUTIONS", "3"))
# read_file is scoped per file + overall request budget so multi-file
# investigations are not blocked after three reads of a different file.
MAX_READ_FILE_TOTAL = int(os.getenv("MAX_READ_FILE_TOTAL", "14"))
# Soft floor per file; scaled up when many required symbols live in one file.
MAX_READS_PER_FILE = int(os.getenv("MAX_READS_PER_FILE", "4"))
MAX_SEARCH_CODE_EXECUTIONS = int(os.getenv("MAX_SEARCH_CODE_EXECUTIONS", "4"))
# Keep at least this many LLM iterations for finalization / honest partials.
FINALIZATION_STEP_RESERVE = int(os.getenv("FINALIZATION_STEP_RESERVE", "1"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "2500"))
MAX_MEMORY_CONTEXT_CHARS = int(os.getenv("MAX_MEMORY_CONTEXT_CHARS", "3500"))


INTENT_GENERAL = "general_conversation"
INTENT_TECHNICAL = "technical_question"
INTENT_CODEBASE = "codebase_investigation"
INTENT_DEBUG = "debugging"
INTENT_CALC = "calculation"
INTENT_CURRENT = "current_information"
INTENT_MEMORY = "memory_operation"

INTENT_TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    INTENT_GENERAL: frozenset(),
    INTENT_TECHNICAL: frozenset(),
    INTENT_CODEBASE: frozenset({"search_code", "read_file"}),
    INTENT_DEBUG: frozenset({"search_code", "read_file", "calculator"}),
    INTENT_CALC: frozenset({"calculator"}),
    INTENT_CURRENT: frozenset({"web_search", "get_current_time"}),
    INTENT_MEMORY: frozenset({
        "search_memory",
        "save_memory",
        "detect_memory_conflict",
        "request_memory_update",
        "confirm_memory_update",
        "reinforce_memory",
        "forget_memory",
        "inspect_memory_metadata",
        "summarize_memories",
        "search_memory_summaries",
        "expand_memory_summary",
    }),
}


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, int(len(str(text)) / GROQ_CHARS_PER_TOKEN) + 1)


def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += 4  # role / framing overhead
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif content is not None:
            total += estimate_tokens(json.dumps(content, default=str))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            try:
                total += estimate_tokens(json.dumps(tool_calls, default=str))
            except Exception:
                total += 80 * len(tool_calls)
        if msg.get("name"):
            total += estimate_tokens(str(msg.get("name")))
    return total


def estimate_tools_tokens(tool_defs: list[dict] | None) -> int:
    if not tool_defs:
        return 0
    try:
        return estimate_tokens(json.dumps(tool_defs, default=str))
    except Exception:
        return 200 * len(tool_defs)


def _is_educational_or_conceptual(text: str) -> bool:
    """True when the user wants an explanation/example, not repo tools.

    Keywords like bug/investigate/fix alone must not force debugging —
    purpose matters: ``Explain how an agent investigates a bug`` is teaching.
    """
    if re.search(
        r"(?i)\b("
        r"explain|describe|difference between|tell me about|"
        r"what is|what's|what are|what does|"
        r"how (does|do|can|would|should|might|could)|"
        r"why (do|does|is|are|would|should)|"
        r"walk me through|help me understand|in simple terms|"
        r"beginner[- ]friendly|for beginners|teach me|"
        r"with (a |an )?(simple |real[- ]world )?example|"
        r"using (a |an )?(simple |real[- ]world )?example|"
        r"give (me )?(a |an )?example"
        r")\b",
        text,
    ):
        return True
    # Hypothetical / conceptual framing without pointing at this repo.
    if re.search(
        r"(?i)\b(in general|conceptually|theoretically|"
        r"an? (ai )?agent can|software engineers typically)\b",
        text,
    ):
        return True
    return False


def _requires_current_information(text: str) -> bool:
    """True when the answer depends on live / changing external facts.

    Must run before educational→technical fallthrough so questions like
    ``What is the current Chief Minister of Telangana?`` are not treated as
    tool-free conceptual answers (parametric knowledge goes stale).
    """
    if not text or not text.strip():
        return False

    # Explicit web / news / weather requests.
    if re.search(
        r"(?i)\b("
        r"search the web|search online|look up online|"
        r"latest news|breaking news|headlines|"
        r"weather|forecast"
        r")\b",
        text,
    ):
        return True

    # Clock / calendar (paired with get_current_time).
    if re.search(
        r"(?i)\b(current time|what time|today'?s date|what(?:'s| is) (the )?date)\b",
        text,
    ):
        return True

    # Live markets / rates / scores.
    if re.search(
        r"(?i)\b("
        r"(usd|inr|eur|gbp|bitcoin|btc|eth|crypto).{0,20}(price|rate|exchange)|"
        r"(exchange rate|stock price|share price|market price)|"
        r"live (score|scores|results?|updates?)"
        r")\b",
        text,
    ):
        return True

    # Current officeholders / political / org leadership.
    if re.search(
        r"(?i)\b("
        r"who is (the )?(current|present)|"
        r"who (is|are) (currently|presently)|"
        r"current (chief minister|prime minister|president|governor|mayor|"
        r"cm|ceo|speaker|chancellor)|"
        r"chief minister of|prime minister of|"
        r"who won\b|election results?"
        r")\b",
        text,
    ):
        return True

    # "current … of …" / latest developments / versions that change.
    if re.search(
        r"(?i)\bcurrent\b.{0,48}\b("
        r"minister|president|governor|mayor|ceo|cm of|"
        r"price of|rate for|version of|standings?|ranking"
        r")\b",
        text,
    ):
        return True

    if re.search(
        r"(?i)\b("
        r"(latest|recent) (developments?|news|release|version|update|updates?)|"
        r"as of today|right now"
        r")\b",
        text,
    ):
        return True

    # Time-bound “what’s new …” without treating product names as repo targets.
    if re.search(
        r"(?i)\b(what(?:'s| is) new)\b.{0,48}\b"
        r"(this (month|week|year)|in the (month|week|year)|lately|recently)\b",
        text,
    ):
        return True

    return False


def _targets_user_repo(text: str) -> bool:
    """True when the user points at their codebase files / this repository.

    Note: bare ``my project`` is often a memory fact (``What is my project?``),
    so only ``in my/the project`` counts as a repo target.
    """
    if re.search(r"\bapp/[\w./\\-]+", text):
        return True
    return bool(
        re.search(
            r"(?i)\b("
            r"in my (code|codebase|repo|repository|project|backend|frontend|file)|"
            r"in (the )?(codebase|repo|repository|project)|"
            r"my (backend|frontend|code|codebase|repo|repository|"
            r"implementation|stack trace|traceback)|"
            r"this (file|function|module|class|implementation)|"
            r"read_file|search_code"
            r")\b",
            text,
        )
    )


def _memory_system_feature_terms() -> str:
    """Feature nouns that mean the memory *subsystem*, not a personal fact."""
    return (
        r"architecture|system|layer|module|service|"
        r"permissions?|scoping|ownership|lifecycle|"
        r"summar(?:y|ies|ization)|conflict(?:s| resolution)?|"
        r"reinforcement|audit|retrieval"
    )


def _requests_codebase_evidence_report(text: str) -> bool:
    """True when the user wants source evidence (paths, lines, PASS/FAIL)."""
    return bool(
        re.search(
            r"(?i)\b("
            r"file paths?|line numbers?|"
            r"PASS\s*/\s*PARTIAL\s*/\s*FAIL|"
            r"read implementations?(?:\s+and\s+tests?)?|"
            r"source (?:code|files?)|"
            r"(?:relevant |unit )?tests?\b.{0,40}\b(?:implementations?|source)|"
            r"implementations?(?:\s+and\s+tests?)?"
            r")\b",
            text,
        )
    )


def _requests_memory_system_code_inspection(text: str) -> bool:
    """True for repository inspection of the agent memory system/service.

    A path mention like ``memory_service.py`` alone is not enough — require
    architecture / multi-feature inspection language so ordinary
    ``read reinforce_memory from memory_service.py`` stays a single-symbol
    investigation.
    """
    if not text:
        return False
    features = _memory_system_feature_terms()
    # Imperative inspect/verify of a memory subsystem feature (code, not facts).
    if re.search(
        rf"(?i)\b(inspect|verify|find|locate|review)\b.{{0,100}}\b"
        rf"memory\s+(?:{features})\b",
        text,
    ):
        return True
    # "read … memory scoping source code" / "memory permissions implementation"
    if re.search(
        rf"(?i)\b(inspect|verify|read|find|locate|review)\b.{{0,120}}\b"
        rf"memory\s+(?:{features})\b.{{0,80}}\b"
        rf"(implementation|source(?:\s+code|\s+files?)?|tests?|"
        rf"file paths?|line numbers?)\b",
        text,
    ):
        return True
    if re.search(
        rf"(?i)\bmemory\s+(?:{features})\b.{{0,60}}\b"
        rf"(implementation|source(?:\s+code|\s+files?)|unit tests?|"
        rf"file paths?|line numbers?|PASS\s*/\s*PARTIAL\s*/\s*FAIL)\b",
        text,
    ):
        return True
    # Permission-/scope-aware retrieval asked as a code inspection topic.
    if re.search(
        r"(?i)\b(permission-aware|scope-aware)\b.{0,60}\b"
        r"(semantic\s+)?(retrieval|search|implementation|source code|tests?)\b",
        text,
    ) and re.search(r"(?i)\b(inspect|verify|read|report)\b", text):
        return True
    # Broad memory-system phrases already used by Stage 19 architecture tests.
    if re.search(
        r"(?i)\b("
        r"memory (architecture|system|layer|module)|"
        r"memory service architecture|"
        r"(inspect|verify)\b.{0,80}\bmemory (architecture|system|service)|"
        r"(inspect|verify|read|find|locate)\b.{0,80}\b"
        r"(memory architecture|complete implementations?|"
        r"relevant tests).{0,40}\bmemor|"
        r"memor(?:y|ies).{0,40}\b(architecture|"
        r"relevant tests|unit tests)"
        r")\b",
        text,
    ):
        return True
    # "Inspect … memory … Report PASS/PARTIAL/FAIL with file paths"
    if (
        re.search(r"(?i)\bmemor(?:y|ies)\b", text)
        and _requests_codebase_evidence_report(text)
        and re.search(r"(?i)\b(inspect|verify|report)\b", text)
        and re.search(
            rf"(?i)\b(?:{features}|implementation|source code)\b",
            text,
        )
        # Educational "explain how … work" stays conceptual.
        and not re.search(
            r"(?i)\b(explain|what is|how (?:does|do|can|would)|teach me|"
            r"difference between)\b",
            text,
        )
    ):
        return True
    return False


def _requests_personal_memory_metadata(text: str) -> bool:
    """True for inspecting the user's stored memory records/metadata.

    Distinct from repository inspection of the memory *service* code.
    """
    if not text or _requests_memory_system_code_inspection(text):
        return False
    # Code-evidence language belongs to codebase investigation, not metadata.
    if _requests_codebase_evidence_report(text) and re.search(
        rf"(?i)\b(?:{_memory_system_feature_terms()}|implementation|"
        rf"source code)\b",
        text,
    ):
        return False
    lowered = text.lower()
    asks_inspect = bool(
        re.search(r"\b(inspect|verify|show|display|report)\b", lowered)
    )
    asks_metadata = bool(
        re.search(
            r"\b(metadata|fact_key|normalized_value|conflict_group_id|"
            r"conflict group|stored values?|actual (?:stored )?values?|"
            r"memory_id|status)\b",
            lowered,
        )
    )
    mentions_memories = bool(
        re.search(r"\b(memor(?:y|ies)|records?|facts?)\b", lowered)
    )
    forbids_modify = bool(
        re.search(
            r"\b(do not modify|don't modify|read-?only|"
            r"without (?:changing|modifying))\b",
            lowered,
        )
    )
    # "Do not modify files" on an implementation inspection is not metadata.
    forbids_only_files = bool(
        re.search(
            r"(?i)\b(do not|don't)\s+modify\s+files?\b|"
            r"\bwithout (?:changing|modifying)\s+files?\b",
            text,
        )
    )
    if asks_inspect and asks_metadata:
        return True
    if asks_inspect and mentions_memories and (
        asks_metadata
        or (forbids_modify and not forbids_only_files)
        or "conflict_group" in lowered
    ):
        return True
    if asks_metadata and mentions_memories and forbids_modify:
        return True
    return False


def _requests_repo_action(text: str) -> bool:
    """True when the user asks to inspect/find concrete source artifacts."""
    if _targets_user_repo(text):
        return True
    if _requests_memory_system_code_inspection(text):
        return True
    if re.search(
        r"(?i)\b("
        r"search (the )?code|"
        r"agent loop|"
        r"implementation of|source (of|for)|"
        # Imperative: find/locate/inspect a symbol, file, or implementation.
        r"(find|locate|show|open|inspect|read)\b.{0,80}\b"
        r"(implementation|function|class|file|definition|module|source|"
        r"code|in the code)|"
        # Find SOME_SYMBOL / find some_name.ext style lookups.
        r"(find|locate|show|open|inspect)\s+"
        r"[A-Za-z_][\w.]*(?:/[A-Za-z0-9_.-]+)*|"
        r"where is (the |my )?\w+"
        r")\b",
        text,
    ):
        return True
    return False


def _requests_debug_action(text: str) -> bool:
    """True for actionable debugging of a concrete error or implementation."""
    if re.search(
        r"(?i)\b("
        r"stack trace|traceback|"
        r"debug (this|my|the)|"
        r"investigate (this|my|the) (error|exception|bug|failure|issue|crash)|"
        r"diagnose (this|my|the)|"
        r"fix (this|my|the) (bug|error|exception|issue|crash)|"
        r"why (is|does|did) (my|this|the)\b|"
        r"(my|this) (code|test|endpoint|tool|function|agent) .{0,40}\b"
        r"(failing|broken|crashing|throwing|errors?)\b"
        r")",
        text,
    ):
        return True
    # "Investigate my tool execution error" — investigate/debug + failure noun
    # with a concrete determiner (my/this), not abstract "a software bug".
    if re.search(
        r"(?i)\b(investigate|debug|diagnose)\b.+\b"
        r"(error|exception|bug|failure|crash|issue)\b",
        text,
    ) and re.search(r"(?i)\b(my|this)\b", text):
        return True
    return False


def classify_intent(user_message: str) -> str:
    """Purpose-based intent router for tool selection.

    Educational / conceptual questions (including ones that mention bugs or
    investigation in the abstract) route to ``technical_question`` with no
    tools. Repo inspection and concrete failure diagnosis activate tools.
    """
    text = (user_message or "").strip()

    if not text:
        return INTENT_GENERAL

    # Repository inspection of the memory *system* is codebase work, not a
    # personal memory-tool operation (avoid fact_key / metadata short-circuit).
    if _requests_memory_system_code_inspection(text):
        return INTENT_CODEBASE

    # Personal stored-memory metadata / fact operations.
    if _requests_personal_memory_metadata(text):
        return INTENT_MEMORY

    # Natural personal-recall requests often name the saved fact directly
    # instead of saying "memory" (for example, "show my preferences" or
    # "retrieve my saved preferences"). Route these to the memory path while
    # keeping unrelated repository searches such as "find my agent loop"
    # outside personal memory retrieval.
    if re.search(
        r"(?i)\b(?:what(?:'s|\s+is|\s+are)|which|show|tell\s+me|"
        r"retrieve|recall|remind\s+me|list|get|find)\b"
        r".{0,55}\b(?:my|me|i)\b.{0,40}\b"
        r"(?:preferences?|favorites?|favourites?|preferred\s+"
        r"(?:language|database|editor|project)|(?:language|database|"
        r"editor|project)\s+i\s+prefer)\b",
        text,
    ) or re.search(
        r"(?i)\b(?:what(?:'s|\s+(?:is|are))?|which)\b.{0,45}\b"
        r"(?:preferences?|favorites?|favourites?)\b.{0,40}\b"
        r"(?:do\s+i\s+have|are\s+mine)\b",
        text,
    ) or re.search(
        r"(?i)\b(?:what|which)\b.{0,45}\b"
        r"(?:programming\s+)?(?:language|database|editor|project)\b"
        r".{0,30}\b(?:do\s+i\s+prefer|is\s+my\s+preferred)\b",
        text,
    ):
        return INTENT_MEMORY

    # Memory operations (personal stored facts / metadata for a fact key)
    if re.search(
        r"(?i)\b("
        r"remember that|forget that|"
        r"forget (?:the |my |this )?memor(?:y|ies)|"
        r"update my|confirm.*update|"
        r"search (?:my )?memor(?:y|ies)|what do you remember|"
        r"inspect.*metadata|"
        r"summarize (?:my )?memor(?:y|ies)|"
        r"retrieve (?:my |the )?(?:preferred )?[\w\s]{0,40}?memor(?:y|ies)|"
        r"(?:get|fetch|look up) (?:my |the )?[\w\s]{0,40}?from memor(?:y|ies)|"
        r"delete (?:a |the |my )?(?:stored )?memor(?:y|ies)"
        r")\b",
        text,
    ) and not _requests_memory_system_code_inspection(text):
        return INTENT_MEMORY

    # Calculations
    if re.search(r"(?i)\b(calculate|compute|calculator)\b", text):
        return INTENT_CALC
    if re.search(r"\d+\s*[\+\-\*\/x×]\s*\d+", text) and len(text) < 80:
        return INTENT_CALC

    # Current / live external information — BEFORE educational fallthrough.
    # Repo-local "current state of my X" stays investigation, not web search.
    if _requires_current_information(text) and not _targets_user_repo(text):
        return INTENT_CURRENT

    # Greetings / small talk
    if re.match(
        r"(?i)^\s*((hi|hello|hey)([,!]|\s)+)?"
        r"(how are you( doing)?|good (morning|afternoon|evening)|"
        r"thanks|thank you|bye|goodbye|"
        r"hi|hello|hey)"
        r"[\s.!?]*$",
        text,
    ):
        return INTENT_GENERAL

    educational = _is_educational_or_conceptual(text)
    repo_action = _requests_repo_action(text)
    debug_action = _requests_debug_action(text)
    # Imperative inspect/find command (sentence-initial), not "how to find…".
    imperative_inspect = bool(
        re.search(
            r"(?i)(?:^|[.!?]\s*)(find|locate|show|open|inspect|read)\b",
            text,
        )
    ) or _targets_user_repo(text)

    # Concrete failure diagnosis wins over educational phrasing
    # ("why is my test failing?" → debugging, not a tutorial).
    if debug_action:
        return INTENT_DEBUG

    # Teaching / examples with no command to inspect this repo → no tools.
    # "Explain how an agent investigates a bug" stays technical even though
    # it contains investigate/fix/bug. Mixed "Find X. Explain Y." keeps tools.
    if educational and not imperative_inspect:
        return INTENT_TECHNICAL

    # Concrete codebase inspection (paths, find symbol, inspect implementation).
    if repo_action or imperative_inspect:
        return INTENT_CODEBASE

    if educational:
        return INTENT_TECHNICAL

    # Short general chat without investigation / technical verbs
    if len(text.split()) <= 8 and not re.search(
        r"(?i)\b(code|file|function|class|tool|error|bug|memory|rag|llm|"
        r"api|database|python|what|explain|how)\b",
        text,
    ):
        return INTENT_GENERAL

    # Default: conceptual technical conversation without tools.
    return INTENT_TECHNICAL


def tools_for_intent(
    intent: str,
    all_tool_definitions: list[dict],
) -> list[dict]:
    """Return tool schemas allowed for this intent (may be empty)."""
    allow = INTENT_TOOL_ALLOWLIST.get(intent)
    if allow is None:
        return list(all_tool_definitions)
    if not allow:
        return []
    selected = []
    for item in all_tool_definitions:
        name = (item.get("function") or {}).get("name")
        if name in allow:
            selected.append(item)
    return selected


def intent_allows_tools(intent: str) -> bool:
    allow = INTENT_TOOL_ALLOWLIST.get(intent, frozenset())
    return bool(allow)


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 40)] + "\n...[truncated for token budget]..."


def compact_memory_context(memory_context: str | None) -> str:
    if not memory_context:
        return ""
    return truncate_text(memory_context, MAX_MEMORY_CONTEXT_CHARS)


def _message_as_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return dict(msg)
    # OpenAI SDK message objects
    data: dict[str, Any] = {
        "role": getattr(msg, "role", None),
        "content": getattr(msg, "content", None),
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        serialized = []
        for tc in tool_calls:
            serialized.append({
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "arguments": getattr(
                        getattr(tc, "function", None), "arguments", "{}"
                    ),
                },
            })
        data["tool_calls"] = serialized
    if getattr(msg, "tool_call_id", None):
        data["tool_call_id"] = msg.tool_call_id
    if getattr(msg, "name", None):
        data["name"] = msg.name
    return data


def _is_tool_sequence_start(msg: dict) -> bool:
    return msg.get("role") == "assistant" and bool(msg.get("tool_calls"))


def compact_conversation_history(
    history: list[Any],
    *,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict]:
    """Keep recent turns; compress older user/assistant text.

    Preserves assistant tool_calls + following tool results as atomic blocks.
    """
    normalized = [_message_as_dict(m) for m in history if m]
    if len(normalized) <= max_messages:
        return normalized

    # Split into blocks: tool sequences stay together.
    blocks: list[list[dict]] = []
    i = 0
    while i < len(normalized):
        msg = normalized[i]
        if _is_tool_sequence_start(msg):
            block = [msg]
            i += 1
            while i < len(normalized) and normalized[i].get("role") == "tool":
                block.append(normalized[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([msg])
            i += 1

    if len(blocks) <= max_messages:
        return [m for b in blocks for m in b]

    older = blocks[: -max_messages]
    recent = blocks[-max_messages:]

    summary_bits: list[str] = []
    for block in older:
        for msg in block:
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                summary_bits.append(f"{role}: {truncate_text(content.strip(), 180)}")
            elif role == "tool":
                summary_bits.append("tool: [result omitted in compacted history]")

    compacted: list[dict] = []
    if summary_bits:
        compacted.append({
            "role": "system",
            "content": (
                "Earlier conversation summary (compacted for token budget):\n"
                + "\n".join(summary_bits[-12:])
            ),
        })
    for block in recent:
        compacted.extend(block)
    return compacted


def prune_messages_for_budget(
    messages: list[Any],
    *,
    tool_defs: list[dict] | None,
    budget: int = GROQ_TPM_BUDGET,
    completion_reserve: int = GROQ_COMPLETION_RESERVE,
) -> list[dict]:
    """Reduce message list until estimated tokens fit the input budget."""
    msgs = [_message_as_dict(m) for m in messages]
    input_budget = max(800, budget - completion_reserve)
    tools_tokens = estimate_tools_tokens(tool_defs)

    def _total() -> int:
        return estimate_messages_tokens(msgs) + tools_tokens

    if _total() <= input_budget:
        return msgs

    # 1) Truncate large contents; keep recent tool evidence longer.
    tool_indices = [
        i for i, m in enumerate(msgs) if m.get("role") == "tool"
    ]
    recent_tool_idxs = set(tool_indices[-3:])
    for idx, msg in enumerate(msgs):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if idx == 0 and msg.get("role") == "system":
            if len(content) > 3500:
                msg["content"] = truncate_text(content, 3500)
            continue
        if msg.get("role") == "tool":
            limit = 1800 if idx in recent_tool_idxs else 500
        else:
            limit = 900
        if len(content) > limit:
            msg["content"] = truncate_text(content, limit)
        if _total() <= input_budget:
            return msgs

    # 2) Drop oldest non-system blocks while preserving the newest tool
    #    sequences (investigation evidence).
    guard = 0
    while _total() > input_budget and len(msgs) > 2 and guard < 60:
        guard += 1
        # Find oldest droppable index after system, preferring non-recent tools.
        drop_at = None
        for i in range(1, len(msgs)):
            if msgs[i].get("role") == "tool" and i in recent_tool_idxs:
                continue
            drop_at = i
            break
        if drop_at is None:
            break
        if _is_tool_sequence_start(msgs[drop_at]):
            end = drop_at + 1
            while end < len(msgs) and msgs[end].get("role") == "tool":
                end += 1
            # Never drop the newest tool sequence.
            if any(j in recent_tool_idxs for j in range(drop_at, end)):
                # Skip to next candidate by shrinking older user/assistant.
                if msgs[drop_at].get("role") in {"user", "assistant"}:
                    content = msgs[drop_at].get("content")
                    if isinstance(content, str) and len(content) > 120:
                        msgs[drop_at]["content"] = truncate_text(content, 120)
                        continue
                break
            del msgs[drop_at:end]
        else:
            del msgs[drop_at]
        tool_indices = [
            i for i, m in enumerate(msgs) if m.get("role") == "tool"
        ]
        recent_tool_idxs = set(tool_indices[-3:])

    # 3) Last resort: hard-trim, but keep a usable recent tool preview.
    if _total() > input_budget:
        for idx, msg in enumerate(msgs):
            if idx == 0:
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            floor = 500 if msg.get("role") == "tool" and idx in recent_tool_idxs else 200
            if len(content) > floor:
                msg["content"] = truncate_text(content, floor)

    return msgs


def is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def classify_groq_rate_limit(exc: BaseException) -> str | None:
    """Return 'tpm' | 'tpd' | 'rpm' | 'rpd' | None for a Groq 429.

    Prefers structured API error fields when present, then message text.
    Never classifies an ambiguous 429 as TPM.
    """
    # Structured metadata (OpenAI/Groq-style error objects).
    structured_bits: list[str] = []
    for attr in ("body", "response", "args", "error"):
        val = getattr(exc, attr, None)
        if val is None:
            continue
        try:
            structured_bits.append(str(val).lower())
        except Exception:
            continue
        # Nested JSON-ish dicts on common SDK attributes.
        if isinstance(val, dict):
            err = val.get("error") if isinstance(val.get("error"), dict) else val
            for key in ("code", "type", "message", "param"):
                if err.get(key) is not None:
                    structured_bits.append(str(err.get(key)).lower())
            # Groq sometimes nests quota labels under error.message already.
    text = " ".join([str(exc).lower(), *structured_bits])
    if "429" not in text and "rate_limit_exceeded" not in text:
        return None
    # Prefer explicit quota labels over generic "token" wording.
    if "tokens per day" in text or re.search(r"\btpd\b", text):
        return "tpd"
    if "requests per day" in text or re.search(r"\brpd\b", text):
        return "rpd"
    if "tokens per minute" in text or re.search(r"\btpm\b", text):
        return "tpm"
    if "requests per minute" in text or re.search(r"\brpm\b", text):
        return "rpm"
    # Ambiguous rate_limit_exceeded without a unit — do not guess TPM.
    return None


def is_tpm_rate_limit_error(exc: BaseException) -> bool:
    """HTTP 429 tokens-per-minute only."""
    return classify_groq_rate_limit(exc) == "tpm"


def is_tpd_rate_limit_error(exc: BaseException) -> bool:
    """HTTP 429 tokens-per-day (daily quota) — do not short-retry."""
    return classify_groq_rate_limit(exc) == "tpd"


def is_rpm_rate_limit_error(exc: BaseException) -> bool:
    return classify_groq_rate_limit(exc) == "rpm"


def is_rpd_rate_limit_error(exc: BaseException) -> bool:
    return classify_groq_rate_limit(exc) == "rpd"


def is_daily_quota_exhausted(exc: BaseException) -> bool:
    return classify_groq_rate_limit(exc) in {"tpd", "rpd"}


def is_token_limit_error(exc: BaseException) -> bool:
    """HTTP 413 / request-too-large — not 429 rate limits."""
    text = str(exc).lower()
    if "429" in text or "rate_limit_exceeded" in text:
        return False
    return (
        "413" in text
        or "request too large" in text
        or "context_length" in text
        or (
            "token" in text
            and ("too large" in text or "context" in text)
            and "per minute" not in text
            and "per day" not in text
        )
    )


def extract_retry_after_seconds(
    exc: BaseException,
    default: float = 5.0,
    *,
    max_wait: float = 30.0,
) -> float:
    """Parse Groq's 'try again in Xs' / 'YmZs' hint when present."""
    text = str(exc)
    # e.g. 25m25.823999999s
    match_m = re.search(
        r"try again in\s+([0-9]+)\s*m\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        text,
        flags=re.IGNORECASE,
    )
    if match_m:
        try:
            total = int(match_m.group(1)) * 60 + float(match_m.group(2))
            return min(max_wait, total + 0.35)
        except ValueError:
            pass
    match = re.search(
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return default
    try:
        return min(max_wait, float(match.group(1)) + 0.35)
    except ValueError:
        return default


def is_tool_choice_error(exc: BaseException) -> bool:
    """Groq 400 when the model emits a tool call without usable tools."""
    text = str(exc).lower()
    return (
        "tool choice is none" in text
        or "tool_use_failed" in text
        or ("400" in text and "tool" in text and "none" in text)
    )


TPM_RETRY_LIMIT = int(os.getenv("GROQ_TPM_RETRY_LIMIT", "2"))
MAX_EVIDENCE_SNIPPET_CHARS = int(os.getenv("MAX_EVIDENCE_SNIPPET_CHARS", "2000"))


# Intentional duplicate/overlap blocks — not execution failures.
BLOCKED_TOOL_REASONS = frozenset({
    "DUPLICATE_TOOL_CALL",
    "OVERLAPPING_FILE_READ",
    "PER_TOOL_EXECUTION_LIMIT",
    "PER_FILE_READ_LIMIT",
    "SAME_FILE_ALREADY_READ",
    "ALREADY_CONFIRMED_UNAVAILABLE",
    "SEARCH_SKIPPED_KNOWN_MISSING",
    "SEARCH_SKIPPED_KNOWN_SYMBOL",
    "SEARCH_SKIPPED_USER_FORBID",
    "REPEATED_FAILED_TOOL_CALL",
    "TOOL_EXECUTION_BUDGET_EXCEEDED",
})


def normalize_search_query(search_text: str) -> str:
    """Collapse a search_code query for dedupe / known-symbol checks."""
    return re.sub(r"[^a-z0-9_]+", "", (search_text or "").lower())


# Known definition lines for high-priority investigation targets.
KNOWN_SYMBOL_LOCATIONS: dict[str, tuple[str, int]] = {
    # Keep these in sync with actual def lines (AST). Stale anchors that
    # fall inside a neighboring function make rebuild windows return None.
    "classify_intent": ("app/services/agent_orchestration.py", 460),
    "classify_groq_rate_limit": ("app/services/agent_orchestration.py", 810),
    "is_tpm_rate_limit_error": ("app/services/agent_orchestration.py", 849),
    "is_tpd_rate_limit_error": ("app/services/agent_orchestration.py", 854),
    "is_rpm_rate_limit_error": ("app/services/agent_orchestration.py", 859),
    "is_rpd_rate_limit_error": ("app/services/agent_orchestration.py", 863),
    "is_daily_quota_exhausted": ("app/services/agent_orchestration.py", 867),
    "execute_tool": ("app/chatbot.py", 1214),
    # TPM/TPD handling lives in the agent LLM error path.
    "rate_limit_retry_handling": ("app/chatbot.py", 3905),
    "read_file": ("app/tools/file_reader.py", 316),
    "ast_enclosing_symbol_span": ("app/tools/file_reader.py", 77),
    "clip_to_function_boundary": ("app/tools/file_reader.py", 195),
    "InvestigationBudgetManager": ("app/services/investigation_state.py", 66),
    "InvestigationChecklist": ("app/services/investigation_state.py", 259),
    "build_budget_from_requirements": ("app/services/investigation_state.py", 420),
    "ReadFileInput": ("app/validation/tool_models.py", 23),
    "canonicalize_read_file_args": ("app/services/agent_orchestration.py", 3465),
    "build_finalization_messages": ("app/services/agent_orchestration.py", 4128),
    # Memory architecture — module-level functions (no MemoryService class).
    "memory_scope_keys": ("app/services/memory_summary_service.py", 216),
    "detect_memory_conflict": ("app/services/memory_conflict_service.py", 578),
    "calculate_memory_freshness_score": (
        "app/services/memory_lifecycle_service.py",
        32,
    ),
    "calculate_memory_priority_score": (
        "app/services/memory_lifecycle_service.py",
        60,
    ),
    "summarize_user_memories": ("app/services/memory_summary_service.py", 1257),
    "reinforce_memory": ("app/services/memory_service.py", 2636),
    "delete_memory": ("app/services/memory_service.py", 3485),
    "record_audit_event": ("app/services/memory_audit_service.py", 81),
    "create_embedding": ("app/services/embedding_service.py", 176),
    "semantic_search_memories": ("app/services/memory_service.py", 2281),
    # Prefer discover_symbol_definition when stale; kept as a fast hint only.
    "add_to_conversation_memory": ("app/chatbot.py", 1095),
    "get_conversation_memory": ("app/chatbot.py", 1114),
}

# Inclusive AST end lines for known symbols (never merge neighbors).
KNOWN_SYMBOL_END_LINES: dict[str, int] = {
    "classify_intent": 584,
    "classify_groq_rate_limit": 846,
    "is_tpm_rate_limit_error": 851,
    "is_tpd_rate_limit_error": 856,
    "is_rpm_rate_limit_error": 860,
    "is_rpd_rate_limit_error": 864,
    "is_daily_quota_exhausted": 868,
    "execute_tool": 1356,
    "memory_scope_keys": 258,
    "detect_memory_conflict": 594,
    "calculate_memory_freshness_score": 57,
    "calculate_memory_priority_score": 93,
    "summarize_user_memories": 1322,
    "reinforce_memory": 2798,
    "delete_memory": 3608,
    "record_audit_event": 131,
    "create_embedding": 213,
    "semantic_search_memories": 2590,
    "add_to_conversation_memory": 1107,
    "get_conversation_memory": 1116,
}

# Cache for filesystem definition discovery (path may move across modules).
_DISCOVERED_SYMBOL_LOCATIONS: dict[str, tuple[str, int]] = {}


def discover_symbol_definition(
    symbol_name: str,
    *,
    search_roots: tuple[str, ...] = ("app",),
) -> tuple[str, int] | None:
    """Locate ``def``/``class`` for a symbol by scanning project modules.

    Does not assume a fixed file (e.g. conversation memory need not live in
    ``app/chatbot.py``). Results are cached for the process lifetime.
    """
    name = str(symbol_name or "").strip()
    if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return None
    cached = _DISCOVERED_SYMBOL_LOCATIONS.get(name)
    if cached:
        return cached
    known = KNOWN_SYMBOL_LOCATIONS.get(name)
    # Validate known hint first; if the def moved, fall through to scan.
    if known:
        try:
            from app.tools.file_reader import load_project_source_text

            src = load_project_source_text(known[0])
            if src:
                lines = src.splitlines()
                idx = max(0, int(known[1]) - 1)
                window = lines[max(0, idx - 2) : idx + 3]
                pattern = re.compile(
                    rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b"
                )
                if any(pattern.search(ln) for ln in window):
                    _DISCOVERED_SYMBOL_LOCATIONS[name] = (
                        str(known[0]).replace("\\", "/"),
                        int(known[1]),
                    )
                    return _DISCOVERED_SYMBOL_LOCATIONS[name]
        except Exception:
            pass
    try:
        from app.tools.file_reader import PROJECT_ROOT
    except Exception:
        return None
    pattern = re.compile(
        rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b"
    )
    root = Path(PROJECT_ROOT)
    candidates: list[tuple[str, int]] = []
    for rel_root in search_roots:
        base = root / rel_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    rel = str(path.relative_to(root)).replace("\\", "/")
                    candidates.append((rel, i))
                    break
    if not candidates:
        return None
    # Prefer non-test app modules; shortest path as a stable tie-break.
    candidates.sort(
        key=lambda c: (
            0 if c[0].startswith("app/") else 1,
            len(c[0]),
            c[0],
            c[1],
        )
    )
    chosen = candidates[0]
    _DISCOVERED_SYMBOL_LOCATIONS[name] = chosen
    return chosen


def resolve_symbol_location(tid: str) -> tuple[str, int] | None:
    """Best-effort (path, def_line) via cache, known hints, then FS discovery."""
    name = str(tid or "").strip()
    if not name:
        return None
    if name in _DISCOVERED_SYMBOL_LOCATIONS:
        return _DISCOVERED_SYMBOL_LOCATIONS[name]
    discovered = discover_symbol_definition(name)
    if discovered:
        return discovered
    known = KNOWN_SYMBOL_LOCATIONS.get(name)
    if known:
        return (str(known[0]).replace("\\", "/"), int(known[1]))
    return None


def symbol_ast_span(tid: str) -> tuple[str, int, int] | None:
    """Return (path, start_line, end_line) for a known symbol, if available."""
    loc = resolve_symbol_location(tid) or KNOWN_SYMBOL_LOCATIONS.get(tid)
    end = KNOWN_SYMBOL_END_LINES.get(tid)
    if not loc or not end:
        return None
    return (str(loc[0]).replace("\\", "/"), int(loc[1]), int(end))


def shrink_read_window_for_char_budget(
    *,
    line_start: int,
    line_end: int,
    remaining_chars: int,
    chars_per_line: int = 80,
) -> tuple[int, int] | None:
    """Shrink a continuation window so it fits the remaining source-char budget.

    Returns ``None`` when even a minimal window cannot fit.
    """
    start = max(1, int(line_start))
    end = max(start, int(line_end))
    remaining = max(0, int(remaining_chars))
    if remaining < 200:
        return None
    max_lines = max(3, remaining // max(20, int(chars_per_line)))
    if end - start + 1 <= max_lines:
        return start, end
    return start, start + max_lines - 1


def record_unread_continuation_gap(
    evidence: dict,
    *,
    tid: str | None,
    file_path: str,
    next_line: int,
    suggested_end_line: int | None = None,
    reason: str = "SOURCE_CHAR_BUDGET_EXCEEDED",
) -> None:
    """Record the exact unread range when a continuation cannot proceed."""
    gap = {
        "target": tid,
        "file_path": str(file_path or "").replace("\\", "/"),
        "unread_from": int(next_line),
        "unread_to": int(suggested_end_line or next_line),
        "reason": reason,
    }
    evidence.setdefault("unread_ranges", []).append(gap)
    evidence["budget_exhaustion_reason"] = reason
    if tid:
        prog = evidence.setdefault("target_progress", {}).setdefault(tid, {})
        prog["function_complete"] = False
        prog["unread_from"] = int(next_line)
        prog["unread_to"] = int(suggested_end_line or next_line)
        prog["budget_blocked_reason"] = reason


def execute_investigation_test_module(
    module_path: str,
    *,
    timeout_sec: float = 30.0,
) -> dict:
    """Execute a test module via unittest and return structured evidence.

    Inspection of source is separate — this records real execution results.
    """
    import unittest
    from pathlib import Path

    path = str(module_path or "").replace("\\", "/")
    result = {
        "module": path,
        "status": TEST_NOT_EXECUTED,
        "passed": [],
        "failed": [],
        "errors": [],
        "message": "",
    }
    if not path:
        result["message"] = "missing test module path"
        return result
    # Resolve to a dotted module under tests/ when possible.
    rel = path
    if rel.startswith("tests/"):
        mod_name = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
    elif rel.endswith(".py"):
        mod_name = Path(rel).stem
        rel = f"tests/{Path(rel).name}"
        mod_name = rel[:-3].replace("/", ".")
    else:
        mod_name = path
    try:
        loader = unittest.defaultTestLoader
        suite = loader.loadTestsFromName(mod_name)
        if suite.countTestCases() == 0:
            result["status"] = TEST_NOT_EXECUTED
            result["message"] = f"no tests loaded from {mod_name}"
            return result
        stream = __import__("io").StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        # Best-effort timed run without a hard kill (Windows-safe).
        outcome = runner.run(suite)
        for test, _trace in outcome.failures:
            result["failed"].append(str(test))
        for test, _trace in outcome.errors:
            result["errors"].append(str(test))
        # Collect passed names from the suite id list minus failures/errors.
        all_ids: list[str] = []
        def _collect(s):
            if hasattr(s, "id"):
                all_ids.append(s.id())
            elif hasattr(s, "__iter__"):
                for c in s:
                    _collect(c)
        _collect(suite)
        bad = set(result["failed"]) | set(result["errors"])
        result["passed"] = [i for i in all_ids if i not in bad]
        if outcome.wasSuccessful():
            result["status"] = TEST_EXECUTION_PASSED
            result["message"] = (
                f"executed {outcome.testsRun} test(s); all passed"
            )
        else:
            result["status"] = TEST_EXECUTION_FAILED
            result["message"] = (
                f"executed {outcome.testsRun} test(s); "
                f"failures={len(outcome.failures)} errors={len(outcome.errors)}"
            )
        del timeout_sec  # reserved for future hard timeouts
        return result
    except Exception as exc:
        result["status"] = TEST_NOT_EXECUTED
        result["message"] = f"could not execute tests: {exc}"[:300]
        return result


def apply_test_execution_result(
    evidence: dict,
    *,
    tid: str,
    execution: dict,
) -> None:
    """Store test execution evidence for a test_module target."""
    bucket = evidence.setdefault("test_executions", {})
    bucket[tid] = dict(execution or {})
    prog = evidence.setdefault("target_progress", {}).setdefault(tid, {})
    prog["test_execution_status"] = str(
        (execution or {}).get("status") or TEST_NOT_EXECUTED
    )
    prog["test_passed"] = list((execution or {}).get("passed") or [])
    prog["test_failed"] = list((execution or {}).get("failed") or [])
    prog["test_errors"] = list((execution or {}).get("errors") or [])


def clip_range_to_symbol_span(
    tid: str,
    *,
    file_path: str,
    line_start: int,
    line_end: int,
) -> tuple[int, int] | None:
    """Clip a read window to the named symbol's AST span, or None if disjoint."""
    span = symbol_ast_span(tid)
    if not span:
        return int(line_start), int(line_end)
    path, start, end = span
    if path.replace("\\", "/").strip().lower() != str(file_path or "").replace(
        "\\", "/"
    ).strip().lower():
        return None
    clipped_start = max(int(line_start), start)
    clipped_end = min(int(line_end), end)
    if clipped_end < clipped_start:
        return None
    return clipped_start, clipped_end


def range_belongs_to_symbol(
    tid: str,
    *,
    file_path: str,
    line_start: int,
    line_end: int,
) -> bool:
    """True when the window overlaps the symbol's own AST span only."""
    clipped = clip_range_to_symbol_span(
        tid,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
    )
    return clipped is not None

# Prefer these when forcing unread-symbol reads.
RATE_LIMIT_SYMBOL_PRIORITY: list[str] = [
    "classify_groq_rate_limit",
    "is_tpm_rate_limit_error",
    "is_tpd_rate_limit_error",
    "is_rpm_rate_limit_error",
    "is_rpd_rate_limit_error",
    "is_daily_quota_exhausted",
    "rate_limit_retry_handling",
]

# Short architecture symbols — prefer before long multi-window targets
# (e.g. read_file) when the remaining tool budget is tight.
ARCHITECTURE_SYMBOL_PRIORITY: list[str] = [
    "InvestigationChecklist",
    "InvestigationBudgetManager",
    "build_budget_from_requirements",
    "ReadFileInput",
    "canonicalize_read_file_args",
    "ast_enclosing_symbol_span",
    "clip_to_function_boundary",
    "build_finalization_messages",
    "classify_intent",
]

# Memory architecture uses module-level service functions (not a MemoryService class).
MEMORY_SYMBOL_PRIORITY: list[str] = [
    # Short conversation-memory helpers first so they are not starved by
    # long multi-window bodies when the source-char budget is tight.
    "add_to_conversation_memory",
    "get_conversation_memory",
    "memory_scope_keys",
    "detect_memory_conflict",
    "calculate_memory_freshness_score",
    "calculate_memory_priority_score",
    "record_audit_event",
    "create_embedding",
    "summarize_user_memories",
    "reinforce_memory",
    "delete_memory",
    "semantic_search_memories",
]

# Explicit test-evidence statuses (inspection ≠ execution).
TEST_SOURCE_INSPECTED = "TEST_SOURCE_INSPECTED"
TEST_EXECUTION_PASSED = "TEST_EXECUTION_PASSED"
TEST_EXECUTION_FAILED = "TEST_EXECUTION_FAILED"
TEST_NOT_EXECUTED = "TEST_NOT_EXECUTED"
TEST_EVIDENCE_STATUSES = frozenset({
    TEST_SOURCE_INSPECTED,
    TEST_EXECUTION_PASSED,
    TEST_EXECUTION_FAILED,
    TEST_NOT_EXECUTED,
})

# Feature topic → primary module-level symbol (+ optional related tests).
MEMORY_FEATURE_TARGETS: list[dict] = [
    {
        "id": "memory_scope_keys",
        "feature": "scoping",
        "aliases": ["memory scoping", "scope", "memory_scope_keys"],
        "related_tests": ["tests/test_memory_summary_scoping.py"],
    },
    {
        "id": "detect_memory_conflict",
        "feature": "conflict_resolution",
        "aliases": [
            "conflict resolution",
            "detect_memory_conflict",
            "memory conflict",
        ],
        "related_tests": ["tests/test_memory_conflict_advanced.py"],
    },
    {
        "id": "calculate_memory_freshness_score",
        "feature": "freshness",
        "aliases": [
            "freshness",
            "calculate_memory_freshness_score",
            "stale memory",
        ],
        "related_tests": ["tests/test_memory_lifecycle_service.py"],
    },
    {
        "id": "calculate_memory_priority_score",
        "feature": "reliability",
        "aliases": [
            "reliability",
            "calculate_memory_priority_score",
            "priority score",
        ],
        "related_tests": ["tests/test_memory_lifecycle_service.py"],
    },
    {
        "id": "summarize_user_memories",
        "feature": "summarization",
        "aliases": [
            "summarization",
            "summarize_user_memories",
            "memory summary",
        ],
        "related_tests": ["tests/test_memory_summarization.py"],
    },
    {
        "id": "reinforce_memory",
        "feature": "reinforcement",
        "aliases": ["reinforcement", "reinforce_memory"],
        "related_tests": ["tests/test_stage19_duplicate_reinforcement.py"],
    },
    {
        "id": "delete_memory",
        "feature": "deletion",
        "aliases": ["deletion", "delete_memory", "forget"],
        "related_tests": ["tests/test_memory_stage17_service.py"],
    },
    {
        "id": "record_audit_event",
        "feature": "source_traceability",
        "aliases": [
            "source traceability",
            "traceability",
            "record_audit_event",
            "audit",
        ],
        "related_tests": ["tests/test_memory_security_and_tools.py"],
    },
    {
        "id": "create_embedding",
        "feature": "embeddings",
        "aliases": ["embeddings", "create_embedding", "embedding"],
        "related_tests": ["tests/test_stage19_retrieval.py"],
    },
    {
        "id": "semantic_search_memories",
        "feature": "semantic_retrieval",
        "aliases": [
            "semantic retrieval",
            "semantic_search_memories",
            "semantic search",
        ],
        "related_tests": ["tests/test_stage19_retrieval.py"],
    },
    {
        "id": "add_to_conversation_memory",
        "feature": "conversation_memory",
        "aliases": [
            "conversation memory",
            "add_to_conversation_memory",
            "get_conversation_memory",
        ],
        "related_tests": ["tests/test_chatbot_memory_stage17.py"],
    },
]

# Symbols known to need multiple continuation windows (~70 lines each).
LONG_MULTIWINDOW_TARGETS: frozenset[str] = frozenset({
    "read_file",
    "execute_tool",
    "rate_limit_retry_handling",
})


def reads_per_file_limit(
    *,
    file_path: str,
    requirements: dict | None = None,
    evidence: dict | None = None,
) -> int:
    """Scale the soft per-file read cap with required symbols in that file.

    Continuations + multiple helpers in one module must not hit
    PER_FILE_READ_LIMIT before every required symbol is inspected.
    """
    base = max(1, MAX_READS_PER_FILE)
    path = str(file_path or "").replace("\\", "/").strip().lower()
    if not path:
        return base
    symbols_on_file = 0
    req = requirements or {}
    ev = evidence or {}
    for item in req.get("required_targets") or []:
        tid = str(item.get("id") or "")
        if not tid:
            continue
        # Known location
        loc = KNOWN_SYMBOL_LOCATIONS.get(tid)
        if loc and str(loc[0]).replace("\\", "/").strip().lower() == path:
            symbols_on_file += 1
            continue
        # Discovered / suggested location
        for hit in (ev.get("discovered") or {}).get(tid) or []:
            if str(hit.get("file_path") or "").replace("\\", "/").strip().lower() == path:
                symbols_on_file += 1
                break
        else:
            for sug in ev.get("suggested_reads") or []:
                if sug.get("target") == tid and str(
                    sug.get("file_path") or ""
                ).replace("\\", "/").strip().lower() == path:
                    symbols_on_file += 1
                    break
    # +2 headroom for span continuations of the largest symbol.
    return max(base, symbols_on_file + 2, min(base + symbols_on_file, 12))


def investigation_read_budget(requirements: dict | None) -> int:
    """Overall read_file budget sized for the required-symbol checklist."""
    req = requirements or {}
    n = len(req.get("required_targets") or [])
    # Each symbol may need ~1–2 windows; keep a floor of MAX_READ_FILE_TOTAL.
    return max(MAX_READ_FILE_TOTAL, n + max(2, n // 2) + 2)


def intent_needs_long_term_memory(
    intent: str,
    user_message: str = "",
) -> bool:
    """Ordinary chat/code/debug/calc should not pull personal LTM.

    Personal-fact questions still load LTM so conflict clarification works.
    Codebase phrases like \"find my agent loop\" do not match.
    """
    if intent in {INTENT_MEMORY}:
        return True
    # Code / calc / current-info investigations stay memory-free.
    if intent in {
        INTENT_CODEBASE,
        INTENT_DEBUG,
        INTENT_CALC,
        INTENT_CURRENT,
    }:
        return False
    lowered = (user_message or "").lower()
    # Explicit personal recall (also drives conflict clarification).
    if re.search(
        r"(?i)\b("
        r"what('s| is) my|"
        r"who am i|"
        r"what do you (know|remember) about me|"
        r"remember about me|"
        r"did i (tell|ask) you|"
        r"ask(?:ed)? you to remember|"
        r"what did i (save|store|remember)|"
        r"my (current )?(primary )?(preferred )?"
        r"(project|name|editor|language|database|city|job|role|"
        r"preference|favorite)"
        r")\b",
        lowered,
    ):
        return True
    # Natural recall phrasing often names the fact before the first-person
    # subject ("Which language do I prefer?", "What project am I working
    # on?"). Keep this narrow to personal fact terms so ordinary technical
    # questions do not trigger a full long-term-memory load.
    personal_fact = (
        r"(?:preferred\s+)?(?:programming\s+)?language|project|editor|database|"
        r"preference|favorite|favourite|city|job|role|company|goal"
    )
    if re.search(
        rf"(?i)\b(?:what|which|tell\s+me|remind\s+me)\b.{{0,55}}"
        rf"\b(?:{personal_fact})\b.{{0,55}}\b(?:my|i|me)\b",
        lowered,
    ) or re.search(
        rf"(?i)\b(?:do\s+you\s+remember|remember|recall)\b.{{0,35}}"
        rf"\b(?:my|what\s+i)\b.{{0,40}}\b(?:{personal_fact})\b",
        lowered,
    ):
        return True
    return False


def should_force_final_answer(
    *,
    step: int,
    max_steps: int,
    successful_searches: int,
    successful_reads: int,
    total_successful_tools: int,
    blocked_in_a_row: int,
    calculator_required: bool = False,
    calculator_used: bool = False,
    pending_continuation: bool = False,
    requirements_met: bool = True,
) -> bool:
    """Reserve the last iteration for answering; stop spinning earlier.

    Do not finalize while a mandatory file continuation is pending or while
    investigation requirements from the user request remain unmet.
    Overlapping-read blocks alone must not end an incomplete investigation.
    """
    reserve = max(1, int(FINALIZATION_STEP_RESERVE))
    last_open = max_steps - reserve
    if pending_continuation and step < last_open:
        return False
    if not requirements_met and step < last_open:
        # Last reserved step(s) still force an honest partial via the caller.
        if calculator_required and not calculator_used:
            return step >= last_open
        if step < last_open:
            return False
    if step >= last_open:
        return True
    # Multi-tool arithmetic still pending — keep tools available.
    if calculator_required and not calculator_used:
        return False
    # Enough code evidence collected (search + multiple targeted reads).
    if (
        requirements_met
        and successful_searches >= 1
        and successful_reads >= 2
        and step >= 3
    ):
        return True
    if requirements_met and total_successful_tools >= 4 and step >= 4:
        return True
    # Model keeps requesting blocked duplicates — only finalize once
    # required source evidence is already collected.
    if (
        requirements_met
        and blocked_in_a_row >= 2
        and total_successful_tools >= 1
        and step >= 2
        and not pending_continuation
    ):
        return True
    return False


_STOPWORD_TARGETS = frozenset({
    "that", "this", "with", "from", "into", "about", "after", "before",
    "where", "which", "while", "when", "your", "their", "have", "been",
    "were", "will", "would", "could", "should", "implement", "implementation",
    "function", "functions", "method", "methods", "class", "source", "code",
    "file", "files", "error", "errors", "request", "requests", "explain",
    "inspect", "read", "find", "locate", "search", "complete", "entire",
    "backend", "frontend", "actual", "relevant", "handling", "detected",
    "educational", "debugging", "questions", "separated", "modify",
})


def _unique_targets(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        key = str(item.get("id") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _target(
    tid: str,
    *,
    aliases: list[str] | None = None,
    kind: str = "symbol",
    related_tests: list[str] | None = None,
    feature: str | None = None,
) -> dict:
    names = [tid] + list(aliases or [])
    row = {
        "id": tid,
        "aliases": names,
        "kind": kind,
        "need_read": True,
    }
    if related_tests:
        row["related_tests"] = list(related_tests)
    if feature:
        row["feature"] = feature
    return row


def _requests_memory_architecture_investigation(text: str) -> bool:
    """True when the user asks to inspect this repo's memory architecture."""
    if not text:
        return False
    if _requests_memory_system_code_inspection(text):
        return True
    return bool(
        re.search(
            r"(?i)\b("
            r"memory architecture|"
            r"memory (scoping|conflict|freshness|reliability|"
            r"summarization|reinforcement|deletion|traceability|"
            r"embeddings?|semantic retrieval|conversation memory)"
            r")\b",
            text,
        )
        and re.search(
            r"(?i)\b(inspect|verify|read|find|locate|report|"
            r"implementation|source|tests?)\b",
            text,
        )
    )


def _append_memory_architecture_targets(
    text: str,
    required_targets: list[dict],
    topics: list[str],
) -> None:
    """Map memory-architecture features to module-level service symbols."""
    if not _requests_memory_architecture_investigation(text):
        return
    topics.append("memory_architecture")
    # Memory-architecture investigations always require related test reads
    # for PASS; wording like "relevant tests" only expands the test set.
    wants_tests = True
    # Umbrella architecture asks include every feature; otherwise only
    # explicitly named features are required.
    inspect_all = bool(re.search(r"(?i)\bmemory architecture\b", text))
    pending_tests: list[tuple[str, str]] = []
    for spec in MEMORY_FEATURE_TARGETS:
        feature = str(spec.get("feature") or "")
        aliases = [str(a) for a in (spec.get("aliases") or []) if a]
        feature_phrases = aliases + [
            feature,
            feature.replace("_", " "),
        ]
        mentioned = inspect_all or any(
            re.search(rf"(?i)\b{re.escape(a)}\b", text)
            for a in feature_phrases
            if a
        )
        if not mentioned:
            continue
        topics.append(feature)
        related = list(spec.get("related_tests") or [])
        required_targets.append(
            _target(
                str(spec["id"]),
                aliases=aliases,
                related_tests=related or None,
                feature=feature,
            )
        )
        for test_path in related:
            pending_tests.append((feature, str(test_path).replace("\\", "/")))

    # Cap unique test modules so the checklist stays within tool budget.
    if wants_tests and pending_tests:
        seen_tests: set[str] = set()
        for feature, test_path in pending_tests:
            if test_path in seen_tests:
                continue
            seen_tests.add(test_path)
            if len(seen_tests) > 6:
                break
            test_id = test_path.rstrip("/").split("/")[-1].removesuffix(".py")
            required_targets.append(
                _target(
                    test_id,
                    aliases=[test_id, test_path],
                    kind="test_module",
                    feature=f"{feature}_tests",
                )
            )


def scope_investigation_requirements_for_intent(
    intent: str,
    requirements: dict | None,
) -> dict:
    """Apply codebase evidence targets only on investigation intents.

    Current/web/calc/memory/general/RAG turns must not invent ``read_file``
    requirements from product names (e.g. ``FastAPI`` in a web-search query)
    or treat the word ``search`` as a mandatory ``search_code`` obligation.
    Preserves Topics 36 / B4–B6 investigation contracts for CODEBASE/DEBUG.
    """
    req = dict(requirements or {})
    if intent in {INTENT_CODEBASE, INTENT_DEBUG}:
        return req
    req["required_targets"] = []
    req["needs_search"] = False
    req["needs_source_reads"] = False
    req["needs_complete_function"] = False
    req["min_source_reads"] = 0
    req["rate_limit_categories"] = []
    if intent != INTENT_CALC:
        req["needs_calculator"] = False
    req["scoped_out_for_intent"] = intent
    return req


def derive_investigation_requirements(user_message: str) -> dict:
    """Track what the original request requires before a complete answer.

    Distinguishes *discovering* symbols (search_code) from *reading* their
    source (read_file). Named implementations and rate-limit categories are
    tracked as required targets that need source evidence.
    """
    text = user_message or ""
    lowered = text.lower()
    needs_complete = bool(
        re.search(
            r"(?i)\b(complete|full|entire|whole)\b.{0,40}\b"
            r"(implementation|function|method|class|source)\b"
            r"|\binspect (its |the )?complete\b"
            r"|\bread sequential\b"
            r"|\bwhen necessary\b"
            r"|\bread (the )?relevant source\b"
            r"|\bdo not modify\b",
            text,
        )
    )
    needs_search = bool(
        re.search(
            r"(?i)\b(find|locate|search|where is|inspect)\b",
            text,
        )
    ) or bool(re.search(r"\bapp/[\w./\\-]+", text))
    needs_calc = bool(
        re.search(r"(?i)\b(calculate|compute|calculator)\b", text)
        or re.search(r"\d+\s*[×x\*]\s*\d+", text)
    )
    topics: list[str] = []
    if re.search(r"(?i)\b(lookup|tool lookup|registry)\b", text):
        topics.append("lookup")
    if re.search(r"(?i)\b(validat|argument)\b", text):
        topics.append("validation")
    if re.search(r"(?i)\b(success|successful execution)\b", text):
        topics.append("success_path")
    if re.search(r"(?i)\b(exception|error.handl|failure)\b", text):
        topics.append("exception_path")

    required_targets: list[dict] = []

    # Explicit snake_case / CamelCase identifiers in the request.
    # Skip path segments (app/foo/bar.py) — those are locations, not symbols.
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b", text):
        name = match.group(1)
        low = name.lower()
        if low in _STOPWORD_TARGETS:
            continue
        start, end = match.span(1)
        # Part of a path or filename: ".../name..." or "name.py" / "name/"
        # Do not treat a sentence period ("execute_tool.") as a path.
        if start > 0 and text[start - 1] in "/\\":
            continue
        if end < len(text) and text[end] in "/\\":
            continue
        if end + 2 < len(text) and text[end] == "." and text[end + 1 : end + 3].isalpha():
            # filename extension like ".py"
            continue
        is_snake = "_" in name
        is_camel = name[:1].islower() and any(c.isupper() for c in name[1:])
        is_pascal = (
            name[:1].isupper()
            and any(c.islower() for c in name[1:])
            and any(c.isupper() for c in name[1:])
        )
        if is_snake or is_camel or is_pascal:
            required_targets.append(
                _target(name, kind="class" if is_pascal else "symbol")
            )
        elif name.startswith(
            ("is_", "classify_", "parse_", "build_", "should_")
        ):
            required_targets.append(_target(name))

    # Architecture topics → concrete symbols (search discovers; reads verify).
    if re.search(r"(?i)\bpydantic\b|\bReadFileInput\b", text):
        required_targets.append(
            _target(
                "ReadFileInput",
                aliases=["Pydantic", "Pydantic validation", "ReadFileInput"],
                kind="class",
            )
        )
        topics.append("validation")
    elif re.search(r"(?i)\bpydantic validation\b|\bstrict validation\b", text):
        required_targets.append(
            _target(
                "ReadFileInput",
                aliases=["Pydantic", "Pydantic validation", "validation"],
                kind="class",
            )
        )
        topics.append("validation")
    if re.search(r"(?i)\bcanonical\b.{0,40}\barg", text):
        required_targets.append(
            _target(
                "canonicalize_read_file_args",
                aliases=["canonical arguments", "canonical argument"],
            )
        )
    if re.search(
        r"(?i)\bAST\b|\bast_enclosing\b|function boundary|boundary detection",
        text,
    ):
        required_targets.append(
            _target(
                "ast_enclosing_symbol_span",
                aliases=["AST", "AST boundary", "function boundary"],
            )
        )
        required_targets.append(
            _target(
                "clip_to_function_boundary",
                aliases=["clip_to_function_boundary", "boundary clip"],
            )
        )
    if re.search(
        r"(?i)\bbudget enforcement\b|InvestigationBudgetManager\b",
        text,
    ):
        required_targets.append(
            _target(
                "InvestigationBudgetManager",
                aliases=["budget enforcement", "InvestigationBudgetManager"],
            )
        )
        required_targets.append(
            _target(
                "build_budget_from_requirements",
                aliases=["build_budget_from_requirements"],
            )
        )
    if re.search(r"(?i)\bInvestigationChecklist\b", text):
        required_targets.append(
            _target(
                "InvestigationChecklist",
                aliases=["InvestigationChecklist", "checklist"],
                kind="class",
            )
        )
    if re.search(r"(?i)\bInvestigationBudgetManager\b", text):
        required_targets.append(
            _target(
                "InvestigationBudgetManager",
                aliases=["InvestigationBudgetManager", "budget manager"],
                kind="class",
            )
        )
    if re.search(r"(?i)\btool-?free finalization\b|\bfinalization\b", text):
        required_targets.append(
            _target(
                "build_finalization_messages",
                aliases=["tool-free finalization", "finalization"],
            )
        )

    # Memory architecture → module-level service functions + related tests.
    _append_memory_architecture_targets(text, required_targets, topics)

    # Intent-router / educational-vs-debug investigations.
    if re.search(
        r"(?i)\b(intent router|classify_intent|educational.{0,40}debug|"
        r"debug.{0,40}educational|purpose-based)\b",
        text,
    ):
        required_targets.append(
            _target("classify_intent", aliases=["intent router", "purpose-based"])
        )
        topics.append("educational_vs_debug")

    # TPM / TPD / RPM / RPD classification + retry handling.
    rate_cats = []
    for cat in ("tpm", "tpd", "rpm", "rpd"):
        if re.search(rf"(?i)\b{cat}\b", text):
            rate_cats.append(cat.upper())
    if rate_cats or re.search(
        r"(?i)\b(rate[- ]limit|groq error-handling|tokens per|"
        r"requests per)\b",
        text,
    ):
        topics.append("rate_limits")
        required_targets.extend([
            _target(
                "classify_groq_rate_limit",
                aliases=["classify_groq_rate_limit", "rate limit classifier"],
            ),
            _target("is_tpm_rate_limit_error", aliases=["is_tpm", "tpm"]),
            _target("is_tpd_rate_limit_error", aliases=["is_tpd", "tpd"]),
            _target("is_rpm_rate_limit_error", aliases=["is_rpm", "rpm"]),
            _target("is_rpd_rate_limit_error", aliases=["is_rpd", "rpd"]),
            _target(
                "rate_limit_retry_handling",
                aliases=[
                    "is_daily_quota_exhausted",
                    "TPM_RETRY",
                    "tpd stops",
                    "daily quota",
                ],
                kind="behavior",
            ),
        ])
        for cat in rate_cats:
            topics.append(cat.lower())

    required_targets = _unique_targets(required_targets)
    needs_source = bool(
        required_targets
        or needs_complete
        or bool(topics)
        or re.search(
            r"(?i)\b(implementation|source code|read the|inspect.{0,20}code)\b",
            text,
        )
    )
    min_reads = 1 if needs_source else 0
    if required_targets:
        min_reads = max(min_reads, min(3, len(required_targets)))

    forbid_search = bool(
        re.search(
            r"(?i)\bdo not search again\b"
            r"|\bdon't search again\b"
            r"|\bdo not search\b"
            r"|\bwithout (further )?search(ing)?\b"
            r"|\bno (more|further) search(ing)?\b",
            text,
        )
    )
    return {
        "needs_search": (
            False
            if forbid_search
            else (needs_search or bool(required_targets))
        ),
        "needs_complete_function": needs_complete or bool(topics),
        "needs_source_reads": needs_source,
        "needs_calculator": needs_calc,
        "topics": topics,
        "required_targets": required_targets,
        "min_source_reads": min_reads,
        "rate_limit_categories": rate_cats,
        "forbid_search": forbid_search,
        "raw": text,
    }


def new_investigation_evidence_state() -> dict:
    """Mutable tracker: discovered (search) vs verified (read_file source).

    ``target_statuses`` holds explicit lifecycle states:
    pending → discovered → reading → verified | failed.
    Search hits alone never set verified.
    """
    return {
        "discovered": {},  # target_id -> [{file_path, line, text}]
        "read_ranges": [],  # [{file_path, line_start, line_end}]
        "verified_targets": set(),  # target ids covered by read_file body
        "search_snippets": [],
        "read_snippets": [],
        "suggested_reads": [],  # [{file_path, line_start, line_end, target}]
        "target_statuses": {},  # tid -> {status, file_path, line_start, ...}
        "budget_exhaustion_reason": None,
        # Class-level ledger: expected range, methods, coverage gaps.
        "class_targets": {},  # tid -> class coverage dict
    }


def parse_search_code_matches(tool_result: dict | None) -> list[dict]:
    """Extract match rows from a search_code tool result."""
    if not isinstance(tool_result, dict):
        return []
    raw = tool_result.get("result")
    payload = raw if isinstance(raw, dict) else None
    if isinstance(raw, str):
        try:
            payload = json.loads(raw.split("\ncontent:\n", 1)[0])
        except Exception:
            payload = None
    if not isinstance(payload, dict) or not payload.get("success"):
        return []
    matches = payload.get("matches") or []
    out: list[dict] = []
    for item in matches:
        if not isinstance(item, dict):
            continue
        path = str(item.get("file_path") or "").replace("\\", "/")
        try:
            line = int(item.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        text = str(item.get("text") or "")
        if path and line > 0:
            out.append({"file_path": path, "line": line, "text": text})
    return out


def _target_aliases(requirements: dict) -> list[tuple[str, list[str]]]:
    rows = []
    for item in requirements.get("required_targets") or []:
        tid = str(item.get("id") or "")
        aliases = [str(a) for a in (item.get("aliases") or [tid]) if a]
        if tid:
            rows.append((tid, aliases))
    return rows


def _match_is_definition(text: str, tid: str) -> bool:
    """True when search text looks like a real def/class header for tid."""
    name = re.escape(str(tid or "").strip())
    if not name:
        return False
    return bool(
        re.search(
            rf"(?m)^\s*(?:async\s+)?(?:def|class)\s+{name}\b",
            text or "",
        )
    )


def _match_is_import_only(text: str, tid: str) -> bool:
    """True for import/from lines that mention tid but are not definitions."""
    blob = text or ""
    if _match_is_definition(blob, tid):
        return False
    low = blob.lower()
    name = str(tid or "").lower()
    if name and name not in low:
        return False
    return bool(
        re.search(r"(?i)^\s*(from\s+\S+\s+import|import\s+)", blob)
        or re.search(r"(?i)\bimport\b", blob)
    )


def register_search_discoveries(
    evidence: dict,
    requirements: dict,
    matches: list[dict],
) -> list[dict]:
    """Record search hits and suggest non-overlapping read windows.

    Discovering a function name is not the same as reading its source.
    Status moves pending → discovered only (never verified).

    Prefer ``def``/``class`` hits over import lines. When only an import is
    found, anchor the suggested read at the known module-level definition
    (memory services are functions, not a ``MemoryService`` class).
    """
    from app.services.investigation_state import InvestigationChecklist

    checklist = InvestigationChecklist(requirements)
    checklist.sync_from_evidence(evidence)
    suggested: list[dict] = []
    # Rank matches so definitions are processed before imports.
    ranked = sorted(
        matches,
        key=lambda m: (
            0
            if any(
                _match_is_definition(m.get("text") or "", tid)
                for tid, _ in _target_aliases(requirements)
            )
            else 1,
            int(m.get("line") or 0),
        ),
    )
    for match in ranked:
        path = str(match["file_path"]).replace("\\", "/")
        line = match["line"]
        text = match.get("text") or ""
        evidence["search_snippets"].append(
            f"search_code hit {path}:{line}: {truncate_text(text, 160)}"
        )
        for tid, aliases in _target_aliases(requirements):
            blob = f"{text} {path}".lower()
            if not (
                any(a.lower() in blob for a in aliases) or tid.lower() in blob
            ):
                continue
            evidence["discovered"].setdefault(tid, []).append(match)

            # Imports are discovery only — read the real definition when known.
            anchor_path, anchor_line = path, int(line)
            if _match_is_import_only(text, tid) or (
                not _match_is_definition(text, tid)
                and tid in KNOWN_SYMBOL_LOCATIONS
            ):
                known = KNOWN_SYMBOL_LOCATIONS.get(tid)
                if known:
                    anchor_path, anchor_line = (
                        str(known[0]).replace("\\", "/"),
                        int(known[1]),
                    )

            window = _suggestion_window_for_symbol(
                anchor_path,
                anchor_line,
                prior=list(evidence.get("read_ranges") or []),
            )
            if window is None:
                start = max(1, int(anchor_line) - 1)
                end = start + 24
            else:
                start = window["line_start"]
                end = window["line_end"]
                anchor_path = window["file_path"]
            suggestion = {
                "file_path": anchor_path,
                "line_start": start,
                "line_end": end,
                "target": tid,
            }
            checklist.mark_discovered(
                tid,
                file_path=anchor_path,
                def_line=int(anchor_line),
                line_start=start,
                line_end=end,
            )
            overlap = any(
                overlapping_read_range(
                    {
                        "file_path": anchor_path,
                        "line_start": start,
                        "line_end": end,
                    },
                    prior,
                )
                for prior in evidence.get("read_ranges") or []
            )
            if not overlap:
                already = any(
                    s.get("target") == tid
                    and str(s.get("file_path") or "").replace("\\", "/")
                    == anchor_path
                    for s in (evidence.get("suggested_reads") or [])
                )
                if not already:
                    suggested.append(suggestion)
                    evidence["suggested_reads"].append(suggestion)
    checklist.apply_to_evidence(evidence)
    return suggested


def _target_def_line(tid: str, evidence: dict) -> tuple[str | None, int | None]:
    """Best-known (path, def_line) for a required symbol."""
    # Prefer evidence / search discoveries, then FS discovery, then hints.
    hits = (evidence.get("discovered") or {}).get(tid) or []
    for hit in hits:
        text = str(hit.get("text") or "")
        if _match_is_definition(text, tid):
            try:
                return (
                    str(hit["file_path"]).replace("\\", "/"),
                    int(hit["line"]),
                )
            except (KeyError, TypeError, ValueError):
                pass
    resolved = resolve_symbol_location(tid)
    if resolved:
        return resolved
    hits = (evidence.get("discovered") or {}).get(tid) or []
    if hits:
        hit = hits[0]
        try:
            return (
                str(hit.get("file_path") or "").replace("\\", "/"),
                int(hit.get("line") or 0) or None,
            )
        except (TypeError, ValueError):
            return None, None
    meta = (evidence.get("target_statuses") or {}).get(tid) or {}
    if meta.get("def_line"):
        return (
            str(meta.get("file_path") or "").replace("\\", "/") or None,
            int(meta["def_line"]),
        )
    return None, None


def _line_covered_by_ranges(
    path: str,
    line: int,
    ranges: list[dict],
) -> bool:
    path_key = str(path or "").replace("\\", "/").strip().lower()
    for rng in ranges:
        r_path, start, end = _read_line_bounds(rng)
        if r_path == path_key and start <= int(line) <= end:
            return True
    return False


def _body_supports_symbol(
    tid: str,
    aliases: list[str],
    body: str,
    *,
    kind: str = "symbol",
) -> bool:
    """True when a source excerpt is enough to treat ``tid`` as verified.

    Symbol targets require a real ``def``/``class`` line (or an unambiguous
    ``def <alias>``). Short aliases like ``tpm`` alone must not verify a
    helper from a neighboring classifier docstring.
    """
    if not body:
        return False
    lowered = body.lower()
    if re.search(
        rf"(?im)^\s*(def|async def|class)\s+{re.escape(tid)}\b",
        body,
    ):
        return True
    # Behavior targets (retry handling) use phrase evidence, not a def name.
    if kind == "behavior" or tid == "rate_limit_retry_handling":
        if re.search(
            r"(?i)(is_daily_quota_exhausted|is_tpd_rate_limit_error|"
            r"is_tpm_rate_limit_error|TPM_RETRY|backing off|"
            r"tokens per day|daily quota exhausted)",
            body,
        ):
            return True
        return False
    for alias in aliases:
        al = str(alias or "").strip()
        if not al or al.lower() == tid.lower():
            continue
        # Only accept another identifier if it appears as a definition.
        if re.search(
            rf"(?im)^\s*(def|async def|class)\s+{re.escape(al)}\b",
            body,
        ):
            return True
    return False


def _is_pascal_case_name(name: str) -> bool:
    text = str(name or "")
    return bool(
        text[:1].isupper()
        and any(c.islower() for c in text[1:])
        and any(c.isupper() for c in text[1:])
    )


def _target_kind(tid: str, requirements: dict | None = None) -> str:
    for item in (requirements or {}).get("required_targets") or []:
        if str(item.get("id") or "") == tid:
            return str(item.get("kind") or "symbol")
    return "class" if _is_pascal_case_name(tid) else "symbol"


def resolve_class_target_metadata(
    tid: str,
    *,
    file_path: str | None = None,
) -> dict | None:
    """Load AST class range + methods for a class target."""
    from app.tools.file_reader import (
        ast_class_definition,
        load_project_source_text,
    )

    path = str(file_path or "").replace("\\", "/").strip()
    if not path:
        loc = KNOWN_SYMBOL_LOCATIONS.get(tid)
        if not loc:
            return None
        path = str(loc[0]).replace("\\", "/")
    source = load_project_source_text(path)
    if not source:
        return None
    info = ast_class_definition(source, tid)
    if not info:
        return None
    methods = list(info.get("methods") or [])
    method_evidence = [
        {
            "name": str(m.get("name")),
            "line_start": int(m["line_start"]),
            "line_end": int(m["line_end"]),
            "purpose": str(m.get("purpose") or f"def {m.get('name')}(...)"),
            "verified": False,
            "evidence_ranges": [],
        }
        for m in methods
        if m.get("name") and m.get("line_start") and m.get("line_end")
    ]
    return {
        "name": info["name"],
        "file_path": path,
        "line_start": int(info["line_start"]),
        "line_end": int(info["line_end"]),
        "methods": methods,
        "discovered_methods": [str(m.get("name")) for m in methods],
        "verified_methods": [],
        "method_evidence": method_evidence,
        "read_ranges": [],
        "missing_ranges": [
            {
                "line_start": int(info["line_start"]),
                "line_end": int(info["line_end"]),
            }
        ],
        # Separate flags: line coverage vs method evidence completeness.
        "source_coverage_complete": False,
        "method_enumeration_complete": bool(method_evidence),
        "method_evidence_complete": False,
        "coverage_complete": False,
        "span_limited_unresolved": False,
    }


def ensure_class_target_metadata(
    evidence: dict,
    requirements: dict,
    *,
    tid: str | None = None,
) -> dict:
    """Ensure ``evidence['class_targets']`` has AST metadata for class ids."""
    class_map = evidence.setdefault("class_targets", {})
    targets = []
    if tid:
        targets = [tid]
    else:
        for item in requirements.get("required_targets") or []:
            cid = str(item.get("id") or "")
            if not cid:
                continue
            if str(item.get("kind") or "") == "class" or _is_pascal_case_name(cid):
                targets.append(cid)
    for cid in targets:
        existing = class_map.get(cid)
        if existing and existing.get("line_end"):
            # Upgrade legacy ledgers that lack explicit method_evidence.
            if existing.get("method_evidence") is None:
                methods = list(existing.get("methods") or [])
                existing["method_evidence"] = [
                    {
                        "name": str(m.get("name")),
                        "line_start": int(m.get("line_start") or 0),
                        "line_end": int(m.get("line_end") or 0),
                        "purpose": str(
                            m.get("purpose") or f"def {m.get('name')}(...)"
                        ),
                        "verified": False,
                        "evidence_ranges": [],
                    }
                    for m in methods
                    if m.get("name")
                ]
                existing.setdefault("source_coverage_complete", False)
                existing.setdefault("method_enumeration_complete", bool(methods))
                existing.setdefault("method_evidence_complete", False)
                existing["coverage_complete"] = False
            continue
        loc = KNOWN_SYMBOL_LOCATIONS.get(cid)
        path = str(loc[0]) if loc else None
        # Prefer a discovered path when known locations are absent.
        if not path:
            hits = (evidence.get("discovered") or {}).get(cid) or []
            if hits:
                path = str(hits[0].get("file_path") or "")
        meta = resolve_class_target_metadata(cid, file_path=path)
        if meta:
            class_map[cid] = meta
    return class_map


def _merge_line_ranges(ranges: list[dict]) -> list[dict]:
    """Merge overlapping/adjacent inclusive line ranges."""
    norm: list[tuple[int, int]] = []
    for item in ranges:
        try:
            a = int(item.get("line_start") or 0)
            b = int(item.get("line_end") or 0)
        except (TypeError, ValueError):
            continue
        if a < 1 or b < a:
            continue
        norm.append((a, b))
    if not norm:
        return []
    norm.sort()
    merged = [list(norm[0])]
    for a, b in norm[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [{"line_start": a, "line_end": b} for a, b in merged]


def _missing_ranges_within(
    *,
    cover_start: int,
    cover_end: int,
    read_ranges: list[dict],
) -> list[dict]:
    """Return gaps inside ``[cover_start, cover_end]`` not covered by reads."""
    if cover_end < cover_start:
        return []
    merged = _merge_line_ranges(read_ranges)
    gaps: list[dict] = []
    cursor = cover_start
    for item in merged:
        a = int(item["line_start"])
        b = int(item["line_end"])
        if b < cover_start or a > cover_end:
            continue
        a = max(a, cover_start)
        b = min(b, cover_end)
        if a > cursor:
            gaps.append({"line_start": cursor, "line_end": a - 1})
        cursor = max(cursor, b + 1)
    if cursor <= cover_end:
        gaps.append({"line_start": cursor, "line_end": cover_end})
    return gaps


def update_class_target_coverage(
    evidence: dict,
    requirements: dict,
    *,
    tid: str,
    file_path: str,
    line_start: int,
    line_end: int,
    span_limited: bool = False,
) -> dict | None:
    """Update class ledger after a successful read window.

    Source-line coverage and per-method evidence are tracked separately.
    A method is verified only when its full AST range is covered by
    successful reads and an explicit evidence record exists.
    """
    ensure_class_target_metadata(evidence, requirements, tid=tid)
    class_map = evidence.setdefault("class_targets", {})
    meta = class_map.get(tid)
    if not meta:
        return None
    path = str(file_path or "").replace("\\", "/").strip().lower()
    meta_path = str(meta.get("file_path") or "").replace("\\", "/").strip().lower()
    if path and meta_path and path != meta_path:
        return meta
    ranges = list(meta.get("read_ranges") or [])
    ranges.append(
        {
            "line_start": int(line_start),
            "line_end": int(line_end),
        }
    )
    ranges = _merge_line_ranges(ranges)
    meta["read_ranges"] = ranges
    class_start = int(meta["line_start"])
    class_end = int(meta["line_end"])
    # Clip recorded ranges to the class span for gap math.
    clipped = []
    for item in ranges:
        a = max(int(item["line_start"]), class_start)
        b = min(int(item["line_end"]), class_end)
        if b >= a:
            clipped.append({"line_start": a, "line_end": b})
    meta["missing_ranges"] = _missing_ranges_within(
        cover_start=class_start,
        cover_end=class_end,
        read_ranges=clipped,
    )
    meta["source_coverage_complete"] = not bool(meta["missing_ranges"])

    # Ensure method_evidence mirrors AST enumeration (never invent counts).
    methods = list(meta.get("methods") or [])
    evidence_by_name = {
        str(e.get("name")): e
        for e in (meta.get("method_evidence") or [])
        if e.get("name")
    }
    method_evidence: list[dict] = []
    verified_methods: list[str] = []
    for method in methods:
        name = str(method.get("name") or "")
        m_start = int(method.get("line_start") or 0)
        m_end = int(method.get("line_end") or 0)
        if not name or m_start < 1 or m_end < m_start:
            continue
        entry = evidence_by_name.get(name) or {
            "name": name,
            "line_start": m_start,
            "line_end": m_end,
            "purpose": str(method.get("purpose") or f"def {name}(...)"),
            "verified": False,
            "evidence_ranges": [],
        }
        entry["line_start"] = m_start
        entry["line_end"] = m_end
        if method.get("purpose") and not entry.get("purpose"):
            entry["purpose"] = str(method.get("purpose"))
        method_gaps = _missing_ranges_within(
            cover_start=m_start,
            cover_end=m_end,
            read_ranges=clipped,
        )
        if not method_gaps:
            # Record the covering read window(s) as explicit evidence.
            covering = [
                r
                for r in clipped
                if int(r["line_start"]) <= m_end
                and int(r["line_end"]) >= m_start
            ]
            entry["evidence_ranges"] = covering
            entry["verified"] = bool(covering) and bool(entry.get("purpose"))
            if entry["verified"]:
                verified_methods.append(name)
        else:
            entry["verified"] = False
            entry["evidence_ranges"] = [
                r
                for r in clipped
                if int(r["line_start"]) <= m_end
                and int(r["line_end"]) >= m_start
            ]
        method_evidence.append(entry)

    meta["method_evidence"] = method_evidence
    meta["verified_methods"] = verified_methods
    meta["discovered_methods"] = [str(m.get("name")) for m in methods if m.get("name")]
    meta["method_enumeration_complete"] = (
        bool(methods)
        and len(method_evidence) == len(meta["discovered_methods"])
        and set(meta["discovered_methods"]) == {
            str(e.get("name")) for e in method_evidence
        }
    )
    meta["method_evidence_complete"] = (
        meta["method_enumeration_complete"]
        and bool(meta["discovered_methods"])
        and set(meta["discovered_methods"]).issubset(set(verified_methods))
        and all(
            e.get("verified") and e.get("purpose") and e.get("evidence_ranges")
            for e in method_evidence
        )
    )
    if span_limited and meta["missing_ranges"]:
        meta["span_limited_unresolved"] = True
    if not meta["missing_ranges"]:
        meta["span_limited_unresolved"] = False
    # PASS-eligible only when source coverage AND method evidence are complete.
    meta["coverage_complete"] = (
        bool(meta["source_coverage_complete"])
        and bool(meta["method_enumeration_complete"])
        and bool(meta["method_evidence_complete"])
        and not meta.get("span_limited_unresolved")
    )
    class_map[tid] = meta
    return meta


def class_target_is_complete(evidence: dict, tid: str) -> bool:
    meta = (evidence.get("class_targets") or {}).get(tid) or {}
    return bool(meta.get("coverage_complete"))


def next_class_continuation(
    evidence: dict,
    requirements: dict,
    *,
    tid: str | None = None,
) -> dict | None:
    """Build a continuation window for the next uncovered class range."""
    ensure_class_target_metadata(evidence, requirements, tid=tid)
    class_map = evidence.get("class_targets") or {}
    ordered: list[str] = []
    if tid:
        ordered = [tid]
    else:
        for item in requirements.get("required_targets") or []:
            cid = str(item.get("id") or "")
            if cid and (
                str(item.get("kind") or "") == "class"
                or _is_pascal_case_name(cid)
            ):
                ordered.append(cid)
        for cid in class_map:
            if cid not in ordered:
                ordered.append(cid)
    for cid in ordered:
        meta = class_map.get(cid) or {}
        if meta.get("coverage_complete"):
            continue
        gaps = list(meta.get("missing_ranges") or [])
        if not gaps:
            # Methods incomplete even if line union looks full.
            if not class_target_is_complete(evidence, cid):
                # Recompute gaps from class span.
                gaps = _missing_ranges_within(
                    cover_start=int(meta.get("line_start") or 1),
                    cover_end=int(meta.get("line_end") or 1),
                    read_ranges=list(meta.get("read_ranges") or []),
                )
        if not gaps:
            continue
        gap = gaps[0]
        next_line = int(gap["line_start"])
        class_end = int(meta.get("line_end") or next_line)
        suggested = min(next_line + 69, class_end)
        path = str(meta.get("file_path") or "").replace("\\", "/")
        if not path or next_line < 1:
            continue
        return {
            "file_path": path,
            "next_line": next_line,
            "suggested_end_line": max(suggested, next_line),
            "target": cid,
            "class_continuation": True,
            "line_end_seen": next_line - 1,
        }
    return None


def synthesize_class_continuation_from_read(
    evidence: dict,
    requirements: dict,
    tool_result: dict | None,
) -> dict | None:
    """If a nested method reported complete mid-class, continue the class.

    Only continues class targets that intersect the current read window.
    Never steals the continuation slot for an unrelated function read.
    """
    if not isinstance(tool_result, dict):
        return None
    raw = tool_result.get("result")
    payload = raw if isinstance(raw, dict) else None
    if isinstance(raw, str):
        try:
            payload = json.loads(raw.split("\ncontent:\n", 1)[0])
        except Exception:
            payload = None
    if not isinstance(payload, dict) or not payload.get("success"):
        return None
    path = str(payload.get("file_path") or "").replace("\\", "/")
    try:
        line_end = int(payload.get("line_end") or 0)
        w_start = int(payload.get("line_start") or 0)
    except (TypeError, ValueError):
        return None
    span_limited = bool(payload.get("span_limited"))
    ensure_class_target_metadata(evidence, requirements)
    touched: list[str] = []
    for cid, meta in list((evidence.get("class_targets") or {}).items()):
        meta_path = str(meta.get("file_path") or "").replace("\\", "/").lower()
        if path.lower() != meta_path:
            continue
        c_start = int(meta.get("line_start") or 0)
        c_end = int(meta.get("line_end") or 0)
        if not (w_start and line_end and c_start and c_end):
            continue
        if line_end < c_start or w_start > c_end:
            continue
        # Require that we have already sighted this class (header/body),
        # so a coincidental overlap with another symbol does not hijack.
        prog = (evidence.get("target_progress") or {}).get(cid) or {}
        if not (
            prog.get("saw_def")
            or prog.get("body")
            or cid in (evidence.get("verified_targets") or set())
        ):
            continue
        update_class_target_coverage(
            evidence,
            requirements,
            tid=cid,
            file_path=path,
            line_start=w_start,
            line_end=line_end,
            span_limited=span_limited,
        )
        touched.append(cid)
    for cid in touched:
        cont = next_class_continuation(evidence, requirements, tid=cid)
        if cont:
            return cont
    return None


def register_read_verification(
    evidence: dict,
    requirements: dict,
    *,
    file_path: str,
    line_start: int,
    line_end: int,
    content: str,
    function_complete: bool | None = None,
    span_limited: bool = False,
) -> list[str]:
    """Mark required targets verified only from read_file source evidence.

    Never verifies from search hits alone. Continuations that omit the
    ``def`` line still verify when:
    - an earlier window already saw the definition (accumulated body), or
    - the known/discovered def line is covered by read ranges and this
      window finishes the function (``function_complete``).
    """
    from app.services.investigation_state import InvestigationChecklist

    checklist = InvestigationChecklist(requirements)
    checklist.sync_from_evidence(evidence)
    path = (file_path or "").replace("\\", "/")
    evidence["read_ranges"].append(
        {
            "file_path": path,
            "line_start": int(line_start),
            "line_end": int(line_end),
        }
    )
    body = content or ""
    evidence["read_snippets"].append(
        f"read_file {path}:{line_start}-{line_end}: "
        f"{truncate_text(body, 360)}"
    )
    progress = evidence.setdefault("target_progress", {})
    newly: list[str] = []
    ranges = list(evidence.get("read_ranges") or [])

    kind_by_id = {
        str(t.get("id")): str(t.get("kind") or "symbol")
        for t in (requirements.get("required_targets") or [])
        if t.get("id")
    }
    ensure_class_target_metadata(evidence, requirements)
    path_key = path.replace("\\", "/").strip().lower()
    for tid, aliases in _target_aliases(requirements):
        kind = kind_by_id.get(tid) or _target_kind(tid, requirements)
        if kind != "class" and _is_pascal_case_name(tid):
            kind = "class"
        prog = progress.setdefault(
            tid,
            {
                "saw_def": False,
                "body": "",
                "file_path": None,
                "line_start": None,
                "line_end": None,
                # None = unknown; False = still open; True = boundary reached.
                "function_complete": None,
            },
        )
        # Test modules: any successful read of the test file is verification.
        if kind == "test_module":
            known = KNOWN_SYMBOL_LOCATIONS.get(tid)
            alias_paths = {
                str(a).replace("\\", "/").strip().lower()
                for a in aliases
                if "/" in str(a) or str(a).endswith(".py")
            }
            if known:
                alias_paths.add(
                    str(known[0]).replace("\\", "/").strip().lower()
                )
            if path_key in alias_paths or any(
                path_key.endswith(p.split("/")[-1]) for p in alias_paths if p
            ):
                prog["saw_def"] = True
                prog["body"] = (prog.get("body") or "") + "\n" + body
                prog["file_path"] = path
                prog["line_start"] = int(line_start)
                prog["line_end"] = int(line_end)
                prog["function_complete"] = True
                if tid not in evidence["verified_targets"]:
                    evidence["verified_targets"].add(tid)
                    newly.append(tid)
                checklist.mark_verified(
                    tid,
                    file_path=path,
                    line_start=int(line_start),
                    line_end=int(line_end),
                )
            continue
        def_path, def_line = _target_def_line(tid, evidence)
        class_meta = (evidence.get("class_targets") or {}).get(tid)
        if kind == "class" and class_meta and not def_line:
            def_path = str(class_meta.get("file_path") or path)
            def_line = int(class_meta.get("line_start") or 0) or None
        on_symbol_file = bool(
            def_path
            and def_path.replace("\\", "/").strip().lower()
            == path.replace("\\", "/").strip().lower()
        )
        # Clip to this symbol's AST span so neighbors never merge ranges.
        clipped = None
        synthetic_def_window = False
        if kind != "class":
            clipped = clip_range_to_symbol_span(
                tid,
                file_path=path,
                line_start=int(line_start),
                line_end=int(line_end),
            )
            if symbol_ast_span(tid) is not None and clipped is None:
                # Allow mock/synthetic windows that clearly contain this def
                # (line numbers may not match production AST in unit tests).
                if re.search(
                    rf"(?im)^\s*(?:async\s+)?def\s+{re.escape(tid)}\b",
                    body or "",
                ):
                    synthetic_def_window = True
                else:
                    # Window is entirely outside this symbol — skip it.
                    continue
        supports_now = _body_supports_symbol(
            tid, aliases, body, kind=kind
        )
        # When the window was clipped, only credit the overlapping lines.
        if clipped:
            eff_start = int(clipped[0])
            eff_end = int(clipped[1])
        else:
            eff_start = int(line_start)
            eff_end = int(line_end)
        covers_def = bool(
            on_symbol_file
            and def_line
            and eff_start <= int(def_line) <= eff_end
        )
        inside_class_span = False
        if kind == "class" and class_meta and on_symbol_file:
            c_start = int(class_meta.get("line_start") or 0)
            c_end = int(class_meta.get("line_end") or 0)
            inside_class_span = bool(
                c_start
                and c_end
                and int(line_end) >= c_start
                and int(line_start) <= c_end
            )
        span = symbol_ast_span(tid)
        span_end = int(span[2]) if span else None
        is_continuation_window = bool(
            on_symbol_file
            and def_line
            and eff_start > int(def_line)
            and (span_end is None or eff_start <= span_end)
            and (
                prog.get("saw_def")
                or prog.get("body")
                or tid in evidence["verified_targets"]
                or inside_class_span
            )
        )
        # Only accumulate for THIS symbol — never attribute a neighbor's
        # window on the same file to an unread target.
        should_accumulate = bool(
            covers_def
            or is_continuation_window
            or (
                supports_now
                and (clipped is not None or symbol_ast_span(tid) is None)
            )
            or (kind == "class" and inside_class_span and (
                prog.get("saw_def") or supports_now or covers_def
            ))
        )
        # Class body windows after the header: accumulate once class seen.
        if (
            kind == "class"
            and inside_class_span
            and (prog.get("saw_def") or supports_now or covers_def)
        ):
            should_accumulate = True

        if should_accumulate:
            prog["body"] = (prog.get("body") or "") + "\n" + body
            prog["file_path"] = path
            prev_start = prog.get("line_start")
            prev_end = prog.get("line_end")
            prog["line_start"] = (
                int(eff_start)
                if prev_start is None
                else min(int(prev_start), int(eff_start))
            )
            prog["line_end"] = (
                int(eff_end)
                if prev_end is None
                else max(int(prev_end), int(eff_end))
            )
            # Never let accumulated end pass the symbol's AST end (real spans).
            if span_end is not None and not synthetic_def_window:
                prog["line_end"] = min(int(prog["line_end"]), span_end)
                if def_line:
                    prog["line_start"] = max(
                        int(prog["line_start"]), int(def_line)
                    )
            if kind == "class":
                # Nested method function_complete must never complete a class.
                update_class_target_coverage(
                    evidence,
                    requirements,
                    tid=tid,
                    file_path=path,
                    line_start=int(line_start),
                    line_end=int(line_end),
                    span_limited=bool(span_limited),
                )
                class_meta = (evidence.get("class_targets") or {}).get(tid)
                if class_meta and class_target_is_complete(evidence, tid):
                    prog["function_complete"] = True
                else:
                    prog["function_complete"] = False
            elif synthetic_def_window:
                # Trust tool boundary flags for synthetic/mock windows.
                if function_complete is True:
                    prog["function_complete"] = True
                elif function_complete is False:
                    prog["function_complete"] = False
            elif span_end is not None and int(prog["line_end"]) >= span_end and (
                prog.get("saw_def") or covers_def
            ):
                # Reached AST end for this symbol — complete regardless of
                # whether a neighbor window also reported complete.
                prog["function_complete"] = True
            elif function_complete is True and (
                span_end is None or int(eff_end) >= int(span_end)
            ):
                prog["function_complete"] = True
            elif (
                function_complete is False
                and prog.get("function_complete") is not True
            ):
                # Explicit open window (needs_continuation) — do not PASS yet.
                prog["function_complete"] = False
            elif (
                span_end is not None
                and int(prog.get("line_end") or 0) < span_end
                and prog.get("function_complete") is not True
            ):
                prog["function_complete"] = False

        if tid in evidence["verified_targets"]:
            # Refresh checklist span so the ledger shows the symbol range only.
            if should_accumulate:
                report_end = int(prog.get("line_end") or eff_end)
                report_start = int(
                    (class_meta or {}).get("line_start")
                    or prog.get("line_start")
                    or eff_start
                )
                if kind == "class" and class_meta:
                    report_end = int(class_meta.get("line_end") or report_end)
                elif span_end is not None:
                    report_end = min(report_end, span_end)
                    if def_line:
                        report_start = max(report_start, int(def_line))
                checklist.mark_verified(
                    tid,
                    file_path=str(prog.get("file_path") or path),
                    line_start=report_start,
                    line_end=report_end,
                )
            continue

        if tid in checklist.targets and should_accumulate:
            checklist.mark_reading(
                tid,
                file_path=path,
                line_start=int(line_start),
                line_end=int(line_end),
            )

        matched = supports_now
        if matched and re.search(
            rf"(?im)^\s*(def|async def|class)\s+{re.escape(tid)}\b",
            body,
        ):
            prog["saw_def"] = True

        # Class targets: header sighting is not verification.
        if kind == "class":
            if prog.get("saw_def") or supports_now or covers_def:
                prog["saw_def"] = True
                update_class_target_coverage(
                    evidence,
                    requirements,
                    tid=tid,
                    file_path=path,
                    line_start=int(line_start),
                    line_end=int(line_end),
                    span_limited=bool(span_limited),
                )
            matched = class_target_is_complete(evidence, tid)
        else:
            # Continuation / coverage path: def already covered, body accumulated.
            if not matched and on_symbol_file and def_line:
                covered = _line_covered_by_ranges(path, def_line, ranges)
                accumulated = prog.get("body") or ""
                if covered and (
                    prog.get("saw_def")
                    or _body_supports_symbol(
                        tid, aliases, accumulated, kind=kind
                    )
                ):
                    if function_complete or len(accumulated) >= 80:
                        matched = True
                # Current window is clearly past the def (continuation) and
                # earlier coverage included the def line.
                if (
                    not matched
                    and covered
                    and int(line_start) > int(def_line)
                    and (
                        prog.get("saw_def")
                        or _body_supports_symbol(
                            tid, aliases, accumulated, kind=kind
                        )
                    )
                ):
                    matched = True

            # Accumulated multi-window body contains the definition.
            if not matched and _body_supports_symbol(
                tid, aliases, prog.get("body") or "", kind=kind
            ):
                matched = True
                if re.search(
                    rf"(?im)^\s*(def|async def|class)\s+{re.escape(tid)}\b",
                    prog.get("body") or "",
                ):
                    prog["saw_def"] = True

        if matched:
            evidence["verified_targets"].add(tid)
            # Prefer accumulated multi-window span over the last window alone,
            # but never report past this symbol's AST end.
            report_start = int(prog.get("line_start") or eff_start)
            report_end = int(prog.get("line_end") or eff_end)
            if kind == "class" and class_meta:
                report_start = int(
                    class_meta.get("line_start") or report_start
                )
                report_end = int(class_meta.get("line_end") or report_end)
            elif span_end is not None:
                report_end = min(report_end, span_end)
                if def_line:
                    report_start = max(report_start, int(def_line))
            checklist.mark_verified(
                tid,
                file_path=str(prog.get("file_path") or path),
                line_start=int(report_start),
                line_end=int(report_end),
            )
            newly.append(tid)
    checklist.apply_to_evidence(evidence)
    return newly


def reconcile_covered_but_unverified(
    evidence: dict,
    requirements: dict,
) -> list[str]:
    """Verify symbols whose def line was already read but status lagged.

    Handles clustered helpers where a prior window covered ``def_line``
    and overlap avoidance returned no new suggestion window.
    """
    newly: list[str] = []
    progress = evidence.setdefault("target_progress", {})
    ranges = list(evidence.get("read_ranges") or [])
    kind_by_id = {
        str(t.get("id")): str(t.get("kind") or "symbol")
        for t in (requirements.get("required_targets") or [])
        if t.get("id")
    }
    for tid, aliases in _target_aliases(requirements):
        if tid in (evidence.get("verified_targets") or set()):
            continue
        kind = kind_by_id.get(tid, "symbol")
        def_path, def_line = _target_def_line(tid, evidence)
        if not def_path or not def_line:
            continue
        if not _line_covered_by_ranges(def_path, def_line, ranges):
            continue
        prog = progress.get(tid) or {}
        body = prog.get("body") or ""
        if not body:
            # Reconstruct from snippets on that file near the def.
            for snip in evidence.get("read_snippets") or []:
                if def_path.split("/")[-1] in snip or def_path in snip:
                    body += "\n" + snip
        if _body_supports_symbol(tid, aliases, body, kind=kind):
            evidence.setdefault("verified_targets", set()).add(tid)
            newly.append(tid)
    if newly:
        from app.services.investigation_state import InvestigationChecklist

        checklist = InvestigationChecklist(requirements)
        checklist.sync_from_evidence(evidence)
        for tid in newly:
            checklist.mark_verified(tid)
        checklist.apply_to_evidence(evidence)
    return newly


def uncovered_required_targets(requirements: dict, evidence: dict) -> list[str]:
    """Required target ids that still lack complete read_file verification.

    Includes symbols that were discovered/partially read but whose
    ``function_complete`` flag is still false (pending continuation).
    """
    missing = []
    verified = evidence.get("verified_targets") or set()
    progress = evidence.get("target_progress") or {}
    for item in requirements.get("required_targets") or []:
        if not item.get("need_read", True):
            continue
        tid = str(item.get("id") or "")
        if not tid:
            continue
        prog = progress.get(tid) or {}
        if tid not in verified:
            missing.append(tid)
            continue
        # Verified name but open body — still needs continuation reads.
        if (
            str(item.get("kind") or "symbol") != "test_module"
            and prog.get("function_complete") is False
        ):
            missing.append(tid)
            continue
        # Implementation verified but required related tests unread.
        related = [
            str(p).replace("\\", "/").strip().lower()
            for p in (item.get("related_tests") or [])
            if p
        ]
        if related:
            read_paths = {
                str(r.get("file_path") or "").replace("\\", "/").strip().lower()
                for r in (evidence.get("read_ranges") or [])
            }
            tests_ok = any(
                any(
                    rp == tp or rp.endswith("/" + tp.split("/")[-1])
                    for rp in read_paths
                )
                for tp in related
            )
            if not tests_ok:
                # Prefer reading the related test module ids when present.
                for titem in requirements.get("required_targets") or []:
                    if str(titem.get("kind") or "") != "test_module":
                        continue
                    test_id = str(titem.get("id") or "")
                    aliases = [
                        str(a).replace("\\", "/").strip().lower()
                        for a in (titem.get("aliases") or [])
                        if a
                    ]
                    if any(
                        any(
                            a == tp or a.endswith("/" + tp.split("/")[-1])
                            for a in aliases
                        )
                        for tp in related
                    ):
                        if test_id and test_id not in verified and test_id not in missing:
                            missing.append(test_id)
    return missing


def format_missing_evidence_nudge(
    requirements: dict,
    evidence: dict,
) -> str | None:
    """System nudge listing unread targets and suggested line ranges."""
    missing = uncovered_required_targets(requirements, evidence)
    if not missing:
        return None
    suggestions = evidence.get("suggested_reads") or []
    lines = [
        "Investigation incomplete: discovering a symbol via search_code is "
        "not enough. You must read_file the source of each required target "
        "before answering. Do not re-read overlapping ranges.",
        f"Still missing source evidence for: {', '.join(missing)}.",
    ]
    unused = []
    for sug in suggestions:
        if sug.get("target") in missing:
            unused.append(
                f"- {sug['target']}: read_file "
                f"{sug['file_path']} lines {sug['line_start']}-{sug['line_end']}"
            )
    if unused:
        lines.append("Suggested non-overlapping reads from search hits:")
        lines.extend(unused[:6])
    else:
        lines.append(
            "Call search_code for each missing symbol, then read_file using "
            "the exact path and line from the match (about 70 lines max)."
        )
    return "\n".join(lines)


def parse_read_file_continuation(tool_result: dict | None) -> dict | None:
    """Extract mandatory continuation metadata from a read_file tool result."""
    if not isinstance(tool_result, dict):
        return None
    raw = tool_result.get("result")
    payload = None
    content_text = ""
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        meta_text, _, body = raw.partition("\ncontent:\n")
        content_text = body or ""
        try:
            payload = json.loads(meta_text)
        except Exception:
            payload = None
    if not isinstance(payload, dict) or not payload.get("success"):
        return None
    # function_complete wins over span_limited — do not keep reading.
    if payload.get("function_complete") and not payload.get("needs_continuation"):
        return None
    if not payload.get("needs_continuation"):
        cont = payload.get("continuation")
        if not isinstance(cont, dict):
            return None
        # Legacy: only continue when body still open — never for span_limited alone.
        if payload.get("span_limited") and not (
            payload.get("truncated") or _content_still_indented(content_text)
        ):
            return None
        if not (
            payload.get("truncated") or _content_still_indented(content_text)
        ):
            return None
    cont = payload.get("continuation") or {}
    next_line = int(
        cont.get("next_line")
        or payload.get("next_line")
        or ((payload.get("line_end") or 0) + 1)
    )
    # Never continue from next_line-1 / line_end.
    line_end = int(payload.get("line_end") or 0)
    if next_line <= line_end:
        next_line = line_end + 1
    suggested = int(
        cont.get("suggested_end_line")
        or (next_line + 69)
    )
    path = str(
        cont.get("file_path") or payload.get("file_path") or ""
    ).replace("\\", "/")
    if not path or next_line < 1:
        return None
    return {
        "file_path": path,
        "next_line": next_line,
        "suggested_end_line": max(suggested, next_line),
        "line_end_seen": line_end,
    }


def _content_still_indented(content: str) -> bool:
    for raw_line in reversed((content or "").splitlines()):
        if not raw_line.strip():
            continue
        return raw_line.startswith((" ", "\t"))
    return False


def investigation_requirements_met(
    requirements: dict,
    *,
    successful_searches: int,
    successful_reads: int,
    pending_continuation: bool,
    calculator_used: bool,
    function_complete: bool,
    evidence: dict | None = None,
) -> bool:
    if requirements.get("needs_search") and successful_searches < 1:
        return False
    if requirements.get("needs_source_reads") or requirements.get(
        "needs_complete_function"
    ):
        min_reads = int(requirements.get("min_source_reads") or 1)
        if successful_reads < min_reads:
            return False
        if pending_continuation:
            return False
        # Single-function complete-window requests still need boundary done,
        # but multi-target investigations rely on verified_targets instead.
        required = requirements.get("required_targets") or []
        if required and evidence is not None:
            if uncovered_required_targets(requirements, evidence):
                return False
        elif requirements.get("needs_complete_function") and not function_complete:
            if not required:
                return False
    if requirements.get("needs_calculator") and not calculator_used:
        return False
    return True


def is_premature_or_generic_answer(
    answer: str,
    *,
    requirements: dict,
    evidence_snippets: list[str],
) -> bool:
    """True when the model stopped without addressing the investigation."""
    text = (answer or "").strip()
    if not text:
        return True
    lowered = text.lower()
    generic = (
        "just let me know" in lowered
        or "what would you like" in lowered
        or "how can i help" in lowered
        or len(text) < 40
    )
    if not evidence_snippets:
        return generic
    if requirements.get("needs_complete_function") or requirements.get("topics"):
        if generic:
            return True
        # Must mention something from the evidence or required topics.
        evidence_blob = " ".join(evidence_snippets).lower()
        mentions_code = any(
            token in lowered
            for token in (
                "execute_tool",
                "read_file",
                "search_code",
                "validation",
                "exception",
                "success",
                "tool_name",
                "def ",
            )
        )
        if not mentions_code and "def " in evidence_blob:
            return True
    return False


def validate_final_answer_text(answer: str) -> str:
    """Remove an unfinished trailing code fence without clipping prose."""
    text = (answer or "").strip()
    if not text:
        return text
    # Drop an unfinished trailing fenced block (odd number of fences).
    fence_count = text.count("```")
    if fence_count % 2 == 1:
        text = text.rsplit("```", 1)[0].rstrip()
        text += (
            "\n\n(Note: an incomplete code example was omitted so the "
            "answer stays complete.)"
        )
    return text.strip()


def normalize_tool_args(arguments: dict | None) -> str:
    safe = {}
    for key, value in (arguments or {}).items():
        if key in {"user_id", "request_id"}:
            continue
        if isinstance(value, str):
            safe[key] = value.strip().lower()
        else:
            safe[key] = value
    try:
        return json.dumps(safe, sort_keys=True, default=str)
    except Exception:
        return str(sorted(safe.items()))


def tool_fingerprint(name: str, arguments: dict | None) -> str:
    return f"{name}:{normalize_tool_args(arguments)}"


def _read_line_bounds(args: dict | None) -> tuple[str, int, int]:
    """Extract path/start/end. Accepts legacy aliases only for reading bounds."""
    args = args or {}
    path = str(args.get("file_path") or "").replace("\\", "/").strip().lower()
    # Prefer canonical ReadFileInput fields; fall back to legacy aliases.
    start_raw = args.get("line_start")
    if start_raw is None:
        start_raw = args.get("start_line")
    end_raw = args.get("line_end")
    if end_raw is None:
        end_raw = args.get("end_line")
    start = int(start_raw if start_raw is not None else 1)
    end = int(end_raw if end_raw is not None else start + 50)
    if end < start:
        start, end = end, start
    return path, start, end


def canonicalize_read_file_args(arguments: dict | None) -> dict:
    """Return only schema-legal read_file fields for Pydantic validation.

    Maps legacy ``start_line``/``end_line`` aliases onto ``line_start``/
    ``line_end`` and drops every other key. Preserves ``extra=\"forbid\"``.
    """
    args = arguments or {}
    path = str(args.get("file_path") or "").replace("\\", "/").strip()
    _, start, end = _read_line_bounds(args)
    return {
        "file_path": path,
        "line_start": int(start),
        "line_end": int(end),
    }


def is_boundary_only_touch(a: dict | None, b: dict | None) -> bool:
    """True when ranges share only a single boundary line (continuation)."""
    path_a, start_a, end_a = _read_line_bounds(a)
    path_b, start_b, end_b = _read_line_bounds(b)
    if not path_a or path_a != path_b:
        return False
    return end_a == start_b or end_b == start_a


def overlapping_read_range(a: dict | None, b: dict | None) -> bool:
    """True when two inclusive line ranges share interior lines.

    Inclusive ranges [start, end]. Sequential continuation is allowed:
    - 750–819 then 820–890 (next_line after prior end)
    - 750–819 then 819–890 (boundary-only → normalize start to 820)
    Genuine interior overlap (750–819 then 800–860) is blocked.
    """
    path_a, start_a, end_a = _read_line_bounds(a)
    path_b, start_b, end_b = _read_line_bounds(b)
    if not path_a or path_a != path_b:
        return False
    # Disjoint or strictly sequential on an inclusive scale:
    # prior ends at E, next starts at E+1 → end_a < start_b.
    if end_a < start_b or end_b < start_a:
        return False
    # One-line boundary touch is continuation, not a duplicate overlap.
    if is_boundary_only_touch(a, b):
        return False
    return True


def actual_read_bounds_from_result(
    arguments: dict | None,
    inner_payload: dict | None,
) -> dict:
    """Prefer line bounds actually returned by read_file (after span limits).

    Requested 750–830 may only return 750–819; overlap checks must use 819.
    Returns canonical ``file_path`` / ``line_start`` / ``line_end`` only.
    """
    args = dict(arguments or {})
    path = str(
        (inner_payload or {}).get("file_path")
        or args.get("file_path")
        or ""
    ).replace("\\", "/").strip()
    start = (inner_payload or {}).get("line_start")
    end = (inner_payload or {}).get("line_end")
    if not isinstance(start, int) or not isinstance(end, int):
        _, start, end = _read_line_bounds(args)
    return {
        "file_path": path or args.get("file_path"),
        "line_start": int(start),
        "line_end": int(end),
    }


def normalize_sequential_read_args(
    arguments: dict | None,
    prior_reads: list[dict],
) -> dict:
    """Advance start to the next unread line when ranges only touch a boundary.

    Example: prior 740–809 + request 809–860 → 810–860.

    Always returns canonical ReadFileInput fields only
    (``file_path``, ``line_start``, ``line_end``) so Pydantic
    ``extra=\"forbid\"`` validation succeeds.
    """
    path, start, end = _read_line_bounds(arguments)
    if not path:
        return canonicalize_read_file_args(arguments)
    for prior in prior_reads:
        p2, _s2, e2 = _read_line_bounds(prior)
        if p2 != path:
            continue
        # Request starts on the prior end line → continue from next unread.
        if start == e2:
            next_line = e2 + 1
            if next_line <= end:
                start = next_line
        # Request ends on the prior start line (reading backwards into prior).
        elif end == _read_line_bounds(prior)[1] and start < end:
            # Leave as-is; substantial overlap detection handles reverse cases.
            pass
    return {
        "file_path": str((arguments or {}).get("file_path") or "").replace(
            "\\", "/"
        ).strip(),
        "line_start": int(start),
        "line_end": int(end),
    }


def investigation_has_sufficient_evidence(
    requirements: dict | None,
    evidence: dict | None,
) -> bool:
    """True when current evidence is enough to skip further search_code.

    Used with user instructions like \"do not search again\": require at least
    one successful class/symbol source read, or known locations for every
    required target so forced reads can proceed without rediscovery.
    """
    req = requirements or {}
    ev = evidence or {}
    targets = [
        str(t.get("id") or "")
        for t in (req.get("required_targets") or [])
        if t.get("id")
    ]
    if not targets:
        return bool(ev.get("read_ranges") or ev.get("read_snippets"))
    verified = set(ev.get("verified_targets") or set())
    class_map = ev.get("class_targets") or {}
    known_or_progress = 0
    for tid in targets:
        if tid in verified:
            known_or_progress += 1
            continue
        meta = class_map.get(tid) or {}
        if meta.get("read_ranges") or meta.get("source_coverage_complete"):
            known_or_progress += 1
            continue
        prog = (ev.get("target_progress") or {}).get(tid) or {}
        if prog.get("body") or prog.get("saw_def"):
            known_or_progress += 1
            continue
        if tid in KNOWN_SYMBOL_LOCATIONS:
            known_or_progress += 1
            continue
        if (ev.get("discovered") or {}).get(tid):
            known_or_progress += 1
            continue
    return known_or_progress >= len(targets)


def should_skip_duplicate_tool(
    name: str,
    arguments: dict | None,
    *,
    fingerprints: set[str],
    prior_reads: list[dict],
    tool_counts: dict[str, int],
    files_already_read: set[str] | None = None,
    read_counts_by_file: dict[str, int] | None = None,
    requirements: dict | None = None,
    evidence: dict | None = None,
    searched_queries: set[str] | None = None,
    read_budget_total: int | None = None,
    budget_manager: Any | None = None,
    is_continuation: bool = False,
) -> tuple[bool, str | None]:
    """Return (skip, reason_code).

    Callers should pass already-normalized read_file args (boundary
    continuation advanced to the next unread line). Exact duplicates and
    substantial interior overlaps are blocked; sequential windows are not.

    When ``budget_manager`` is provided it is the source of truth for
    per-file / total / continuation / search limits. Legacy kwargs remain
    for unit tests that construct limits inline.
    """
    del files_already_read  # retained for call-site compatibility
    fp = tool_fingerprint(name, arguments)
    if fp in fingerprints:
        return True, "DUPLICATE_TOOL_CALL"

    if name == "read_file":
        path, _, _ = _read_line_bounds(arguments)
        if budget_manager is not None:
            ok, reason = budget_manager.allow_read(
                path, is_continuation=is_continuation
            )
            if not ok:
                return True, reason
        else:
            per_file = int((read_counts_by_file or {}).get(path, 0))
            file_limit = reads_per_file_limit(
                file_path=path,
                requirements=requirements,
                evidence=evidence,
            )
            if per_file >= file_limit:
                return True, "PER_FILE_READ_LIMIT"
            total_reads = int(tool_counts.get("read_file", 0))
            budget = int(
                read_budget_total
                if read_budget_total is not None
                else investigation_read_budget(requirements)
            )
            if total_reads >= budget:
                return True, "PER_TOOL_EXECUTION_LIMIT"
        for prior in prior_reads:
            if overlapping_read_range(arguments, prior):
                return True, "OVERLAPPING_FILE_READ"
        return False, None

    if name == "search_code":
        req = requirements or {}
        ev = evidence or {}
        # Honor explicit user instruction when enough evidence already exists.
        if req.get("forbid_search") and investigation_has_sufficient_evidence(
            req, ev
        ):
            return True, "SEARCH_SKIPPED_USER_FORBID"
        if budget_manager is not None:
            ok, reason = budget_manager.allow_search()
            if not ok:
                return True, reason
        elif tool_counts.get(name, 0) >= MAX_SEARCH_CODE_EXECUTIONS:
            return True, "PER_TOOL_EXECUTION_LIMIT"
        query = str((arguments or {}).get("search_text") or "")
        qnorm = normalize_search_query(query)
        if qnorm and searched_queries is not None and qnorm in searched_queries:
            return True, "SEARCH_SKIPPED_KNOWN_SYMBOL"
        # Already discovered / verified required symbols — do not re-search.
        known_ids = set((ev.get("discovered") or {}).keys()) | set(
            ev.get("verified_targets") or set()
        )
        # Also honor explicit checklist statuses when present.
        for tid, meta in (ev.get("target_statuses") or {}).items():
            st = str((meta or {}).get("status") or "")
            if st in {"discovered", "reading", "verified", "failed"}:
                known_ids.add(tid)
        for tid in known_ids:
            tnorm = normalize_search_query(tid)
            if tnorm and (tnorm == qnorm or tnorm in qnorm or qnorm in tnorm):
                return True, "SEARCH_SKIPPED_KNOWN_SYMBOL"
        # After discovery/seeds exist, prioritize read_file over more search.
        uncovered = uncovered_required_targets(req, ev)
        if uncovered and (
            ev.get("suggested_reads")
            or any(tid in KNOWN_SYMBOL_LOCATIONS for tid in uncovered)
        ):
            searches_done = int(tool_counts.get(name, 0))
            if searches_done >= 1 or all(
                tid in KNOWN_SYMBOL_LOCATIONS for tid in uncovered
            ):
                # Do not chase a missing class when module-level defs are known.
                if re.search(r"(?i)^\s*class\s+\w+", query or ""):
                    return True, "SEARCH_SKIPPED_READ_PRIORITY"
                if searches_done >= 1:
                    return True, "SEARCH_SKIPPED_READ_PRIORITY"
        return False, None

    if tool_counts.get(name, 0) >= MAX_SAME_TOOL_EXECUTIONS:
        return True, "PER_TOOL_EXECUTION_LIMIT"
    return False, None


def next_unread_suggested_read(
    requirements: dict,
    evidence: dict,
    *,
    prefer_targets: list[str] | None = None,
    exclude_targets: set[str] | None = None,
) -> dict | None:
    """Pick the next non-overlapping suggested read for an uncovered target.

    Prefers ``prefer_targets`` order (e.g. rate-limit helpers) then the
    remaining checklist order. ``exclude_targets`` skips symbols whose
    forced read was already attempted/blocked this turn.
    """
    missing = set(uncovered_required_targets(requirements, evidence))
    if exclude_targets:
        missing -= set(exclude_targets)
    if not missing:
        return None
    prior = list(evidence.get("read_ranges") or [])
    ordered: list[str] = []
    for tid in prefer_targets or []:
        if tid in missing and tid not in ordered:
            ordered.append(tid)
    for tid in uncovered_required_targets(requirements, evidence):
        if tid in missing and tid not in ordered:
            ordered.append(tid)

    suggestions = list(evidence.get("suggested_reads") or [])
    for tid in ordered:
        for sug in suggestions:
            if str(sug.get("target") or "") != tid:
                continue
            candidate = {
                "file_path": sug.get("file_path"),
                "line_start": int(sug.get("line_start") or 1),
                "line_end": int(sug.get("line_end") or 70),
                "target": tid,
            }
            if any(overlapping_read_range(candidate, p) for p in prior):
                # Rebuild a shifted window around the known/suggested line.
                anchor = int(sug.get("line_start") or 1) + 2
                loc = KNOWN_SYMBOL_LOCATIONS.get(tid)
                if loc:
                    anchor = int(loc[1])
                rebuilt = _suggestion_window_for_symbol(
                    str(sug.get("file_path") or ""),
                    anchor,
                    prior=prior,
                )
                if not rebuilt:
                    continue
                candidate = {**rebuilt, "target": tid}
                if any(overlapping_read_range(candidate, p) for p in prior):
                    continue
            return candidate
    return None


DIRECT_ANSWER_SYSTEM_PROMPT = """
You are an AI Software Engineer assistant.

Answer technical and conversational questions directly in clear natural
language. You do NOT have tools for this turn.

Hard rules:
- Keep beginner explanations clear and concise (usually under ~400 words).
- Prefer short paragraphs and a simple real-world analogy when teaching.
- Do not generate pseudocode or long code samples unless the user asked
  for code, or a tiny snippet is truly necessary.
- Never leave an unfinished code fence or truncated example.
- Do not emit tool calls, function calls, tool JSON, or tool_use markup.
- Do not invent personal facts about the user.
- Do not claim you searched memory or the codebase unless evidence is provided.
- Do not invent product features or claim this repository implements a
  behavior unless the user asked you to inspect source and evidence is
  present.

AI agent memory topics (tool-free conceptual answers):
When the user asks about memory scoping, conflict resolution, freshness,
reliability, summarization, reinforcement, source traceability, or deletion
in an AI/agent context, explain them as **agent long-term memory** concepts
— never as programming-language variable scope (global/local) or generic
OS/process memory.

Cover, in plain language when asked:
- Scoping: memories belong to a user/tenant/project (and often a fact key),
  not a shared global bag of everyone’s facts. Missing ownership or tenant
  never makes a memory public — access is denied until scope is explicit.
- Conflict resolution: contradictory facts for the same key are detected,
  held for review, and updated only with confirmation — not silently overwritten.
- Freshness: newer or reinforced facts outrank stale ones; outdated values
  should age out or be superseded.
- Reliability: confidence/importance and corroboration affect whether a fact
  is trusted in answers.
- Summarization: long histories may be compacted into summaries while keeping
  a path back to details.
- Reinforcement: repeated consistent statements strengthen a memory’s weight.
- Source traceability: remember where a fact came from (user utterance / tool)
  so answers can be audited.
- Deletion: forget/remove flows deactivate or erase a fact deliberately and
  stop retrieving it afterward.
""".strip()


FINALIZATION_SYSTEM_PROMPT = """
You are an AI Software Engineer assistant writing a final answer from evidence.

CRITICAL: You have NO tools. You cannot call functions. You cannot search or
read files. Respond with natural language prose only.

Ground every claim:
1. VERIFIED — only facts present in the SOURCE_EXCERPTS below.
2. INFERRED — reasonable conclusions from those excerpts; label as inferred.
3. MISSING — symbols listed under MISSING_SOURCE_EVIDENCE; say they were not
   inspected. Do not invent their bodies.

Hard evidence rules:
- Imports, function names, and search hits are discovery only — never PASS.
- PASS requires complete implementation read, related test inspection, AND
  successful test execution (TEST_EXECUTION_PASSED) when tests exist.
- TEST_SOURCE_INSPECTED means the test file was read — never that tests passed.
- Complete source alone or inspected-but-unexecuted tests → PARTIAL.
- Pending continuations / unread ranges are PARTIAL — never PASS.
- Each File/lines range must belong to that named symbol only.
- Copy Evidence Ledger statuses exactly; do not upgrade MISSING/PARTIAL to PASS.

Never emit tool calls, function-call JSON, tool_use markup, or requests to
run search_code / read_file / calculator.
""".strip()


def _next_known_symbol_line(path: str, after_line: int) -> int | None:
    """Return the nearest later known boundary on the same path, if any.

    Prefers the next symbol's def line, and also respects known AST end
    lines so a suggestion window never spills into a neighbor function.
    """
    path_key = str(path or "").replace("\\", "/").strip().lower()
    later = [
        int(ln)
        for p, ln in KNOWN_SYMBOL_LOCATIONS.values()
        if str(p).replace("\\", "/").strip().lower() == path_key
        and int(ln) > int(after_line)
    ]
    # Also stop just after any known symbol end that starts at/before after_line.
    for tid, (p, start) in KNOWN_SYMBOL_LOCATIONS.items():
        if str(p).replace("\\", "/").strip().lower() != path_key:
            continue
        end = KNOWN_SYMBOL_END_LINES.get(tid)
        if end and int(start) <= int(after_line) <= int(end):
            # Current symbol end + 1 is the first line we must not include.
            later.append(int(end) + 1)
        elif end and int(end) > int(after_line) and int(start) > int(after_line):
            later.append(int(start))
    return min(later) if later else None


def _suggestion_window_for_symbol(
    path: str,
    line: int,
    *,
    prior: list[dict] | None = None,
    span: int = 24,
) -> dict | None:
    """Build a compact canonical window that still covers ``line``.

    Ends before the next known symbol on the same file when possible, and
    advances ``line_start`` past prior overlapping reads so clustered helpers
    remain individually readable.
    """
    path_n = str(path or "").replace("\\", "/").strip()
    path_key = path_n.lower()
    target_line = max(1, int(line))
    start = max(1, target_line - 1)
    end = start + max(8, int(span)) - 1
    nxt = _next_known_symbol_line(path_n, target_line)
    if nxt is not None:
        end = min(end, nxt - 1)
    # Always include at least the def line (+ a few body lines).
    end = max(end, target_line + 3)
    if nxt is not None:
        end = min(end, nxt - 1)
    for p in prior or []:
        p_path, p_start, p_end = _read_line_bounds(p)
        if not p_path or p_path != path_key:
            continue
        # Prior already covers this definition — nothing new to read.
        if p_start <= target_line <= p_end:
            return None
        if overlapping_read_range(
            {"file_path": path_n, "line_start": start, "line_end": end},
            p,
        ):
            start = max(start, p_end + 1)
            end = max(end, start + 3)
            if nxt is not None:
                end = min(end, nxt - 1)
    if start > target_line or end < target_line or end < start:
        return None
    return {
        "file_path": path_n,
        "line_start": int(start),
        "line_end": int(end),
    }


def seed_known_symbol_suggestions(
    requirements: dict,
    evidence: dict,
) -> list[dict]:
    """Add non-overlapping read suggestions for required known symbols."""
    ensure_class_target_metadata(evidence, requirements)
    missing = set(uncovered_required_targets(requirements, evidence))
    prior = list(evidence.get("read_ranges") or [])
    added: list[dict] = []
    kind_by_id = {
        str(t.get("id")): str(t.get("kind") or "symbol")
        for t in (requirements.get("required_targets") or [])
        if t.get("id")
    }
    alias_by_id = {
        str(t.get("id")): [str(a) for a in (t.get("aliases") or []) if a]
        for t in (requirements.get("required_targets") or [])
        if t.get("id")
    }
    progress = evidence.get("target_progress") or {}
    for tid in list(missing):
        loc = KNOWN_SYMBOL_LOCATIONS.get(tid)
        class_meta = (evidence.get("class_targets") or {}).get(tid)
        prog = progress.get(tid) or {}
        # Incomplete symbol: continue from the next unread line inside AST span.
        if (
            kind_by_id.get(tid) != "test_module"
            and prog.get("function_complete") is False
            and prog.get("file_path")
            and prog.get("line_end")
        ):
            span = symbol_ast_span(tid)
            next_line = int(prog["line_end"]) + 1
            path = str(prog["file_path"]).replace("\\", "/")
            if span and next_line <= int(span[2]):
                end = min(next_line + 69, int(span[2]))
                candidate = {
                    "file_path": path,
                    "line_start": next_line,
                    "line_end": end,
                    "target": tid,
                }
                if not any(
                    overlapping_read_range(candidate, p) for p in prior
                ):
                    already = any(
                        s.get("target") == tid
                        and int(s.get("line_start") or 0) == next_line
                        for s in (evidence.get("suggested_reads") or [])
                    )
                    if not already:
                        evidence.setdefault("suggested_reads", []).append(
                            candidate
                        )
                        prior.append(candidate)
                        added.append(candidate)
                continue
        # Test modules: open the test file from path aliases.
        if kind_by_id.get(tid) == "test_module" and not loc:
            for alias in alias_by_id.get(tid) or []:
                if "/" in alias or alias.endswith(".py"):
                    loc = (alias.replace("\\", "/"), 1)
                    break
        if class_meta and class_meta.get("file_path") and class_meta.get("line_start"):
            path = str(class_meta["file_path"]).replace("\\", "/")
            line = int(class_meta["line_start"])
        elif loc:
            path, line = loc
        else:
            # Flexible discovery — do not assume a hardcoded module path.
            resolved = resolve_symbol_location(tid)
            if not resolved:
                continue
            path, line = resolved
        # Class targets: first window toward class end; do not stop at a
        # neighboring known symbol that sits inside the class body.
        if class_meta and class_meta.get("line_end"):
            start = max(1, int(line) - 1)
            end = min(int(class_meta["line_end"]), start + 69)
            end = max(end, int(line) + 3)
            for p in prior:
                p_path, p_start, p_end = _read_line_bounds(p)
                if not p_path or p_path != path.replace("\\", "/").lower():
                    continue
                if p_start <= int(line) <= p_end:
                    # Header already covered — continuation handles the rest.
                    window = None
                    break
                if overlapping_read_range(
                    {"file_path": path, "line_start": start, "line_end": end},
                    p,
                ):
                    start = max(start, p_end + 1)
                    end = max(end, start + 3)
                    end = min(end, int(class_meta["line_end"]))
            else:
                if start <= int(line) <= end:
                    window = {
                        "file_path": path,
                        "line_start": int(start),
                        "line_end": int(end),
                    }
                else:
                    window = None
            if window is None:
                cont = next_class_continuation(
                    evidence, requirements, tid=tid
                )
                if cont:
                    window = {
                        "file_path": cont["file_path"],
                        "line_start": cont["next_line"],
                        "line_end": cont["suggested_end_line"],
                    }
                else:
                    reconcile_covered_but_unverified(evidence, requirements)
                    continue
        else:
            window = _suggestion_window_for_symbol(path, line, prior=prior)
            if not window:
                # Def already covered by a prior range — try to close the status.
                reconcile_covered_but_unverified(evidence, requirements)
                continue
        candidate = {**window, "target": tid}
        if any(overlapping_read_range(candidate, p) for p in prior):
            reconcile_covered_but_unverified(evidence, requirements)
            continue
        # Avoid duplicate suggestions for the same target+range.
        already = any(
            s.get("target") == tid
            and str(s.get("file_path") or "").replace("\\", "/") == path
            for s in (evidence.get("suggested_reads") or [])
        )
        if already:
            continue
        evidence.setdefault("suggested_reads", []).append(candidate)
        # Do not mark seeded locations as discovered — that would block the
        # first search_code for the symbol. Discovery comes from search hits
        # or verified reads; seeds only provide forced-read targets.
        # Treat freshly seeded windows as reserved so later seeds shift.
        prior.append({
            "file_path": candidate["file_path"],
            "line_start": candidate["line_start"],
            "line_end": candidate["line_end"],
        })
        added.append(candidate)
    return added


def build_chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    tools: list | None = None,
    tool_choice: str | dict | None = None,
    allow_tools: bool = False,
) -> dict:
    """Build exact Groq chat.completions.create kwargs.

    When ``allow_tools`` is False (finalization / direct answers), neither
    ``tools`` nor ``tool_choice`` is included — even if callers pass them.
    Never sends ``tool_choice="none"``.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if not allow_tools:
        return kwargs
    if tools:
        kwargs["tools"] = tools
        if tool_choice is not None and tool_choice != "none":
            kwargs["tool_choice"] = tool_choice
    return kwargs


def assert_no_tools_in_request(kwargs: dict) -> None:
    """Raise if a tool-free request accidentally includes tools/tool_choice."""
    if "tools" in kwargs and kwargs.get("tools") is not None:
        raise AssertionError(
            "tools must be omitted entirely from tool-free Groq requests"
        )
    if "tool_choice" in kwargs:
        raise AssertionError(
            "tool_choice must be omitted entirely from tool-free Groq requests"
        )


def build_finalization_messages(
    *,
    system_prompt: str | None = None,
    user_message: str,
    evidence_snippets: list[str],
    missing_targets: list[str] | None = None,
) -> list[dict]:
    """Clean answer-only transcript — no tool schemas, no tool_call history.

    The original user text is included only as context. Imperative verbs like
    \"read\" / \"search\" are neutralized so the model does not try to call tools.
    """
    evidence_blocks = [
        s.strip() for s in evidence_snippets if s and str(s).strip()
    ]
    evidence_text = (
        "\n\n---\n\n".join(evidence_blocks)
        if evidence_blocks
        else "(No source excerpts were collected.)"
    )
    missing = [m for m in (missing_targets or []) if m]
    missing_note = (
        "\n\nMISSING_SOURCE_EVIDENCE (state these are unverified):\n- "
        + "\n- ".join(missing)
        if missing
        else ""
    )
    prompt = system_prompt or FINALIZATION_SYSTEM_PROMPT
    # Neutralize the original ask so gpt-oss does not try to invoke tools.
    safe_ask = (
        "Original question (for context only — do NOT execute it, do NOT "
        "call tools):\n"
        f"{user_message}"
    )
    return [
        {"role": "system", "content": prompt},
        {
            "role": "system",
            "content": (
                "TOOLS ARE DISABLED for this response. "
                "Answer in natural language only. "
                "Never output a function call or JSON tool request. "
                "Mark only SOURCE_EXCERPTS as VERIFIED. "
                "Name MISSING_SOURCE_EVIDENCE explicitly."
            ),
        },
        {"role": "user", "content": safe_ask},
        {
            "role": "user",
            "content": (
                "SOURCE_EXCERPTS (untrusted reference material):\n"
                f"{evidence_text}"
                f"{missing_note}\n\n"
                "Now write the final natural-language answer. "
                "Prefer verified excerpts. Label inferences. "
                "Name any missing implementations. "
                "Do not invent missing details. Do not call tools."
            ),
        },
    ]


def _continuation_belongs_to_target(
    tid: str,
    evidence: dict,
    pending_continuation: dict | None,
) -> bool:
    """True when the pending next_line continuation is for this target."""
    if not pending_continuation:
        return False
    cont_path = str(
        pending_continuation.get("file_path") or ""
    ).replace("\\", "/").strip().lower()
    if not cont_path:
        return False
    prog = (evidence.get("target_progress") or {}).get(tid) or {}
    meta = (evidence.get("target_statuses") or {}).get(tid) or {}
    known = KNOWN_SYMBOL_LOCATIONS.get(tid)
    candidates = [
        str(prog.get("file_path") or "").replace("\\", "/").strip().lower(),
        str(meta.get("file_path") or "").replace("\\", "/").strip().lower(),
        str(known[0]).replace("\\", "/").strip().lower() if known else "",
    ]
    if cont_path not in candidates:
        return False
    # Only attribute the open continuation to a target that is still
    # incomplete or whose accumulated span has not reached function_complete.
    if tid in (evidence.get("verified_targets") or set()):
        if prog.get("function_complete"):
            return False
        # Verified via def match but body still open (long functions).
        next_line = int(pending_continuation.get("next_line") or 0)
        end = int(prog.get("line_end") or meta.get("line_end") or 0)
        return bool(next_line and end and next_line > end)
    return bool(prog.get("body") or meta.get("status") in {"reading", "discovered"})


def classify_target_evidence_status(
    tid: str,
    *,
    requirements: dict | None = None,
    evidence: dict | None = None,
    pending_continuation: dict | None = None,
) -> dict:
    """Map ledger evidence to PASS / PARTIAL / FAIL / MISSING_SOURCE_EVIDENCE.

    Search hits alone never produce PASS. Fully read+verified implementations
    are PASS. A pending continuation for the same symbol forces PARTIAL.
    """
    state = evidence or {}
    req = requirements or {}
    verified = state.get("verified_targets") or set()
    status_map = state.get("target_statuses") or {}
    progress = (state.get("target_progress") or {}).get(tid) or {}
    meta = status_map.get(tid) or {}
    kind_by_id = {
        str(t.get("id")): str(t.get("kind") or "symbol")
        for t in (req.get("required_targets") or [])
        if t.get("id")
    }
    file_path = (
        progress.get("file_path")
        or meta.get("file_path")
        or (KNOWN_SYMBOL_LOCATIONS.get(tid) or (None, None))[0]
    )
    line_start = progress.get("line_start") or meta.get("line_start")
    line_end = progress.get("line_end") or meta.get("line_end")
    # Ledger ranges must belong to this symbol only (never neighbors).
    if file_path and line_start and line_end:
        clipped = clip_range_to_symbol_span(
            tid,
            file_path=str(file_path),
            line_start=int(line_start),
            line_end=int(line_end),
        )
        if clipped:
            line_start, line_end = clipped
        elif symbol_ast_span(tid) is not None:
            # Stale merged range that no longer overlaps — fall back to AST.
            span = symbol_ast_span(tid)
            if span:
                file_path, line_start, line_end = span[0], span[1], span[2]
    cont_for_target = _continuation_belongs_to_target(
        tid, state, pending_continuation
    )
    row = {
        "id": tid,
        "kind": kind_by_id.get(tid, "symbol"),
        "file_path": str(file_path).replace("\\", "/") if file_path else None,
        "line_start": int(line_start) if line_start else None,
        "line_end": int(line_end) if line_end else None,
        "function_complete": bool(progress.get("function_complete")),
        "implementation_complete": bool(
            progress.get("function_complete") and tid in verified
        ),
        "continuation": None,
        "gaps": [],
        "related_tests": [],
        "test_execution_status": None,
        "status": "MISSING_SOURCE_EVIDENCE",
        "reason": "no successful source read",
    }
    if progress.get("unread_from"):
        row["gaps"].append(
            f"unread {progress.get('unread_from')}-{progress.get('unread_to')}"
        )
        row["unread_from"] = progress.get("unread_from")
        row["unread_to"] = progress.get("unread_to")
    # Class-level ledger overrides function_complete from nested methods.
    class_meta = (state.get("class_targets") or {}).get(tid)
    if row["kind"] == "class" or class_meta:
        ensure_class_target_metadata(state, req, tid=tid)
        class_meta = (state.get("class_targets") or {}).get(tid) or class_meta
    if class_meta:
        row["file_path"] = class_meta.get("file_path") or row["file_path"]
        row["line_start"] = class_meta.get("line_start") or row["line_start"]
        row["line_end"] = class_meta.get("line_end") or row["line_end"]
        method_evidence = list(class_meta.get("method_evidence") or [])
        missing_methods = [
            str(e.get("name"))
            for e in method_evidence
            if not e.get("verified")
        ] or [
            m
            for m in (class_meta.get("discovered_methods") or [])
            if m not in set(class_meta.get("verified_methods") or [])
        ]
        row["class_ledger"] = {
            "expected_range": (
                f"{class_meta.get('line_start')}-{class_meta.get('line_end')}"
            ),
            "read_ranges": list(class_meta.get("read_ranges") or []),
            "discovered_methods": list(
                class_meta.get("discovered_methods") or []
            ),
            "verified_methods": list(class_meta.get("verified_methods") or []),
            "method_evidence": method_evidence,
            "missing_ranges": list(class_meta.get("missing_ranges") or []),
            "source_coverage_complete": bool(
                class_meta.get("source_coverage_complete")
            ),
            "method_enumeration_complete": bool(
                class_meta.get("method_enumeration_complete")
            ),
            "method_evidence_complete": bool(
                class_meta.get("method_evidence_complete")
            ),
            "span_limited_unresolved": bool(
                class_meta.get("span_limited_unresolved")
            ),
        }
        # PASS only with complete source coverage AND complete method evidence.
        if (
            class_meta.get("coverage_complete")
            and class_meta.get("source_coverage_complete")
            and class_meta.get("method_enumeration_complete")
            and class_meta.get("method_evidence_complete")
            and tid in verified
        ):
            row["status"] = "PASS"
            row["reason"] = (
                "complete class range covered with explicit evidence for "
                f"all {len(method_evidence)} methods"
            )
            row["function_complete"] = True
            return row
        if class_meta.get("read_ranges") or progress.get("body"):
            reasons: list[str] = []
            if not class_meta.get("source_coverage_complete"):
                gap_txt = ", ".join(
                    f"{g.get('line_start')}-{g.get('line_end')}"
                    for g in (class_meta.get("missing_ranges") or [])
                ) or "unknown"
                reasons.append(f"incomplete source coverage; missing lines {gap_txt}")
            else:
                reasons.append("source coverage complete")
            if not class_meta.get("method_enumeration_complete"):
                reasons.append("method enumeration incomplete")
            if not class_meta.get("method_evidence_complete"):
                reasons.append(
                    "method evidence incomplete"
                    + (
                        f" (unverified: {', '.join(missing_methods)})"
                        if missing_methods
                        else ""
                    )
                )
            if class_meta.get("span_limited_unresolved"):
                reasons.append("span_limited unresolved")
            # Source complete but methods/reporting incomplete → PARTIAL.
            row["status"] = "PARTIAL"
            row["reason"] = "; ".join(reasons) or "incomplete class investigation"
            if cont_for_target and pending_continuation:
                row["continuation"] = {
                    "next_line": pending_continuation.get("next_line"),
                    "suggested_end_line": pending_continuation.get(
                        "suggested_end_line"
                    ),
                    "needs_continuation": True,
                }
            return row
        row["status"] = "MISSING_SOURCE_EVIDENCE"
        row["reason"] = "no successful source read of class body"
        return row

    if cont_for_target and pending_continuation:
        row["continuation"] = {
            "next_line": pending_continuation.get("next_line"),
            "suggested_end_line": pending_continuation.get(
                "suggested_end_line"
            ),
            "needs_continuation": True,
        }
        row["status"] = "PARTIAL"
        row["reason"] = (
            f"pending continuation at line "
            f"{pending_continuation.get('next_line')}"
        )
        return row
    if tid in verified:
        # Symbol targets that explicitly still have an open function window
        # must stay PARTIAL. Unknown (None) allows PASS for short helpers /
        # reconcile paths that never recorded a continuation flag.
        incomplete_symbol = (
            row["kind"] == "symbol"
            and progress.get("function_complete") is False
        )
        # Also PARTIAL when known AST end is not fully covered.
        span = symbol_ast_span(tid)
        if (
            not incomplete_symbol
            and row["kind"] == "symbol"
            and span
            and line_end
            and int(line_end) < int(span[2])
            and progress.get("function_complete") is not True
        ):
            incomplete_symbol = True
        if incomplete_symbol:
            unread_from = progress.get("unread_from")
            unread_to = progress.get("unread_to")
            reason = (
                "source was read but function_complete is false "
                "(continuation still required)"
            )
            if unread_from:
                reason += (
                    f"; unread range {unread_from}"
                    f"-{unread_to or unread_from}"
                )
            if progress.get("budget_blocked_reason"):
                reason += f"; blocked={progress.get('budget_blocked_reason')}"
            row["status"] = "PARTIAL"
            row["reason"] = reason
            row["function_complete"] = False
            row["implementation_complete"] = False
            if unread_from:
                row["unread_from"] = unread_from
                row["unread_to"] = unread_to
                row.setdefault("gaps", []).append(
                    f"unread {unread_from}-{unread_to or unread_from}"
                )
            return row
        if row["kind"] == "test_module":
            exec_status = str(
                progress.get("test_execution_status")
                or (state.get("test_executions") or {}).get(tid, {}).get(
                    "status"
                )
                or TEST_NOT_EXECUTED
            )
            row["test_execution_status"] = exec_status
            row["test_names_passed"] = list(
                progress.get("test_passed")
                or (state.get("test_executions") or {}).get(tid, {}).get(
                    "passed"
                )
                or []
            )
            row["test_names_failed"] = list(
                progress.get("test_failed")
                or (state.get("test_executions") or {}).get(tid, {}).get(
                    "failed"
                )
                or []
            )
            row["function_complete"] = True
            if exec_status == TEST_EXECUTION_PASSED:
                row["status"] = TEST_EXECUTION_PASSED
                row["reason"] = (
                    "test source inspected and unittest execution passed"
                )
            elif exec_status == TEST_EXECUTION_FAILED:
                row["status"] = TEST_EXECUTION_FAILED
                row["reason"] = (
                    "test source inspected but unittest execution failed"
                )
            elif tid in verified or progress.get("body"):
                row["status"] = TEST_SOURCE_INSPECTED
                row["reason"] = (
                    "test source inspected via read_file; "
                    "tests not executed (inspection ≠ pass)"
                )
            else:
                row["status"] = TEST_NOT_EXECUTED
                row["reason"] = "test module not inspected or executed"
            return row
        # Implementation PASS requires related test inspection + execution.
        related = []
        for item in req.get("required_targets") or []:
            if str(item.get("id") or "") == tid:
                related = [
                    str(p).replace("\\", "/").strip().lower()
                    for p in (item.get("related_tests") or [])
                    if p
                ]
                break
        if related:
            read_paths = {
                str(r.get("file_path") or "").replace("\\", "/").strip().lower()
                for r in (state.get("read_ranges") or [])
            }
            tests_inspected = any(
                any(
                    rp == tp or rp.endswith("/" + tp.split("/")[-1])
                    for rp in read_paths
                )
                for tp in related
            )
            # Map related test files to test_module execution statuses.
            exec_ok = False
            exec_failed = False
            related_test_ids: list[str] = []
            for titem in req.get("required_targets") or []:
                if str(titem.get("kind") or "") != "test_module":
                    continue
                aliases = [
                    str(a).replace("\\", "/").strip().lower()
                    for a in (titem.get("aliases") or [])
                    if a
                ]
                if any(
                    any(
                        a == tp or a.endswith("/" + tp.split("/")[-1])
                        for a in aliases
                    )
                    for tp in related
                ):
                    related_test_ids.append(str(titem.get("id") or ""))
            for test_id in related_test_ids:
                tprog = (state.get("target_progress") or {}).get(test_id) or {}
                tex = (state.get("test_executions") or {}).get(test_id) or {}
                st = str(
                    tprog.get("test_execution_status")
                    or tex.get("status")
                    or ""
                )
                if st == TEST_EXECUTION_PASSED:
                    exec_ok = True
                if st == TEST_EXECUTION_FAILED:
                    exec_failed = True
            row["related_tests"] = related
            row["related_test_ids"] = related_test_ids
            if not tests_inspected:
                row["status"] = "PARTIAL"
                row["reason"] = (
                    "implementation read; relevant tests not inspected"
                )
                row["function_complete"] = bool(
                    progress.get("function_complete")
                )
                return row
            if exec_failed:
                row["status"] = "PARTIAL"
                row["reason"] = (
                    "implementation read; related tests executed and failed"
                )
                return row
            if not exec_ok:
                row["status"] = "PARTIAL"
                row["reason"] = (
                    "implementation and test source inspected; "
                    "tests not executed (TEST_SOURCE_INSPECTED ≠ PASS)"
                )
                row["function_complete"] = True
                return row
        row["status"] = "PASS"
        row["reason"] = (
            "complete implementation read; related tests inspected and "
            "executed successfully"
            if related
            else "complete source implementation read and verified"
        )
        row["function_complete"] = True
        row["implementation_complete"] = True
        if related:
            row["gaps"] = []
        return row
    if progress.get("body") or meta.get("status") in {"reading", "verified"}:
        row["status"] = "PARTIAL"
        row["reason"] = (
            "source was read but implementation remains unverified"
            if tid not in verified
            else "source read incomplete"
        )
        return row
    if meta.get("status") == "failed":
        row["status"] = "FAIL"
        row["reason"] = str(meta.get("failure_reason") or "requirement violated")
        return row
    if meta.get("status") == "discovered" or tid in (state.get("discovered") or {}):
        row["status"] = "MISSING_SOURCE_EVIDENCE"
        row["reason"] = "search discovery only — no successful source read"
        return row
    return row


def build_evidence_ledger_rows(
    requirements: dict | None,
    evidence: dict | None,
    *,
    pending_continuation: dict | None = None,
) -> list[dict]:
    """Per-target evidence rows for the authoritative final report."""
    req = requirements or {}
    state = evidence or {}
    rows: list[dict] = []
    seen: set[str] = set()
    for item in req.get("required_targets") or []:
        tid = str(item.get("id") or "")
        if not tid or tid in seen:
            continue
        seen.add(tid)
        rows.append(
            classify_target_evidence_status(
                tid,
                requirements=req,
                evidence=state,
                pending_continuation=pending_continuation,
            )
        )
    # Include verified extras that were not in the original checklist.
    for tid in sorted(state.get("verified_targets") or set()):
        if tid in seen:
            continue
        seen.add(tid)
        rows.append(
            classify_target_evidence_status(
                tid,
                requirements=req,
                evidence=state,
                pending_continuation=pending_continuation,
            )
        )
    return rows


def format_evidence_ledger_report(
    rows: list[dict],
    *,
    budget_reason: str | None = None,
    pending_continuation: dict | None = None,
) -> str:
    """Markdown table of authoritative PASS/PARTIAL/FAIL/MISSING statuses."""
    lines = [
        "| Target | File / lines | Status | Evidence note |",
        "|---|---|---|---|",
    ]
    for row in rows:
        loc = "*not inspected*"
        if row.get("file_path"):
            loc = str(row["file_path"])
            if row.get("line_start") and row.get("line_end"):
                loc += f":{row['line_start']}-{row['line_end']}"
        note = str(row.get("reason") or "")
        cont = row.get("continuation") or {}
        if cont.get("needs_continuation"):
            note += (
                f"; needs_continuation=true next_line="
                f"{cont.get('next_line')}"
            )
            if cont.get("suggested_end_line"):
                note += f" suggested_end={cont.get('suggested_end_line')}"
        if row.get("unread_from"):
            note += (
                f"; unread={row.get('unread_from')}-{row.get('unread_to')}"
            )
        if row.get("test_execution_status"):
            note += f"; test_execution={row.get('test_execution_status')}"
        if row.get("related_tests"):
            note += "; related_tests=" + ",".join(
                str(t) for t in row.get("related_tests") or []
            )
        if row.get("function_complete") and row.get("status") == "PASS":
            note += "; function_complete=true"
        class_ledger = row.get("class_ledger") or {}
        if class_ledger:
            verified_n = len(class_ledger.get("verified_methods") or [])
            discovered_n = len(class_ledger.get("discovered_methods") or [])
            evidence_n = len(
                [
                    e
                    for e in (class_ledger.get("method_evidence") or [])
                    if e.get("verified") and e.get("evidence_ranges")
                ]
            )
            # Never advertise N/N unless explicit per-method evidence exists.
            if (
                class_ledger.get("method_evidence_complete")
                and evidence_n == discovered_n
                and discovered_n > 0
            ):
                note += (
                    f"; class_expected={class_ledger.get('expected_range')}"
                    f"; methods_verified={verified_n}/{discovered_n} "
                    f"(explicit evidence for each)"
                )
            else:
                note += (
                    f"; class_expected={class_ledger.get('expected_range')}"
                    f"; source_coverage="
                    f"{'complete' if class_ledger.get('source_coverage_complete') else 'incomplete'}"
                    f"; methods_with_evidence={evidence_n}/{discovered_n}"
                )
        lines.append(
            f"| `{row.get('id')}` | {loc} | {row.get('status')} | {note} |"
        )
        # Explicit per-method evidence table — required for class PASS claims.
        method_evidence = list(class_ledger.get("method_evidence") or [])
        if method_evidence:
            lines.append("")
            lines.append(
                f"**Methods for `{row.get('id')}`** "
                f"(verified only with complete source range evidence):"
            )
            lines.append(
                "| Method | Lines | Verified | Purpose |"
            )
            lines.append("|---|---|---|---|")
            for entry in method_evidence:
                lines.append(
                    f"| `{entry.get('name')}` | "
                    f"{entry.get('line_start')}-{entry.get('line_end')} | "
                    f"{'yes' if entry.get('verified') else 'no'} | "
                    f"{str(entry.get('purpose') or '').replace('|', '/')} |"
                )
            lines.append("")
    footer = []
    if pending_continuation:
        footer.append(
            f"**Pending continuation:** "
            f"`{pending_continuation.get('file_path')}` "
            f"next_line={pending_continuation.get('next_line')} "
            f"suggested_end={pending_continuation.get('suggested_end_line')} "
            f"(finalization blocked from claiming PASS for that symbol)."
        )
    if budget_reason:
        footer.append(f"**Budget / stop reason:** `{budget_reason}`")
    missing = [
        r["id"] for r in rows if r.get("status") == "MISSING_SOURCE_EVIDENCE"
    ]
    partial = [r["id"] for r in rows if r.get("status") == "PARTIAL"]
    passed = [r["id"] for r in rows if r.get("status") == "PASS"]
    # Summary counts are derived only from ledger row statuses.
    counts = summarize_evidence_ledger_counts(rows)
    summary_parts = [
        f"PASS={counts['PASS']}",
        f"PARTIAL={counts['PARTIAL']}",
        f"MISSING_SOURCE_EVIDENCE={counts['MISSING_SOURCE_EVIDENCE']}",
    ]
    if counts.get("FAIL"):
        summary_parts.append(f"FAIL={counts['FAIL']}")
    for key in (
        TEST_SOURCE_INSPECTED,
        TEST_EXECUTION_PASSED,
        TEST_EXECUTION_FAILED,
        TEST_NOT_EXECUTED,
    ):
        if counts.get(key):
            summary_parts.append(f"{key}={counts[key]}")
    footer.append("**Summary:** " + " ".join(summary_parts))
    if missing:
        footer.append(
            "MISSING_SOURCE_EVIDENCE (no successful read_file): "
            + ", ".join(f"`{m}`" for m in missing)
        )
    return "\n".join(lines) + ("\n\n" + "\n\n".join(footer) if footer else "")


def summarize_evidence_ledger_counts(rows: list[dict]) -> dict[str, int]:
    """Count ledger statuses — source of truth for summaries."""
    counts = {
        "PASS": 0,
        "PARTIAL": 0,
        "FAIL": 0,
        "MISSING_SOURCE_EVIDENCE": 0,
        TEST_SOURCE_INSPECTED: 0,
        TEST_EXECUTION_PASSED: 0,
        TEST_EXECUTION_FAILED: 0,
        TEST_NOT_EXECUTED: 0,
    }
    for row in rows:
        status = str(row.get("status") or "MISSING_SOURCE_EVIDENCE")
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return counts


def assert_ledger_ranges_match_symbols(rows: list[dict]) -> list[str]:
    """Return problems when a ledger range is outside the named symbol span."""
    problems: list[str] = []
    for row in rows:
        tid = str(row.get("id") or "")
        if not tid or row.get("kind") == "test_module":
            continue
        span = symbol_ast_span(tid)
        if not span or not row.get("line_start") or not row.get("line_end"):
            continue
        path, start, end = span
        row_path = str(row.get("file_path") or "").replace("\\", "/")
        if row_path and row_path != path:
            problems.append(
                f"{tid}: path {row_path} != {path}"
            )
        if int(row["line_start"]) < start or int(row["line_end"]) > end:
            problems.append(
                f"{tid}: reported {row['line_start']}-{row['line_end']} "
                f"outside AST span {start}-{end}"
            )
        if int(row["line_end"]) < int(row["line_start"]):
            problems.append(f"{tid}: inverted range")
    return problems


def build_evidence_ledger_report(
    requirements: dict | None,
    evidence: dict | None,
    *,
    pending_continuation: dict | None = None,
) -> str:
    """Authoritative investigation status table from the evidence ledger."""
    state = evidence or {}
    rows = build_evidence_ledger_rows(
        requirements,
        state,
        pending_continuation=pending_continuation,
    )
    return format_evidence_ledger_report(
        rows,
        budget_reason=state.get("budget_exhaustion_reason"),
        pending_continuation=pending_continuation,
    )


def reconcile_answer_with_evidence_ledger(
    answer: str,
    *,
    requirements: dict | None,
    evidence: dict | None,
    pending_continuation: dict | None = None,
) -> str:
    """Ensure final report statuses match the evidence ledger.

    Fully read+verified targets are never labeled MISSING_SOURCE_EVIDENCE.
    The ledger table is authoritative; model prose is narrative only.
    Imports / search hits alone never justify narrative PASS.
    """
    ledger = build_evidence_ledger_report(
        requirements,
        evidence,
        pending_continuation=pending_continuation,
    )
    narrative = (answer or "").strip()
    rows = build_evidence_ledger_rows(
        requirements,
        evidence,
        pending_continuation=pending_continuation,
    )
    status_by_id = {
        str(r.get("id") or ""): str(r.get("status") or "")
        for r in rows
        if r.get("id")
    }
    # Feature aliases used in memory-architecture narratives.
    feature_to_targets: dict[str, list[str]] = {}
    for item in (requirements or {}).get("required_targets") or []:
        tid = str(item.get("id") or "")
        feature = str(item.get("feature") or "").replace("_", " ")
        if tid and feature:
            feature_to_targets.setdefault(feature.lower(), []).append(tid)
            feature_to_targets.setdefault(
                feature.lower().replace(" ", "_"), []
            ).append(tid)

    for row in rows:
        tid = str(row.get("id") or "")
        if not tid:
            continue
        esc = re.escape(tid)
        status = str(row.get("status") or "")
        if status == "PASS":
            narrative = re.sub(
                rf"(?im)(\|?\s*`?{esc}`?[^\n]*\|\s*)MISSING_SOURCE_EVIDENCE\b",
                rf"\1PASS",
                narrative,
            )
            narrative = re.sub(
                rf"(?im)^(\s*[-*]\s*`?{esc}`?\s*[:\-].*)\bMISSING_SOURCE_EVIDENCE\b",
                rf"\1PASS",
                narrative,
            )
            continue
        # Demote unsupported PASS claims for non-PASS ledger targets.
        if status in {"MISSING_SOURCE_EVIDENCE", "PARTIAL", "FAIL"}:
            narrative = re.sub(
                rf"(?im)(\|[^|\n]*`?{esc}`?[^|\n]*\|\s*)\*\*?PASS\*\*?",
                rf"\1{status}",
                narrative,
            )
            narrative = re.sub(
                rf"(?im)(\|[^|\n]*`?{esc}`?[^|\n]*\|\s*)PASS\b",
                rf"\1{status}",
                narrative,
            )

    # Demote feature-level PASS claims when no matching target is PASS.
    for feature, tids in feature_to_targets.items():
        if any(status_by_id.get(t) == "PASS" for t in tids):
            continue
        replacement = "MISSING_SOURCE_EVIDENCE"
        if any(status_by_id.get(t) == "PARTIAL" for t in tids):
            replacement = "PARTIAL"
        esc_feat = re.escape(feature)
        narrative = re.sub(
            rf"(?im)(\|[^|\n]*\*\*?{esc_feat}\*\*?[^|\n]*\|\s*)\*\*?PASS\*\*?",
            rf"\1{replacement}",
            narrative,
        )
        narrative = re.sub(
            rf"(?im)(\|[^|\n]*{esc_feat}[^|\n]*\|\s*)\*\*?PASS\*\*?",
            rf"\1{replacement}",
            narrative,
        )
        narrative = re.sub(
            rf"(?im)(\|[^|\n]*{esc_feat}[^|\n]*\|\s*)PASS\b",
            rf"\1{replacement}",
            narrative,
        )

    # Empty / all-missing ledger → never allow narrative PASS from imports.
    if not rows or all(
        r.get("status") == "MISSING_SOURCE_EVIDENCE" for r in rows
    ):
        narrative = re.sub(
            r"(?im)(\|[^|\n]+\|\s*)\*\*?PASS\*\*?",
            r"\1MISSING_SOURCE_EVIDENCE",
            narrative,
        )
        narrative = re.sub(
            r"(?im)(\|[^|\n]+\|\s*)PASS\b",
            r"\1MISSING_SOURCE_EVIDENCE",
            narrative,
        )

    header = (
        "## Evidence ledger (authoritative — from successful read_file windows)\n\n"
        + ledger
    )
    if not narrative:
        return header
    return (
        header
        + "\n\n## Model synthesis (narrative only; cannot override ledger statuses)\n\n"
        + narrative
    )


def build_finalization_messages_with_checklist(
    *,
    system_prompt: str | None = None,
    user_message: str,
    evidence_snippets: list[str],
    missing_targets: list[str] | None = None,
    evidence_state: dict | None = None,
    pending_continuation: dict | None = None,
    requirements: dict | None = None,
) -> list[dict]:
    """Like ``build_finalization_messages`` but includes the evidence ledger."""
    msgs = build_finalization_messages(
        system_prompt=system_prompt,
        user_message=user_message,
        evidence_snippets=evidence_snippets,
        missing_targets=missing_targets,
    )
    state = evidence_state or {}
    ledger = build_evidence_ledger_report(
        requirements,
        state,
        pending_continuation=pending_continuation,
    )
    msgs.insert(2, {
        "role": "system",
        "content": (
            "EVIDENCE LEDGER (authoritative — copy these statuses exactly; "
            "do NOT mark a PASS target as MISSING_SOURCE_EVIDENCE or PARTIAL):\n"
            + ledger
        ),
    })
    return msgs


def build_blocked_tool_result(
    *,
    reason: str,
    prior_result: str | None = None,
) -> dict:
    """Structured non-failure response for intentional duplicate blocking."""
    preview = truncate_text(prior_result or "", 1600) if prior_result else None
    payload = {
        "success": True,
        "blocked": True,
        "error_code": reason,
        "message": (
            "This tool call was blocked because the evidence was already "
            "retrieved (duplicate, overlapping read, or budget). "
            "Do NOT call another tool. Produce the final answer now using "
            "the prior tool results."
        ),
    }
    if preview:
        payload["prior_result_preview"] = preview
    return {
        "success": True,
        "blocked": True,
        "result": json.dumps(payload, ensure_ascii=False),
    }


def build_partial_evidence_answer(
    user_message: str,
    *,
    evidence_snippets: list[str],
    reason: str,
    function_complete: bool = False,
    pending_continuation: bool | dict | None = False,
    requirements: dict | None = None,
    evidence_state: dict | None = None,
) -> str:
    """Evidence-based answer without another Groq call when quota/budget blocks."""
    return build_offline_evidence_answer(
        user_message,
        evidence_snippets=evidence_snippets,
        reason=reason,
        function_complete=function_complete,
        pending_continuation=pending_continuation,
        requirements=requirements,
        evidence_state=evidence_state,
    )


def build_offline_evidence_answer(
    user_message: str,
    *,
    evidence_snippets: list[str],
    reason: str,
    function_complete: bool = False,
    pending_continuation: bool | dict | None = False,
    requirements: dict | None = None,
    evidence_state: dict | None = None,
) -> str:
    """Structured partial from collected tool evidence — no LLM required."""
    snippets = [s.strip() for s in evidence_snippets if s and str(s).strip()]
    state = evidence_state or {}
    req = requirements or {}
    read_snips = list(state.get("read_snippets") or [])
    search_snips = list(state.get("search_snippets") or [])
    # Prefer explicitly tracked read_file windows as verified evidence.
    if not read_snips:
        read_snips = [
            s for s in snippets if s.lower().startswith("read_file")
        ]
    if not search_snips:
        search_snips = [
            s for s in snippets if s.lower().startswith("search_code")
        ]

    if reason in {"tpd", "daily_quota", "rpd"}:
        header = (
            "Groq's **daily token quota (TPD)** is exhausted, so I stopped "
            "further model requests for this task instead of retrying. "
            "Here is a structured answer from the tool evidence already collected."
        )
    elif reason == "tpm":
        header = (
            "Groq's **tokens-per-minute (TPM)** limit blocked another model "
            "call after bounded retries. Here is what the collected evidence shows."
        )
    elif reason == "token_limit":
        header = (
            "The model request was too large (HTTP 413). "
            "Answering from the evidence gathered so far."
        )
    elif reason == "tool_choice_finalization":
        header = (
            "The model tried to call a tool during the tool-free finalization "
            "step, so I did not retry that request. Here is an honest answer "
            "from the source excerpts already collected."
        )
    else:
        header = (
            "I could not complete every investigation step, so here is the "
            "best evidence-based answer available now."
        )

    verified_lines = [
        f"- {truncate_text(snip, 420)}" for snip in read_snips[-8:]
    ]
    verified_block = (
        "\n\n**Verified from read_file source** "
        "(search hits alone are not verified implementation):\n"
        + "\n".join(verified_lines)
        if verified_lines
        else "\n\n**Verified from read_file source:** none collected yet."
    )

    discovered_lines = [
        f"- {truncate_text(snip, 220)}" for snip in search_snips[-6:]
    ]
    discovered_block = (
        "\n\n**Discovered via search_code** (names/paths only; not verified):\n"
        + "\n".join(discovered_lines)
        if discovered_lines
        else ""
    )

    pending_cont = None
    if pending_continuation:
        # Accept bool (legacy) or the continuation dict for ledger rows.
        if isinstance(pending_continuation, dict):
            pending_cont = pending_continuation
        else:
            pending_cont = {"file_path": "?", "next_line": "?", "needs_continuation": True}

    rows = build_evidence_ledger_rows(
        req, state, pending_continuation=pending_cont if isinstance(pending_continuation, dict) else None
    ) if req else []
    # When only a bool was passed, still treat pending as incomplete overall.
    missing = [
        r["id"] for r in rows if r.get("status") == "MISSING_SOURCE_EVIDENCE"
    ] if rows else (uncovered_required_targets(req, state) if req else [])
    partial_ids = [r["id"] for r in rows if r.get("status") == "PARTIAL"]
    passed_ids = [r["id"] for r in rows if r.get("status") == "PASS"]

    # Never list a PASS target under missing evidence.
    if missing:
        missing_block = (
            "\n\n**MISSING_SOURCE_EVIDENCE** (no successful read_file): "
            + ", ".join(missing)
            + ". Do not treat those implementations as verified."
        )
    else:
        missing_block = ""

    if rows:
        checklist_block = (
            "\n\n**Evidence ledger (authoritative):**\n"
            + format_evidence_ledger_report(
                rows,
                budget_reason=state.get("budget_exhaustion_reason"),
                pending_continuation=(
                    pending_cont if isinstance(pending_continuation, dict) else None
                ),
            )
        )
    else:
        checklist_block = ""
    budget_reason = state.get("budget_exhaustion_reason")
    budget_block = (
        f"\n\n**Budget:** stopped with reason `{budget_reason}`."
        if budget_reason
        else ""
    )

    if (
        function_complete
        and not pending_continuation
        and read_snips
        and not missing
        and not partial_ids
        and (not req.get("required_targets") or passed_ids)
    ):
        assumptions = (
            "\n\n**Assumptions / limits:** The source windows above were "
            "collected successfully. Narrative flow beyond those snippets is "
            "inferred and provisional."
        )
        if passed_ids:
            completeness = (
                "\n\n**Completeness:** Required source windows were covered "
                f"(PASS={len(passed_ids)})."
            )
        else:
            completeness = (
                "\n\n**Completeness:** Source windows were collected, but this "
                "offline answer is still **PARTIAL** without a final verified "
                "synthesis."
            )
    elif pending_continuation or missing or partial_ids:
        assumptions = (
            "\n\n**Assumptions:** Any explanatory wording beyond quoted "
            "read_file evidence is inferred, not verified."
        )
        missing_note = (
            f" MISSING_SOURCE_EVIDENCE: {', '.join(missing)}."
            if missing
            else ""
        )
        partial_note = (
            f" PARTIAL: {', '.join(partial_ids)}."
            if partial_ids
            else ""
        )
        reason_note = (
            f" Exhaustion reason: `{budget_reason}`."
            if budget_reason
            else ""
        )
        completeness = (
            "\n\n**Completeness: PARTIAL** — investigation did not verify "
            "every requested target."
            f"{missing_note}{partial_note}{reason_note}"
        )
    else:
        assumptions = (
            "\n\n**Assumptions:** Any explanatory wording beyond quoted "
            "read_file evidence is inferred, not verified."
        )
        completeness = (
            "\n\n**Completeness:** Partial — not presented as fully verified."
        )

    follow = (
        "\n\nWhen your Groq daily quota resets, ask me to continue from the "
        "same symbol and I will synthesize a fuller explanation."
        if reason in {"tpd", "daily_quota", "rpd"}
        else (
            "\n\nIf you want, ask me to continue from a specific unread symbol "
            "or file range and I will dig in more narrowly."
        )
    )
    return (
        f"{header}{verified_block}{discovered_block}{missing_block}"
        f"{checklist_block}{budget_block}"
        f"{assumptions}{completeness}{follow}"
    )


CONVERSATION_STYLE_ADDENDUM = """
============================================================
CONVERSATIONAL STYLE
============================================================

Communicate like a helpful technical assistant:
- Be friendly, professional, and concise.
- Lead with a direct answer, then add brief explanation if useful.
- Ask a follow-up question only when essential information is missing.
- Use examples when teaching concepts.
- Do not use unnecessary tables for simple answers.
- Never end with robotic status lines like "Tool execution completed".
- Never fabricate code, search results, tool output, or personal facts.
- For greetings and ordinary conceptual questions, answer directly
  without tools.
- For codebase questions, prefer search_code then targeted read_file.
""".strip()
