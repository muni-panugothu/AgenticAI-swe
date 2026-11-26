"""Phase B4 — multi-intent understanding and task decomposition.

Deterministic, no provider calls. Produces a validated ``TaskDecomposition``
for routing/planning visibility. Never executes tools, never grants
permissions, and never replaces Topic 35 AgentPlanning.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.input_normalization import normalize_user_input
from app.validation.task_models import (
    DecomposedTask,
    DependencyKind,
    TaskCategory,
    TaskDecomposition,
    TaskSourceProvenance,
)

# Split only when these suggest multiple actions.
_MULTI_SPLIT = re.compile(
    r"(?:,\s*(?:and\s+)?|\s+and\s+then\s+|\s+then\s+|\s+;\s+|\s+and\s+)",
    flags=re.IGNORECASE,
)

_NUMBERED = re.compile(
    r"(?:^|\n)\s*(?:\d+[\).\]]\s+|[-*]\s+)",
)

_ACTION_VERB = re.compile(
    r"(?i)^\s*("
    r"build|create|implement|add|write|connect|integrate|configure|"
    r"generate|make|design|develop|setup|set\s+up|explain|document|"
    r"summarize|analyse|analyze|compare|recommend|find|search|"
    r"test|review|debug|fix|refactor|convert|rewrite|calculate|"
    r"compute|remember|forget|apply|deploy"
    r")\b"
)

# (category, pattern, weight) — applied to a segment.
_CATEGORY_CUES: list[tuple[TaskCategory, str, float]] = [
    ("testing", r"(?i)\b((unit )?tests?|pytest|jest|test coverage)\b", 3.0),
    (
        "documentation",
        r"(?i)\b(documentation|document|docs|readme|how to run|setup guide|explain how)\b",
        3.0,
    ),
    # Framework/stack cues outrank bare "login" so UI/API work is not mislabeled.
    ("api_development", r"(?i)\b(fastapi|flask|django|express|rest api|api endpoints?|graphql)\b", 3.6),
    ("code_generation", r"(?i)\b(react|vue|angular|tailwind|tsx|jsx)\b", 3.6),
    ("code_generation", r"(?i)\b(login page|ui component|front[- ]?end)\b", 3.2),
    ("authentication", r"(?i)\b(jwt|oauth|authentication|sign[- ]?in)\b", 3.2),
    ("authentication", r"(?i)\b(auth|token)\b", 2.4),
    ("authentication", r"(?i)\blogin\b", 2.0),
    ("authorization", r"(?i)\b(authori[sz]ation|rbac|permissions?|acls?|role-based)\b", 2.6),
    ("database_integration", r"(?i)\b(mongodb|postgres(ql)?|mysql|redis|database|db\b|connect .+db)\b", 2.8),
    ("code_modification", r"(?i)\b(make (it|this) |refactor|shorten|improve|modify|convert to)\b", 2.4),
    (
        "debugging",
        r"(?i)\b(debug|fix|stack trace|traceback|exception|why (is|does))\b",
        2.8,
    ),
    (
        "debugging",
        r"(?i)\b(bug|error).{0,40}\b(code|repo|project|file|backend|frontend)\b",
        3.4,
    ),
    ("code_review", r"(?i)\b(code review|review (this|my) (code|pr))\b", 2.4),
    ("research_current_information", r"(?i)\b(current|latest|who is|weather|news|cm\b|chief minister)\b", 2.6),
    ("rag_document", r"(?i)\b(document|handbook|pdf|uploaded|summarize .{0,20}(doc|file|pdf))\b", 2.4),
    ("data_analysis", r"(?i)\b(analy[sz]e|analysis|insights?|metrics|chart)\b", 2.2),
    ("personal_memory", r"(?i)\b(remember that|forget|my preference|search memor)\b", 2.6),
    ("project_work", r"(?i)\b(my project|listifys|in (the )?project)\b", 2.2),
    ("comparison", r"(?i)\b(compare|versus|vs\.?|trade-?offs?|recommend one)\b", 2.6),
    ("explanation", r"(?i)\b(explain|what is|how does|teach|describe)\b", 2.2),
    ("calculation", r"(?i)\b(calculate|compute|\d+\s*[\+\-\*/×x]\s*\d+)\b", 2.8),
]

# Soft required edges when both categories appear (evidence = co-presence).
_CATEGORY_DEPS: list[tuple[TaskCategory, TaskCategory, DependencyKind]] = [
    ("database_integration", "api_development", "required"),
    ("authentication", "api_development", "required"),
    ("authentication", "database_integration", "optional"),
    ("authorization", "authentication", "required"),
    ("testing", "api_development", "required"),
    ("testing", "authentication", "required"),
    ("testing", "code_generation", "required"),
    ("testing", "database_integration", "optional"),
    ("documentation", "api_development", "required"),
    ("documentation", "authentication", "optional"),
    ("documentation", "code_generation", "optional"),
    ("documentation", "database_integration", "optional"),
    ("authentication", "code_generation", "optional"),
]

_MISSING_DB = re.compile(
    r"(?i)\b(with a database|connect( it)?|a db)\b"
)
_MISSING_STACK = re.compile(
    r"(?i)\b(login system|an api|the api|connect it|test everything|all these changes)\b"
)
_INJECTION = re.compile(
    r"(?i)\b(ignore (all )?(previous|prior|safety)|system prompt|jailbreak|"
    r"dump all memor|bypass (tool|allowlist|acl)|grant .{0,20}permission)"
)


def _topo_order(tasks: list[DecomposedTask]) -> list[str]:
    """Kahn topological order; stable by original order_hint then id."""
    ids = [t.id for t in tasks]
    indeg = {t.id: 0 for t in tasks}
    kids: dict[str, list[str]] = {t.id: [] for t in tasks}
    by_id = {t.id: t for t in tasks}
    for t in tasks:
        for d in t.depends_on:
            kids[d].append(t.id)
            indeg[t.id] += 1
    ready = sorted(
        [i for i, n in indeg.items() if n == 0],
        key=lambda i: (by_id[i].order_hint if by_id[i].order_hint is not None else 999, i),
    )
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for c in kids[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
                ready.sort(
                    key=lambda i: (
                        by_id[i].order_hint if by_id[i].order_hint is not None else 999,
                        i,
                    )
                )
    if len(out) != len(ids):
        # Cycle should already be rejected by model; fall back to hint order.
        return [t.id for t in sorted(tasks, key=lambda x: (x.order_hint or 0, x.id))]
    return out


def _classify_segment(segment: str) -> tuple[TaskCategory, float, list[str]]:
    scores: dict[str, float] = {}
    signals: list[str] = []
    for cat, pattern, weight in _CATEGORY_CUES:
        if re.search(pattern, segment):
            scores[cat] = scores.get(cat, 0.0) + weight
            signals.append(f"cat:{cat}")
    if not scores:
        return "general", 0.0, signals
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, float(scores[best]), signals  # type: ignore[return-value]


def _extract_entities(segment: str) -> list[str]:
    ents: list[str] = []
    for pat in (
        r"(?i)\b(FastAPI|Flask|Django|React|Vue|Angular|Tailwind(?: CSS)?|"
        r"MongoDB|PostgreSQL|MySQL|Redis|JWT|OAuth|Listifys|"
        r"Telangana|Andhra Pradesh|Hyderabad)\b",
        r"\b[\w.-]+\.(?:py|tsx?|jsx?|pdf|md)\b",
    ):
        for m in re.finditer(pat, segment):
            val = m.group(0)
            if val not in ents:
                ents.append(val)
    return ents


def _split_segments(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    # Numbered / bulleted lists → one segment per item.
    if _NUMBERED.search(cleaned) and len(_NUMBERED.findall(cleaned)) >= 2:
        parts = _NUMBERED.split(cleaned)
        segs = [p.strip(" \n\t,.") for p in parts if p and p.strip(" \n\t,.")]
        if len(segs) >= 2:
            return segs

    # Conjunction / comma splits when enough action-like pieces remain.
    raw_parts = [p.strip(" \n\t,.") for p in _MULTI_SPLIT.split(cleaned) if p.strip()]
    if len(raw_parts) >= 2:
        # Keep parts that look like actions or are substantial.
        actionish = [
            p
            for p in raw_parts
            if _ACTION_VERB.search(p)
            or len(p.split()) >= 3
            or _classify_segment(p)[1] >= 2.0
        ]
        if len(actionish) >= 2:
            return actionish
        # "A, B, and C" where first holds the verb and others are noun phrases.
        if len(raw_parts) >= 2 and _ACTION_VERB.search(raw_parts[0]):
            # Promote noun-phrase tails into actions by inheriting a verb cue.
            verb_m = _ACTION_VERB.match(raw_parts[0])
            verb = verb_m.group(1) if verb_m else "complete"
            promoted: list[str] = [raw_parts[0]]
            for tail in raw_parts[1:]:
                if _ACTION_VERB.search(tail):
                    promoted.append(tail)
                elif len(tail.split()) >= 2:
                    promoted.append(f"{verb} {tail}")
                else:
                    promoted.append(tail)
            if len(promoted) >= 2:
                return promoted

    return [cleaned]


def _expected_outputs(category: TaskCategory, segment: str) -> list[str]:
    mapping: dict[TaskCategory, list[str]] = {
        "testing": ["test suite or test cases"],
        "documentation": ["setup/run instructions"],
        "authentication": ["auth implementation"],
        "database_integration": ["database connection/config"],
        "api_development": ["API endpoints or project structure"],
        "code_generation": ["source code"],
        "comparison": ["comparison and recommendation"],
        "explanation": ["explanation"],
        "research_current_information": ["current factual answer"],
        "rag_document": ["document-grounded summary or answers"],
        "calculation": ["numeric result"],
        "data_analysis": ["analysis findings"],
    }
    outs = list(mapping.get(category, ["completed task output"]))
    if re.search(r"(?i)\bquestions?\b", segment):
        outs.append("generated questions")
    if re.search(r"(?i)\bresponsive\b", segment):
        outs.append("responsive layout")
    if re.search(r"(?i)\bvalidation\b", segment):
        outs.append("input validation")
    return outs


def _acceptance(category: TaskCategory, segment: str) -> list[str]:
    criteria: list[str] = []
    if category == "testing":
        criteria.append("tests cover the implemented auth/API behavior")
    if category == "documentation":
        criteria.append("includes concrete run/setup steps for this stack")
    if category == "authentication" and re.search(r"(?i)\bjwt\b", segment):
        criteria.append("JWT issue and validate paths are defined")
    if re.search(r"(?i)\btailwind\b", segment):
        criteria.append("uses Tailwind utility classes")
    if re.search(r"(?i)\bresponsive\b", segment):
        criteria.append("layout works on narrow viewports")
    return criteria


def _infer_dependencies(
    tasks: list[DecomposedTask],
) -> tuple[list[DecomposedTask], list[str]]:
    """Add evidence-based edges; do not invent when categories alone don't apply."""
    signals: list[str] = []
    by_cat: dict[str, list[str]] = {}
    for t in tasks:
        by_cat.setdefault(t.category, []).append(t.id)

    # Explicit user ordering via order_hint: earlier tasks may be soft predecessors
    # only when later tasks are testing/docs/auth building on earlier build tasks.
    updated = [t.model_copy(deep=True) for t in tasks]
    by_id = {t.id: t for t in updated}

    for later_cat, earlier_cat, kind in _CATEGORY_DEPS:
        laters = by_cat.get(later_cat) or []
        earliers = by_cat.get(earlier_cat) or []
        if not laters or not earliers:
            continue
        for lid in laters:
            for eid in earliers:
                if eid == lid:
                    continue
                # Prefer earlier order_hint as predecessor.
                if (
                    by_id[eid].order_hint is not None
                    and by_id[lid].order_hint is not None
                    and by_id[eid].order_hint > by_id[lid].order_hint
                ):
                    continue
                if eid not in by_id[lid].depends_on:
                    by_id[lid].depends_on.append(eid)
                    by_id[lid].dependency_kind = kind
                    signals.append(f"dep:{lid}->{eid}:{kind}")

    # Preserve explicit user sequence as optional deps when uncertain categories.
    ordered = sorted(
        updated,
        key=lambda t: (t.order_hint if t.order_hint is not None else 0, t.id),
    )
    if len(ordered) >= 2:
        # If no deps were inferred at all, mark ordering uncertain — do not invent
        # a full chain, but record sequential optional edges only for
        # build→test/doc style when adjacent.
        for i in range(1, len(ordered)):
            prev, cur = ordered[i - 1], ordered[i]
            if cur.depends_on:
                continue
            if cur.category in {"testing", "documentation"} and prev.category in {
                "api_development",
                "authentication",
                "code_generation",
                "database_integration",
                "code_modification",
            }:
                cur.depends_on.append(prev.id)
                cur.dependency_kind = "required"
                signals.append(f"dep_seq:{cur.id}->{prev.id}")

    for t in updated:
        t.independent = not t.depends_on and all(
            t.id not in other.depends_on for other in updated if other.id != t.id
        )
        # Independent means no deps and nothing depends on it? Spec says
        # "identify independent tasks" — tasks with empty depends_on that can
        # start immediately.
        t.independent = len(t.depends_on) == 0

    return updated, signals


def _clarifications_for(
    text: str,
    tasks: list[DecomposedTask],
) -> tuple[list[DecomposedTask], list[str]]:
    prompts: list[str] = []
    updated = [t.model_copy(deep=True) for t in tasks]

    vague_db = bool(_MISSING_DB.search(text)) and not any(
        t.category == "database_integration"
        and any(
            e.casefold() in {"mongodb", "postgresql", "mysql", "redis"}
            for e in t.entities
        )
        for t in updated
    )
    if vague_db or (
        re.search(r"(?i)\bwith a database\b", text)
        and not re.search(r"(?i)\b(mongodb|postgres|mysql|redis)\b", text)
    ):
        for t in updated:
            if t.category in {"database_integration", "authentication", "api_development"}:
                t.needs_clarification = True
                t.clarification_prompt = (
                    "Which database should I use (for example MongoDB or PostgreSQL)?"
                )
                prompts.append(t.clarification_prompt)
                break

    if re.search(r"(?i)\bconnect it\b", text) and not re.search(
        r"(?i)\b(mongodb|postgres|mysql|redis|database)\b", text
    ):
        for t in updated:
            if "connect" in t.description.casefold():
                t.needs_clarification = True
                t.clarification_prompt = (
                    "What should the API connect to (database, service, or URL)?"
                )
                prompts.append(t.clarification_prompt)
                break

    if re.search(r"(?i)\bin my project\b", text) and not re.search(
        r"(?i)\bproject\s+[A-Za-z][A-Za-z0-9_-]{1,40}\b|\blistifys\b",
        text,
    ):
        for t in updated:
            if t.category in {"project_work", "code_modification", "code_generation"}:
                t.needs_clarification = True
                t.clarification_prompt = "Which project should I apply these changes to?"
                prompts.append(t.clarification_prompt)
                break

    if re.search(r"(?i)\btest everything\b", text):
        for t in updated:
            if t.category == "testing":
                t.needs_clarification = True
                t.clarification_prompt = (
                    "What should the tests cover, and which stack/framework is in use?"
                )
                prompts.append(t.clarification_prompt)
                break

    # Deduplicate prompts
    uniq: list[str] = []
    for p in prompts:
        if p and p not in uniq:
            uniq.append(p)
    return updated, uniq


def decompose_request(
    message: str | None,
    *,
    normalized_message: str | None = None,
    trusted_instruction: str | None = None,
    primary_intent: str | None = None,
    contextual_resolution: dict[str, Any] | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
    tenant_id: str | None = None,
) -> TaskDecomposition:
    """Decompose a user request into validated tasks.

    Uses only the trusted current-user instruction. Historical assistant /
    document / code content from B3 must not be treated as new instructions.
    ``user_id`` / ``project_id`` / ``tenant_id`` are audit signals only.
    """
    del project_id, tenant_id
    original = "" if message is None else str(message)
    if trusted_instruction is not None:
        instruction = str(trusted_instruction).strip()
    elif normalized_message is not None:
        instruction = str(normalized_message).strip()
    else:
        instruction = normalize_user_input(original).normalized.strip()

    # Strip injection phrases from the instruction surface (defense in depth).
    safe_instruction = _INJECTION.sub("[redacted-untrusted]", instruction)

    base = TaskDecomposition(
        original_request=original,
        trusted_instruction=safe_instruction,
        primary_intent=primary_intent or "general",
        signals=[f"user_scope:{user_id or 'caller'}"],
        planning_hints={},
    )

    # Never promote B3 untrusted history into tasks.
    if contextual_resolution:
        if contextual_resolution.get("history_is_untrusted"):
            base.signals.append("history_untrusted_ignored_for_tasks")
        # Prefer B3 current_user_instruction when present.
        cui = str(contextual_resolution.get("current_user_instruction") or "").strip()
        if cui and not trusted_instruction:
            safe_instruction = _INJECTION.sub("[redacted-untrusted]", cui)
            base.trusted_instruction = safe_instruction

    if not safe_instruction:
        base.confidence = "low"
        base.confidence_explanation = "empty_instruction"
        base.needs_clarification = True
        base.clarification_prompts = ["Please describe what you need."]
        return base

    # Follow-ups that are single modifications should stay single-intent.
    if contextual_resolution and contextual_resolution.get("is_follow_up"):
        ftype = str(contextual_resolution.get("follow_up_type") or "")
        if ftype in {
            "modification",
            "style_change",
            "language_change",
            "continuation",
            "expansion",
            "entity_substitution",
            "code_continuation",
            "document_continuation",
            "project_reference",
            "multi_turn_clarification",
            "comparison_continuation",
        }:
            # Only split follow-ups when the trusted instruction itself lists
            # multiple new actions.
            segs = _split_segments(safe_instruction)
            if len(segs) < 2:
                task = DecomposedTask(
                    id="t1",
                    description=safe_instruction,
                    category=_classify_segment(safe_instruction)[0],
                    entities=_extract_entities(safe_instruction),
                    independent=True,
                    expected_outputs=_expected_outputs(
                        _classify_segment(safe_instruction)[0], safe_instruction
                    ),
                    provenance=TaskSourceProvenance(
                        source_kind="contextual_resolution",
                        source_span=safe_instruction[:200],
                        notes=f"follow_up:{ftype}",
                    ),
                    confidence="high",
                    confidence_explanation="single_follow_up_action",
                    order_hint=0,
                )
                base.tasks = [task]
                base.execution_order = ["t1"]
                base.independent_task_ids = ["t1"]
                base.is_multi_intent = False
                base.ordering_confidence = "high"
                base.confidence = "high"
                base.confidence_score = 0.9
                base.confidence_explanation = "b3_follow_up_single_action"
                base.signals.append(f"follow_up:{ftype}")
                base.planning_hints = {
                    "task_ids": ["t1"],
                    "is_multi_intent": False,
                    "advisory_only": True,
                }
                return base

    segments = _split_segments(safe_instruction)

    # Single-intent: one segment / one logical action.
    if len(segments) <= 1:
        seg = segments[0] if segments else safe_instruction
        cat, score, signals = _classify_segment(seg)
        task = DecomposedTask(
            id="t1",
            description=seg,
            category=cat,
            entities=_extract_entities(seg),
            independent=True,
            expected_outputs=_expected_outputs(cat, seg),
            acceptance_criteria=_acceptance(cat, seg),
            provenance=TaskSourceProvenance(
                source_kind="user_instruction",
                source_span=seg[:240],
            ),
            confidence="high" if score >= 2.0 else "medium",
            confidence_explanation=f"single_segment;score={score:.2f}",
            order_hint=0,
        )
        tasks, clar_prompts = _clarifications_for(safe_instruction, [task])
        base.tasks = tasks
        base.execution_order = [tasks[0].id]
        base.independent_task_ids = [tasks[0].id]
        base.is_multi_intent = False
        base.ordering_confidence = "high"
        base.confidence = tasks[0].confidence
        base.confidence_score = 0.88 if score >= 2.0 else 0.65
        base.confidence_explanation = "single_intent_no_split"
        base.signals.extend(signals)
        base.needs_clarification = bool(clar_prompts)
        base.clarification_prompts = clar_prompts
        base.planning_hints = {
            "task_ids": [tasks[0].id],
            "is_multi_intent": False,
            "advisory_only": True,
        }
        return base

    built: list[DecomposedTask] = []
    all_signals: list[str] = ["multi_segment"]
    for idx, seg in enumerate(segments):
        cat, score, signals = _classify_segment(seg)
        all_signals.extend(signals)
        tid = f"t{idx + 1}"
        built.append(
            DecomposedTask(
                id=tid,
                description=seg[0].upper() + seg[1:] if seg else seg,
                category=cat,
                entities=_extract_entities(seg),
                depends_on=[],
                independent=True,
                expected_outputs=_expected_outputs(cat, seg),
                acceptance_criteria=_acceptance(cat, seg),
                provenance=TaskSourceProvenance(
                    source_kind="user_instruction",
                    source_span=seg[:240],
                ),
                confidence="high" if score >= 2.2 else "medium",
                confidence_explanation=f"segment={idx};score={score:.2f}",
                order_hint=idx,
            )
        )

    built, dep_signals = _infer_dependencies(built)
    all_signals.extend(dep_signals)
    built, clar_prompts = _clarifications_for(safe_instruction, built)

    order = _topo_order(built)
    independent = [t.id for t in built if not t.depends_on]

    # Ordering confidence: high when every required edge is category-evidenced;
    # uncertain when many tasks lack deps and categories are mixed without edges.
    if any(t.dependency_kind == "uncertain" for t in built if t.depends_on):
        ordering_confidence = "uncertain"
    elif dep_signals:
        ordering_confidence = "high" if len(built) <= 5 else "medium"
    else:
        ordering_confidence = "uncertain"
        all_signals.append("ordering_uncertain_no_strong_deps")

    # Conflicting instructions (e.g. two mutually exclusive DB choices).
    dbs = {
        e.casefold()
        for t in built
        for e in t.entities
        if e.casefold() in {"mongodb", "postgresql", "mysql", "redis"}
    }
    if len(dbs) >= 2 and re.search(r"(?i)\bonly\b|\binstead\b", safe_instruction):
        clar_prompts.append(
            "Which database should be primary for this request?"
        )
        for t in built:
            if t.category == "database_integration":
                t.needs_clarification = True
                t.clarification_prompt = clar_prompts[-1]

    primary = primary_intent or built[0].category
    base.primary_intent = str(primary)
    base.is_multi_intent = True
    base.tasks = built
    base.execution_order = order
    base.independent_task_ids = independent
    base.ordering_confidence = ordering_confidence  # type: ignore[assignment]
    base.needs_clarification = bool(clar_prompts) or any(
        t.needs_clarification for t in built
    )
    base.clarification_prompts = clar_prompts
    base.confidence = "high" if len(built) >= 2 and ordering_confidence != "uncertain" else "medium"
    if clar_prompts:
        base.confidence = "medium"
    base.confidence_score = min(1.0, 0.45 + 0.1 * len(built))
    base.confidence_explanation = (
        f"multi_intent tasks={len(built)};ordering={ordering_confidence}"
    )
    base.signals.extend(all_signals)
    base.planning_hints = {
        "task_ids": [t.id for t in built],
        "execution_order": order,
        "independent_task_ids": independent,
        "is_multi_intent": True,
        "ordering_confidence": ordering_confidence,
        "advisory_only": True,
        "categories": [t.category for t in built],
    }
    return base
