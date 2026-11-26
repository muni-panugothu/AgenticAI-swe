"""Typed per-step tool selection (Topic 39).

Wraps existing ``resolve_tool_choice`` policy without inventing new selection
intelligence. Request-level catalogs remain owned by AgentRouting (Topic 36).
Planning FORCE_* paths bypass this selector in ``chat_with_ai``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


MODE_DISABLED = "disabled"
MODE_FORCED = "forced"
MODE_AUTO = "auto"
MODE_FINALIZE_NONE = "finalize_none"

SELECTION_MODES = frozenset({
    MODE_DISABLED,
    MODE_FORCED,
    MODE_AUTO,
    MODE_FINALIZE_NONE,
})


ResolveToolChoiceFn = Callable[..., tuple[str | dict, str | None]]


@dataclass
class ToolSelectionDecision:
    """Per-step Groq tool-selection outcome for one agent LLM turn."""

    mode: str
    tool_choice: str | dict[str, Any] | None = None
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    nudge: str | None = None
    reason: str = ""


def _forced_tool_name(tool_choice: str | dict[str, Any] | None) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    fn = tool_choice.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        return str(name) if name else None
    return None


def select_tools_for_step(
    *,
    tools_enabled: bool,
    user_message: str,
    step: int,
    max_steps: int,
    calculator_used: bool,
    active_definitions: list[dict[str, Any]] | None,
    force_final: bool = False,
    resolve_tool_choice_fn: ResolveToolChoiceFn | None = None,
) -> ToolSelectionDecision:
    """Select tool schemas and tool_choice for one LLM step.

    Preserves existing semantics:
    - tools disabled → no definitions, no tool_choice
    - ``resolve_tool_choice`` → ``none`` → finalize mode with empty definitions
      (caller must not send schemas with tool_choice=none)
    - otherwise the routing-supplied ``active_definitions`` catalog is kept

    Pass ``resolve_tool_choice_fn=resolve_tool_choice`` from ``chatbot`` in
    production so calculator / multi-tool predicates are the live ones.
    """
    catalog = list(active_definitions or [])

    if not tools_enabled:
        return ToolSelectionDecision(
            mode=MODE_DISABLED,
            tool_choice=None,
            tool_definitions=[],
            nudge=None,
            reason="tools_disabled",
        )

    resolve = resolve_tool_choice_fn
    if resolve is None:
        # Late import avoids circular import at module load; production
        # chatbot always injects the live function explicitly.
        from app.chatbot import resolve_tool_choice as resolve

    tool_choice, nudge = resolve(
        user_message=user_message,
        step=step,
        max_steps=max_steps,
        calculator_used=calculator_used,
        force_final=force_final,
    )

    if tool_choice == "none":
        return ToolSelectionDecision(
            mode=MODE_FINALIZE_NONE,
            tool_choice="none",
            tool_definitions=[],
            nudge=nudge,
            reason="tool_choice_none_finalize",
        )

    forced_name = _forced_tool_name(tool_choice)
    if forced_name:
        return ToolSelectionDecision(
            mode=MODE_FORCED,
            tool_choice=tool_choice,
            tool_definitions=catalog,
            nudge=nudge,
            reason=f"forced_tool:{forced_name}",
        )

    return ToolSelectionDecision(
        mode=MODE_AUTO,
        tool_choice=tool_choice if tool_choice is not None else "auto",
        tool_definitions=catalog,
        nudge=nudge,
        reason="tool_choice_auto",
    )
