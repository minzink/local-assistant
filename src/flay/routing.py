"""Routing a request to an action with a form instead of a free tool call.

The model fills in ``{action, project, text}``; the server enforces the structure, and the
project can only be one of the user's real project names (or empty). Free tool calls let
small models invent names, pick a project the user never mentioned, or describe the
action in words instead of calling the tool.
"""

from __future__ import annotations

from typing import Any

from flay.core.language import Language
from flay.core.messages import Message, Role, ToolCall
from flay.prompt import LANGUAGE_NAMES

# What each routable action needs from the form.
ACTIONS = {
    "open_project": "open a project in PhpStorm",
    "close_project": "close a project that is open in PhpStorm",
    "send_to_claude": (
        "give Claude Code (an AI coding agent in PhpStorm's terminal) an instruction; text = "
        "the instruction as one complete, precise sentence"
    ),
    "claude_status": "ask what Claude Code is doing or whether it is done",
    "remember": "store a fact about the user; text = the fact as one short sentence",
    "forget": "delete a stored fact; text = the fact in the user's words",
}
NONE = "none"


def schema(actions: list[str], projects: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            # First, in the model's own words: deciding the action straight away made it
            # fall back to the first listed one whenever the verb was garbled.
            "request": {"type": "string"},
            "action": {"type": "string", "enum": [*actions, NONE]},
            "project": {"type": "string", "enum": ["", *projects]},
            "text": {"type": "string"},
        },
        "required": ["request", "action", "project", "text"],
        "additionalProperties": False,
    }


def messages(
    request: str,
    language: Language,
    actions: list[str],
    projects: list[str],
    state: str | None = None,
    previous: list[Message] | None = None,
) -> list[Message]:
    described = "\n".join(f"- {name}: {ACTIONS[name]}" for name in actions)
    earlier = "".join(
        f"\n{'User' if m.role is Role.USER else 'Assistant'}: {m.text}" for m in previous or []
    )
    instructions = (
        "You decide what a voice assistant for software development should do. Speech "
        "recognition often garbles words, especially project names and the first verb "
        '("She K Command" can be "Schließe KX Command").\n'
        f"Actions:\n{described}\n- {NONE}: anything else – conversation, questions (also "
        'how to do something), statements, or a request the user takes back ("never '
        'mind", "vergiss es").\n'
        "project: the project the user means, from the allowed values; garbled or spelled "
        'names count ("K X E W O" = "kx-ewo"). Leave it empty if the user names none – '
        "never pick a project only because it is open.\n"
        f"Projects: {', '.join(projects)}.\n"
        "Fill in request first: what the user wants, in a few English words."
        f"\nWrite text in {LANGUAGE_NAMES[language]}; empty if the action needs none."
        + (f"\n{state}" if state else "")
        + (f"\nThe conversation so far:{earlier}" if earlier else "")
    )
    return [
        Message.from_text(Role.SYSTEM, instructions),
        Message.from_text(Role.USER, request, language),
    ]


def to_call(answer: dict[str, Any]) -> ToolCall | None:
    """The tool call the form asks for; None for "no action"."""
    action = str(answer.get("action") or NONE)
    if action not in ACTIONS:
        return None
    project = str(answer.get("project") or "")
    text = str(answer.get("text") or "").strip()
    arguments: dict[str, Any]
    if action in ("open_project", "close_project"):
        arguments = {"name": project}
    elif action == "send_to_claude":
        arguments = {"instruction": text} | ({"project": project} if project else {})
    elif action == "claude_status":
        arguments = {"project": project} if project else {}
    else:
        arguments = {"fact": text}
    return ToolCall("routed", action, arguments)
