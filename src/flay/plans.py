"""Requests with several steps: "Öffne flay, dann starte die Tests und danach schick Claude
die Fehler".

The request is split at words that order steps ("dann", "danach", "anschließend", "then").
Flay says the steps, asks first if one of them is far-reaching (sending to Claude, running
commands, closing), and does them one after another; each step sees what the earlier ones
reported, and a step that fails stops the rest.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

# Words between steps; "und" alone is not one ("Firefox und Slack nebeneinander").
_THEN = re.compile(
    r"\s*[,;]?\s*\b(?:und dann|und danach|und anschließend|danach|anschließend|dann|"
    r"und zum schluss|zum schluss|zuletzt|and then|then|after that|afterwards|finally)\b"
    r"\s*[,:]?\s*",
    re.IGNORECASE,
)
_FIRST = re.compile(r"^\W*(?:zuerst|erst(?:mal)?|als erstes|first(?:ly)?)\b[,:]?\s*", re.IGNORECASE)
# Steps that change more than the view: confirmed before the plan starts.
FAR_REACHING = frozenset(
    {"send_to_claude", "run_command", "close_project", "use_keyboard", "fill_web_form",
     "workflow", "ssh_connect"}
)  # fmt: skip
MAX_STEPS = 5


# "und" between two actions: only used when the parts ask for different tools (see
# ``plan_intents``), so "Leg Firefox und Slack nebeneinander" stays one request.
_AND = re.compile(r"\s*,?\s*\b(?:und|and)\b\s*", re.IGNORECASE)


def split_steps(text: str, at_and: bool = False) -> list[str]:
    """The steps of a request, in order; one item if it has no ordering words.

    ``at_and``: also split at "und" ("Schau dir den Bildschirm an und kopier den Text an
    Claude").
    """
    parts = [p.strip(" ,.;") for p in _THEN.split(_FIRST.sub("", text.strip()))]
    if at_and:
        parts = [piece.strip(" ,.;") for part in parts for piece in _AND.split(part)]
    return [p for p in parts if len(p.split()) >= 2]


def plan_intents(
    steps: Sequence[str], intent: Callable[[str], str | None], distinct: bool = False
) -> list[str | None] | None:
    """The tool of each step if this is a plan: at least two steps, two of them actions
    (``distinct``: actions of two different tools, and every step an action)."""
    if not 2 <= len(steps) <= MAX_STEPS:
        return None
    intents = [intent(step) for step in steps]
    if distinct:
        tools = [i for i in intents if i is not None]
        return intents if len(tools) == len(intents) and len(set(tools)) >= 2 else None
    return intents if sum(i is not None for i in intents) >= 2 else None


def describe(steps: Sequence[str], language: str) -> str:
    """The plan in one sentence: "Erstens: öffne flay. Zweitens: …"."""
    ordinals = (
        ["Erstens", "Zweitens", "Drittens", "Viertens", "Fünftens"]
        if language == "de"
        else ["First", "Second", "Third", "Fourth", "Fifth"]
    )
    return " ".join(f"{ordinals[i]}: {step}." for i, step in enumerate(steps))
