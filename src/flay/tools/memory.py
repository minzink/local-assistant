"""Remember and forget facts on request ("Merk dir …", "Vergiss …")."""

from __future__ import annotations

import difflib
import re
from typing import Any

from flay.llm.base import ToolSpec
from flay.memory.base import Fact, FactMemory, sounds_important
from flay.tools.base import ToolResult, speech

# A description must share this much with a fact to count as meaning it.
MIN_SIMILARITY = 0.5
# The best match must lead the runner-up by this much, otherwise Flay asks.
CLEAR_LEAD = 0.15


# Filler words that say nothing about which fact is meant.
_STOPWORDS = frozenset(
    "dass ich der die das den dem des ein eine einen und oder mein meine meinen mir mich "
    "nutzer benutzer user the that and my me".split()
)


def _words(text: str) -> set[str]:
    # Four-letter stems match inflections like "heiße"/"heißt" or "Geburtstag"/"Geburtstags".
    words = re.findall(r"[a-zäöüß0-9]+", text.lower())
    return {word[:4] for word in words if len(word) > 2 and word not in _STOPWORDS}


def similarity(description: str, fact: str) -> float:
    a, b = description.lower(), fact.lower()
    if a in b or b in a:
        return 1.0
    words_a, words_b = _words(a), _words(b)
    overlap = len(words_a & words_b) / len(words_a) if words_a else 0.0
    return max(overlap, difflib.SequenceMatcher(None, a, b).ratio())


def match_facts(description: str, facts: list[Fact]) -> list[Fact]:
    """Facts the description clearly refers to; several if it is ambiguous."""
    scored = sorted(
        ((similarity(description, fact.text), fact) for fact in facts),
        key=lambda pair: pair[0],
        reverse=True,
    )
    candidates = [(score, fact) for score, fact in scored if score >= MIN_SIMILARITY]
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] >= CLEAR_LEAD:
        return [candidates[0][1]]
    return [fact for _, fact in candidates]


_FACT = {"type": "string", "description": "The fact as one short sentence"}


class RememberTool:
    def __init__(self, memory: FactMemory) -> None:
        self._memory = memory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="remember",
            description=(
                "Store a fact in long-term memory, when the user asks you to remember "
                "something. Write it as one short sentence in the user's language, e.g. "
                "'Lea hat am 3. März Geburtstag.'"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fact": _FACT,
                    "important": {
                        "type": "boolean",
                        "description": (
                            "true for a lasting rule or instruction for you ('in Zukunft "
                            "nie …', 'immer …') or if the user says it is important"
                        ),
                    },
                },
                "required": ["fact"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        fact = " ".join(str(arguments.get("fact", "")).split())
        if not fact:
            return ToolResult("No fact given.", ok=False)
        important = arguments.get("important") is True or sounds_important(fact)
        if not self._memory.remember(fact, important=important):
            return ToolResult(
                f"Already known: {fact}",
                action=f"Gedächtnis: schon bekannt – „{fact}“",
                say=speech("Das weiß ich schon.", "I already know that."),
            )
        if important:
            return ToolResult(
                f"Remembered as important: {fact}",
                action=f"Gedächtnis: gemerkt (wichtig) – „{fact}“",
                say=speech(
                    "Okay, das merke ich mir als wichtig.", "Okay, I'll remember that as important."
                ),
            )
        return ToolResult(
            f"Remembered: {fact}",
            action=f"Gedächtnis: gemerkt – „{fact}“",
            say=speech("Okay, merke ich mir.", "Okay, I'll remember that."),
        )


class ForgetTool:
    def __init__(self, memory: FactMemory) -> None:
        self._memory = memory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="forget",
            description=(
                "Delete a fact from long-term memory when the user asks you to forget it. "
                "Describe the fact in the user's words; it is matched against the stored facts."
            ),
            parameters={
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact to forget"}},
                "required": ["fact"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        description = str(arguments.get("fact", "")).strip()
        facts = self._memory.facts()
        matches = match_facts(description, facts)
        if not matches:
            return ToolResult(
                f"No stored fact matches '{description}'; nothing was deleted. "
                f"Stored facts: {'; '.join(f.text for f in facts) or 'none'}.",
                ok=False,
                action=f"Gedächtnis: nichts gefunden zu „{description}“",
                say=speech(
                    "Dazu habe ich mir nichts gemerkt.", "I haven't stored anything about that."
                ),
            )
        if len(matches) > 1:
            options = "; ".join(f.text for f in matches)
            return ToolResult(
                f"'{description}' could mean several facts, nothing was deleted. Ask the user "
                f"which one: {options}",
                ok=False,
                action=f"Gedächtnis: mehrdeutig, nichts gelöscht – {options}",
                say=speech(
                    f"Das passt auf mehrere Einträge: {options}. Welchen soll ich vergessen?",
                    f"That matches several entries: {options}. Which one should I forget?",
                ),
            )
        fact = matches[0]
        self._memory.forget([fact.id])
        return ToolResult(
            f"Forgot: {fact.text}",
            action=f"Gedächtnis: vergessen – „{fact.text}“",
            say=speech(f"Okay, vergessen: {fact.text}", f"Okay, forgotten: {fact.text}"),
        )
