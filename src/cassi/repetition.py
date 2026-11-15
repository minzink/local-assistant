"""Notice when a reply repeats a sentence of the last replies – measured, never cut.

Small models copy their own earlier replies word for word ("Es ist spät, aber ich bleibe
wach, solange du brauchst." at the end of every answer). Cassi keeps the model from it at the
cause – the prompt asks for no stock sign-offs, older replies go into the context condensed
(``cassi.prompt``) – and only logs what still slips through, so it can be measured. Nothing
the assistant says is removed.

Short sentences ("Alles klar.", "Erledigt.") may repeat – they are natural; the first sentence
of a reply never counts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Sentences of fewer words may repeat ("Gern geschehen.").
MIN_WORDS = 4
# How many earlier replies count.
REPLIES = 3

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"\w+")


def key(sentence: str) -> str:
    """The sentence's words, compared without case, punctuation or spacing."""
    return " ".join(_WORD.findall(sentence.lower()))


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


class RepeatGuard:
    """Knows the sentences of the last replies; tells which new ones are repeats."""

    def __init__(self, earlier_replies: Iterable[str], min_words: int = MIN_WORDS) -> None:
        self._min_words = min_words
        self._seen = {
            key(sentence)
            for reply in list(earlier_replies)[-REPLIES:]
            for sentence in sentences(reply)
            if len(key(sentence).split()) >= min_words
        }
        self._kept = 0

    def repeated(self, sentence: str) -> bool:
        """Whether to leave ``sentence`` out; the first one of a reply always stays."""
        words = key(sentence)
        repeat = self._kept > 0 and len(words.split()) >= self._min_words and words in self._seen
        if not repeat:
            self._kept += 1
        return repeat


def repeated_sentences(text: str, earlier_replies: Iterable[str]) -> list[str]:
    """The sentences of ``text`` already said in the last replies (for measuring)."""
    guard = RepeatGuard(earlier_replies)
    return [s for s in sentences(text) if guard.repeated(s)]
