"""Split a streamed LLM reply into sentences so speech can start after the first one."""

from __future__ import annotations

import re
from collections.abc import Iterator

# Lowercase abbreviations (without the final dot) that end with a dot but not a sentence.
ABBREVIATIONS = frozenset(
    {
        "z.b", "d.h", "u.a", "usw", "bzw", "ca", "dr", "nr", "prof", "evtl", "ggf", "inkl",
        "vgl", "etc", "e.g", "i.e", "mr", "mrs", "ms", "vs", "st", "jr", "sr",
    }
)  # fmt: skip

_BOUNDARY = re.compile(r"[.!?…]+[\"'»«“”)\]]*\s+|\n+")
_MARKDOWN = re.compile(r"[*_#`>|]+")
_EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f]")
_SPACES = re.compile(r"\s+")
# Clause separators where a sentence may be cut: comma, semicolon, colon, dash.
_CLAUSE = re.compile(r"[,;:]\s+|\s[–—-]\s+")
MIN_CLAUSE_CHARS = 20
# The first chunk is cut at its first clause once it has this many characters, so speech
# starts before the first sentence is complete; "Ja, gerne." stays whole.
MIN_FIRST_CLAUSE_CHARS = 15
_LIST_MARKER = re.compile(r"^(?:[-•]|\d+\.)\s+")


def clean_for_speech(text: str) -> str:
    """Remove markdown and emojis that TTS would read out or stumble over."""
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    return _LIST_MARKER.sub("", _SPACES.sub(" ", text).strip())


def _is_sentence_end(text: str, dot_index: int) -> bool:
    if text[dot_index] != ".":
        return True
    word = text[:dot_index].rsplit(maxsplit=1)[-1] if text[:dot_index].strip() else ""
    if not word:
        return True
    # "am 3. Oktober" – ordinal numbers end with a dot in German.
    if word[-1].isdigit():
        return False
    # Single letters ("A. Merkel") and known abbreviations.
    stripped = word.lower().lstrip("(\"'»“")
    return not (len(stripped) == 1 or stripped in ABBREVIATIONS)


class SentenceSplitter:
    """Yields sentences from streamed text.

    The first chunk ends at its first clause separator (from ``MIN_FIRST_CLAUSE_CHARS`` on),
    so speech starts early. Later sentences growing beyond ``max_chars`` are cut at their last
    clause separator: small models often answer in a single long one.
    """

    def __init__(self, min_chars: int = 2, max_chars: int = 160) -> None:
        self._buffer = ""
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._emitted = False

    def feed(self, delta: str) -> Iterator[str]:
        """Add streamed text; yields every sentence completed by it."""
        self._buffer += delta
        start = 0
        for match in _BOUNDARY.finditer(self._buffer):
            punctuation_start = match.start()
            if match.group().strip() and not _is_sentence_end(self._buffer, punctuation_start):
                continue
            sentence = clean_for_speech(self._buffer[start : match.end()])
            if len(sentence) >= self._min_chars:
                yield from self._emit(sentence)
                start = match.end()
        self._buffer = self._buffer[start:]

        if not self._emitted:
            clause = _CLAUSE.search(self._buffer, MIN_FIRST_CLAUSE_CHARS)
            if clause:
                yield from self._emit(clean_for_speech(self._buffer[: clause.end()]))
                self._buffer = self._buffer[clause.end() :]
        elif len(self._buffer) > self._max_chars:
            clauses = list(_CLAUSE.finditer(self._buffer, MIN_CLAUSE_CHARS))
            if clauses:
                cut = clauses[-1].end()
                yield from self._emit(clean_for_speech(self._buffer[:cut]))
                self._buffer = self._buffer[cut:]

    def _emit(self, text: str) -> Iterator[str]:
        self._emitted = True
        yield text

    def flush(self) -> Iterator[str]:
        """Yield whatever is left once the stream has ended."""
        sentence = clean_for_speech(self._buffer)
        self._buffer = ""
        if sentence:
            yield sentence
