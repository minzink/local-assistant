"""Clean up transcripts with what Flay knows: the user's project names and tech terms.

Speech recognition does not know "symfony" or "kx-ewo" and writes what they sound like
("Symphony"). Such words are replaced by the known spelling, so tools find the project and
Claude gets the right name. Filler and noise ("Hm.", "Matchup.") heard while listening
continuously are recognized here too.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Callable, Iterable

log = logging.getLogger(__name__)

# Terms that are no project name but come up when talking about projects.
TECH_TERMS = ("PhpStorm", "Symfony", "README", "Laravel", "Composer", "Docker")
# Words shorter than this are left alone: too many ordinary words look alike.
MIN_LENGTH = 5
SIMILARITY = 0.8

_WORD = re.compile(r"[\wäöüÄÖÜß]+")
# "Claude" right after "sag"/"frag"/"tell" comes out as "Lord", "Lot" or "Claw" for every
# recognizer tried (measured on the user's own recordings); in that spot it means Claude.
_CLAUDE_MISHEARD = re.compile(
    r"\b(sag|sagt|sage|frag|fragt|frage|schreib|schreibt|schreibe|tell|ask)(\s+)"
    r"(lord|lot|lott|claw|clo|klo|laude)\b",
    re.IGNORECASE,
)
# Words around project names that look like parts of them but must stay as they are.
PROTECTED = frozenset("projekt projekte projekts project projects formen ordner folder".split())

# One-word utterances that mean something on their own; anything else of one word heard
# while listening continuously is most likely noise.
SHORT_REPLIES = frozenset(
    "ja nein jo jep nö nee okay ok stopp stop halt danke hallo hi hey weiter genau richtig "
    "falsch gut super perfekt passt erledigt fertig abbrechen yes no yeah yep nope thanks "
    "hello continue right wrong sure fine great done cancel".split()
)


def _key(word: str) -> str:
    """Spelling-insensitive form: "Möwe" and "moewe" compare equal."""
    word = word.lower()
    for umlaut, plain in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        word = word.replace(umlaut, plain)
    return word


class Vocabulary:
    def __init__(self, names: Callable[[], Iterable[str]]) -> None:
        # Called for every utterance: project lists change while Flay runs.
        self._names = names

    def prompt(self) -> str:
        """The known words as a prompt for Whisper, which then spells them this way."""
        names = list(dict.fromkeys(self._names()))[:40]
        return ", ".join(["Claude", "PhpStorm", *TECH_TERMS[1:], *names]) + "."

    def terms(self) -> dict[str, str]:
        """Known words by their spelling-insensitive key."""
        terms: dict[str, str] = {}
        for name in [*self._names(), *TECH_TERMS]:
            for part in [name, *re.split(r"[-_. ]+", name)]:
                if len(part) >= MIN_LENGTH - 1:
                    terms.setdefault(_key(part), part)
        return terms

    def correct(self, text: str) -> str:
        text = _CLAUDE_MISHEARD.sub(r"\1\2Claude", text)
        terms = self.terms()
        if not terms:
            return text
        words = list(_WORD.finditer(text))
        replacements: list[tuple[int, int, str]] = []
        skip = False
        for index, match in enumerate(words):
            if skip:
                skip = False
                continue
            # "Read Me" → "README": two words heard for one term.
            if index + 1 < len(words):
                following = words[index + 1]
                joined = _key(match.group() + following.group())
                gap = text[match.end() : following.start()]
                if gap.strip() == "" and joined in terms and _key(match.group()) not in terms:
                    replacements.append((match.start(), following.end(), terms[joined]))
                    skip = True
                    continue
            # A guess from similarity alone needs company: a neighbour that surely is a
            # project word ("Stadt Wolkental", "Symphony KX Forms") or "Projekt" before it.
            # Otherwise ordinary words that only look alike would be changed.
            neighbours = [words[i].group() for i in (index - 1, index + 1) if 0 <= i < len(words)]
            context = any(_key(n) in terms or _key(n) in PROTECTED for n in neighbours)
            if term := self._closest(match.group(), terms, fuzzy=context):
                replacements.append((match.start(), match.end(), term))
        for start, end, term in reversed(replacements):
            log.debug("Spracherkennung korrigiert: %s → %s", text[start:end], term)
            text = text[:start] + term + text[end:]
        return text

    @staticmethod
    def _closest(word: str, terms: dict[str, str], fuzzy: bool) -> str | None:
        key = _key(word)
        if key in PROTECTED:
            return None
        if key in terms:
            # Known already; only fix the spelling ("möwe" → "moewe").
            term = terms[key]
            return term if term != word and term.lower() != word.lower() else None
        if not fuzzy or len(word) < MIN_LENGTH or not word.isalpha():
            return None
        close = difflib.get_close_matches(key, list(terms), n=2, cutoff=SIMILARITY)
        # Two candidates that fit equally well: better not guess.
        if len(close) != 1:
            return None
        term = terms[close[0]]
        return term.capitalize() if word[0].isupper() and term.islower() else term


def is_noise(text: str, known: Iterable[str] = ()) -> bool:
    """A lone word that means nothing on its own ("Hm.", "Matchup.")."""
    words = _WORD.findall(text)
    if len(words) != 1:
        return not words
    word = words[0].lower()
    return word not in SHORT_REPLIES and not word.isdigit() and word not in {_key(k) for k in known}
