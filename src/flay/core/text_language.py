"""Guess whether typed text is German or English (speech gets its language from Whisper)."""

from __future__ import annotations

import re

from flay.core.language import Language

_GERMAN = frozenset(
    (
        "der die das und ist ich du nicht ein eine mit auf für was wie wer wo warum bitte "
        "danke hallo mir mich dir dich mein dein sie wir ihr es zu von den dem im auch noch "
        "nur schon kannst kann bist sind habe hast hat wird werden heute morgen gestern ja "
        "nein"
    ).split()
)
_ENGLISH = frozenset(
    (
        "the and is i you not a an with on for what how who where why please thanks thank "
        "hello me my your it we they are am to of in also only already can could will would "
        "have has do does today tomorrow yesterday yes no"
    ).split()
)
_WORD = re.compile(r"[a-zäöüß']+")


def detect_text_language(text: str, fallback: Language) -> Language:
    lowered = text.lower()
    if any(char in lowered for char in "äöüß"):
        return Language.DE
    words = _WORD.findall(lowered)
    german = sum(word in _GERMAN for word in words)
    english = sum(word in _ENGLISH for word in words)
    if german == english:
        return fallback
    return Language.DE if german > english else Language.EN
