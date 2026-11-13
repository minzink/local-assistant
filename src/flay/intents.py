"""Which of the computer tools a request clearly asks for (apps, keyboard, PhpStorm, ...).

Like ``flay.assistant.tool_intent`` for projects and Claude: a clear request is routed to
its tool directly, so the model only fills in the arguments. Unclear ones are left to the
model, which sees all tools.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from flay.tools.apps import App, named_app

_APP_VERBS = re.compile(
    r"\b(?:öffne\w*|oeffne\w*|starte?\w*|open|launch|wechsel\w*|switch|geh\w* zu|zeig\w*|"
    r"bring\w*|hol\w*|schließ\w*|schliess\w*|beende\w*|quit|close|minimier\w*|maximier\w*|"
    r"minimi[sz]e|maximi[sz]e|verschieb\w*|schieb\w*|leg\w*|pack\w*|move|nebeneinander|"
    r"side by side|vollbild|full ?screen)\b|\bmach\w*\b.*\b(?:zu|auf)\W*$",
    re.IGNORECASE,
)
_PROJECT_WORD = re.compile(r"\b(?:projekt\w*|project\w*)\b", re.IGNORECASE)
_FILE_IN_IDE = re.compile(
    r"\b(?:öffne\w*|open|zeig\w*|spring\w*|geh\w*)\b.*\b(?:datei|klasse|class|file|controller|"
    r"service|entity|interface|trait|template)\b",
    re.IGNORECASE,
)
_IDE_PROBLEMS = re.compile(
    r"\b(?:fehler|probleme?|warnungen|errors?|problems?|warnings?)\b.*"
    r"\b(?:phpstorm|datei|file|editor|code|hier)\b|\b(?:phpstorm|inspection)\b.*\b(?:fehler|"
    r"probleme?|meldet|errors?|problems?)\b",
    re.IGNORECASE,
)
_GIT_STATUS = re.compile(
    r"\bgit[- ]?status\b|\b(?:was|welche)\b.*\b(?:geändert|änderungen|uncommitted|"
    r"nicht committet|committed)\b|\bwhat(?:'s| has)? changed\b",
    re.IGNORECASE,
)
_RUN = re.compile(
    r"\b(?:lass\w*|starte?\w*|führ\w*|ausführ\w*|run|execute|mach\w*)\b", re.IGNORECASE
)
_RUN_CONFIG = re.compile(
    r"\b(?:tests?|testsuite|phpunit|konfiguration|run configuration|ausführungskonfiguration)\b",
    re.IGNORECASE,
)
_TIMER = re.compile(
    r"\berinner(?:e|n)? mich\b|\bremind me\b|\b(?:timer|wecker|countdown|eieruhr)\b|"
    r"\berinnerungen\b|\breminders?\b",
    re.IGNORECASE,
)
_SYSTEM = re.compile(
    r"\b(?:lauter|leiser|lautstärke|volume|louder|quieter|stumm\w*|mute|unmute|"
    r"ton (?:aus|an|ab)|musik (?:an|aus|stopp\w*|pausier\w*|weiter)|pausier\w*|"
    r"(?:nächste[sn]?|vorherige[sn]?|letzte[sn]?) (?:lied|song|titel|track)|"
    r"(?:next|previous) (?:song|track)|sperr\w* (?:den |meinen )?(?:bildschirm|rechner|computer)|"
    r"(?:bildschirm|rechner|computer) sperren|lock (?:the |my )?(?:screen|computer))\b",
    re.IGNORECASE,
)
_DICTATE = re.compile(r"^\W*(?:schreib\w*|tipp\w*|diktier\w*|type|write)\b", re.IGNORECASE)
_NOT_DICTATION = re.compile(
    r"\b(?:claude|gedächtnis|memory|mail|e-mail|nachricht an|merk)\b", re.IGNORECASE
)
_SHORTCUT = re.compile(
    r"\b(?:speicher(?:n|e|t)?|save|rückgängig|undo|wiederherstell\w*|redo|"
    r"alles (?:markieren|auswählen)|select all|neue[rn]? tab|new tab|tab (?:schließen|zu)|"
    r"close (?:the )?tab|neu laden|reload|aktualisier\w*|kopier\w*|copy|einfüg\w*|paste|"
    r"ausschneid\w*|escape|abbrechen)\b",
    re.IGNORECASE,
)
_DOCUMENT = (
    r"(?:pdf|dokument\w*|bild\w*|foto\w*|rechnung\w*|excel|tabelle\w*|präsentation\w*|"
    r"download\w*|schreibtisch|desktop|word-datei)"
)
# A document from the user's folders (not a file of a project): opened with its app.
_OPEN_FILE = re.compile(
    rf"\b(?:öffne\w*|open|zeig\w*)\b.*\b{_DOCUMENT}\b|"
    rf"\bwo (?:ist|liegt|sind|finde ich)\b.*\b(?:{_DOCUMENT}|datei\w*)\b",
    re.IGNORECASE,
)
_SELECTION = re.compile(
    r"\b(?:markiert\w*|markierung|ausgewählt\w*|auswahl|selektiert\w*|selected|selection|"
    r"highlighted)\b",
    re.IGNORECASE,
)
_SHORT = 6
# "Zeig mir, was in Firefox offen ist" asks about the app, it does not switch to it.
_QUESTION = re.compile(r"\b(?:was|welche\w*|wie|wo|warum|what|which|how|where|why)\b|\?", re.I)
# "Speichere, dass …" is a fact to remember, not Cmd+S.
_THAT = re.compile(r"\b(?:dass|that)\b", re.IGNORECASE)


def mentions_selection(text: str) -> bool:
    """The request is about the text the user has selected ("Übersetz das Markierte")."""
    return _SELECTION.search(text) is not None


def computer_intent(
    text: str,
    apps: Sequence[App] = (),
    projects: Sequence[str] = (),
    commands: Sequence[str] = (),
) -> str | None:
    """The computer tool a request clearly asks for, if any."""
    lowered = text.casefold()
    words = len(text.split())
    if commands and _RUN.search(text):
        for name in commands:
            if re.search(rf"\b{re.escape(name.casefold())}\b", lowered):
                return "run_command"
    if _TIMER.search(text):
        return "timer"
    if _SYSTEM.search(text):
        return "system_control"
    if _DICTATE.search(text) and not _NOT_DICTATION.search(text):
        return "use_keyboard"
    if _GIT_STATUS.search(text):
        return "phpstorm"
    if _OPEN_FILE.search(text) and "phpstorm" not in lowered:
        return "find_file"
    if _FILE_IN_IDE.search(text) or _IDE_PROBLEMS.search(text):
        return "phpstorm"
    if _RUN.search(text) and _RUN_CONFIG.search(text):
        return "phpstorm"
    if words <= _SHORT and _SHORTCUT.search(text) and not _THAT.search(text):
        return "use_keyboard"
    if _APP_VERBS.search(text) and not _PROJECT_WORD.search(text) and not _QUESTION.search(text):
        app = named_app(text, list(apps))
        named_project = any(
            re.search(rf"(?<!\w){re.escape(p.casefold())}(?!\w)", lowered) for p in projects
        )
        if app is not None and not named_project:
            return "control_window"
    return None
