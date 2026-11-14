"""Clear requests carried out without the model: "Öffne das Projekt flay", "Mach lauter".

Once a pattern has recognized the tool (cassi.intents), the model only fills in its
arguments – ~1 s, of which ~0.75 s go to writing the ~27 tokens of the call (M2 Pro,
qwen3-vl:8b). For many requests the arguments are as clear as the tool: a project named
exactly, an app with a verb, a shortcut word. Those are filled in here, in microseconds.

Only when nothing is left to interpret: a name that fits two projects, a request with more
in it than one action, anything to be phrased (a task for Claude, a fact) still goes to the
model. ``None`` means: ask the model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from cassi.tools.apps import App, named_app
from cassi.tools.phpstorm import mention

Arguments = dict[str, Any]

# Requests longer than this carry more than one clear action: the model reads them.
MAX_WORDS = 9


def direct_arguments(
    tool: str,
    text: str,
    projects: Sequence[str] = (),
    apps: Sequence[App] = (),
) -> Arguments | None:
    """The arguments of ``tool`` for ``text`` if they are unambiguous, else None."""
    if len(text.split()) > MAX_WORDS and tool not in ("use_keyboard",):
        return None
    handler = _HANDLERS.get(tool)
    return handler(text, projects, apps) if handler else None


# Projects ----------------------------------------------------------------------------------


def _project(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    """The one project the request names literally ("flay", "stadt wolkenthal")."""
    named = _named(text, projects)
    return {"name": named[0]} if len(named) == 1 else None


def _named(text: str, projects: Sequence[str]) -> list[str]:
    """The projects the request names – [] if it names one only roughly next to another:
    "kx forms" is exactly project-symfony-kx-forms but roughly lib-react-kx-forms-app too,
    and Cassi asks which."""
    found = {p: how for p in projects if (how := mention(p, text)) is not None}
    if any(how == "fuzzy" for how in found.values()):
        return []  # only roughly named: the model and the usual checks decide
    return list(found)


def _claude_status(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    named = _named(text, projects)
    if len(named) > 1 or (not named and any(mention(p, text) for p in projects)):
        return None
    return {"project": named[0]} if named else {}


# Apps and windows --------------------------------------------------------------------------

_WINDOW_ACTIONS = [
    ("side_by_side", r"\b(?:nebeneinander|side by side)\b"),
    ("minimize", r"\b(?:minimier\w*|minimi[sz]e)\b"),
    ("maximize", r"\b(?:maximier\w*|maximi[sz]e|vollbild|full ?screen)\b"),
    ("move", r"\b(?:schieb\w*|verschieb\w*|leg\w*|pack\w*|wirf|move)\b"),
    ("close", r"\b(?:schließ\w*|schliess\w*|beende\w*|quit|close)\b|^\W*mach\w*\b.*\bzu\W*$"),
    ("open", r"\b(?:öffne\w*|oeffne\w*|starte?\w*|open|launch)\b|^\W*mach\w*\b.*\bauf\W*$"),
    ("switch", r"\b(?:wechsel\w*|switch|geh\w* zu|zeig\w*|hol\w*|bring\w*)\b"),
]
_MONITOR_WORDS = [
    ("left", r"\blink\w*\b|\bleft\b"),
    ("right", r"\brecht\w*\b|\bright\b"),
    ("other", r"\bander\w*\b|\bother\b"),
    ("1", r"\b(?:erst\w*|haupt\w*|first|main)\s+(?:bildschirm|monitor|screen|display)"),
    ("2", r"\b(?:zweit\w*|second)\b|\b(?:bildschirm|monitor|screen|display)\s+2\b"),
    ("3", r"\b(?:dritt\w*|third)\b|\b(?:bildschirm|monitor|screen|display)\s+3\b"),
]


def _window(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    actions = [name for name, pattern in _WINDOW_ACTIONS if re.search(pattern, text, re.I)]
    if not actions:
        return None
    action = actions[0]  # most specific first: "schieb … nebeneinander" is side_by_side
    app = named_app(text, list(apps))
    if app is None:
        return None
    if action == "side_by_side":
        rest = re.sub(re.escape(app.name), " ", text, flags=re.I)
        other = named_app(rest, list(apps))
        if other is None:
            return None
        return {"action": action, "app": app.name, "other_app": other.name}
    if action == "move":
        where = [name for name, pattern in _MONITOR_WORDS if re.search(pattern, text, re.I)]
        if len(where) != 1:
            return None
        return {"action": action, "app": app.name, "monitor": where[0]}
    if len(actions) > 1 and not (action == "close" and "mach" in text.casefold()):
        # Two verbs ("Öffne Firefox und zeig …"): the model sorts that out.
        return None
    return {"action": action, "app": app.name}


# Keyboard ----------------------------------------------------------------------------------

_SHORTCUTS = [
    ("delete_word", r"\b(?:streich|lösch)\w* (?:das )?letzte wort|delete (?:the )?last word"),
    ("select_all", r"\balles (?:markieren|auswählen)\b|\bselect all\b"),
    ("new_tab", r"\bneue[rn]? tab\b|\bnew tab\b"),
    ("close_tab", r"\btab (?:schließen|zu)\b|\bclose (?:the )?tab\b"),
    ("reopen_tab", r"\btab wieder (?:öffnen|auf)\b|\breopen (?:the )?tab\b"),
    ("redo", r"\bwiederherstell\w*\b|\bredo\b"),
    ("undo", r"\brückgängig\b|\bundo\b"),
    ("save", r"^\W*(?:speicher(?:n|e|t)?|save)\W*$"),
    ("reload", r"\bneu laden\b|\breload\b|\baktualisier\w*\b"),
    ("copy", r"^\W*(?:kopier\w*|copy)\W*$"),
    ("paste", r"^\W*(?:einfüg\w*|füg\w* ein|paste)\W*$"),
    ("cut", r"^\W*(?:ausschneid\w*|schneid\w* aus|cut)\W*$"),
    ("escape", r"^\W*(?:escape|abbrechen)\W*$"),
]
# "Schreib: Viele Grüße" – the colon marks where the dictated text begins.
_DICTATED = re.compile(r"^\W*(?:schreib\w*|tipp\w*|diktier\w*|type|write)\b[^:]{0,20}:\s*(.+)$",
                       re.IGNORECASE | re.DOTALL)  # fmt: skip


def _keyboard(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    if found := _DICTATED.match(text.strip()):
        return {"text": found.group(1).strip()}
    if len(text.split()) > 5:
        return None
    hits = [name for name, pattern in _SHORTCUTS if re.search(pattern, text.strip(), re.I)]
    return {"shortcut": hits[0]} if len(hits) == 1 else None


# System ------------------------------------------------------------------------------------

_SYSTEM = [
    ("lock_screen", r"\bsperr\w*\b|\block\b"),
    ("unmute", r"\bton (?:an|wieder an)\b|\bunmute\b|\bstumm\w* aus\b"),
    ("mute", r"\bstumm\b|\bton aus\b|\bmute\b"),
    ("next_track", r"\b(?:nächste[sn]?|next)\s+(?:lied|song|titel|track)\b"),
    ("previous_track", r"\b(?:vorherige[sn]?|letzte[sn]?|previous)\s+(?:lied|song|titel|track)\b"),
    ("play_pause", r"\bpausier\w*\b|\bmusik (?:an|aus|stopp\w*|weiter)\b|\bpause\b|\bplay\b"),
    ("volume_up", r"\blauter\b|\blouder\b|\bturn (?:it )?up\b"),
    ("volume_down", r"\bleiser\b|\bquieter\b|\bturn (?:it )?down\b"),
]
_LEVEL = re.compile(r"\b(?:lautstärke|volume)\b.*?\b(\d{1,3})\s*(?:%|prozent|percent)?", re.I)


def _system(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    if found := _LEVEL.search(text):
        level = int(found.group(1))
        return {"action": "set_volume", "level": level} if 0 <= level <= 100 else None
    hits = [name for name, pattern in _SYSTEM if re.search(pattern, text, re.I)]
    return {"action": hits[0]} if len(hits) == 1 else None


# Timer -------------------------------------------------------------------------------------

_IN = re.compile(r"\bin\s+(\d{1,3})\s*(minuten?|min|stunden?|std|minutes?|hours?)\b", re.I)
_ABOUT = re.compile(r"\b(?:an|wegen|zu|about|to|of)\s+(.+?)\W*$", re.I)


def _timer(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    if re.search(r"\b(?:welche|list|lösch|cancel|stopp?|abbrechen)\w*\b", text, re.I):
        return None  # listing and cancelling: the model picks which
    found = _IN.search(text)
    if found is None:
        return None
    amount = int(found.group(1))
    minutes = amount * 60 if found.group(2).lower().startswith(("stund", "std", "hour")) else amount
    after = text[found.end() :]
    about = _ABOUT.search(after)
    arguments: Arguments = {"action": "set", "minutes": minutes}
    if about:
        arguments["text"] = re.sub(
            r"^(?:den|die|das|the)\s+", "", about.group(1).strip(), flags=re.I
        )
    return arguments


# Cassi's own settings -----------------------------------------------------------------------

_SETTINGS = [
    ("speed", "slower", r"\blangsamer\b|\bslower\b"),
    ("speed", "faster", r"\bschneller\b|\bfaster\b"),
    ("length", "short", r"\bkürzer\b|\bknapper\b|\bshorter\b"),
    ("length", "detailed", r"\bausführlich\w*\b|\blänger\b|\bin detail\b|\blonger\b"),
    ("screen", "pause", r"\bnicht mehr\b.*\b(?:bildschirm|screen)\b|\b(?:don't|stop) look"),
    ("screen", "resume", r"\bdarfst wieder\b|\bwieder (?:auf den )?(?:bildschirm|screen)\b"),
    ("voice", "next", r"\b(?:andere|neue) stimme\b|\b(?:other|different) voice\b"),
]


def _settings(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    hits = [(key, value) for key, value, pattern in _SETTINGS if re.search(pattern, text, re.I)]
    if len({key for key, _ in hits}) != 1 or len(hits) != 1:
        return None
    return {hits[0][0]: hits[0][1]}


# Calendar ----------------------------------------------------------------------------------

_WHEN = [
    ("tomorrow", r"\bmorgen\b|\btomorrow\b"),
    ("week", r"\bwoche\b|\bweek\b"),
    ("next", r"\bnächste[nr]?\s+termin\b|\bnext (?:appointment|meeting)\b"),
    ("today", r"\bheute\b|\btoday\b"),
]


def _calendar(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    if re.search(r"\bübermorgen\b|\b\d{1,2}\.\s*\d{1,2}\.|\b(?:montag|dienstag|mittwoch|"
                 r"donnerstag|freitag|samstag|sonntag)\b", text, re.I):  # fmt: skip
        return None  # a date to work out: the model
    hits = [when for when, pattern in _WHEN if re.search(pattern, text, re.I)]
    if len(hits) > 1:
        return None
    return {"when": hits[0] if hits else "today"}


# PhpStorm: the questions without a name in them ------------------------------------------

_GIT = re.compile(r"\bgit[- ]?status\b|\b(?:geändert|änderungen|uncommitted|changed)\b", re.I)
_PROBLEMS = re.compile(r"\b(?:fehler|probleme?|warnungen|errors?|problems?|warnings?)\b", re.I)


def _ide(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    named = _named(text, projects)
    if len(named) > 1 or (not named and any(mention(p, text) for p in projects)):
        return None
    where = {"project": named[0]} if named else {}
    git, problems = _GIT.search(text), _PROBLEMS.search(text)
    if git and not problems:
        return {"action": "git_status", **where}
    if problems and not git and not re.search(r"\b(?:öffne|open|datei \w+)\b", text, re.I):
        return {"action": "problems", **where}
    return None  # a file or run configuration to name: the model


# Workflows: finishing and listing need no name -------------------------------------------


def _workflow(text: str, projects: Sequence[str], apps: Sequence[App]) -> Arguments | None:
    if re.search(r"^\W*(?:ablauf|workflow)\s+(?:fertig|ende|beendet|stopp?)\W*$|"
                 r"\b(?:done|finish)\w*\b.*\b(?:workflow|recording)\b", text, re.I):  # fmt: skip
        return {"action": "stop"}
    if re.search(r"\bwelche\b.*\babl[aä]uf\w*\b|\blist\w*\b.*\bworkflows?\b", text, re.I):
        return {"action": "list"}
    return None  # a name to pick out: the model


_HANDLERS = {
    "phpstorm": _ide,
    "workflow": _workflow,
    "open_project": _project,
    "close_project": _project,
    "claude_status": _claude_status,
    "control_window": _window,
    "use_keyboard": _keyboard,
    "system_control": _system,
    "timer": _timer,
    "adjust_settings": _settings,
    "calendar": _calendar,
}
