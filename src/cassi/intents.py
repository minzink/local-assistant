"""What a request asks for: the patterns that route it to a tool without asking the model.

All fixed patterns live here (computer tools, projects, Claude, web pages, files, memory),
so overlaps show up in one place and in one test table (tests/test_patterns.py).

Computer tools (apps, keyboard, PhpStorm, ...):

Like ``cassi.assistant.tool_intent`` for projects and Claude: a clear request is routed to
its tool directly, so the model only fills in the arguments. Unclear ones are left to the
model, which sees all tools.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Coroutine, Sequence
from typing import Any

from cassi.core.language import Language
from cassi.tools.apps import App, named_app
from cassi.tools.base import ToolResult, speech
from cassi.tools.screen import AT_POINTER
from cassi.tools.web import (
    find_url,
)

log = logging.getLogger(__name__)

_APP_VERBS = re.compile(
    r"\b(?:öffne\w*|oeffne\w*|starte?\w*|open|launch|wechsel\w*|switch|geh\w* zu|zeig\w*|"
    r"bring\w*|hol\w*|schließ\w*|schliess\w*|beende\w*|quit|close|minimier\w*|maximier\w*|"
    r"minimi[sz]e|maximi[sz]e|verschieb\w*|schieb\w*|leg\w*|pack\w*|move|nebeneinander|"
    r"side by side|vollbild|full ?screen)\b|"
    # "Mach Slack auf" – not "Mach in Slack einen neuen Kanal auf" (a task inside the app).
    r"^\W*mach\w*\s+(?:(?:mal|bitte|das|den|die|doch)\s+)*\S+(?:\s+\S+)?\s+(?:zu|auf)\W*$",
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
# Not "mach": "Mach in Slack einen Kanal namens test auf" runs nothing.
_RUN = re.compile(r"\b(?:lass\w*|starte?\w*|führ\w*|ausführ\w*|run|execute)\b", re.IGNORECASE)
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
    r"ausschneid\w*|escape|abbrechen|(?:streich|lösch)\w* (?:das )?letzte wort|"
    r"delete (?:the )?last word)\b",
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
# "Leg eine Taste an, mit der du in OBS die Aufnahme startest", "Mach dir einen Hotkey für …".
_NEW_KEY = re.compile(
    r"\bvirtuelle\w*\s+tasten?\b|\bvirtual\s+keys?\b|"
    r"\b(?:leg\w*|erstell\w*|mach\w*|richt\w*|create|make)\b.{0,30}"
    r"\b(?:tasten?|hotkeys?|tastenkürzel|key)\b.{0,30}\b(?:an|ein|für|for|to)\b",
    re.IGNORECASE,
)
# "Drück Enter", "Drück dreimal Pfeil runter", "Press ctrl shift t".
_PRESS = re.compile(
    r"^\W*(?:\w+\W+)?(?:drück\w*|press)\b(?!\s+(?:auf|on)\b)"
    r"(?!.*\b(?:knopf|button|link|schaltfläche)\b)",
    re.IGNORECASE,
)
# "Zeig mir, was in Firefox offen ist" asks about the app, it does not switch to it.
_QUESTION = re.compile(r"\b(?:was|welche\w*|wie|wo|warum|what|which|how|where|why)\b|\?", re.I)
# "Speichere, dass …" is a fact to remember, not Cmd+S.
_THAT = re.compile(r"\b(?:dass|that)\b", re.IGNORECASE)


# How Cassi itself speaks or whether it may look: adjust_settings, not the computer's settings.
_CASSI_SETTINGS = re.compile(
    r"\b(?:sprich|sprech\w*|rede|red|speak|talk)\b.*\b(?:langsamer|schneller|normal\w*|"
    r"slower|faster)\b|\b(?:andere|neue|deine) stimme\b|\bstimme (?:wechseln|ändern)\b|"
    r"\b(?:other|different) voice\b|\bantworte\w*\b.*\b(?:kürzer|länger|ausführlich\w*|"
    r"knapper|normal\w*)\b|\b(?:answer|reply)\b.*\b(?:shorter|longer|in detail)\b|"
    r"\b(?:schau|guck|sieh|look)\w*\b.*\bnicht mehr\b.*\b(?:bildschirm|screen)\b|"
    r"\b(?:don't|do not|stop) look\w*\b.*\bscreen\b|"
    r"\bdarfst wieder\b.*\b(?:bildschirm|screen|schauen|gucken)\b",
    re.IGNORECASE,
)


def about_flay_settings(text: str) -> bool:
    return _CASSI_SETTINGS.search(text) is not None


def mentions_selection(text: str) -> bool:
    """The request is about the text the user has selected ("Übersetz das Markierte")."""
    return _SELECTION.search(text) is not None


def computer_intent(
    text: str,
    apps: Sequence[App] = (),
    projects: Sequence[str] = (),
    commands: Sequence[str] = (),
    keys: Sequence[str] = (),
) -> str | None:
    """The computer tool a request clearly asks for, if any. ``keys``: the user's virtual
    keys ("Nächste Folie")."""
    from cassi.tools.virtual_keys import named_key

    lowered = text.casefold()
    words = len(text.split())
    if keys and named_key(text, keys) is not None:
        return "use_keyboard"
    if _PRESS.search(text) or _NEW_KEY.search(text):
        return "use_keyboard"
    if commands and _RUN.search(text):
        for name in commands:
            if re.search(rf"\b{re.escape(name.casefold())}\b", lowered):
                return "run_command"
    if _CASSI_SETTINGS.search(text):
        return "adjust_settings"
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


# Requests the patterns above miss but that may still be about the computer: the model
# classifies them (cassi.classify). Words of the computer, or a request that starts with a
# verb of doing something; plain chat ("Erzähl mir einen Witz") does not get there.
_COMPUTER_WORDS = re.compile(
    r"\b(?:fenster\w*|window\w*|bildschirm\w*|monitor\w*|screen\w*|display\w*|app\w*|"
    r"programme?|programms|anwendung\w*|einstellung\w*|settings?|menü\w*|menu\w*|tab\w*|klick\w*|"
    r"click\w*|datei\w*|ordner\w*|folder\w*|files?|download\w*|lautstärke\w*|volume|musik|"
    r"music|lied\w*|song\w*|erinner\w*|remind\w*|termin\w*|kalender\w*|calendar|"
    r"appointment\w*|meeting\w*|besprechung\w*|stimme|voice|sprich|speak|zwischenablage|"
    r"clipboard|kopier\w*|copy|einfüg\w*|paste|wecker|timer|bescheid|tipp\w*|markier\w*|"
    r"scroll\w*|dunkelmodus|dark mode|wlan|wifi|bluetooth|helligkeit|brightness|"
    r"benachrichtigung\w*|notification\w*|desktop|schreibtisch|browser|slack|mail\w*|"
    r"spotify|teams|zoom|finder|explorer|antwort\w*|kürzer|ausführlich\w*|langsamer|"
    r"schneller|shorter|slower|faster)\b",
    re.IGNORECASE,
)
_DOING = re.compile(
    r"^\W*(?:bitte\s+)?(?:öffne|oeffne|starte?|mach|schalt\w*|stell\w*|schieb|wirf|hol|"
    r"zeig|such|find|leg|pack|setz|dreh|spiel|stopp?|pausier|wechsel|geh|klick|schreib|"
    r"tipp|kopier|füg|lösch|verschieb|sortier|aktivier|deaktivier|turn|open|start|switch|"
    r"move|put|set|show|find|play|stop|pause|close|type|write|copy|paste)\b",
    re.IGNORECASE,
)
_CLIPBOARD = re.compile(
    r"\b(?:zwischenablage|clipboard|(?:was )?(?:ich )?(?:gerade |eben )?kopiert\w*|"
    r"das kopierte|(?:i )?(?:just )?copied)\b",
    re.IGNORECASE,
)


def may_be_computer_task(text: str, apps: Sequence[App] = ()) -> bool:
    """Worth asking the model whether this is about the computer."""
    if len(text.split()) < 2:
        return False
    if _COMPUTER_WORDS.search(text) or _DOING.search(text):
        return True
    return named_app(text, list(apps)) is not None


def mentions_clipboard(text: str) -> bool:
    """The request is about what the user copied ("Übersetz die Zwischenablage")."""
    return _CLIPBOARD.search(text) is not None


# Requests about projects, Claude, the web, files, memory (formerly in cassi.assistant) ----

SCREEN_TOOL = "look_at_screen"
# Answers to a permission question of Claude ("Ja", "Erlaub das", "Ja, immer", "Nein").
_CLAUDE_YES = re.compile(
    r"^\W*(?:ja|jo|jep|yes|yep|ok(?:ay)?|klar|gerne?|mach|los)\b|"
    r"\b(?:erlaub\w*|genehmig\w*|freigeb\w*|gib\w* (?:es |das )?frei|approve|allow)\b",
    re.IGNORECASE,
)
_CLAUDE_ALWAYS = re.compile(
    r"\b(?:immer|always|nicht mehr fragen|don't ask again|dauerhaft)\b", re.IGNORECASE
)
_CLAUDE_NO = re.compile(
    r"^\W*(?:nein|nee|nö|no|nope)\b|\b(?:ablehn\w*|lehn\w* (?:es |das )?ab|nicht erlaub\w*|"
    r"verbiet\w*|deny|decline)\b",
    re.IGNORECASE,
)
# Longer requests are first cut down to their core (see Assistant._core).
LONG_REQUEST_WORDS = 25
# A plan step that asks for a picture of the screen.
_LOOK_STEP = re.compile(
    r"\b(?:screenshot\w*|bildschirmfoto\w*|bildschirmaufnahme\w*)\b|"
    r"\b(?:schau|guck|sieh|look)\w*\b.*\b(?:screen|bildschirm|monitor)\b",
    re.IGNORECASE,
)
# What a plan step passes on to the next: tool results, and what a screenshot showed.
REPORT_CHARS = 300
SEEN_CHARS = 2000
_SCREEN_WORDS = re.compile(
    r"bildschirm|screen|monitor|display|fenster|window|was siehst du|siehst du (das|hier|da|mal)"
    r"|schau (dir )?(mal )?(das|hier|an)|guck (dir )?(mal )?(das|hier|an)"
    r"|what do you see|look at (this|that|my)",
    re.IGNORECASE,
)


# Tools that act on the computer themselves: a word like "Monitor" in their request ("schieb
# Firefox auf den linken Monitor") is no reason for a screenshot.
COMPUTER_TOOLS = frozenset(
    {
        "control_window",
        "use_keyboard",
        "phpstorm",
        "run_command",
        "operate_computer",
        "calendar",
        "adjust_settings",
        "find_file",
        "timer",
        "system_control",
    }
)


async def _discard(work: Coroutine[Any, Any, Any]) -> None:
    await work


def mentions_screen(text: str) -> bool:
    return _SCREEN_WORDS.search(text) is not None or AT_POINTER.search(text) is not None


# "Welche Programme sind offen?" is answered from the window list: no screenshot, which
# costs the model seconds (~7.5 s on an M2 Pro) and tells it less about names.
_WHAT_IS_OPEN = re.compile(
    r"\bwelche\s+(?:programme|apps?|anwendungen|fenster)\b"
    r"|\bwas\s+(?:ist|habe ich|hab ich|hast du)\s+(?:\w+\s+){0,2}(?:offen|geöffnet|auf)\b"
    r"|\bwhat(?:'s| is| do i have)\s+(?:\w+\s+){0,2}open\b"
    r"|\bwhich\s+(?:apps?|programs?|applications|windows)\b",
    re.IGNORECASE,
)


def asks_what_is_open(text: str) -> bool:
    return _WHAT_IS_OPEN.search(text) is not None


# "Claude" as Whisper tends to write it in German speech. "Cloud" only counts without an
# article ("die Cloud" is the cloud).
_CLAUDE = re.compile(
    r"\b(?:claude?|clau?de|clod+|klod+|klaud|claud|kloud)\b|\bclaw(?=-?zeile)"
    r"|(?<!\bder )(?<!\bdie )(?<!\bzur )(?<!\beine )(?<!\bthe )(?<!\ba )\bcloud\b",
    re.IGNORECASE,
)
_ORDER_WORDS = re.compile(
    r"\b(?:sag\w*|schreib\w*|frag\w*|gib|gibst|geben|gebe|gebt|teil\w*|bitt\w*|beauftrag\w*|lass\w*"
    r"|anweisung|auftrag|kopier\w*|übertrag\w*|tell|ask|give|send|have|let|copy)\b",
    re.IGNORECASE,
)
# Handing something over: only for web pages ("schick Claude die Schritte von …"). In general
# "Schick das Gleiche an Claude" needs the conversation, which the direct routing lacks.
_HAND_OVER = re.compile(
    r"\b(?:schick\w*|übergib|übergeb\w*|weiterleit\w*|leite\w*\b.*\bweiter|"
    r"forward|pass)\b",
    re.IGNORECASE,
)
_STATUS_WORDS = re.compile(
    r"\b(?:fertig|status|stand|soweit|so weit|done|finished|doing|macht|ready)\b", re.IGNORECASE
)


_REMEMBER = re.compile(
    r"\b(?:merk\w*|notier\w*|remember|note that)\b|\bspeicher\w* (?:dir|ab)\b.*\b(?:dass|that)\b|"
    # "Trag in deinem Gedächtnis ein, dass …" – a request, not a question about memory.
    r"^(?![^?]*\?\s*$).*\b(?:trag\w*|speicher\w*|schreib\w*|save|store)\b.*"
    r"\b(?:gedächtnis|memory)\b",
    re.IGNORECASE,
)
_FORGET = re.compile(
    r"\b(?:vergiss|vergesse|vergessen|forget)\b|\blösch\w*\b.*\b(?:gedächtnis|gemerkt)",
    re.IGNORECASE,
)


_PROJECT = r"\b(?:projekt\w*|project\w*)\b"
_CLOSE_PROJECT = re.compile(
    r"\b(?:schließ\w*|schliess\w*|close)\b.*"
    + _PROJECT
    + "|"
    + _PROJECT
    + r".*\b(?:schließen|schliessen|zumachen|zu machen|close)\b",
    re.IGNORECASE,
)
_OPEN_PROJECT = re.compile(
    r"\b(?:öffne\w*|oeffne\w*|open|starte?)\b.*"
    + _PROJECT
    + "|"
    + _PROJECT
    + r".*\b(?:öffnen|oeffnen|aufmachen|auf machen|open)\b",
    re.IGNORECASE,
)
_CLOSE_VERB = re.compile(r"\b(?:schließ\w*|schliess\w*|zumachen|zu machen|close)\b", re.IGNORECASE)
_OPEN_VERB = re.compile(r"\b(?:öffne\w*|oeffne\w*|aufmachen|auf machen|open)\b", re.IGNORECASE)
_PROJECT_TOOLS = ("close_project", "open_project")
# How long "close it" still refers to the project request before it.
FOLLOW_UP_SECONDS = 120.0


# Replies that claim an action happened. Without a tool call in the same turn the claim
# is made up ("Das Projekt ist jetzt geschlossen." after misheard speech), so the tool is
# called after all. Only first person or completed forms count, so explanations like "du
# schließt ein Projekt über File" do not trigger anything.
# The apostrophe is required: German "Im Editor ist … geöffnet" is no "I'm".
_CLAIM_MARKER = re.compile(
    r"\b(?:ich|i|i['’]ve|i['’]ll|i['’]m|jetzt|nun|wurden?|erfolgreich|erledigt|now|"
    r"has been|successfully|done)\b",
    re.IGNORECASE,
)
# Describing what is on screen ("Ich sehe, dass flay geöffnet ist") claims no action.
_DESCRIPTION = re.compile(
    r"\b(?:sehe|siehst|sieht|zu sehen|erkenne|i see|i can see|visible|shows?|showing)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(?:nicht|nichts|kein\w*|not|no|cannot|can't|couldn't|kann|konnte|leider)\b|\?",
    re.IGNORECASE,
)
_CLAIMS = {
    "close_project": re.compile(
        r"\b(?:schließe|schliesse|geschlossen|closed?|closing)\b.*"
        + _PROJECT
        + "|"
        + _PROJECT
        + r".*\b(?:schließe|schliesse|geschlossen|closed)\b",
        re.IGNORECASE,
    ),
    "open_project": re.compile(
        r"\b(?:öffne|geöffnet|opened|opening)\b.*"
        + _PROJECT
        + "|"
        + _PROJECT
        + r".*\b(?:öffne|geöffnet|opened)\b",
        re.IGNORECASE,
    ),
    "send_to_claude": re.compile(
        r"\bclaude\b.*\b(?:geschickt|gesendet|eingegeben|eingetippt|weitergegeben|übergeben|"
        r"sent|typed|passed)\b|\b(?:sent|passed)\b.*\bclaude\b",
        re.IGNORECASE,
    ),
    "remember": re.compile(
        r"\b(?:gemerkt|notiert|merke ich mir|remembered|noted|will remember)\b", re.IGNORECASE
    ),
    "forget": re.compile(r"\b(?:vergessen|gelöscht|forgotten|deleted)\b", re.IGNORECASE),
}


def fixed_reply(results: list[ToolResult], language: Language) -> str | None:
    """What to say about tool results, if every one of them says it itself."""
    if not results or any(result.say is None for result in results):
        return None
    texts = []
    for result in results:
        assert result.say is not None
        texts.append(result.say.get(language) or next(iter(result.say.values())))
    return " ".join(texts)


def claimed_tool(sentence: str) -> str | None:
    """Tool whose action a sentence of the reply claims to have done, if any."""
    if (
        not _CLAIM_MARKER.search(sentence)
        or _NEGATION.search(sentence)
        or _DESCRIPTION.search(sentence)
    ):
        return None
    return next((name for name, claim in _CLAIMS.items() if claim.search(sentence)), None)


def project_follow_up(text: str) -> str | None:
    """Project tool a follow-up without a project name asks for ("schließ es doch")."""
    if _CLOSE_VERB.search(text):
        return "close_project"
    if _OPEN_VERB.search(text):
        return "open_project"
    return None


# "Wie schließe ich ein Projekt?" asks for an explanation, not for closing one.
_HOW_TO = re.compile(r"^\W*(?:wie|how)\b.*\b(?:ich|man|i|you|one)\b", re.IGNORECASE)
# "Ich habe flay gestern geöffnet" tells what happened, it asks for nothing: earlier
# commands that sounded alike must not turn it into one.
_NARRATION = re.compile(
    r"\b(?:(?:habe|hab|hatte|haben|hatten)\b(?:\s+\S+){0,6}?\s+ge\w+(?:t|en)|"
    r"gestern|vorhin|letzte woche|"
    r"i (?:have |had |'ve )?(?:opened|closed|sent|told|asked)|yesterday|last week)\b",
    re.IGNORECASE,
)
# The user takes the request back at the end: "…, never mind" / "ach, vergiss es".
_CANCEL = re.compile(
    r"\b(?:never ?mind|forget it|vergiss es|doch nicht|schon gut|lass (?:es|mal)|egal)\W*$",
    re.IGNORECASE,
)
# Tools whose project argument must be something the user actually said; required = the
# tool cannot act without it.
_PROJECT_ARGUMENTS = {
    "open_project": ("name", True),
    "close_project": ("name", True),
    "send_to_claude": ("project", False),
    "claude_status": ("project", False),
}


CANCELLED = speech("Okay, dann nicht.", "Okay, never mind.")
# Answers to "Meinst du …?": short, and starting with the yes or no.
_YES = re.compile(
    r"^\W*(?:ja|jo|jep|jawohl|genau|richtig|korrekt|stimmt|mach(?:\s+das)?|bitte|okay|ok|"
    r"yes|yeah|yep|sure|correct|right|do it)\b",
    re.IGNORECASE,
)
_NO = re.compile(r"^\W*(?:nein|nee|nö|no|nope|falsch|nicht|abbrechen|cancel)\b", re.IGNORECASE)
# How long a question like "Meinst du …?" waits for its answer.
CONFIRM_SECONDS = 60.0
# Id of a call the user confirmed: it runs without asking again.
CONFIRMED = "confirmed"
# Actions whose request is kept as an example (not remember/forget: their content is the
# point, not how the request was worded).
LEARNABLE = frozenset(
    {
        "open_project",
        "close_project",
        "send_to_claude",
        "claude_status",
        "look_at_screen",
        "ssh_connect",
    }
)
# The user says the last action was wrong.
_CORRECTION = re.compile(
    r"^\W*(?:nein|nee|falsch|stopp|halt|nicht das|das (?:meinte|wollte) ich nicht|"
    r"das war falsch|no|wrong|not that)\b",
    re.IGNORECASE,
)
# A request that led to no action counts as tried again by the next one within this time.
RETRY_SECONDS = 60.0
# The user explicitly wants a thought-through answer.
_THINK = re.compile(
    r"\b(?:denk\w*\s+(?:\w+\s+){0,3}?(?:nach|drüber|darüber)|überleg\w*|gründlich|"
    r"schritt für schritt|think (?:about|it through|carefully|hard)|step by step|"
    r"take your time)\b",
    re.IGNORECASE,
)
THINKING = speech("Moment, ich überlege kurz.", "One moment, let me think.")


def wants_thinking(text: str) -> bool:
    return bool(_THINK.search(text))


# "…, mach es wieder zu": a story with a request at its end.
_MAKE_VERB = re.compile(r"\bmach\w*\b.*\b(?:zu|auf)\b", re.IGNORECASE)


def narrates(text: str) -> bool:
    """Tells what happened ("hab flay gestern geöffnet") without asking for anything."""
    return (
        bool(_NARRATION.search(text))
        and project_follow_up(text) is None
        and not _MAKE_VERB.search(text)
    )


def wants_examples(text: str) -> bool:
    """If earlier commands may help with this request (not for how-tos and stories)."""
    return not _HOW_TO.search(text) and not narrates(text)


def cancelled(text: str) -> bool:
    return bool(_CANCEL.search(text.strip()))


# "Klick auf Git", "Rechtsklick auf die Datei", "click the bell icon".
_CLICK = re.compile(
    r"\b(?:klick\w*|click\w*|rechtsklick\w*|doppelklick\w*|right[- ]click\w*|"
    r"double[- ]click\w*|drück\w*\s+(?:\w+\s+){0,2}?auf|tipp\w*\s+(?:\w+\s+){0,2}?auf)\b",
    re.IGNORECASE,
)
_SCROLL = re.compile(
    r"\b(?:scroll\w*|runterscroll\w*|hochscroll\w*|blätter\w*|page (?:down|up))\b",
    re.IGNORECASE,
)
# A web page: a link, a URL or an address ("heise punkt de").
# "linke", "linken", "links" are the left side, not links.
_WEB_WORDS = re.compile(
    r"\b(?:link(?!e[nrsm]?\b|s\b)\w*|(?:die|den|alle|the|all) links|url|webseite\w*|"
    r"website\w*|web ?page\w*|homepage|internetseite\w*)\b",
    re.IGNORECASE,
)
# What to do with a page beyond opening it: read it.
_READ = re.compile(
    r"\b(?:les\w*|lies|fass\w*|zusammen\w*|inhalt\w*|steht|summar\w*|read\w*|content|"
    r"what does|was sagt|erklär\w*|explain\w*|vergleich\w*|compar\w*|unterschied\w*|"
    r"differen\w*|such\w*|search\w*|find\w*|siehst|sieht|zu sehen|see|sag\w* mir|tell me)\b",
    re.IGNORECASE,
)
# The page open in the browser right now: "von dieser Seite", "this page".
_THIS_PAGE = re.compile(
    r"\b(?:diese[rnms]?|aktuelle[nr]?|offene[nr]?|geöffnete[nr]?)\s+(?:web|internet)?seite\b|"
    r"\b(?:this|the current|the open)\s+(?:web ?)?page\b|\b(?:im|in the) browser\b",
    re.IGNORECASE,
)
SCREEN_ACTIONS = frozenset(
    {"click_on_screen", "scroll_screen", "open_url", "read_webpage", "fill_web_form"}
)
# "Trag bei Name Max ein", "Füll das Formular aus", "fill in the form": with a form, field or
# page named, it is about a web form (not memory: "trag in deinem Gedächtnis ein").
_FILL = re.compile(
    r"\b(?:trag\w*|füll\w*|schreib\w*)\b.*\b(?:ein|aus)\b|\b(?:ausfüll\w*|eintrag\w*)\b|"
    r"\bfill\b.*\b(?:in|out)\b|\benter\b.*\b(?:into|in)\b",
    re.IGNORECASE,
)
_FORM_WORDS = re.compile(
    r"\b(?:formular\w*|feld\w*|form|forms|field\w*|kontaktformular\w*)\b", re.IGNORECASE
)


# A local file: "die README von flay", "das PDF auf dem Schreibtisch", "notes.md".
_FILE_WORDS = re.compile(
    r"\b(?:datei\w*|pdf|dokument\w*|readme|changelog|lizenz\w*|license|notiz\w*|file|"
    r"document\w*|word-datei)\b|\w\.(?:md|txt|pdf|docx|py|php|json|ya?ml|toml|csv)\b|"
    # "die Roadmap im Projekt flay": something inside a project is a file there.
    r"\b(?:im|in dem|aus dem|vom|in the|from the) (?:projekt|project) \w+",
    re.IGNORECASE,
)


# "Sag Claude, er soll die README zusammenfassen", "Tell Claude to update the changelog":
# Claude does the reading, Cassi only passes the task on.
_CLAUDE_DOES = re.compile(r"\b(?:er|sie|es) soll\w*\b|\bto (?!claude\b)[a-z]+", re.IGNORECASE)


def file_intent(text: str) -> bool:
    """Cassi is to read a local file (no web address in the request)."""
    if find_url(text) is not None or not _FILE_WORDS.search(text):
        return False
    if _HOW_TO.search(text) or narrates(text):
        return False
    if _CLAUDE.search(text) and _CLAUDE_DOES.search(text):
        return False
    return bool(_READ.search(text))


# "Merk dir einen Ablauf", "Mach den Ablauf Release bauen", "Ablauf fertig".
_WORKFLOW = re.compile(r"\b(?:abl[aä]uf\w*|workflow\w*|makro\w*|macro\w*)\b", re.IGNORECASE)
_WORKFLOW_ACTIONS = re.compile(
    r"\b(?:merk\w*|zeig\w*|nimm\w*|zeichne\w*|aufnehm\w*|aufzeichn\w*|mach\w*|führ\w*|"
    r"start\w*|spiel\w*|wiederhol\w*|lösch\w*|fertig|ende|stopp?\w*|beend\w*|speicher\w*|"
    r"welche|kennst|liste\w*|record\w*|run|play|repeat|delete|list|done|finish\w*|save)\b",
    re.IGNORECASE,
)


def fill_intent(text: str) -> bool:
    """The request fills in a web form."""
    if not _FILL.search(text) or _HOW_TO.search(text) or narrates(text):
        return False
    return bool(_FORM_WORDS.search(text) or _THIS_PAGE.search(text) or find_url(text))


READING = speech("Ich öffne die Seite und lese sie.", "I'm opening the page and reading it.")
READING_OPEN = speech("Ich lese die Seite.", "I'm reading the page.")


def web_intent(text: str) -> str | None:
    """open_url or read_webpage if the request is about a web page.

    read_webpage covers everything beyond opening: reading, answering, scrolling or clicking
    on the page and handing what was found to Claude.
    """
    if fill_intent(text):
        return None  # fill_web_form, routed like other actions
    url = find_url(text)
    link = url is not None or bool(_WEB_WORDS.search(text))
    if not link and not _THIS_PAGE.search(text):
        return None
    if _HOW_TO.search(text) or narrates(text):
        return None
    if _READ.search(text) or to_claude(text):
        return "read_webpage"
    if _CLICK.search(text) or _SCROLL.search(text):
        # "Öffne heise.de und klick auf Impressum" is a task on that page; "Klick auf den
        # Link Impressum" is just a click.
        return "read_webpage" if url else None
    return "open_url" if link else None


# "SSH auf prod", "Verbinde dich mit dem Server …", "connect to staging".
_SSH = re.compile(
    r"\bs\s?s\s?h\b|\b(?:verbind\w*|connect\w*|einlogg\w*|logg? dich)\b", re.IGNORECASE
)


# Coding without naming Claude: "Implementiere im Projekt flay einen Export-Button", "Behebe
# den Bug im Login", "Schreib Tests für den Warenkorb". Code is always Claude's job – Cassi
# only words the task. Verbs that mean code on their own …
_CODE_VERB_ALONE = re.compile(
    r"\b(?:implementier(?:e|en|st|t)?|programmier(?:e|en|st|t)?|refaktorier(?:e|en|st|t)?|"
    r"refactor(?:e|ing)?|debugg?(?:e|en|st|t|ing)?|implement|umsetzen|setz\w*\b.{0,60}\bum|"
    r"cod(?:e|en|iere|ier)\b(?!\s*(?:für|for|vom|des|der)\b))\b",
    re.IGNORECASE,
)
# … and everyday verbs that do only together with something in code.
_CODE_VERB = re.compile(
    r"\b(?:(?:ein|aus|um|an|hinzu|ab|auf|nach|dazu)?(?:zu)?(?:bau\w*|schreib\w*|füg\w*|"
    r"leg\w*|änder\w*|pass\w*)|fix\w*|beheb\w*|reparier\w*|ergänz\w*|erstell\w*|"
    r"erweiter\w*|entfern\w*|optimier\w*|aktualisier\w*|"
    r"build|write|add|fix|create|change|update|remove|extend)\b",
    re.IGNORECASE,
)
_CODE_THING = re.compile(
    r"\b(?:funktion\w*|feature\w*|klasse\w*|methode\w*|bugs?|bugfix\w*|fehler\w*|tests?|"
    r"unit-?tests?|testf[aä]ll\w*|endpoints?|endpunkt\w*|api|schnittstelle\w*|komponente\w*|"
    r"\w*button\w*|knopf|migration\w*|controller\w*|service\w*|modul\w*|skript\w*|script\w*|"
    r"datenbank\w*|validierung\w*|css|template\w*|code|tickets?|startseite\w*|login|"
    r"formular\w*|readme|changelog|doku\w*|abhängigkeit\w*|dependenc\w*|function\w*|"
    r"class\w*|methods?|components?|database|page)\b"
    r"|\b(?:im|in dem|ins|in the|in)\s+(?:projekt|project|repo)\b",
    re.IGNORECASE,
)
# Things outside code the same verbs make: "Erstelle einen Termin", "Schreib eine Mail".
_NOT_CODE = re.compile(
    r"\b(?:timer\w*|wecker|erinnerung\w*|termin\w*|kalender\w*|e?-?mails?(?!-)|nachricht\w*|"
    r"notiz\w*|einkaufsliste|playlist|alarm|reminder|appointment|message)\b",
    re.IGNORECASE,
)
# "Ist der Bug behoben?", "Warum schlägt der Test fehl?" ask about code, they order nothing.
_ASKS_ABOUT = re.compile(
    r"^\W*(?:ist|sind|hat|haben|hast|wurde|wurden|war|was|warum|wieso|weshalb|wo|wer|"
    r"welche\w*|gibt es|is|are|has|have|did|does|do|was|were|what|why|where|who|which)\b",
    re.IGNORECASE,
)


def coding_intent(text: str) -> bool:
    """The request wants code written or changed – Claude's job, named or not."""
    if any(p.search(text) for p in (_NOT_CODE, _ASKS_ABOUT, _HOW_TO, _CANCEL)):
        return False
    if narrates(text):
        return False
    if _CODE_VERB_ALONE.search(text):
        return True
    return bool(_CODE_VERB.search(text) and _CODE_THING.search(text))


def tool_intent(text: str) -> str | None:
    """Tool a request clearly asks for (Claude, projects, remember, forget), if any."""
    if claude := claude_intent(text):
        return claude
    if _HOW_TO.search(text):
        return None
    if coding_intent(text):
        return "send_to_claude"
    if _SSH.search(text) and not narrates(text):
        return "ssh_connect"
    if _WORKFLOW.search(text) and _WORKFLOW_ACTIONS.search(text) and not narrates(text):
        return "workflow"
    if fill_intent(text):
        return "fill_web_form"
    if _CLICK.search(text) and not narrates(text):
        return "click_on_screen"
    if _SCROLL.search(text) and not narrates(text):
        return "scroll_screen"
    if _CLOSE_PROJECT.search(text):
        return "close_project"
    if _OPEN_PROJECT.search(text):
        return "open_project"
    if _FORGET.search(text):
        return "forget"
    if _REMEMBER.search(text):
        return "remember"
    return None


def to_claude(text: str) -> bool:
    """The request hands something to Claude ("gib es an Claude", "schick Claude …")."""
    return (
        claude_intent(text) == "send_to_claude"
        or bool(_CLAUDE.search(text) and _HAND_OVER.search(text))
        or coding_intent(text)
    )


def claude_intent(text: str) -> str | None:
    """Tool that a request about Claude Code clearly asks for, if any."""
    if not _CLAUDE.search(text):
        return None
    if _ORDER_WORDS.search(text):
        return "send_to_claude"
    if _STATUS_WORDS.search(text):
        return "claude_status"
    return None
