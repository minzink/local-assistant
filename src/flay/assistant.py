"""Orchestrator: utterance → STT → LLM → TTS → speaker, with per-stage timings."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from flay import routing
from flay.audio.base import AudioBuffer, AudioOutput
from flay.audio.vad import VoiceActivityDetector
from flay.browsing import Finding, browse, browse_text
from flay.config import Settings
from flay.core.events import (
    AssistantDelta,
    AssistantMessage,
    AssistantState,
    EventBus,
    ListeningChanged,
    MuteChanged,
    Notice,
    RecordingChanged,
    StateChanged,
    ToolUsed,
    UserMessage,
    WakeWordChanged,
)
from flay.core.language import Language
from flay.core.messages import ImagePart, Message, Role, TextPart, ToolCall, ToolOutcome
from flay.core.sentences import SentenceSplitter
from flay.core.text_language import detect_text_language
from flay.filler import FillerVoice, TurnFiller
from flay.input.base import UtteranceSource
from flay.llm.base import ChatModel, StructuredChat
from flay.llm.gate import synthesis
from flay.memory.base import Fact, Memory, ShortTermMemory
from flay.prompt import LANGUAGE_NAMES, build_messages, format_examples
from flay.stt.base import SpeechToText, Transcript
from flay.stt.vocabulary import Vocabulary, is_noise
from flay.tools.base import ToolRegistry, ToolResult, speech, spoken
from flay.tools.pages import Page
from flay.tools.phpstorm import Project, match_projects, mention
from flay.tools.screen import Window, screen_arguments
from flay.tools.web import (
    clipboard_url,
    find_url,
    find_urls,
    last_url,
    look_at_page,
    wants_clipboard,
)
from flay.tts.base import TextToSpeech
from flay.verbs import CANONICAL, MAX_WORDS, VERB_TOOLS, candidates
from flay.wake import FOLLOW_UP_SECONDS as WAKE_FOLLOW_UP_SECONDS
from flay.wake import strip_wake_word

log = logging.getLogger(__name__)

MIN_UTTERANCE_SECONDS = 0.3
# Below this the recording is effectively mute (denied permission, wrong device, muted
# mic). Whisper would hallucinate words like "you" from it, so it is not transcribed.
SILENCE_DBFS = -60.0
# How long the answer may wait for the computer's state (open projects, Claude sessions)
# once speech recognition is done; it is read in parallel and usually ready by then.
STATE_TIMEOUT = 0.3
# Tool round trips per turn, e.g. read a web page, scroll on, then type into Claude.
MAX_TOOL_ROUNDS = 5
SCREEN_TOOL = "look_at_screen"
_SCREEN_WORDS = re.compile(
    r"bildschirm|screen|monitor|display|fenster|window|was siehst du|siehst du (das|hier|da|mal)"
    r"|schau (dir )?(mal )?(das|hier|an)|guck (dir )?(mal )?(das|hier|an)"
    r"|what do you see|look at (this|that|my)",
    re.IGNORECASE,
)


def mentions_screen(text: str) -> bool:
    return _SCREEN_WORDS.search(text) is not None


# "Claude" as Whisper tends to write it in German speech. "Cloud" only counts without an
# article ("die Cloud" is the cloud).
_CLAUDE = re.compile(
    r"\b(?:claude?|clau?de|clod+|klod+|klaud|claud|kloud)\b"
    r"|(?<!\bder )(?<!\bdie )(?<!\bzur )(?<!\beine )(?<!\bthe )(?<!\ba )\bcloud\b",
    re.IGNORECASE,
)
_ORDER_WORDS = re.compile(
    r"\b(?:sag\w*|schreib\w*|frag\w*|gib|gibst|geben|gebe|gebt|teil\w*|bitt\w*|beauftrag\w*|lass\w*"
    r"|anweisung|auftrag|tell|ask|give|send|have|let)\b",
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
    r"\b(?:merk\w*|notier\w*|remember|note that)\b|"
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
_CLAIM_MARKER = re.compile(
    r"\b(?:ich|i|i'?ve|i'?ll|i'?m|jetzt|nun|wurden?|erfolgreich|erledigt|now|has been|"
    r"successfully|done)\b",
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
    if not _CLAIM_MARKER.search(sentence) or _NEGATION.search(sentence):
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
_WEB_WORDS = re.compile(
    r"\b(?:link\w*|url|webseite\w*|website\w*|web ?page\w*|homepage|internetseite\w*)\b",
    re.IGNORECASE,
)
# What to do with a page beyond opening it: read it.
_READ = re.compile(
    r"\b(?:les\w*|lies|fass\w*|zusammen\w*|inhalt\w*|steht|summar\w*|read\w*|content|"
    r"what does|was sagt|erklär\w*|explain\w*|vergleich\w*|compar\w*|unterschied\w*|"
    r"differen\w*|such\w*|search\w*|find\w*)\b",
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
# Claude does the reading, Flay only passes the task on.
_CLAUDE_DOES = re.compile(r"\b(?:er|sie|es) soll\w*\b|\bto (?!claude\b)[a-z]+", re.IGNORECASE)


def file_intent(text: str) -> bool:
    """Flay is to read a local file (no web address in the request)."""
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


def tool_intent(text: str) -> str | None:
    """Tool a request clearly asks for (Claude, projects, remember, forget), if any."""
    if claude := claude_intent(text):
        return claude
    if _HOW_TO.search(text):
        return None
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
    return claude_intent(text) == "send_to_claude" or bool(
        _CLAUDE.search(text) and _HAND_OVER.search(text)
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


@dataclass
class TurnTimings:
    clock: Callable[[], float]
    start: float = 0.0
    marks: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start = self.clock()

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, self.clock() - self.start)

    def summary(self) -> str:
        return " | ".join(f"{name} {seconds:.2f}s" for name, seconds in self.marks.items())


class UnbackedClaimError(Exception):
    """The reply claimed an action no tool performed."""

    def __init__(self, tool: str) -> None:
        super().__init__(tool)
        self.tool = tool


class Assistant:
    """Runs conversation turns from voice (utterance source) or typed text (``submit_text``).

    Only one turn runs at a time; a new one replaces whatever is still being answered.
    Microphone and speaker can be muted independently; with a muted speaker replies are
    generated as text only.
    """

    def __init__(
        self,
        settings: Settings,
        source: UtteranceSource,
        stt: SpeechToText,
        llm: ChatModel,
        tts: TextToSpeech,
        output: AudioOutput,
        vad: VoiceActivityDetector | None = None,
        memory: Memory | None = None,
        tools: ToolRegistry | None = None,
        events: EventBus | None = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        vocabulary: Vocabulary | None = None,
        state: Callable[[], Coroutine[Any, Any, str | None]] | None = None,
        projects: Callable[[], list[str]] | None = None,
        wake_word_available: bool = True,
    ) -> None:
        self._settings = settings
        # Listen for "Hey Flay" while listening continuously (switchable in UI and tray).
        self._wake_word = settings.input.continuous.wake_word and wake_word_available
        self._wake_word_available = wake_word_available
        self._source = source
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._output = output
        self._vad = vad
        self._clock = clock
        self._now = now
        self._memory: Memory = memory or ShortTermMemory()
        self._tools = tools or ToolRegistry()
        self._vocabulary = vocabulary
        self._state = state
        # The user's project names: the only values routing may choose from.
        self._projects = projects
        # The current request's action was recognized by a learned verb ("She K NVS").
        self._by_learned_verb = False
        self._state_task: asyncio.Task[str | None] | None = None
        self.events = events or EventBus()
        self._turn: asyncio.Task[None] | None = None
        self._last_activity = time.monotonic()
        self._prefetch: tuple[bytes, asyncio.Task[Transcript | None]] | None = None
        self._microphone_muted = False
        self._speaker_muted = False
        # Last request that named a project, for follow-ups like "versuch's nochmal".
        self._project_request: tuple[str, float] | None = None
        self._fillers = FillerVoice(tts, settings.tts.filler)
        # Language of the last turn: a filler due before speech recognition is done uses it.
        self._language = settings.language.default
        # Tools run in the current turn, remembered with the reply.
        self._actions: list[ToolOutcome] = []
        # What the user said in this turn and just before: project names must come from it.
        self._heard = ""
        # An action waiting for the user's yes ("Meinst du …?"), since when, and the
        # request it came from.
        self._pending: tuple[ToolCall, float, str] | None = None
        # Until when an utterance without "Hey Flay" counts (after a bare "Hey Flay").
        self._wake_follow_up = 0.0
        # This turn's request, and actions of the last turn that become examples unless
        # the user contradicts them now.
        self._request = ""
        self._to_learn: list[tuple[str, ToolCall]] = []
        # The last request if it led to no action, and when: "Sta flay" → "Was meinst du?"
        # → "Starte flay" – the first one meant the same and is learned with it.
        self._unanswered: tuple[str, float] | None = None
        # Fire-and-forget work, referenced so it is not garbage-collected mid-way.
        self._tasks: set[asyncio.Task[None]] = set()
        # Work started for the current turn only.
        self._turn_tasks: list[asyncio.Task[Any]] = []
        self.state = AssistantState.IDLE

    @property
    def pending_confirmation(self) -> ToolCall | None:
        return self._pending[0] if self._pending else None

    @property
    def history(self) -> list[Message]:
        return list(self._memory.context().recent)

    async def prepare_fillers(self) -> None:
        """Synthesize the filler phrases ahead, so none costs time when it is needed."""
        if self._fillers.enabled:
            await self._fillers.prepare()

    # Controls ---------------------------------------------------------------------------

    @property
    def microphone_muted(self) -> bool:
        return self._microphone_muted

    @property
    def speaker_muted(self) -> bool:
        return self._speaker_muted

    def set_muted(self, *, microphone: bool | None = None, speaker: bool | None = None) -> None:
        if microphone is not None:
            self._microphone_muted = microphone
        if speaker is not None:
            self._speaker_muted = speaker
            if speaker:
                # Stop talking right away; the reply continues as text.
                self._output.stop()
        self.events.publish(MuteChanged(self._microphone_muted, self._speaker_muted))

    @property
    def continuous(self) -> bool | None:
        """Whether continuous listening is on; None if the input cannot do it."""
        value = getattr(self._source, "continuous", None)
        return value if isinstance(value, bool) else None

    def set_continuous(self, enabled: bool) -> None:
        switch = getattr(self._source, "set_continuous", None)
        if not callable(switch):
            self._notice("Dauerhaftes Zuhören ist mit dieser Eingabe nicht möglich.")
            return
        switch(enabled)
        self.events.publish(ListeningChanged(enabled))

    @property
    def wake_word(self) -> bool | None:
        """Whether continuous listening waits for "Hey Flay"; None if the model is missing."""
        return self._wake_word if self._wake_word_available else None

    def set_wake_word(self, enabled: bool) -> None:
        if not self._wake_word_available:
            self._notice("Das Aktivierungswort fehlt – bitte einmal 'flay setup' ausführen.")
            return
        if enabled == self._wake_word:
            return
        self._wake_word = enabled
        self._wake_follow_up = 0.0
        log.info("Aktivierungswort „Hey Flay“ %s", "an" if enabled else "aus")
        # The continuous listener is built for one mode; a running one starts anew.
        restart = getattr(self._source, "restart_continuous", None)
        if self.continuous and callable(restart):
            restart()
        self.events.publish(WakeWordChanged(enabled))

    @property
    def workflow_recording(self) -> str | None:
        """Name of the workflow being recorded, if any."""
        recording = getattr(self._tools.get("workflow"), "recording", None)
        return recording if isinstance(recording, str) else None

    def interrupt(self) -> None:
        """The push-to-talk key went down: stop speaking and show that we listen."""
        if self._microphone_muted:
            return
        self._output.stop()
        self._set_state(AssistantState.LISTENING)

    # Turns ------------------------------------------------------------------------------

    async def run(self) -> None:
        keep_warm = asyncio.create_task(self.keep_warm(self._settings.llm.keep_warm_minutes * 60))
        try:
            async for utterance in self._source.utterances():
                if self._microphone_muted:
                    log.debug("Mikrofon stumm geschaltet, Aufnahme verworfen")
                    continue
                recognized = None
                if self._needs_wake_word():
                    recognized = await self._says_wake_word(utterance)
                    if recognized is None:
                        # Someone nearby: Flay's current answer goes on undisturbed.
                        continue
                await self._start_turn(self.handle(utterance, recognized))
            # Source exhausted: let the last answer finish.
            if self._turn is not None:
                await asyncio.wait([self._turn])
        finally:
            keep_warm.cancel()
            await self._cancel_turn()

    async def submit_text(self, text: str) -> None:
        """Answer typed text; replaces a turn that is still running."""
        if text.strip():
            await self._start_turn(self.handle_text(text.strip()))

    async def handle(
        self, audio: AudioBuffer, transcript: asyncio.Task[Transcript | None] | None = None
    ) -> None:
        await self._turn_scope(self._handle_audio(audio, transcript))

    async def handle_text(self, text: str) -> None:
        await self._turn_scope(self._handle_text(text))

    async def _start_turn(self, turn: Coroutine[Any, Any, None]) -> None:
        await self._cancel_turn()
        self._turn = asyncio.create_task(turn)
        self._turn.add_done_callback(_log_failure)

    async def _turn_scope(self, turn: Coroutine[Any, Any, None]) -> None:
        self._memory.turn_started()
        # Read in parallel to speech recognition, so it costs no time.
        self._state_task = asyncio.create_task(self._state()) if self._state else None
        try:
            await turn
        finally:
            if self._state_task is not None:
                self._state_task.cancel()
            # Searches a fixed reply did not need must not outlive the turn.
            for task in self._turn_tasks:
                task.cancel()
            self._turn_tasks.clear()
            self._memory.turn_finished()
            self._set_state(AssistantState.IDLE)
            self._last_activity = time.monotonic()

    async def keep_warm(self, idle_seconds: float, check_every: float = 30.0) -> None:
        """While idle, periodically send the usual prompt so the next answer starts fast."""
        warm = getattr(self._llm, "warm", None)
        if not callable(warm) or idle_seconds <= 0:
            return
        while True:
            await asyncio.sleep(check_every)
            idle = time.monotonic() - self._last_activity
            if idle < idle_seconds or (self._turn is not None and not self._turn.done()):
                continue
            await self._warm()
            # The memory's embedding model is unloaded after idle time just like the LLM.
            warm_memory = getattr(self._memory, "warm_up", None)
            if callable(warm_memory):
                await warm_memory()
            self._last_activity = time.monotonic()

    async def _warm(self) -> None:
        """Send the usual prompt without generating: loads the model and caches the prompt."""
        warm = getattr(self._llm, "warm", None)
        if not callable(warm):
            return
        language = self._settings.language.default
        messages = build_messages(
            self._settings.persona,
            self._memory.context(),
            Message.from_text(Role.USER, ".", language),
            language,
            self._now(),
            tools=bool(self._tools),
        )
        try:
            await warm(messages, self._tools.specs)
            log.debug("LLM warm gehalten")
        except Exception as exc:
            log.debug("Warmhalten fehlgeschlagen: %s", exc)

    async def _handle_text(self, text: str) -> None:
        timings = TurnTimings(self._clock)
        language = detect_text_language(text, self._settings.language.default)
        await self._answer(text, language, "text", timings)

    async def _handle_audio(
        self, audio: AudioBuffer, recognized: asyncio.Task[Transcript | None] | None = None
    ) -> None:
        if audio.duration == 0:
            self._notice("Keine Audiodaten vom Mikrofon erhalten – Eingabegerät prüfen.")
            return
        if audio.duration < MIN_UTTERANCE_SECONDS:
            log.debug("Aufnahme zu kurz (%.2fs), ignoriert", audio.duration)
            return
        log.debug("Aufnahme %.1fs, Pegel %.0f dBFS", audio.duration, audio.level_dbfs)
        if audio.level_dbfs < SILENCE_DBFS:
            self._notice(
                f"Aufnahme ist (fast) stumm ({audio.level_dbfs:.0f} dBFS). Mikrofon-Berechtigung "
                "und Eingabegerät prüfen, z.B. mit 'flay mic-test'."
            )
            return

        self._set_state(AssistantState.THINKING)
        timings = TurnTimings(self._clock)
        filler = self._start_filler()
        try:
            prefetched = recognized or self._take_prefetch(audio)
            if prefetched is not None:
                log.debug("Transkript aus der Vorab-Erkennung übernommen")
            transcript = await (prefetched or self._recognize(audio))
            timings.mark("stt")
            if transcript is None:
                self._notice("Keine Sprache erkannt.")
                return
            if not transcript.text:
                self._notice("Nichts verstanden.")
                return
            text = transcript.text
            known: list[str] = []
            if self._vocabulary is not None:
                text = self._vocabulary.correct(text)
                known = list(self._vocabulary.terms().values())
            if self.continuous and is_noise(text, known):
                # Listening all the time picks up noise that recognition turns into a word.
                log.info("Als Geräusch verworfen: %s", text)
                return
            # "Hey Flay, öffne …": the wake word is not part of the request.
            command = strip_wake_word(text)
            if command == text and self._needs_wake_word():
                # The detector only filters; without the wake word in the transcript it was
                # someone talking nearby.
                log.info("Ohne „Hey Flay“ im Text – ignoriert: %s", text)
                return
            self._wake_follow_up = 0.0
            if command != text:
                if not command:
                    # Just "Hey Flay": the next utterance is the command.
                    self._wake_follow_up = time.monotonic() + WAKE_FOLLOW_UP_SECONDS
                    await self._respond(
                        Message.from_text(Role.USER, text, transcript.language),
                        transcript.language,
                        timings,
                        filler=filler,
                        fixed=speech("Ja?", "Yes?")[transcript.language],
                    )
                    return
                text = command
            if filler is not None:
                filler.language = transcript.language
            await self._answer(text, transcript.language, "voice", timings, filler)
        finally:
            if filler is not None:
                filler.cancel()

    def _needs_wake_word(self) -> bool:
        """Listening continuously with the wake word, and no "Ja?" waiting for an answer."""
        if not (self.continuous and self._wake_word):
            return False
        return time.monotonic() > self._wake_follow_up

    def _start_filler(self) -> TurnFiller | None:
        if not self._fillers.enabled or self._speaker_muted:
            return None

        def playing(on: bool) -> None:
            if not self._speaker_muted:
                self._set_state(AssistantState.SPEAKING if on else AssistantState.THINKING)

        return TurnFiller(
            self._fillers, self._output, self._language, time.monotonic, on_play=playing
        )

    async def _recognize(self, audio: AudioBuffer) -> Transcript | None:
        """VAD trim + speech-to-text; None if the recording holds no speech."""
        if self._vad is not None:
            speech = await self._vad.trim(audio)
            if speech is None:
                return None
            log.debug("Sprache %.1fs von %.1fs", speech.duration, audio.duration)
            audio = speech
        hint = None if self._settings.language.auto_detect else self._settings.language.default
        return await self._stt.transcribe(audio, hint)

    def prefetch(self, audio: AudioBuffer) -> None:
        """Start recognizing a tentative utterance (continuous listening, first pause).

        If the user stops talking, the final utterance is this audio plus silence and the
        transcript is ready earlier; if they continue, the result is dropped.
        """
        if self._microphone_muted:
            return
        if self._prefetch is not None:
            self._prefetch[1].cancel()
        task = asyncio.create_task(self._recognize(audio))
        task.add_done_callback(lambda done: done.cancelled() or done.exception())
        self._prefetch = (audio.pcm, task)

    async def _says_wake_word(self, audio: AudioBuffer) -> asyncio.Task[Transcript | None] | None:
        """Recognize first, then decide – before a new turn cancels the running one.

        Returns the recognition for the turn (it is not done twice), or None if the
        transcript does not start with the wake word.
        """
        task = asyncio.create_task(self._recognize(audio))
        try:
            transcript = await asyncio.shield(task)
        except Exception:  # recognition failed: the turn reports it
            return task
        if transcript is None or not transcript.text:
            return None
        if strip_wake_word(transcript.text) == transcript.text:
            log.info("Ohne „Hey Flay“ im Text – ignoriert: %s", transcript.text)
            return None
        return task

    def _take_prefetch(self, audio: AudioBuffer) -> asyncio.Task[Transcript | None] | None:
        prefetch, self._prefetch = self._prefetch, None
        if prefetch is None:
            return None
        pcm, task = prefetch
        if audio.pcm.startswith(pcm):
            return task
        task.cancel()
        return None

    async def _answer(
        self,
        text: str,
        language: Language,
        source: Literal["voice", "text"],
        timings: TurnTimings,
        filler: TurnFiller | None = None,
    ) -> None:
        log.info("Du (%s): %s", language, text)
        self.events.publish(UserMessage(text, language, source))
        if self.state is not AssistantState.SPEAKING:  # a filler may be playing
            self._set_state(AssistantState.THINKING)
        self._language = language
        acted = any(outcome.ok for outcome in self._actions)
        self._unanswered = (
            (self._request, time.monotonic()) if self._request and not acted else None
        )
        self._actions = []
        self._settle_examples(text)
        self._request = text
        earlier = [m.text for m in self.history if m.role is Role.USER][-2:]
        self._heard = " ".join([*earlier, text])
        state = await self._current_state()
        # Searching memories runs while the request is routed; it is collected at the end.
        recall = getattr(self._memory, "recall", None)
        recalling = self._turn_task(recall(text)) if callable(recall) else None
        find_examples = getattr(self._memory, "examples", None)
        finding = (
            self._turn_task(find_examples(text))
            if callable(find_examples) and wants_examples(text)
            else None
        )
        examples: tuple[Fact, ...] = ()
        user_message = Message.from_text(Role.USER, text, language)
        if (answer := await self._confirmation(text, language)) is not None:
            await self._respond(user_message, language, timings, filler=filler, fixed=answer)
            return
        if cancelled(text):
            # Taken back before it was finished: nothing to do, and nothing to guess.
            log.info("Anfrage zurückgenommen")
            await self._respond(
                user_message, language, timings, filler=filler, fixed=CANCELLED[language]
            )
            return
        if file_intent(text) and "read_file" in self._tools:
            await self._file(user_message, text, language, timings, filler, state)
            return
        web = web_intent(text)
        if (
            web is not None
            and web in self._tools
            and await self._web(user_message, text, web, language, timings, filler, state)
        ):
            return
        intent, request = self._intent(text)
        # "Klick im PhpStorm-Fenster auf Git": the click looks itself.
        screen = (
            SCREEN_TOOL in self._tools and mentions_screen(text) and intent not in SCREEN_ACTIONS
        )
        if filler is not None:
            filler.action = screen or (intent is not None and intent in self._tools)
        # A reply that is fixed by what the tools reported, without asking the model.
        fixed: str | None = None
        prelude: list[Message] = []
        # A story names projects and verbs like a command, but asks for no action.
        exclude: set[str] = set(routing.ACTIONS) if intent is None and narrates(text) else set()
        try:
            # Heard exactly like this and confirmed before: that beats a learned verb.
            unclear = intent is None or self._by_learned_verb
            known = self._known_action(text) if unclear and not screen else None
            if known is not None:
                # Heard exactly like this before, and it surely meant this action.
                log.info("Bekannte Bitte: %s(%s)", known.name, known.arguments)
                prelude, results = await self._act(known, language)
                timings.mark("tool")
                fixed = fixed_reply(results, language)
            if screen:
                user_message, shot = await self._attach_screenshot(user_message)
                timings.mark("screenshot")
                if not shot.images:
                    # Without the picture the model would make up what it "sees".
                    fixed = fixed_reply([shot], language)
            if fixed is None and intent is not None and intent in self._tools:
                # The weighing step sees how such requests were meant before.
                examples = await self._recalled(finding)
            if fixed is None and (projects := self._routable_projects(text, intent)):
                assert intent is not None
                available, call = await self._route(request, language, state, projects, intent)
                timings.mark("route")
                if available and call is None:
                    # The form said "no action": the answer must not act after all.
                    exclude = set(routing.ACTIONS)
                elif available and call is not None:
                    prelude, results = await self._act(call, language)
                    timings.mark("tool")
                    fixed = fixed_reply(results, language)
                elif intent is not None and intent in self._tools:
                    routed = Message.from_text(Role.USER, request, language)
                    prelude, results = await self._route_tool(
                        intent, routed, language, state, examples
                    )
                    timings.mark("tool")
                    fixed = fixed_reply(results, language)
            elif fixed is None and intent is not None and intent in self._tools:
                routed = Message.from_text(Role.USER, request, language)
                prelude, results = await self._route_tool(intent, routed, language, state, examples)
                timings.mark("tool")
                fixed = fixed_reply(results, language)
        except asyncio.CancelledError:
            # Interrupted before any answer: keep the request, the next one may build on it.
            self._remember(user_message, "", language, interrupted=True)
            raise
        think = fixed is None and self._settings.llm.thinking.enabled and wants_thinking(text)
        memories = await self._recalled(recalling) if fixed is None else ()
        if fixed is None and not examples:
            examples = await self._recalled(finding)
        completed = await self._respond(
            user_message,
            language,
            timings,
            prelude,
            filler,
            fixed=fixed,
            state=state,
            exclude=exclude,
            think=think,
            memories=memories,
            examples=examples,
        )
        timings.mark("done" if completed else "interrupted")
        log.info("Zeiten: %s", timings.summary())
        if think and self._settings.llm.thinking.model:
            # The server may have unloaded the normal model for the thinking one (Ollama
            # does when memory is short); reload it now, not at the next request.
            self._background(self._warm())

    async def _respond(
        self,
        user_message: Message,
        language: Language,
        timings: TurnTimings,
        prelude: list[Message] | None = None,
        filler: TurnFiller | None = None,
        fixed: str | None = None,
        state: str | None = None,
        exclude: set[str] | None = None,
        think: bool = False,
        memories: Sequence[Fact] = (),
        examples: Sequence[Fact] = (),
        announce: str | None = None,
        work: Callable[
            [Callable[[str], None]], Coroutine[Any, Any, tuple[list[Message], str | None]]
        ]
        | None = None,
    ) -> bool:
        """Stream LLM → sentences → TTS → speaker concurrently.

        Speech starts with the first complete sentence while the LLM keeps generating and
        TTS synthesizes the following sentences ahead of playback. ``fixed`` is spoken as
        it is, without the model. Returns False if the reply was interrupted.

        ``announce`` is said first; ``work`` (slow actions such as reading a web page) runs
        while it is spoken and returns the prelude for the model and/or a fixed reply. It
        gets a function to say what it does next ("Ich klicke auf Installation.").
        """
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        # Audio chunks with the number of the sentence they belong to.
        audio: asyncio.Queue[tuple[int, AudioBuffer] | None] = asyncio.Queue()
        parts: list[str] = []
        finished = False
        delivered = False
        # Sentences sent to TTS, and how many of them started playing: if the user cuts
        # Flay off, only those were heard.
        synthesized: list[str] = []
        heard = 0
        # Sentences count for model requests (flay.llm.gate) only while they are voiced.
        voiced = False

        splitter = SentenceSplitter()

        def emit(text: str, check_claims: bool = False, new_round: bool = False) -> None:
            if new_round and parts and not parts[-1][-1:].isspace():
                # Text of a new round continues the previous one.
                text = " " + text
            parts.append(text)
            self.events.publish(AssistantDelta(text))
            speak(splitter.feed(text), check_claims)

        def speak(complete: Iterable[str], check_claims: bool = False) -> None:
            for sentence in complete:
                tool = claimed_tool(sentence) if check_claims else None
                if tool is not None and tool in self._tools and tool not in (exclude or ()):
                    raise UnbackedClaimError(tool)
                timings.mark("first_sentence")
                if voiced:
                    synthesis.queued()
                sentences.put_nowait(sentence)

        async def converse(conversation: list[Message], check_claims: bool) -> None:
            # Tool calls Flay already made for this request (see _route_tool) follow the
            # request; the model must not repeat them. Earlier requests' actions are in the
            # history too, but they do not rule anything out now.
            last_request = max(
                (i for i, m in enumerate(conversation) if m.role is Role.USER), default=0
            )
            done = {call.name for m in conversation[last_request:] for call in m.tool_calls}
            # Actions the routing form ruled out for this request.
            done |= exclude or set()
            # The model may call tools, see their results and continue; the last round
            # offers no tools so it has to answer in words.
            for round_ in range(MAX_TOOL_ROUNDS + 1):
                specs = (
                    [s for s in self._tools.specs if s.name not in done]
                    if round_ < MAX_TOOL_ROUNDS
                    else []
                )
                round_text: list[str] = []
                calls: list[ToolCall] = []
                stream = (
                    self._llm.stream(conversation, specs, think=True)
                    if think
                    else self._llm.stream(conversation, specs)
                )
                async for event in stream:
                    if isinstance(event, ToolCall):
                        calls.append(event)
                        continue
                    timings.mark("llm_first_token")
                    emit(event.text, check_claims, new_round=not round_text)
                    round_text.append(event.text)
                if not calls:
                    # The last sentence only comes out of the splitter now – a reply that
                    # is a single made-up "Ich schließe das Projekt …" must be checked too.
                    speak(splitter.flush(), check_claims)
                    break
                check_claims = False
                for call in calls:
                    # Slow tools (reading a page) say what they do before they start.
                    announcement = getattr(self._tools.get(call.name), "announcement", None)
                    if callable(announcement) and (said := announcement(call.arguments)):
                        emit(said.get(language) or next(iter(said.values())), new_round=True)
                        speak(splitter.flush())
                conversation.append(
                    Message(
                        Role.ASSISTANT,
                        (TextPart("".join(round_text)),),
                        language,
                        tool_calls=tuple(calls),
                    )
                )
                messages, results = await self._run_tools(calls)
                if (say := fixed_reply(results, language)) is not None:
                    # The tools said what happened; another model round would only
                    # paraphrase it – slower and sometimes wrong.
                    emit(say, new_round=True)
                    return
                conversation.extend(messages)

        def progress(text: str) -> None:
            emit(text, new_round=True)
            speak(splitter.flush())

        async def generate() -> None:
            nonlocal finished, fixed, prelude
            try:
                if announce:
                    emit(announce)
                    speak(splitter.flush())
                if work is not None:
                    prelude, fixed = await work(progress)
                if fixed is not None:
                    emit(fixed, new_round=bool(announce))
                else:
                    await answer_with_model()
                speak(splitter.flush())
                timings.mark("llm_done")
                finished = True
            finally:
                sentences.put_nowait(None)

        async def answer_with_model() -> None:
            base = build_messages(
                self._settings.persona,
                self._memory.context(),
                user_message,
                language,
                self._now(),
                tools=bool(self._tools),
                state=state,
                memories=memories,
                examples=examples,
            )
            if think:
                # Thinking takes seconds; say so right away instead of falling silent.
                emit(THINKING[language])
                speak(splitter.flush())
            try:
                await converse([*base, *(prelude or [])], check_claims=not prelude)
            except UnbackedClaimError as claim:
                # Made-up success: do what was claimed and answer from the real result.
                log.info("Antwort behauptete %s ohne Werkzeug – führe es aus", claim.tool)
                parts.clear()
                list(splitter.flush())  # the unspoken rest of the made-up reply
                routed, results = await self._route_tool(claim.tool, user_message, language, state)
                timings.mark("tool")
                if (say := fixed_reply(results, language)) is not None:
                    emit(say)
                    return
                if not routed:
                    # A note from the user's side: many chat templates (e.g. Qwen3.5)
                    # reject system messages after the start of the conversation.
                    routed = [
                        Message.from_text(
                            Role.USER,
                            "(Note: no action was performed for this request. Do not claim "
                            "otherwise; ask me what exactly to do.)",
                        )
                    ]
                await converse([*base, *routed], check_claims=False)

        async def synthesize() -> None:
            nonlocal voiced
            try:
                while (sentence := await sentences.get()) is not None:
                    synthesized.append(sentence)
                    async for chunk in self._tts.synthesize(sentence, language):
                        audio.put_nowait((len(synthesized), chunk))
                    synthesis.synthesized()
            finally:
                voiced = False
                synthesis.reset()
                audio.put_nowait(None)

        async def chunks() -> AsyncIterator[AudioBuffer]:
            nonlocal heard
            item: tuple[int, AudioBuffer] | None = None
            wait = filler.remaining() if filler is not None else None
            try:
                item = await asyncio.wait_for(audio.get(), wait)
            except TimeoutError:
                # Still nothing to say: a filler first, so the user knows Flay is on it.
                assert filler is not None
                for chunk in await filler.take():
                    self._set_state(AssistantState.SPEAKING)
                    timings.mark("filler")
                    yield chunk
                item = await audio.get()
            while item is not None:
                if self.state is not AssistantState.SPEAKING:
                    self._set_state(AssistantState.SPEAKING)
                timings.mark("first_audio")
                heard = max(heard, item[0])
                yield item[1]
                item = await audio.get()

        try:
            async with asyncio.TaskGroup() as group:
                generating = group.create_task(generate())
                if self._speaker_muted:
                    delivered = await _finished(generating)
                    return delivered
                voiced = True
                synthesizing = group.create_task(synthesize())
                if filler is not None:
                    await filler.hand_over()
                if await self._output.play(chunks()):
                    delivered = True
                    return True
                synthesizing.cancel()
                if self._speaker_muted:
                    # Muted while speaking: the reply continues as text.
                    delivered = await _finished(generating)
                    return delivered
                generating.cancel()
                return False
        finally:
            text = "".join(parts).strip()
            if text:
                self.events.publish(AssistantMessage(text, language, interrupted=not finished))
            if delivered or self._speaker_muted:
                # Read as text (muted): it got as far as it was generated.
                self._remember(user_message, text, language, interrupted=not delivered)
            else:
                said = " ".join(synthesized[:heard]).strip()
                self._remember(user_message, said, language, interrupted=True)

    def _remember(
        self, user_message: Message, reply: str, language: Language, interrupted: bool
    ) -> None:
        actions = tuple(self._actions)
        if not reply and not interrupted and not actions:
            return
        if reply:
            note = " [unterbrochen]" if interrupted else ""
            log.info("%s: %s%s", self._settings.persona.name, reply, note)
        # Only the words are remembered; screenshots are not sent again later.
        self._memory.record(
            Message.from_text(Role.USER, user_message.text, language),
            Message(
                Role.ASSISTANT,
                (TextPart(reply),),
                language,
                interrupted=interrupted,
                actions=actions,
            ),
        )

    async def _recalled(self, task: asyncio.Task[Any] | None) -> tuple[Fact, ...]:
        """The memories found for this request, if the search is done in time."""
        if task is None:
            return ()
        timeout = self._settings.memory.recall.timeout_ms / 1000
        try:
            found = await asyncio.wait_for(asyncio.shield(task), timeout)
        except Exception as exc:
            log.debug("Erinnerungen nicht rechtzeitig gefunden: %s", exc or type(exc).__name__)
            return ()
        if found:
            log.debug("Erinnerungen: %s", " | ".join(fact.text for fact in found))
        return tuple(found)

    async def _web(
        self,
        user_message: Message,
        text: str,
        tool: str,
        language: Language,
        timings: TurnTimings,
        filler: TurnFiller | None,
        state: str | None,
    ) -> bool:
        """Open a web page, or work on one and answer or hand the result to Claude.

        Done here rather than left to the model: the chain open → read (screenshots) →
        scroll/click until found → Claude has to happen in this order every time, and the
        model would otherwise type the bare link into Claude. Returns False if the request
        is not for a web page after all ("diese Seite" with no browser in front).
        """
        url = await self._find_url(text)
        current = url is None and tool == "read_webpage" and bool(_THIS_PAGE.search(text))
        if current and not _WEB_WORDS.search(text) and not await self._browser_in_front():
            return False
        if url is None and not current:
            reply = speech(
                "Welchen Link soll ich öffnen? Du kannst ihn auch kopieren und sagen: "
                "der Link aus der Zwischenablage.",
                "Which link should I open? You can also copy it and say: the link from "
                "the clipboard.",
            )
            await self._respond(
                user_message, language, timings, filler=filler, fixed=reply[language]
            )
            return True
        if tool == "open_url" and url is not None:
            prelude, results = await self._act(ToolCall("web", "open_url", {"url": url}), language)
            timings.mark("tool")
            await self._respond(
                user_message, language, timings, prelude, filler, fixed_reply(results, language)
            )
            return True
        for_claude = to_claude(text) and "send_to_claude" in self._tools

        # "Vergleiche python.org und ruby-lang.org": the first page is read as usual, the
        # others alongside.
        others = [u for u in find_urls(text) if u != url] if url else []

        async def read(say: Callable[[str], None]) -> tuple[list[Message], str | None]:
            try:
                return await work(say)
            finally:
                # Tabs opened in the background for this request are closed again.
                finish = getattr(self._tools.get("read_webpage"), "finish", None)
                if callable(finish):
                    await finish()

        async def work(say: Callable[[str], None]) -> tuple[list[Message], str | None]:
            arguments = {"url": url} if url else {}
            prelude, results = await self._act(ToolCall("web", "read_webpage", arguments), language)
            timings.mark("tool")
            if not results or not all(result.ok for result in results):
                return prelude, fixed_reply(results, language)
            finding = await self._browse(text, url, results[0], language, say, for_claude, others)
            if not finding.text:
                unread = speech(
                    "Ich konnte auf der Seite nichts lesen.",
                    "I couldn't read anything on the page.",
                )
                return [], unread[language]
            if not for_claude:
                return [], finding.text
            source = speech("Quelle", "Source")[language]
            page = finding.url or url or self._page_title() or ""
            instruction = f"{finding.text} ({source}: {page})" if page else finding.text
            _, sent = await self._act(
                ToolCall("web-claude", "send_to_claude", {"instruction": instruction}), language
            )
            return [], fixed_reply(sent, language)

        await self._respond(
            user_message,
            language,
            timings,
            filler=filler,
            state=state,
            announce=(READING if url else READING_OPEN)[language],
            work=read,
        )
        return True

    async def _file(
        self,
        user_message: Message,
        text: str,
        language: Language,
        timings: TurnTimings,
        filler: TurnFiller | None,
        state: str | None,
    ) -> None:
        """Find and read a local file, then answer or hand the result to Claude."""
        for_claude = to_claude(text) and "send_to_claude" in self._tools
        reader = self._tools.get("read_file")

        async def work(say: Callable[[str], None]) -> tuple[list[Message], str | None]:
            # The model only turns the request into a file name and a place.
            prelude, results = await self._route_tool("read_file", user_message, language, state)
            timings.mark("tool")
            if not results or not all(result.ok for result in results):
                missing = speech("Welche Datei soll ich lesen?", "Which file should I read?")[
                    language
                ]
                return prelude, fixed_reply(results, language) or missing
            if getattr(reader, "page", None) is None:
                return prelude, None  # the model answers from what the tool reported
            path = getattr(reader, "path", None)
            if isinstance(path, Path):
                say(speech(f"Ich lese {path.name}.", f"I'm reading {path.name}.")[language])
            finding = await browse_text(
                self._llm,
                text,
                reader,  # type: ignore[arg-type]
                language,
                say,
                for_claude=for_claude,
                rules=self._rules(),
                max_steps=self._settings.tools.web.max_steps,
            )
            return [], await self._deliver(finding, for_claude, str(path or ""), language)

        await self._respond(
            user_message,
            language,
            timings,
            filler=filler,
            state=state,
            announce=speech("Ich suche die Datei.", "I'm looking for the file.")[language],
            work=work,
        )

    async def _deliver(
        self, finding: Finding, for_claude: bool, source: str, language: Language
    ) -> str | None:
        """What was found: said, or typed into Claude with its source."""
        if not finding.text:
            return speech("Ich konnte darin nichts lesen.", "I couldn't read anything in it.")[
                language
            ]
        if not for_claude:
            return finding.text
        label = speech("Quelle", "Source")[language]
        instruction = f"{finding.text} ({label}: {source})" if source else finding.text
        _, sent = await self._act(
            ToolCall("file-claude", "send_to_claude", {"instruction": instruction}), language
        )
        return fixed_reply(sent, language)

    async def _browser_in_front(self) -> bool:
        from flay.tools.web import front_browser

        try:
            return await asyncio.to_thread(front_browser) is not None
        except Exception as exc:  # no window list (permissions, headless)
            log.debug("Browserfenster nicht ermittelbar: %s", exc)
            return False

    def _page_title(self) -> str | None:
        reader = self._tools.get("read_webpage")
        page = getattr(reader, "page", None)
        if isinstance(page, Page) and page.url:
            return page.url
        window = getattr(reader, "window", None)
        return window.title if isinstance(window, Window) and window.title else None

    async def _browse(
        self,
        request: str,
        url: str | None,
        seen: ToolResult,
        language: Language,
        say: Callable[[str], None],
        for_claude: bool,
        others: Sequence[str] = (),
    ) -> Finding:
        """Follow links, scroll or click on the page that was read until it has the answer."""
        reader = self._tools.get("read_webpage")
        if getattr(reader, "page", None) is not None:
            loaded: list[Page] = []
            load = getattr(reader, "load", None)
            for other in others:
                if callable(load) and (page := await load(other)) is not None:
                    loaded.append(page)
                else:
                    log.info("Vergleichsseite nicht lesbar: %s", other)
            return await browse_text(
                self._llm,
                request,
                reader,  # type: ignore[arg-type]
                language,
                say,
                for_claude=for_claude,
                rules=self._rules(),
                max_steps=self._settings.tools.web.max_steps,
                others=loaded,
            )
        window = getattr(reader, "window", None)
        if not isinstance(window, Window):
            window = None
        max_size = self._settings.tools.screen.max_size

        async def act(call: ToolCall) -> ToolResult:
            result = await self._execute(call)
            if result.confirm is not None:
                # Not asked mid-task; the model is told to go on without it.
                self._pending = None
            return result

        async def look() -> ImagePart | None:
            return await look_at_page(max_size)

        return await browse(
            self._llm,
            request,
            url or (window.title if window else "the page open in the browser"),
            seen,
            language,
            self._tools.specs,
            act,
            say,
            for_claude=for_claude,
            look=look if window is not None else None,
            window=window.app if window is not None else None,
            rules=self._rules(),
            max_steps=self._settings.tools.web.max_steps,
        )

    async def _find_url(self, text: str) -> str | None:
        """The address the user means: said or typed, copied, or in Claude's answer."""
        if url := find_url(text):
            return url
        if wants_clipboard(text):
            return await asyncio.to_thread(clipboard_url)
        screen_text = getattr(self._tools.get("claude_status"), "screen_text", None)
        if _CLAUDE.search(text) and not _THIS_PAGE.search(text) and callable(screen_text):
            # "Öffne den Link aus Claudes Antwort"
            return last_url(await screen_text() or "")
        return None

    async def _current_state(self) -> str | None:
        """What is open right now, if it was read in time; never holds up the answer."""
        task, self._state_task = self._state_task, None
        if task is None:
            return None
        try:
            return await asyncio.wait_for(task, STATE_TIMEOUT)
        except Exception as exc:
            log.debug("Zustand nicht rechtzeitig gelesen: %s", exc or type(exc).__name__)
            return None

    async def _attach_screenshot(self, message: Message) -> tuple[Message, ToolResult]:
        """Add a screenshot when the user talks about the screen.

        Models do not reliably choose the screen tool on their own ("look at my screen" was
        sometimes only answered with words), so explicit mentions do not depend on it.
        """
        list_apps = getattr(self._tools.get(SCREEN_TOOL), "apps", None)
        apps: list[str] = await asyncio.to_thread(list_apps) if callable(list_apps) else []
        # Which part of the screen ("the PhpStorm window", "second monitor") is read from the
        # words directly: no extra model call, so no extra waiting.
        call = ToolCall("auto_screenshot", SCREEN_TOOL, screen_arguments(message.text, apps))
        result = await self._execute(call)
        if not result.images:
            note = f"\n(Screenshot failed: {result.text})"
            return replace(message, parts=(*message.parts, TextPart(note))), result
        return replace(message, parts=(*message.parts, *result.images)), result

    def _intent(self, text: str) -> tuple[str | None, str]:
        """Tool to force for this request, and the request text the tool call is made from.

        A follow-up like "prüf nochmal, ob du es schließen kannst" names no project; without
        a tool call the model would just claim success, so it is routed with the previous
        project request as context.
        """
        intent = tool_intent(text)
        now = time.monotonic()
        self._by_learned_verb = False
        if intent in _PROJECT_TOOLS:
            self._project_request = (text, now)
            return intent, text
        if intent is not None:
            self._project_request = None
            return intent, text
        if (learned := self._learned_verb(text)) is not None:
            self._project_request = (learned[1], now)
            self._by_learned_verb = True
            return learned
        if self._project_request is not None:
            previous, at = self._project_request
            follow_up = project_follow_up(text)
            if follow_up is not None and now - at <= FOLLOW_UP_SECONDS:
                return follow_up, f"{previous}\n{text}"
        return None, text

    def _learned_verb(self, text: str) -> tuple[str, str] | None:
        """A verb learned from earlier requests with a project name: "She K NVS",
        "Mach K NVS zu" (see flay.verbs)."""
        verb = getattr(self._memory, "verb", None)
        if not callable(verb) or self._projects is None:
            return None
        for word, rest in candidates(text):
            tool = verb(word)
            if tool is not None and any(mention(name, rest) for name in self._projects()):
                log.info("Gelerntes Verb: „%s“ → %s", word, tool)
                return tool, f"{CANONICAL[tool]} {rest}"
        return None

    def _rules(self) -> str:
        important = getattr(self._memory.context(), "important", ())
        if not important:
            return ""
        listed = "\n".join(f"- {fact.text}" for fact in important)
        return f"\nRules the user set – follow them:\n{listed}"

    def _routable_projects(self, text: str, intent: str | None) -> list[str]:
        """The project names to route with – if this request should go through the form.

        It does when a rule recognized an action, or when the user names one of the
        projects (with a garbled verb, "She K Command", no rule matches). Plain talk
        skips it, so its first words come as fast as before.
        """
        if (
            not self._settings.llm.structured_routing
            or self._projects is None
            or not isinstance(self._llm, StructuredChat)
        ):
            return []
        if intent is None or intent not in routing.ACTIONS:
            return []
        return self._projects()

    async def _route(
        self,
        request: str,
        language: Language,
        state: str | None,
        projects: list[str],
        intent: str,
    ) -> tuple[bool, ToolCall | None]:
        """Fill in the routing form; (False, None) if the server could not."""
        assert isinstance(self._llm, StructuredChat)
        actions = [intent]
        previous = [m for m in self.history if m.role in (Role.USER, Role.ASSISTANT)][-2:]
        try:
            answer = await self._llm.complete_json(
                routing.messages(request, language, actions, projects, state, previous),
                routing.schema(actions, projects),
                name="route",
            )
        except Exception as exc:
            log.warning("Weiterleitung per Formular fehlgeschlagen: %s", exc)
            return False, None
        log.debug("Weiterleitung: %s", answer)
        return True, routing.to_call(answer)

    async def _act(
        self, call: ToolCall, language: Language
    ) -> tuple[list[Message], list[ToolResult]]:
        assistant = Message(Role.ASSISTANT, (TextPart(""),), language, tool_calls=(call,))
        messages, results = await self._run_tools([call])
        return [assistant, *messages], results

    async def _route_tool(
        self,
        name: str,
        user_message: Message,
        language: Language,
        state: str | None = None,
        examples: Sequence[Fact] = (),
    ) -> tuple[list[Message], list[ToolResult]]:
        """Let the model fill in one specific tool call, in a fresh context.

        The conversation history can talk the model out of acting (earlier replies that only
        *claimed* to act get imitated), and Ollama ignores ``tool_choice``. A minimal prompt
        with just this tool and the request makes the call reliable.
        """
        spec = next(s for s in self._tools.specs if s.name == name)
        router = Message.from_text(
            Role.SYSTEM,
            f"Turn the user's request into exactly one call of the tool '{name}'. Write any "
            f"instruction or fact as a complete, precise sentence in "
            f"{LANGUAGE_NAMES[language]}. Only name a project if the user names one."
            # Lets the model write the exact name of the project the user means.
            + (f"\n{state}" if state else "")
            # E.g. "never tell Claude to commit" must shape the instruction typed into Claude.
            + self._rules()
            + (f"\n{shown}" if (shown := format_examples(examples)) else ""),
        )
        request = Message.from_text(Role.USER, user_message.text, language)
        calls = [
            event
            async for event in self._llm.stream([router, request], [spec])
            if isinstance(event, ToolCall) and event.name == name
        ]
        if not calls:
            log.debug("Werkzeug %s wurde nicht erzwungen: Modell lieferte keinen Aufruf", name)
            return [], []
        return await self._act(calls[0], language)

    async def _run_tools(self, calls: list[ToolCall]) -> tuple[list[Message], list[ToolResult]]:
        """Execute tool calls; returns the result messages for the conversation."""
        messages: list[Message] = []
        results: list[ToolResult] = []
        images: list[ImagePart] = []
        for call in calls:
            result = await self._execute(call)
            results.append(result)
            messages.append(Message(Role.TOOL, (TextPart(result.text),), tool_call_id=call.id))
            images.extend(result.images)
        if images:
            tools = ", ".join(
                dict.fromkeys(c.name for c, r in zip(calls, results, strict=True) if r.images)
            )
            # Tool messages carry text only in the chat API; images follow as a user turn.
            messages.append(Message(Role.USER, (TextPart(f"Screenshots from {tools}:"), *images)))
        return messages, results

    async def _execute(self, call: ToolCall) -> ToolResult:
        call, refusal = self._check_project(call)
        result = refusal or await self._tools.execute(call)
        if result.ok and call.name in LEARNABLE and call.id != CONFIRMED:
            # Becomes an example if the next request does not say it was wrong.
            self._to_learn.append((self._request, call))
            if (tried := self._tried_before(call)) is not None:
                log.info("Vorher nicht verstanden: „%s“ meinte dasselbe", tried)
                self._to_learn.append((tried, call))
        self.events.publish(
            ToolUsed(call.name, call.arguments, result.text, result.ok, result.action)
        )
        if call.name == "workflow":
            self.events.publish(RecordingChanged(self.workflow_recording))
        # What happened stays in the conversation, so later turns do not have to guess.
        self._actions.append(ToolOutcome(call, result.text, result.ok))
        if result.confirm is not None:
            # The tool asks first ("Soll ich wirklich …?"); a yes runs this call.
            self._pending = (result.confirm, time.monotonic(), self._request)
        return result

    def _tried_before(self, call: ToolCall) -> str | None:
        """The request just before, if it was this one not understood ("Sta flay").

        Only for opening and closing projects, and only if that request was short, came
        just before and named the same project; it is learned (or rejected) with this one.
        """
        if self._unanswered is None or call.name not in VERB_TOOLS:
            return None
        earlier, at = self._unanswered
        self._unanswered = None
        name = str(call.arguments.get("name", ""))
        if (
            time.monotonic() - at > RETRY_SECONDS
            or not name
            or len(earlier.split()) > MAX_WORDS
            or _compact_words(earlier) == _compact_words(self._request)
            or cancelled(earlier)
            or _CORRECTION.search(earlier)
            or not wants_examples(earlier)
            or mention(name, earlier) is None
        ):
            return None
        return earlier

    def _turn_task(self, work: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(work)
        self._turn_tasks.append(task)
        return task

    def _background(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _settle_examples(self, text: str) -> None:
        """Last turn's actions become examples – unless this request says they were wrong."""
        candidates, self._to_learn = self._to_learn, []
        learn = getattr(self._memory, "learn", None)
        unlearn = getattr(self._memory, "unlearn", None)
        if not callable(learn) or not callable(unlearn):
            return
        wrong = bool(_CORRECTION.search(text))
        for request, call in candidates:
            if wrong:
                unlearn(request, call)
            else:
                learn(request, call)

    def _known_action(self, text: str) -> ToolCall | None:
        known_action = getattr(self._memory, "known_action", None)
        found = known_action(text) if callable(known_action) else None
        if found is None:
            return None
        call, confirmed = found
        # Confirmed with "ja" before: no need to ask again.
        return ToolCall(CONFIRMED, call.name, call.arguments) if confirmed else call

    async def _confirmation(self, text: str, language: Language) -> str | None:
        """The reply if ``text`` answers a pending "Meinst du …?"; None otherwise."""
        pending, self._pending = self._pending, None
        if pending is None or time.monotonic() - pending[1] > CONFIRM_SECONDS:
            return None
        if len(text.split()) <= 6 and _YES.search(text):
            _, results = await self._act(pending[0], language)
            if all(result.ok for result in results):
                learn = getattr(self._memory, "learn", None)
                if callable(learn):
                    learn(pending[2], pending[0], confirmed=True)
            return fixed_reply(results, language)
        if _NO.search(text):
            return CANCELLED[language]
        # Something else: the question is dropped, the new request counts.
        return None

    def _check_project(self, call: ToolCall) -> tuple[ToolCall, ToolResult | None]:
        """Keep the model from acting on a project the user never named.

        With the open projects in view, a model sometimes simply picks one of them
        ("Mach mal Moewe auf" opened project-symfony-kx-ewo). A required project that
        does not appear in the user's words stops the call with a question; an optional
        one is dropped, so the tool falls back to its default (e.g. the Claude in view).
        """
        if call.id == CONFIRMED:
            return call, None
        if call.name == "ssh_connect":
            return call, self._check_host(call)
        argument, required = _PROJECT_ARGUMENTS.get(call.name, ("", False))
        value = str(call.arguments.get(argument) or "") if argument else ""
        if not value:
            return call, None
        # Judged by the project the name stands for: "KEWO" is literally in "Schließe
        # KEWO", but "project-symfony-kx-ewo" only roughly.
        name = self._resolve(value)
        how = mention(name, self._heard)
        if how == "fuzzy" and call.name == "close_project":
            return call, self._ask_first(call, name)
        if how is not None:
            return call, None
        log.info("%s: Projekt „%s“ wurde nicht genannt", call.name, value)
        if not required:
            arguments = {k: v for k, v in call.arguments.items() if k != argument}
            return ToolCall(call.id, call.name, arguments), None
        return call, ToolResult(
            f"Nothing was done: the user did not mention a project like '{value}'. Ask "
            "which project they mean.",
            ok=False,
            action=f"Nicht ausgeführt: Projekt „{value}“ wurde nicht genannt",
            say=speech("Welches Projekt meinst du?", "Which project do you mean?"),
        )

    def _check_host(self, call: ToolCall) -> ToolResult | None:
        """A server the user never named is not connected to (the model may pick one)."""
        value = str(call.arguments.get("host") or "")
        if not value:
            return None
        tool = self._tools.get(call.name)
        find = getattr(tool, "find", None)
        hosts = find(value) if callable(find) else []
        names = [value, *(h.alias for h in hosts), *(h.hostname for h in hosts if h.hostname)]
        heard = _compact_words(self._heard)
        if any(_compact_words(n) in heard or mention(n, self._heard) for n in names):
            return None
        log.info("ssh_connect: Server „%s“ wurde nicht genannt", value)
        return ToolResult(
            f"Nothing was done: the user did not name a server like '{value}'. Ask which.",
            ok=False,
            action=f"Nicht ausgeführt: Server „{value}“ wurde nicht genannt",
            say=speech("Mit welchem Server?", "Which server?"),
        )

    def _resolve(self, value: str) -> str:
        """The full project name ``value`` stands for, if exactly one fits."""
        names = self._projects() if self._projects else []
        matches = match_projects(value, [Project(name, Path(name)) for name in names])
        return matches[0].name if len(matches) == 1 else value

    def _ask_first(self, call: ToolCall, name: str) -> ToolResult:
        """Closing a project the user's words only roughly name: ask before doing it."""
        self._pending = (
            ToolCall(CONFIRMED, call.name, {"name": name}),
            time.monotonic(),
            self._request,
        )
        log.info("%s: „%s“ nur ungefähr genannt – frage nach", call.name, name)
        return ToolResult(
            f"Nothing was done yet: asked the user to confirm closing '{name}'.",
            ok=False,
            action=f"PhpStorm: Rückfrage vor dem Schließen von „{name}“",
            say=speech(f"Meinst du {spoken(name)}?", f"Do you mean {spoken(name)}?"),
        )

    def _set_state(self, state: AssistantState) -> None:
        self.state = state
        self.events.publish(StateChanged(state))

    def _notice(self, text: str) -> None:
        log.info(text)
        self.events.publish(Notice(text))

    async def _cancel_turn(self) -> None:
        if self._turn is None or self._turn.done():
            return
        self._output.stop()
        self._turn.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._turn


async def _finished(task: asyncio.Task[None]) -> bool:
    await task
    return True


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        log.error("Fehler bei der Verarbeitung: %s", exc, exc_info=exc)


def _compact_words(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())
