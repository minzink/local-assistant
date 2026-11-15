"""Orchestrator: utterance → STT → LLM → TTS → speaker, with per-stage timings."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from cassi import perf, routing
from cassi.audio.base import AudioBuffer, AudioOutput
from cassi.audio.vad import VoiceActivityDetector
from cassi.browsing import Finding, browse, browse_text
from cassi.classify import classify
from cassi.config import PersonaSettings, Settings
from cassi.core.events import (
    AssistantDelta,
    AssistantMessage,
    AssistantState,
    ContextCleared,
    EventBus,
    ListeningChanged,
    MuteChanged,
    Notice,
    RecordingChanged,
    StateChanged,
    ToolUsed,
    TurnProfiled,
    UserMessage,
    WakeWordChanged,
)
from cassi.core.language import Language
from cassi.core.messages import (
    ImagePart,
    Message,
    Modality,
    Role,
    TextPart,
    ToolCall,
    ToolOutcome,
)
from cassi.core.sentences import SentenceSplitter
from cassi.core.text_language import detect_text_language
from cassi.direct import direct_arguments
from cassi.filler import FillerVoice, TurnFiller
from cassi.input.base import UtteranceSource
from cassi.intents import (
    _CLAUDE,
    _CLAUDE_ALWAYS,
    _CLAUDE_NO,
    _CLAUDE_YES,
    _CORRECTION,
    _HOW_TO,
    _NO,
    _PROJECT_ARGUMENTS,
    _PROJECT_TOOLS,
    _THIS_PAGE,
    _WEB_WORDS,
    _YES,
    CANCELLED,
    COMPUTER_TOOLS,
    CONFIRM_SECONDS,
    CONFIRMED,
    FOLLOW_UP_SECONDS,
    LEARNABLE,
    LONG_REQUEST_WORDS,
    READING,
    READING_OPEN,
    RETRY_SECONDS,
    SCREEN_ACTIONS,
    SCREEN_TOOL,
    THINKING,
    _discard,
    asks_what_is_open,
    cancelled,
    claimed_tool,
    computer_intent,
    file_intent,
    fixed_reply,
    may_be_computer_task,
    mentions_clipboard,
    mentions_screen,
    mentions_selection,
    narrates,
    project_follow_up,
    to_claude,
    tool_intent,
    wants_examples,
    wants_thinking,
    web_intent,
)
from cassi.llm.base import ChatModel, StructuredChat, ToolSpec
from cassi.llm.gate import synthesis
from cassi.memory.base import Fact, Memory, ShortTermMemory
from cassi.planning import Planner
from cassi.prompt import LANGUAGE_NAMES, build_messages, format_examples
from cassi.repetition import RepeatGuard
from cassi.routines import Routines
from cassi.stt.base import SpeechToText, Transcript
from cassi.stt.vocabulary import Vocabulary, is_noise
from cassi.tools.base import ToolRegistry, ToolResult, speech, spoken
from cassi.tools.keyboard import (
    clipboard_text,
    describe_clipboard,
    describe_selection,
    selected_text,
)
from cassi.tools.pages import Page
from cassi.tools.phpstorm import Project, match_projects, mention
from cassi.tools.preferences import Preferences
from cassi.tools.screen import Window, screen_arguments
from cassi.tools.web import (
    clipboard_url,
    find_url,
    find_urls,
    last_url,
    look_at_page,
    wants_clipboard,
)
from cassi.tts.base import TextToSpeech
from cassi.verbs import CANONICAL, MAX_WORDS, VERB_TOOLS, candidates, heard_verb
from cassi.wake import FOLLOW_UP_SECONDS as WAKE_FOLLOW_UP_SECONDS
from cassi.wake import strip_wake_word, wake_pattern

log = logging.getLogger(__name__)

# Tools whose result the model talks about afterwards: routed with the conversation's own
# prompt start (Assistant._route_in_conversation), so that answer finds it cached. Measured
# with `cassi eval`: the answer after them came after 0.3 s instead of ~5 s; for actions with
# a fixed reply the longer routing prompt only cost ~0.5 s.
ANSWERED_TOOLS = frozenset(
    {"claude_status", "look_at_screen", "click_on_screen", "scroll_screen", "calendar"}
)  # fmt: skip
# How long after a turn, and how often, the prompt cache is checked (see _rewarm_when_idle).
REWARM_WATCH_SECONDS = 30.0
REWARM_CHECK_SECONDS = 0.5

MIN_UTTERANCE_SECONDS = 0.3
# Below this the recording is effectively mute (denied permission, wrong device, muted
# mic). Whisper would hallucinate words like "you" from it, so it is not transcribed.
SILENCE_DBFS = -60.0
# How long the answer may wait for the computer's state (open projects, Claude sessions)
# once speech recognition is done; it is read in parallel and usually ready by then.
STATE_TIMEOUT = 0.3
# Tool round trips per turn, e.g. read a web page, scroll on, then type into Claude.
MAX_TOOL_ROUNDS = 5


@dataclass
class TurnTimings:
    clock: Callable[[], float]
    start: float = 0.0
    marks: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start = self.clock()

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, self.clock() - self.start)
        perf.mark(name)

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
        preferences: Preferences | None = None,
        routines: Routines | None = None,
    ) -> None:
        # Plans with a name, set up by voice ("Merk dir als Feierabend: …").
        self._routines = routines
        self._settings = settings
        self._wake_spoken = wake_pattern(settings.input.continuous.wake_pattern)
        # Changed by voice ("Antworte kürzer"): the answer length goes into the prompt.
        self._preferences = preferences
        # Listen for "Hey Cassi" while listening continuously (switchable in UI and tray).
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
        self._overview_task: asyncio.Task[str | None] | None = None
        self.events = events or EventBus()
        self._turn: asyncio.Task[None] | None = None
        # Refills the server's prompt cache after a turn (see _rewarm_when_idle).
        self._rewarm: asyncio.Task[None] | None = None
        # Plans of several steps and routines (cassi.planning).
        self._planner = Planner(self)
        # Things to say on Cassi's own initiative, and the task that says them when idle.
        self._announcement_queue: list[Mapping[Language, str]] = []
        self._announcer: asyncio.Task[None] | None = None
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
        # The replies before this turn, read once per turn in _answer (cassi.repetition).
        self._earlier_replies: list[str] = []
        # Until when an utterance without "Hey Cassi" counts (after a bare "Hey Cassi").
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
    def awaiting_answer(self) -> bool:
        """Cassi asked a yes/no question: a tool call or a plan waits for the answer."""
        return self._pending is not None or self._planner.pending is not None

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
        """Whether continuous listening waits for "Hey Cassi"; None if the model is missing."""
        return self._wake_word if self._wake_word_available else None

    def set_wake_word(self, enabled: bool) -> None:
        if not self._wake_word_available:
            self._notice("Das Aktivierungswort fehlt – bitte einmal 'cassi setup' ausführen.")
            return
        if enabled == self._wake_word:
            return
        self._wake_word = enabled
        self._wake_follow_up = 0.0
        log.info("Aktivierungswort „Hey Cassi“ %s", "an" if enabled else "aus")
        # The continuous listener is built for one mode; a running one starts anew.
        restart = getattr(self._source, "restart_continuous", None)
        if self.continuous and callable(restart):
            restart()
        self.events.publish(WakeWordChanged(enabled))

    def set_system_prompt(self, text: str) -> None:
        """From the next request on (an adapter learned something, see Adapter.run)."""
        persona = self._settings.persona.model_copy(update={"system_prompt": text})
        self._settings = self._settings.model_copy(update={"persona": persona})
        log.info("System-Prompt aktualisiert (%d Zeichen)", len(text))

    def new_session(self, reason: str = "") -> None:
        """Start a new conversation: what was said leaves the context (learned facts stay),
        and a question that waited for "yes" is dropped – a "yes" now must not run it."""
        self._memory.new_session()
        self._earlier_replies = []
        self._pending = None
        self._planner.pending = None
        self._wake_follow_up = 0.0
        log.info("Neues Gespräch%s", f" ({reason})" if reason else "")
        self.events.publish(ContextCleared(reason))

    @property
    def name(self) -> str:
        """The name shown in the chat and the tray (an adapter like Flay sets its own)."""
        return self._settings.persona.name

    @property
    def wake_phrase(self) -> str:
        return self._settings.input.continuous.wake_phrase

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
                        # Someone nearby: Cassi's current answer goes on undisturbed.
                        continue
                await self._start_turn(self.handle(utterance, recognized))
            # Source exhausted: let the last answer finish.
            if self._turn is not None:
                await asyncio.wait([self._turn])
        finally:
            keep_warm.cancel()
            await self._cancel_turn()

    def announce(self, say: Mapping[Language, str]) -> None:
        """Say something on Cassi's own initiative (a timer, Claude is done) once idle."""
        self._announcement_queue.append(say)
        if self._announcer is None or self._announcer.done():
            self._announcer = asyncio.get_running_loop().create_task(self._announce_when_idle())

    async def wait_until_idle(self, poll: float = 0.2) -> None:
        """Until no turn runs, nothing is spoken and no workflow is being recorded."""
        while (
            (self._turn is not None and not self._turn.done())
            or self.state is not AssistantState.IDLE
            or self.workflow_recording is not None
            or self._announcement_queue
        ):
            await asyncio.sleep(poll)

    async def _announce_when_idle(self) -> None:
        while self._announcement_queue:
            # Never into an answer: wait until the current turn is over.
            while (self._turn is not None and not self._turn.done()) or (
                self.state is not AssistantState.IDLE
            ):
                await asyncio.sleep(0.2)
            say = self._announcement_queue.pop(0)
            text = say.get(self._language) or next(iter(say.values()))
            await self._start_turn(self._speak_announcement(text))
            if self._turn is not None:
                await asyncio.wait([self._turn])

    async def _speak_announcement(self, text: str) -> None:
        language = self._language
        self._notice(text)
        self.events.publish(AssistantMessage(text, language))
        if self._speaker_muted:
            return
        self._set_state(AssistantState.SPEAKING)
        try:

            async def chunks() -> AsyncIterator[AudioBuffer]:
                async for chunk in self._tts.synthesize(text, language):
                    yield chunk

            await self._output.play(chunks())
        finally:
            self._set_state(AssistantState.IDLE)

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
        if self._rewarm is not None:
            self._rewarm.cancel()  # the new request goes first
        self._memory.turn_started()
        # Read in parallel to speech recognition, so it costs no time.
        self._state_task = asyncio.create_task(self._state()) if self._state else None
        # The window list too: screen requests need it, and it costs ~0.3 s to read.
        self._overview_task = (
            asyncio.create_task(self._read_window_overview())
            if SCREEN_TOOL in self._tools
            else None
        )
        try:
            await turn
        finally:
            if self._state_task is not None:
                self._state_task.cancel()
            if self._overview_task is not None:
                self._overview_task.cancel()
            # Searches a fixed reply did not need must not outlive the turn.
            for task in self._turn_tasks:
                task.cancel()
            self._turn_tasks.clear()
            self._memory.turn_finished()
            self._set_state(AssistantState.IDLE)
            self._last_activity = time.monotonic()
            if self._settings.llm.rewarm:
                self._rewarm = asyncio.create_task(self._rewarm_when_idle())

    async def _rewarm_when_idle(self) -> None:
        """Put the conversation's prompt start back into the server's cache.

        The server keeps one cached prompt start. Routing, classifying and the memory's
        upkeep (summaries, ~3 s after a turn) each push the conversation's ~2000 tokens out;
        the next answer would read them again (~5 s on an M2 Pro instead of ~0.7 s). While
        nothing else runs, Cassi sends that start once more; a new request cancels it.
        """
        deadline = time.monotonic() + REWARM_WATCH_SECONDS
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(REWARM_CHECK_SECONDS)
                if perf.SERVER.active or perf.cache_fits_conversation():
                    continue
                if self._turn is not None and not self._turn.done():
                    return
                log.debug("Gesprächsanfang wieder in den Cache gelegt")
                await self._warm()
        except asyncio.CancelledError:
            pass

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

    async def prewarm(self) -> None:
        """Put the conversation's prompt start into the server's cache (at the start)."""
        await self._warm()
        log.debug("Gesprächsanfang vorab in den Cache gelegt")

    async def _warm(self) -> None:
        """Send the usual prompt without generating: loads the model and caches the prompt."""
        warm = getattr(self._llm, "warm", None)
        if not callable(warm):
            return
        language = self._settings.language.default
        messages = build_messages(
            self._persona,
            self._memory.context(),
            Message.from_text(Role.USER, ".", language),
            language,
            self._now(),
            tools=bool(self._tools),
            verbatim_replies=self._settings.memory.verbatim_replies,
        )
        try:
            await warm(messages, self._conversation_specs)
            log.debug("LLM warm gehalten")
        except Exception as exc:
            log.debug("Warmhalten fehlgeschlagen: %s", exc)

    async def _handle_text(self, text: str) -> None:
        profile = perf.begin()
        timings = TurnTimings(self._clock)
        language = detect_text_language(text, self._settings.language.default)
        try:
            await self._answer(text, language, "text", timings)
        finally:
            self._finish_profile(profile)

    def _finish_profile(self, profile: perf.Profile) -> None:
        """Log where the time went and hand it to the history (`cassi stats --perf`)."""
        if not profile.marks and not profile.calls:
            return  # nothing was answered (noise, too short)
        profile.mark("end")
        log.info("Profil: %s", profile.summary())
        self.events.publish(TurnProfiled(profile.to_dict()))

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
                "und Eingabegerät prüfen, z.B. mit 'cassi mic-test'."
            )
            return

        self._set_state(AssistantState.THINKING)
        profile = perf.begin()
        profile.marks["recorded"] = round(audio.duration, 2)  # how long the user spoke
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
            # "Hey Cassi, öffne …": the wake word is not part of the request.
            command = strip_wake_word(text, self._wake_spoken)
            if command == text and self._needs_wake_word():
                # The detector only filters; without the wake word in the transcript it was
                # someone talking nearby.
                log.info("Ohne Aktivierungswort im Text – ignoriert: %s", text)
                return
            self._wake_follow_up = 0.0
            if command != text:
                if not command:
                    # Just "Hey Cassi": the next utterance is the command.
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
            self._finish_profile(profile)

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
            with perf.span("vad"):
                speech = await self._vad.trim(audio)
            if speech is None:
                return None
            log.debug("Sprache %.1fs von %.1fs", speech.duration, audio.duration)
            audio = speech
        hint = None if self._settings.language.auto_detect else self._settings.language.default
        with perf.span("stt"):
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
        if strip_wake_word(transcript.text, self._wake_spoken) == transcript.text:
            log.info("Ohne Aktivierungswort im Text – ignoriert: %s", transcript.text)
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
        history = self.history
        earlier = [m.text for m in history if m.role is Role.USER][-2:]
        self._heard = " ".join([*earlier, text])
        # For the repetition guard of the reply (one read of the memory per turn).
        self._earlier_replies = [m.text for m in history if m.role is Role.ASSISTANT]
        # Classifying an unusual computer request costs the model ~0.4 s: it runs while the
        # computer's state is read (PhpStorm, Claude sessions).
        classifying = self._turn_task(self._classified(text)) if self._may_classify(text) else None
        with perf.span("state"):
            state = await self._current_state()
        # Searching memories runs while the request is routed; it is collected at the end.
        recall = getattr(self._memory, "recall", None)
        recalling = self._turn_task(_timed("recall", recall(text))) if callable(recall) else None
        find_examples = getattr(self._memory, "examples", None)
        finding = (
            self._turn_task(find_examples(text))
            if callable(find_examples) and wants_examples(text)
            else None
        )
        examples: tuple[Fact, ...] = ()
        user_message = Message.from_text(Role.USER, text, language)
        if (plan := self._planner.confirmed_plan(text)) is not None:
            await self._planner.run_plan(user_message, plan, language, timings, filler, state)
            return
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
        if self._routines is not None and await self._planner.routine(
            user_message, text, language, timings, filler, state
        ):
            return
        if (claude_answer := self._claude_answer(text)) is not None:
            # "Ja" while Claude waits for a permission (and Cassi asked nothing itself).
            answered, results = await self._act(claude_answer, language)
            timings.mark("tool")
            await self._respond(
                user_message, language, timings, answered, filler,
                fixed=fixed_reply(results, language),
            )  # fmt: skip
            return
        if (steps := self._planner.plan_steps(text)) is not None:
            await self._planner.plan(user_message, steps, language, timings, filler, state)
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
        if intent is None and len(text.split()) > LONG_REQUEST_WORDS and not narrates(text):
            # A long, rambling request: its core decides what to do.
            with perf.span("core"):
                core = await self._core(text, language)
            if core is not None:
                intent, request = self._intent(core)
                if intent is None and (classified := await self._classified(core)) is not None:
                    intent, request = classified, core
        if classifying is not None:
            classified = await classifying if intent is None else None
            if classifying is not None and not classifying.done():
                classifying.cancel()
            if classified is not None:
                intent, request = classified, text
        overview_only = SCREEN_TOOL in self._tools and intent is None and asks_what_is_open(text)
        # "Klick im PhpStorm-Fenster auf Git": the click looks itself.
        screen = (
            SCREEN_TOOL in self._tools
            and mentions_screen(text)
            and intent not in SCREEN_ACTIONS
            and intent not in COMPUTER_TOOLS
            and not overview_only
            # "Ich habe gestern Slack auf den zweiten Bildschirm geschoben" shows nothing.
            and not narrates(text)
        )
        if "use_keyboard" in self._tools and mentions_selection(text):
            # "Übersetz das Markierte": the selection, copied from the front app. Sent with
            # the request, never stored.
            selection = describe_selection(await asyncio.to_thread(selected_text))
            state = f"{state} {selection}" if state else selection
        if mentions_clipboard(text):
            # "Fass zusammen, was ich kopiert habe": the clipboard, sent with the request.
            copied = describe_clipboard(await asyncio.to_thread(clipboard_text))
            state = f"{state} {copied}" if state else copied
        if screen or overview_only:
            # Which windows are where, by name: the model need not read titles from pixels,
            # and "the left monitor" means something. Sent with the request, never stored.
            overview = await self._window_overview()
            if overview:
                state = f"{state} {overview}" if state else overview
            elif overview_only:
                # No window list on this system: look at the screen as before.
                screen = mentions_screen(text) and intent not in SCREEN_ACTIONS
        if filler is not None:
            filler.action = screen or (intent is not None and intent in self._tools)
        # A reply that is fixed by what the tools reported, without asking the model.
        fixed: str | None = None
        prelude: list[Message] = []
        # A story names projects and verbs like a command, but asks for no action.
        exclude: set[str] = (
            {*routing.ACTIONS, *SCREEN_ACTIONS, SCREEN_TOOL}
            if intent is None and narrates(text)
            else set()
        )
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
        # Cassi off, only those were heard.
        synthesized: list[str] = []
        heard = 0
        # Sentences count for model requests (cassi.llm.gate) only while they are voiced.
        voiced = False

        splitter = SentenceSplitter()
        # Small models copy their own earlier replies word for word. Only noticed and logged
        # (cassi.repetition) – nothing Cassi says is cut; the prompt and the history keep
        # the model from it.
        guard = RepeatGuard(self._earlier_replies)

        def emit(text: str, check_claims: bool = False, new_round: bool = False) -> None:
            if new_round and parts and not parts[-1][-1:].isspace():
                # Text of a new round continues the previous one.
                text = " " + text
            parts.append(text)
            self.events.publish(AssistantDelta(text))
            speak(splitter.feed(text), check_claims)

        def speak(complete: Iterable[str], check_claims: bool = False) -> None:
            for sentence in complete:
                if guard.repeated(sentence):
                    log.info("Wiederholung erkannt (nur gemeldet): %s", sentence)
                tool = claimed_tool(sentence) if check_claims else None
                if tool is not None and tool in self._tools and tool not in (exclude or ()):
                    raise UnbackedClaimError(tool)
                timings.mark("first_sentence")
                if voiced:
                    synthesis.queued()
                sentences.put_nowait(sentence)

        async def converse(conversation: list[Message], check_claims: bool) -> None:
            # Tool calls Cassi already made for this request (see _route_tool) follow the
            # request; the model must not repeat them. Earlier requests' actions are in the
            # history too, but they do not rule anything out now.
            last_request = max(
                (i for i, m in enumerate(conversation) if m.role is Role.USER), default=0
            )
            done = {call.name for m in conversation[last_request:] for call in m.tool_calls}
            ran = set(done)  # really carried out for this request
            # Actions the routing form ruled out for this request.
            done |= exclude or set()
            # The model may call tools, see their results and continue. The tool list stays
            # the same in every round and every turn: it sits at the start of the prompt, and
            # a changed list makes the server read the whole prompt again (~5 s for 2000
            # tokens on an M2 Pro instead of ~0.7 s). Calls of tools that are done or ruled
            # out are dropped instead.
            for round_ in range(MAX_TOOL_ROUNDS + 1):
                # The very last round (rare) offers none, so the model has to answer.
                specs = self._conversation_specs if round_ < MAX_TOOL_ROUNDS else []
                round_text: list[str] = []
                calls: list[ToolCall] = []
                stream = (
                    self._llm.stream(conversation, specs, think=True)
                    if think
                    else self._llm.stream(conversation, specs)
                )
                with perf.span("answer"):
                    async for event in stream:
                        if isinstance(event, ToolCall):
                            calls.append(event)
                            continue
                        timings.mark("llm_first_token")
                        emit(event.text, check_claims, new_round=not round_text)
                        round_text.append(event.text)
                dropped = [c for c in calls if c.name in done]
                calls = [c for c in calls if c not in dropped]
                if dropped and not calls and not round_text and round_ < MAX_TOOL_ROUNDS:
                    # Only a call that may not happen: ask for words, at the end of the
                    # prompt (its start stays cached). Said truthfully – "already done" made
                    # the model claim actions after a mere story.
                    names = ", ".join(sorted({c.name for c in dropped}))
                    repeated = {c.name for c in dropped} & ran
                    log.debug("Werkzeugaufruf verworfen: %s", names)
                    note = (
                        f"(That was already done above: {names}. Do not call it again; answer "
                        "in words.)"
                        if repeated
                        else "(I am not asking you to do anything – nothing was done, and "
                        "nothing should be. Just reply in words; do not say you did or will "
                        "do anything.)"
                    )
                    conversation.append(Message.from_text(Role.USER, note))
                    continue
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
                self._persona,
                self._memory.context(),
                user_message,
                language,
                self._now(),
                tools=bool(self._tools),
                verbatim_replies=self._settings.memory.verbatim_replies,
                state=state,
                memories=memories,
                examples=examples,
            )
            if think:
                # Thinking takes seconds; say so right away instead of falling silent.
                emit(THINKING[language])
                speak(splitter.flush())
            try:
                # A reply about a screenshot describes what is open; it claims nothing.
                described = Modality.IMAGE in user_message.modalities
                await converse(
                    [*base, *(prelude or [])], check_claims=not prelude and not described
                )
            except UnbackedClaimError as claim:
                parts.clear()
                list(splitter.flush())  # the unspoken rest of the made-up reply
                asked_for_nothing = narrates(user_message.text) or bool(
                    _HOW_TO.search(user_message.text)
                )
                if asked_for_nothing:
                    # "Ich habe gestern Moewe geöffnet", "Wie öffne ich …?": the user asked
                    # for no action – a made-up one is not carried out, the model answers
                    # again (with the note below).
                    log.info("Antwort behauptete %s nach einer Erzählung/Frage", claim.tool)
                    routed: list[Message] = []
                    results: list[ToolResult] = []
                else:
                    # Made-up success: do what was claimed and answer from the real result.
                    log.info("Antwort behauptete %s ohne Werkzeug – führe es aus", claim.tool)
                    routed, results = await self._route_tool(
                        claim.tool, user_message, language, state
                    )
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
                            "otherwise; "
                            + (
                                "just reply to what I said.)"
                                if asked_for_nothing
                                else "ask me what exactly to do.)"
                            ),
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
                # Still nothing to say: a filler first, so the user knows Cassi is on it.
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
            with perf.span("web"):
                finding = await self._browse(
                    text, url, results[0], language, say, for_claude, others
                )
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
            with perf.span("file"):
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
        from cassi.tools.web import front_browser

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
        tool = self._tools.get(SCREEN_TOOL)
        list_apps = getattr(tool, "apps", None)
        apps: list[str] = await asyncio.to_thread(list_apps) if callable(list_apps) else []
        list_monitors = getattr(tool, "monitors", None)
        monitors = await asyncio.to_thread(list_monitors) if callable(list_monitors) else []
        # Which part of the screen ("the PhpStorm window", "the left monitor") is read from
        # the words directly: no extra model call, so no extra waiting.
        arguments = screen_arguments(message.text, apps, monitors)
        call = ToolCall("auto_screenshot", SCREEN_TOOL, arguments)
        result = await self._execute(call)
        if not result.images:
            note = f"\n(Screenshot failed: {result.text})"
            return replace(message, parts=(*message.parts, TextPart(note))), result
        return replace(message, parts=(*message.parts, *result.images)), result

    async def _window_overview(self) -> str | None:
        """The window list read at the start of the turn (or now, if there was none)."""
        task, self._overview_task = self._overview_task, None
        if task is not None:
            try:
                return await task
            except asyncio.CancelledError:
                pass
        return await self._read_window_overview()

    async def _read_window_overview(self) -> str | None:
        """The open windows per monitor, if the screen tool can list them."""
        overview = getattr(self._tools.get(SCREEN_TOOL), "overview", None)
        if not callable(overview):
            return None
        try:
            result = await asyncio.to_thread(overview)
        except Exception as exc:
            log.debug("Fensterliste nicht lesbar: %s", exc)
            return None
        return result if isinstance(result, str) else None

    def _intent(self, text: str) -> tuple[str | None, str]:
        """Tool to force for this request, and the request text the tool call is made from.

        A follow-up like "prüf nochmal, ob du es schließen kannst" names no project; without
        a tool call the model would just claim success, so it is routed with the previous
        project request as context.
        """
        intent = tool_intent(text)
        now = time.monotonic()
        self._by_learned_verb = False
        if intent is None and (computer := self._computer_intent(text)) is not None:
            self._project_request = None
            return computer, text
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
        if self._projects is not None and (heard := heard_verb(text, self._projects())):
            log.info("Verb am Klang erkannt: „%s“ → %s", text, heard[1])
            self._project_request = (heard[1], now)
            return heard
        if self._project_request is not None:
            previous, at = self._project_request
            follow_up = project_follow_up(text)
            if follow_up is not None and now - at <= FOLLOW_UP_SECONDS:
                return follow_up, f"{previous}\n{text}"
        return None, text

    @property
    def _persona(self) -> PersonaSettings:
        """The persona, with the answer length the user asked for."""
        persona = self._settings.persona
        style = self._preferences.style() if self._preferences is not None else ""
        if not style:
            return persona
        return persona.model_copy(update={"system_prompt": f"{persona.system_prompt} {style}"})

    @property
    def _conversation_specs(self) -> list[ToolSpec]:
        """The tools the model may pick on its own. The computer tools are only reached
        through their recognized requests: offered freely, the model took "Ich habe flay
        gestern geöffnet" for phpstorm and "Mach mal Moewe auf" for system_control
        (cassi eval). It also keeps ~1000 tokens out of every request."""
        return [s for s in self._tools.specs if s.name not in COMPUTER_TOOLS]

    def _claude_answer(self, text: str) -> ToolCall | None:
        """answer_claude for a short yes / always / no while Claude waits for one."""
        tool = self._tools.get("answer_claude")
        if tool is None or not getattr(tool, "waiting", None) or len(text.split()) > 8:
            return None
        if _CLAUDE_ALWAYS.search(text):
            answer = "always"
        elif _CLAUDE_NO.search(text):
            answer = "no"
        elif _CLAUDE_YES.search(text):
            answer = "yes"
        else:
            return None
        return ToolCall("claude-answer", "answer_claude", {"answer": answer})

    def _may_classify(self, text: str) -> bool:
        """Worth classifying: no pattern recognizes it, and it sounds like the computer.
        Pure checks only – the routing itself runs later."""
        if not self._settings.llm.classify_requests or not isinstance(self._llm, StructuredChat):
            return False
        if tool_intent(text) or web_intent(text) or file_intent(text):
            return False
        return self._computer_intent(text) is None

    async def _core(self, text: str, language: Language) -> str | None:
        """What a long request asks for, in one sentence (None if nothing)."""
        if not isinstance(self._llm, StructuredChat):
            return None
        schema = {
            "type": "object",
            "properties": {"request": {"type": "string"}},
            "required": ["request"],
            "additionalProperties": False,
        }
        messages = [
            Message.from_text(
                Role.SYSTEM,
                "The user spoke a long request. Write what they want done now as one short "
                f"imperative sentence in {LANGUAGE_NAMES[language]}, keeping names, addresses "
                "and quoted text exactly. Empty if they ask for nothing.",
            ),
            Message.from_text(Role.USER, text),
        ]
        try:
            answer = await self._llm.complete_json(messages, schema, "core")
        except Exception as exc:
            log.debug("Kern der Anfrage nicht ermittelt: %s", exc)
            return None
        core = str(answer.get("request") or "").strip()
        log.info("Kern der Anfrage: %s", core or "(nichts)")
        return core or None

    async def _classified(self, text: str) -> str | None:
        """A computer action for a request the patterns missed, decided by the model."""
        if not self._settings.llm.classify_requests or not isinstance(self._llm, StructuredChat):
            return None
        if narrates(text) or _HOW_TO.search(text):
            return None
        apps = getattr(self._tools.get("control_window"), "_apps", None) or []
        if not may_be_computer_task(text, apps):
            return None
        with perf.span("classify"):
            return await classify(self._llm, text, [s.name for s in self._tools.specs])

    def _computer_intent(self, text: str) -> str | None:
        """Apps, keyboard, PhpStorm, timers ...: a tool the request clearly asks for."""
        windows = self._tools.get("control_window")
        apps = getattr(windows, "_apps", None)
        if windows is not None and apps is None:
            # Read once in the background (~0.3 s); until then apps are not recognized.
            load = getattr(windows, "app_names", None)
            if callable(load):
                self._background(_discard(asyncio.to_thread(load)))
        commands = getattr(self._tools.get("run_command"), "names", [])
        intent = computer_intent(
            text, apps or [], self._projects() if self._projects else [], commands
        )
        return intent if intent is not None and intent in self._tools else None

    def _learned_verb(self, text: str) -> tuple[str, str] | None:
        """A verb learned from earlier requests with a project name: "She K NVS",
        "Mach K NVS zu" (see cassi.verbs)."""
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
        apps = getattr(self._tools.get("control_window"), "_apps", None) or []
        projects = self._projects() if self._projects is not None else []
        if (direct := direct_arguments(name, user_message.text, projects, apps)) is not None:
            # Nothing left to interpret: no model needed (~1 s saved, see cassi.direct).
            log.info("Ohne Modell: %s(%s)", name, direct)
            perf.mark("direct")
            return await self._act(ToolCall("direct", name, direct), language)
        if (
            self._settings.llm.shared_prefix_routing
            and name in ANSWERED_TOOLS
            and any(s.name == name for s in self._conversation_specs)
        ):
            shared = await self._route_in_conversation(name, user_message, language, state)
            if shared is not None:
                return await self._act(shared, language)
        spec = next(s for s in self._tools.specs if s.name == name)
        brief = await self._claude_context(language) if name == "send_to_claude" else ""
        router = Message.from_text(
            Role.SYSTEM,
            (
                brief
                or f"Turn the user's request into exactly one call of the tool '{name}'. "
                f"Write any instruction or fact as a complete, precise sentence in "
                f"{LANGUAGE_NAMES[language]}."
            )
            + " Only name a project if the user names one."
            # Lets the model write the exact name of the project the user means.
            + (f"\n{state}" if state else "")
            # E.g. "never tell Claude to commit" must shape the instruction typed into Claude.
            + self._rules()
            + (f"\n{shown}" if (shown := format_examples(examples)) else ""),
        )
        request = Message.from_text(Role.USER, user_message.text, language)
        with perf.span("route"):
            calls = [
                event
                async for event in self._llm.stream([router, request], [spec])
                if isinstance(event, ToolCall) and event.name == name
            ]
        if not calls:
            log.debug("Werkzeug %s wurde nicht erzwungen: Modell lieferte keinen Aufruf", name)
            return [], []
        return await self._act(calls[0], language)

    async def _route_in_conversation(
        self, name: str, user_message: Message, language: Language, state: str | None
    ) -> ToolCall | None:
        """The tool call, asked for with the conversation's own prompt plus one line.

        The server keeps only the start of the last prompt. A separate routing prompt
        pushes the conversation's ~2000 tokens out, and the answer after the action reads
        them all again (~5 s on an M2 Pro). With the same start – system prompt, tool list,
        history, the request – routing reads only the last line, and the answer after it
        finds everything cached.
        """
        base = build_messages(
            self._persona,
            self._memory.context(),
            user_message,
            language,
            self._now(),
            tools=bool(self._tools),
            verbatim_replies=self._settings.memory.verbatim_replies,
            state=state,
        )
        brief = await self._claude_context(language) if name == "send_to_claude" else ""
        order = Message.from_text(
            Role.USER,
            f"(Call the tool '{name}' now for my request above – no text. "
            + (brief if brief else f"Write any instruction or fact as a complete, precise "
               f"sentence in {LANGUAGE_NAMES[language]}.")
            + " Only name a project I named."
            + self._rules()
            + ")",
            language,
        )  # fmt: skip
        with perf.span("route_shared"):
            calls = [
                event
                async for event in self._llm.stream([*base, order], self._conversation_specs)
                if isinstance(event, ToolCall)
            ]
        wanted = next((c for c in calls if c.name == name), None)
        if wanted is None:
            log.debug("Weiterleitung im Gespräch ohne %s (%s) – eigener Prompt", name, calls)
        return wanted

    async def _claude_context(self, language: Language) -> str:
        """How to write a task for Claude Code, with what Claude cannot know itself.

        Claude does not hear the conversation: "mach das wie eben besprochen" must arrive
        as the actual task – goal, place in the code, known errors, when it is done.
        """
        lines = [
            "Turn the user's request into exactly one call of the tool 'send_to_claude'. The "
            "instruction goes to Claude Code, an AI agent working in the project's code, which "
            "does not hear this conversation. Write it so Claude can act on it alone: what to "
            "do and why, where (file, class, function – if known from the request or the "
            "context below), what is known (error messages, what the user saw, earlier "
            "results), and how to tell it is done (expected behaviour, tests). Keep exactly "
            "the user's intent; do not invent requirements or extra steps. One paragraph "
            f"without line breaks, imperative, in {LANGUAGE_NAMES[language]}. If the user "
            "dictates the wording, keep it word for word."
        ]
        recent = [m for m in self.history if m.role in (Role.USER, Role.ASSISTANT)][-4:]
        if recent:
            said = " | ".join(
                f"{'User' if m.role is Role.USER else self.name}: {m.text[:300]}" for m in recent
            )
            lines.append(f"Conversation just before (for references like 'das'): {said}")
        editor = await asyncio.to_thread(_phpstorm_title)
        if editor:
            lines.append(f"PhpStorm shows: {editor}")
        return "\n".join(lines)

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


async def _timed(phase: str, work: Coroutine[Any, Any, Any]) -> Any:
    """Await ``work`` as a phase of the turn's profile (cassi.perf)."""
    with perf.span(phase):
        return await work


def _phpstorm_title() -> str | None:
    """The title of PhpStorm's front window ("flay – assistant.py"), if it is open."""
    try:
        from cassi.tools.screen import list_windows

        window = next(
            (w for w in list_windows() if "phpstorm" in f"{w.app} {w.title}".lower()), None
        )
    except Exception:
        return None
    return window.title if window is not None and window.title else None


def _compact_words(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.lower())
