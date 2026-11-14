"""Settings: built-in defaults, overridden by an optional YAML file.

Lookup order for the YAML file: explicit path, ``$CASSI_CONFIG``, ``./cassi.yaml``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from platformdirs import user_data_path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
)

from cassi.core.language import Language
from cassi.core.messages import Modality
from cassi.input.base import InputMode
from cassi.llm.base import ModelCapabilities

DEFAULT_CONFIG_FILE = Path("cassi.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LanguageSettings(_Section):
    default: Language = Language.DE
    # Detect de/en per utterance and answer in the same language.
    auto_detect: bool = True


def _default_push_to_talk_key() -> str:
    # Mac keyboards map F-keys to media functions and have no right Ctrl.
    return "alt_r" if sys.platform == "darwin" else "ctrl_r"


class ContinuousSettings(_Section):
    # A pause this long ends an utterance. Shorter answers faster, but cuts off people
    # who pause mid-sentence.
    end_silence_ms: int = Field(default=600, ge=200, le=3000)
    min_speech_ms: int = Field(default=250, ge=50)
    max_utterance_seconds: float = Field(default=30, gt=0)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    # Interrupt Cassi by speaking while it talks. Needs a headset: through speakers Cassi
    # hears itself. Off: Cassi ignores speech while it talks.
    barge_in: bool = True
    # Only what starts with "Hey Cassi" counts ("Hey Cassi" alone: the next utterance). Cassi's
    # own detector (``cassi train-wake``) filters, the transcript decides; it also interrupts
    # Cassi while it talks.
    wake_word: bool = False
    # Higher: fewer false wake-ups, but "Hey Cassi" is missed more often.
    wake_threshold: float = Field(default=0.7, gt=0, lt=1)
    # Another wake word, e.g. from an adapter: its trained detector (.npz as written by
    # ``cassi train-wake``) and a regular expression for what speech recognition makes of it at
    # the start of a transcript. None: "Hey Cassi" (own training in the data folder, else the
    # bundled detector).
    wake_model: Path | None = None
    wake_pattern: str | None = None
    # How the wake word is shown in the chat and the tray (an adapter sets its own).
    wake_phrase: str = "Hey Cassi"

    @field_validator("wake_pattern")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(f"kein gültiger regulärer Ausdruck: {exc}") from exc
        return value


class InputSettings(_Section):
    # push_to_talk: hold a key. continuous: always listening, speech detected by VAD.
    mode: InputMode = InputMode.PUSH_TO_TALK
    continuous: ContinuousSettings = ContinuousSettings()
    # hotkey: hold a global key (macOS needs the Input Monitoring permission).
    # terminal: press Enter to start and again to stop; works without permissions.
    trigger: Literal["hotkey", "terminal"] = "hotkey"
    push_to_talk_key: str = Field(default_factory=_default_push_to_talk_key)


class AudioSettings(_Section):
    input_device: str | int | None = None
    output_device: str | int | None = None
    # Silero VAD: trims silence and drops recordings without speech.
    vad: bool = True


class STTSettings(_Section):
    # parakeet: fast (~0.1 s, CPU, macOS + Windows), slightly less accurate in German.
    # auto: Whisper – mlx-whisper on Apple Silicon, faster-whisper elsewhere (~0.6 s).
    backend: Literal["auto", "faster-whisper", "mlx-whisper", "parakeet", "canary", "voxtral"] = (
        "parakeet"
    )
    model: str = "large-v3-turbo"
    # Whisper only: pass the project names and tech terms as a prompt, so they are spelled
    # the way Cassi knows them.
    vocabulary_prompt: bool = False


class ThinkingSettings(_Section):
    """Thinking before answering, for requests the user explicitly wants thought through.

    Never for the first words: Cassi says "Moment, ich überlege kurz" at once and then
    answers with thinking. Measured for recognizing commands it did not help (Qwen3.5-9B:
    60 % instead of 70 % on hard cases, 3–13 s), so it is only used when asked for.
    """

    enabled: bool = False
    # Another model on the same server for thinking (e.g. a "thinking" variant); empty =
    # the normal model with the switches below.
    model: str | None = None
    # Request fields that switch thinking on, e.g. for Qwen3.5 on llama.cpp:
    # {"chat_template_kwargs": {"enable_thinking": true}}.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Budget for thinking plus answer.
    max_tokens: int = Field(default=2000, gt=0)


class LLMSettings(_Section):
    # Any server with an OpenAI-compatible API: Ollama, llama.cpp, LM Studio, mlx-lm, vLLM.
    # In-process backends (e.g. transformers, MLX) can be added as further literals.
    backend: Literal["openai-compatible"] = "openai-compatible"
    # 127.0.0.1 rather than localhost: Ollama only listens on IPv4, localhost may be ::1 first.
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: SecretStr | None = None
    # Qwen3-VL (Apache-2.0): text, images and tool calls in one model. The "instruct"
    # variant answers directly; the default "qwen3-vl:8b" tag always thinks first (6–10 s).
    model: str = "qwen3-vl:8b-instruct"
    input_modalities: frozenset[Modality] = frozenset({Modality.TEXT, Modality.IMAGE})
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    # After this long without a conversation Cassi sends the server a tiny request with the
    # usual prompt: the model stays loaded (Ollama unloads after 5 min) and the prompt's
    # start stays cached. 0 = off.
    keep_warm_minutes: float = Field(default=4, ge=0)
    max_tokens: int = Field(default=400, gt=0)
    # Context length the server gives the model: longer requests are shortened before
    # sending (oldest messages first), instead of being rejected by the server. 8192 is
    # what Cassi sets up for Ollama on GPUs below 24 GB; raise it for a server with more.
    # 0 = never shorten.
    context_tokens: int = Field(default=8192, ge=0)
    # Sent with every request as it is, for server-specific switches, e.g. for Qwen3.5 on
    # llama.cpp: {"chat_template_kwargs": {"enable_thinking": false}}.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Route recognized actions through a form whose project field only allows the user's
    # project names (needs a server with JSON-schema output). Measured with
    # qwen3-vl:8b-instruct it was no better and 0.7 s slower than a plain tool call, so off;
    # worth re-checking with `cassi eval` for other models.
    structured_routing: bool = False
    # Requests no fixed pattern recognizes, but that could be about the computer ("Wirf
    # Spotify auf den zweiten Bildschirm"), are classified by the model first (~0.4 s,
    # only for such requests). Off: only the patterns route computer actions.
    classify_requests: bool = True
    # Route recognized actions with the conversation's own prompt plus one line, so the
    # server's cached prompt start stays valid for the answer after it (see cassi eval).
    shared_prefix_routing: bool = True
    # After a turn whose routing (or the memory's upkeep) pushed the conversation out of the
    # server's prompt cache, send its start again while idle: the next answer starts fast.
    rewarm: bool = True
    thinking: ThinkingSettings = ThinkingSettings()

    @field_validator("input_modalities")
    @classmethod
    def _requires_text(cls, value: frozenset[Modality]) -> frozenset[Modality]:
        if Modality.TEXT not in value:
            raise ValueError("input_modalities must contain 'text'")
        return value

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(input_modalities=self.input_modalities)


class FillerSettings(_Section):
    """A short phrase while an action is still being prepared, so the user knows Cassi heard
    them. Only for spoken input.

    Answers need none: their first sentence is spoken as soon as the model has written it
    (~0.9 s after the user stopped, M2 Pro) – a real reply instead of a canned "Hm". An
    instruction to keep that sentence extra short brought no measurable gain and made the
    model claim actions more often, so there is none. Phrases for answers can still be set
    (``phrases``)."""

    enabled: bool = True
    # Played only if nothing is spoken this long after the user stopped speaking. Most
    # actions are done before (clear ones without the model: cassi.direct).
    delay_ms: int = Field(default=1200, ge=0, le=5000)
    phrases: dict[Language, list[str]] = Field(
        default_factory=lambda: {language: list[str]() for language in Language}
    )
    # When the request asks for an action (open a project, tell Claude, ...).
    action_phrases: dict[Language, list[str]] = Field(
        default_factory=lambda: {
            Language.DE: ["Moment.", "Einen Moment.", "Okay, Moment."],
            # Short ones: the answer waits until the filler is finished.
            Language.EN: ["One moment.", "Just a moment.", "Okay, moment."],
        }
    )


class VoiceSettings(_Section):
    # The voice for both languages: "own" (recorded with `cassi voice record`), a bundled one
    # ("ramona", "hokuspokus" – female, "thorsten" – male) or a WAV file with its transcript
    # in a .txt next to it. "auto": the own recording if there is one, else "ramona".
    name: str = "auto"
    # Semitones up (+) or down (-), and the speaking rate (1.2 = 20 % faster).
    pitch: float = Field(default=0.0, ge=-12, le=12)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class TTSSettings(_Section):
    # qwen3: Qwen3-TTS (Apache 2.0) clones one voice for German and English; fast on Apple
    # Silicon, slow with PyTorch on NVIDIA. piper: one voice per language, fast on any CPU.
    # auto: qwen3 on Apple Silicon, else piper.
    backend: Literal["auto", "qwen3", "piper"] = "auto"
    voice: VoiceSettings = VoiceSettings()
    # Piper voices, one per language (the qwen3 voice comes from ``voice``).
    voices: dict[Language, str] = Field(
        default_factory=lambda: {
            # Cassi's own Ramona voice (voices/README.md): the same voice as Qwen3-TTS, one
            # model for both languages, first sound after ~0.1 s on a CPU. It comes with the
            # repository (Git LFS); without it Cassi speaks with Kerstin (de, CC0) and Kristin
            # (en, public domain). Male alternative: "de_DE-thorsten-medium" (CC0).
            Language.DE: "cassi-ramona-medium",
            Language.EN: "cassi-ramona-medium",
        }
    )
    filler: FillerSettings = FillerSettings()

    @field_validator("voices")
    @classmethod
    def _covers_all_languages(cls, value: dict[Language, str]) -> dict[Language, str]:
        missing = set(Language) - value.keys()
        if missing:
            raise ValueError(f"no voice configured for: {', '.join(sorted(missing))}")
        return value


class PersonaSettings(_Section):
    name: str = "Cassi"
    # Written in English because small models follow English instructions most reliably;
    # the reply language is appended per turn.
    system_prompt: str = (
        "You are Cassi, a helpful voice assistant. Your replies are spoken aloud: answer in "
        "one or two short, natural sentences, without markdown, lists or emojis."
    )
    # Free-form mood dimensions on a 0–10 scale, e.g. {"cheerful": 7, "curious": 5}.
    # Replaces the former EmotionEncoder: the values are written into the system prompt.
    mood: dict[str, int] = Field(default_factory=dict)

    @field_validator("mood")
    @classmethod
    def _mood_range(cls, value: dict[str, int]) -> dict[str, int]:
        for name, level in value.items():
            if not 0 <= level <= 10:
                raise ValueError(f"mood '{name}' must be between 0 and 10, got {level}")
        return value


class RecallSettings(_Section):
    """Finding memories for each request (RAG) instead of putting all of them in the prompt."""

    enabled: bool = True
    # Embedding server (OpenAI-compatible); empty = the LLM's server.
    base_url: str | None = None
    api_key: SecretStr | None = None
    # Qwen3-Embedding 0.6B: Apache 2.0, multilingual, ~640 MB (`ollama pull` it once).
    # "auto": its 2k-context variant from `cassi setup` (1 GB instead of 4.8 GB), else it.
    model: str = "auto"
    # Memories handed to the model per request.
    limit: int = Field(default=6, ge=1, le=30)
    # Cosine similarity below which a memory never counts as related by meaning; above it,
    # only memories close to the best match count.
    min_similarity: float = Field(default=0.30, ge=0.0, le=1.0)
    # How long the answer may wait for the search once the request is known.
    timeout_ms: int = Field(default=300, ge=0)


class MemorySettings(_Section):
    # Persist the conversation and long-term facts in <data_dir>/memory.sqlite3.
    enabled: bool = True
    # Messages of the current session sent verbatim; older ones are summarized.
    recent_messages: int = Field(default=12, ge=2)
    # A pause longer than this starts a new session (fresh context, facts are kept).
    session_timeout_minutes: float = Field(default=30, gt=0)
    summarize: bool = True
    # Let the LLM pick durable facts ("my daughter is called Lea") from conversations.
    extract_facts: bool = True
    max_facts: int = Field(default=50, ge=0)
    # Background work (summaries, facts) waits for this much silence after a turn.
    idle_seconds: float = Field(default=3.0, ge=0)
    recall: RecallSettings = RecallSettings()


ToolName = Literal[
    "open_project",
    "close_project",
    "send_to_claude",
    "claude_status",
    "look_at_screen",
    "click_on_screen",
    "scroll_screen",
    "open_url",
    "read_webpage",
    "fill_web_form",
    "read_file",
    "workflow",
    "remember",
    "forget",
    "ssh_connect",
    "control_window",
    "use_keyboard",
    "phpstorm",
    "run_command",
    "find_file",
    "timer",
    "system_control",
    "operate_computer",
    "calendar",
    "adjust_settings",
    "answer_claude",
]


class PhpStormToolSettings(_Section):
    # Command that opens a folder in PhpStorm; default: Toolbox script or `phpstorm`.
    launcher: Path | None = None
    # PhpStorm config dir with options/recentProjects.xml; default: newest version.
    config_dir: Path | None = None
    # PhpStorm's built-in MCP server (Settings → Tools → MCP Server): URL and token from the
    # copied client configuration. Lets Cassi run commands in PhpStorm's terminal.
    # Use it to open Claude in a PhpStorm terminal tab when no session runs.
    use_mcp: bool = True
    # null = find it automatically (ports 64342–64461).
    mcp_url: str | None = None
    # Only needed if the server runs in "restricted mode".
    mcp_token: SecretStr | None = None


class ClaudeToolSettings(_Section):
    # auto: type into a `cassi claude` tmux session if one runs, otherwise use the clipboard.
    transport: Literal["auto", "tmux", "clipboard"] = "auto"
    # No session for the named project: start Claude in a background tmux session and type
    # there. Without a project name Cassi asks which project is meant.
    autostart: bool = True


class ScreenToolSettings(_Section):
    # 1 = primary monitor, 2 = second, ..., 0 = all monitors combined.
    monitor: int = Field(default=1, ge=0)
    max_size: int = Field(default=1280, ge=256)


class PointerToolSettings(_Section):
    # Model that finds things in screenshots (click_on_screen); null = llm.model. It must
    # accept images and answer Qwen-VL style points (0–1000). "auto": the 4B variant from
    # `cassi setup` if the server has it (twice as fast, fits next to the 8B), else llm.model.
    model: str | None = "auto"
    # A second look at a crop around the first guess when accessibility cannot confirm
    # the element: hits small targets reliably, costs a second image (~8 s on an M2 Pro).
    refine: bool = True
    # After a click, compare the screen before and after (~0.6 s): if nothing changed, Cassi
    # says so instead of reporting success. Needs the screen recording permission.
    verify: bool = True
    # Find labels the accessibility interfaces miss (PhpStorm's toolbars …) with the
    # system's text recognition (~0.3 s) before asking the vision model (~8 s).
    ocr: bool = True


class WebToolSettings(_Section):
    # Screens read_webpage reads at most (scrolling down); each costs the model ~8 s.
    max_pages: int = Field(default=3, ge=1, le=10)
    # Scroll/click steps on a page until the requested information shows (each ~8 s).
    max_steps: int = Field(default=4, ge=0, le=10)
    # Load pages the user names directly over HTTP (invisible, ~0.2 s; no logins, no
    # JavaScript). Off: only the user's browser loads pages.
    direct: bool = True
    # Read pages as text from the browser's accessibility tree (macOS), in the background.
    browser_text: bool = True
    # Page text the model reads at once (~4 characters per token): of a long page the parts
    # about the request first; more on request.
    max_chars: int = Field(default=8000, ge=1000, le=60000)
    # Close the tabs Cassi opened in the background once a request is done.
    close_tabs: bool = True


class SshToolSettings(_Section):
    # Servers Cassi can connect to; null = ~/.ssh/config (Windows: %USERPROFILE%\\.ssh\\config).
    config_file: Path | None = None
    # auto: Terminal.app (macOS), Windows Terminal or a console window (Windows).
    terminal: Literal["auto", "terminal", "iterm", "windows-terminal", "console"] = "auto"


class CommandSettings(_Section):
    # Run in a shell, as written; only commands listed here can be run at all.
    command: str
    # Folder to run it in; default: where Cassi was started.
    cwd: Path | None = None
    timeout_seconds: float = Field(default=600, gt=0)


class FileSearchSettings(_Section):
    # Folders find_file searches (a few levels deep); default: Downloads, Desktop, Documents.
    roots: list[Path] = Field(default_factory=list)


class NotificationSettings(_Section):
    # Say when a Claude session (started with `cassi claude`) is done or asks for approval.
    claude: bool = True
    # Only work that took at least this long is worth "Claude ist fertig".
    claude_min_work_seconds: float = Field(default=20, ge=0)


class ComputerSettings(_Section):
    # operate_computer: steps (look, click, type …) Cassi may take for one task before it
    # stops and says how far it got.
    max_steps: int = Field(default=12, ge=1, le=40)
    # Model that picks the moves; null = llm.model. The 4B variant ("auto": if `cassi setup`
    # made it) is ~1/3 faster per move but picked 2 of 5 first moves wrong (8B: 1 of 5).
    model: str | None = None
    # Solutions that worked are kept and shown to the model for similar tasks.
    learn: bool = True


class CalendarSettings(_Section):
    # Calendars to leave out (by name), e.g. ["Feiertage", "Geburtstage"].
    hidden: list[str] = Field(default_factory=list)


class ToolsSettings(_Section):
    enabled: list[ToolName] = Field(default_factory=lambda: list(get_args(ToolName)))
    phpstorm: PhpStormToolSettings = PhpStormToolSettings()
    claude: ClaudeToolSettings = ClaudeToolSettings()
    screen: ScreenToolSettings = ScreenToolSettings()
    pointer: PointerToolSettings = PointerToolSettings()
    web: WebToolSettings = WebToolSettings()
    ssh: SshToolSettings = SshToolSettings()
    # Commands run_command may run, by name, e.g.
    # {"tests": {"command": "composer test", "cwd": "~/projects/shop"}}. Empty: no run_command.
    commands: dict[str, CommandSettings] = Field(default_factory=dict)
    files: FileSearchSettings = FileSearchSettings()
    notifications: NotificationSettings = NotificationSettings()
    computer: ComputerSettings = ComputerSettings()
    calendar: CalendarSettings = CalendarSettings()


class UISettings(_Section):
    # Local web UI (chat history, text input, mute switches); only reachable from this PC.
    enabled: bool = True
    port: int = Field(default=8765, ge=1024, le=65535)
    open_browser: bool = True
    # Menu bar (macOS) / tray (Windows) icon with state, mute switches and quit.
    tray: bool = True


class UpdateSettings(_Section):
    # A running Cassi switches to a newer installed version by itself (after `cassi update`,
    # a git pull or a new installation) once it has nothing to do, and says so in the chat.
    # Off while developing, if every commit should not restart it.
    switch: bool = True
    check_seconds: float = Field(default=60, ge=5)


class PrivacySettings(_Section):
    # No screenshot (nor OCR) while one of these apps – or a window whose title contains one
    # of these words – is visible in the captured part of the screen. Compared without case,
    # parts of names count ("keepass" covers "KeePassXC").
    private_apps: list[str] = Field(
        default_factory=lambda: [
            "1Password", "Bitwarden", "KeePass", "Dashlane", "LastPass", "Enpass",
            "Keychain Access", "Schlüsselbundverwaltung", "Passwords", "Passwörter",
        ]
    )  # fmt: skip
    private_titles: list[str] = Field(
        default_factory=lambda: [
            "Online-Banking", "Onlinebanking", "Banking", "Sparkasse", "Volksbank", "PayPal",
            "Kontoauszug", "Privates Surfen", "Private Browsing", "Privates Fenster",
            "InPrivate", "Inkognito", "Incognito",
        ]
    )  # fmt: skip
    # Keep a log of requests (text, recognized action, tools, duration, errors) for
    # `cassi history` and `cassi stats`, on this computer only. Days to keep it; 0 = no log.
    history_days: int = Field(default=30, ge=0)


class Settings(_Section):
    language: LanguageSettings = LanguageSettings()
    input: InputSettings = InputSettings()
    audio: AudioSettings = AudioSettings()
    stt: STTSettings = STTSettings()
    llm: LLMSettings = LLMSettings()
    tts: TTSSettings = TTSSettings()
    persona: PersonaSettings = PersonaSettings()
    memory: MemorySettings = MemorySettings()
    ui: UISettings = UISettings()
    updates: UpdateSettings = UpdateSettings()
    tools: ToolsSettings = ToolsSettings()
    privacy: PrivacySettings = PrivacySettings()
    # Personal data: memory, history, own recordings and own models. Export/import carries it
    # between computers (`cassi export`); it never belongs in the repository.
    data_dir: Path = Field(default_factory=lambda: default_data_dir())
    # Downloaded models (Piper voices, speech detection, wake word features). None: in
    # data_dir. An adapter with its own data_dir (Flay) points here to Cassi's, so nothing
    # is downloaded twice.
    downloads_dir: Path | None = None
    # Installed adapters to use (packages that personalize Cassi, e.g. "flay"); None = all
    # installed ones, [] = none. See cassi.adapters.
    adapters: list[str] | None = None

    @property
    def downloads(self) -> Path:
        return self.downloads_dir or self.data_dir


def default_data_dir(name: str = "cassi") -> Path:
    """The data folder of Cassi, or of an adapter with its own (``default_data_dir("flay")``)."""
    return user_data_path(name, appauthor=False)


def resolve_config_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return path
    if env_path := os.environ.get("CASSI_CONFIG"):
        return Path(env_path)
    return DEFAULT_CONFIG_FILE if DEFAULT_CONFIG_FILE.is_file() else None


def load_settings(path: Path | None = None) -> Settings:
    config_path = resolve_config_path(path)
    if config_path is None:
        return Settings()

    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings.model_validate(data)
