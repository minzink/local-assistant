"""adjust_settings: how Cassi speaks, changed by voice ("Sprich langsamer", "Antworte kürzer").

Kept in data_dir/preferences.json, so they hold after a restart, and applied right away:
the speaking rate and voice in the TTS, the answer length in the system prompt, the screen
pause in cassi.tools.screen.PRIVACY.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from cassi.llm.base import ToolSpec
from cassi.tools.base import ToolResult, speech

if TYPE_CHECKING:
    from cassi.config import Settings

log = logging.getLogger(__name__)

Length = Literal["short", "normal", "detailed"]
SPEED_STEP = 0.15
MIN_SPEED, MAX_SPEED = 0.6, 1.8
STYLES: dict[str, str] = {
    "short": "The user wants very short answers: one sentence whenever possible.",
    "normal": "",
    "detailed": (
        "The user wants detailed answers: explain in up to six sentences, still without "
        "lists or markdown."
    ),
}
# Voices that come with Cassi (Qwen3-TTS reference recordings / Piper models).
QWEN_VOICES = ("ramona", "hokuspokus", "thorsten", "own")


@dataclass
class Preferences:
    speed: float | None = None
    voice: str | None = None
    length: str = "normal"
    screen_paused: bool = False
    path: Path | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, data_dir: Path) -> Preferences:
        path = data_dir / "preferences.json"
        found = cls(path=path)
        if path.is_file():
            try:
                raw = json.loads(path.read_text("utf-8"))
                found = cls(
                    speed=raw.get("speed"),
                    voice=raw.get("voice"),
                    length=raw.get("length") or "normal",
                    screen_paused=bool(raw.get("screen_paused")),
                    path=path,
                )
            except (ValueError, TypeError) as exc:
                log.warning("Einstellungen nicht lesbar (%s): %s", path, exc)
        return found

    def save(self) -> None:
        if self.path is None:
            return
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1), "utf-8")

    def applied(self, settings: Settings) -> Settings:
        """The settings with the spoken changes (rate, voice) laid over them."""
        tts = settings.tts
        voice = tts.voice
        if self.speed:
            voice = voice.model_copy(update={"speed": self.speed})
        voices = tts.voices
        if self.voice:
            if self.voice in QWEN_VOICES:
                voice = voice.model_copy(update={"name": self.voice})
            else:
                voices = dict.fromkeys(voices, self.voice)
        return settings.model_copy(
            update={"tts": tts.model_copy(update={"voice": voice, "voices": voices})}
        )

    def style(self) -> str:
        """The answer length, as an instruction for the system prompt."""
        return STYLES.get(self.length, "")


class AdjustSettingsTool:
    def __init__(
        self,
        preferences: Preferences,
        tts: Callable[[], Any] | None = None,
        voices: Callable[[], list[str]] | None = None,
    ) -> None:
        self.preferences = preferences
        # The running TTS (set once it exists): set_speed / set_voice if it supports them.
        self._tts = tts
        self._voices = voices or (lambda: list(QWEN_VOICES))

    def attach(self, tts: Callable[[], Any], voices: Callable[[], list[str]]) -> None:
        self._tts, self._voices = tts, voices

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="adjust_settings",
            description=(
                "Change how Cassi itself works: speaking 'speed' (slower, faster, normal), "
                "'voice' (a voice name or 'next'), answer 'length' (short, normal, detailed), "
                "'screen' (pause: Cassi must not look at the screen; resume)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "speed": {"type": "string", "enum": ["slower", "faster", "normal"]},
                    "voice": {"type": "string", "description": "Voice name, or 'next'"},
                    "length": {"type": "string", "enum": ["short", "normal", "detailed"]},
                    "screen": {"type": "string", "enum": ["pause", "resume"]},
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        said: list[tuple[str, str]] = []
        speed = str(arguments.get("speed") or "")
        if speed:
            said.append(self._speed(speed))
        voice = str(arguments.get("voice") or "").strip()
        if voice:
            outcome = self._voice(voice)
            if isinstance(outcome, ToolResult):
                return outcome
            said.append(outcome)
        length = str(arguments.get("length") or "")
        if length in STYLES:
            self.preferences.length = length
            said.append(
                {
                    "short": ("Ich antworte ab jetzt kürzer.", "I'll keep my answers short."),
                    "normal": ("Ich antworte wieder normal lang.", "Back to normal answers."),
                    "detailed": ("Ich antworte ab jetzt ausführlicher.", "I'll go into detail."),
                }[length]
            )
        screen = str(arguments.get("screen") or "")
        if screen in ("pause", "resume"):
            from cassi.tools.screen import PRIVACY

            PRIVACY.paused = self.preferences.screen_paused = screen == "pause"
            said.append(
                ("Ich schaue nicht mehr auf den Bildschirm.", "I won't look at the screen.")
                if screen == "pause"
                else (
                    "Ich darf wieder auf den Bildschirm schauen.",
                    "I may look at the screen again.",
                )
            )
        if not said:
            return ToolResult(
                "Nothing to change was named.",
                ok=False,
                say=speech("Was soll ich ändern?", "What should I change?"),
            )
        self.preferences.save()
        de = " ".join(s[0] for s in said)
        en = " ".join(s[1] for s in said)
        return ToolResult(f"Changed: {en}", action=f"Einstellungen: {de}", say=speech(de, en))

    def _speed(self, how: str) -> tuple[str, str]:
        current = self.preferences.speed or 1.0
        if how == "normal":
            new = 1.0
        else:
            step = SPEED_STEP if how == "faster" else -SPEED_STEP
            new = round(min(MAX_SPEED, max(MIN_SPEED, current + step)), 2)
        self.preferences.speed = new
        tts = self._tts() if self._tts else None
        if tts is not None and callable(getattr(tts, "set_speed", None)):
            tts.set_speed(new)
        if how == "normal":
            return ("Ich spreche wieder normal schnell.", "Back to my normal pace.")
        if new == current:
            return (
                "Schneller geht es nicht." if how == "faster" else "Langsamer geht es nicht.",
                "I can't go faster." if how == "faster" else "I can't go slower.",
            )
        return (
            ("Ich spreche jetzt schneller.", "I'm speaking faster now.")
            if how == "faster"
            else ("Ich spreche jetzt langsamer.", "I'm speaking slower now.")
        )

    def _voice(self, wanted: str) -> tuple[str, str] | ToolResult:
        voices = self._voices()
        if not voices:
            return ToolResult(
                "No other voices are available.",
                ok=False,
                say=speech("Ich habe keine andere Stimme.", "I have no other voice."),
            )
        current = self.preferences.voice
        if wanted.casefold() in ("next", "andere", "other", "nächste"):
            index = voices.index(current) + 1 if current in voices else 1
            chosen = voices[index % len(voices)]
        else:
            matches = [v for v in voices if wanted.casefold() in v.casefold()]
            if not matches:
                listed = ", ".join(voices)
                return ToolResult(
                    f"No voice '{wanted}'. Available: {listed}.",
                    ok=False,
                    say=speech(
                        f"Eine Stimme {wanted} habe ich nicht. Ich habe: {listed}.",
                        f"I don't have a voice {wanted}. I have: {listed}.",
                    ),
                )
            chosen = matches[0]
        self.preferences.voice = chosen
        tts = self._tts() if self._tts else None
        if tts is not None and callable(getattr(tts, "set_voice", None)):
            tts.set_voice(chosen)
        return (f"So klingt die Stimme {chosen}.", f"This is the voice {chosen}.")
