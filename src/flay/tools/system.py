"""Volume, music and locking the screen."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech

log = logging.getLogger(__name__)

ACTIONS = (
    "volume_up",
    "volume_down",
    "set_volume",
    "mute",
    "unmute",
    "play_pause",
    "next_track",
    "previous_track",
    "lock_screen",
)
# One step of "lauter"/"leiser" in percent.
STEP = 10
SAY = {
    "volume_up": ("Lauter.", "Louder."),
    "volume_down": ("Leiser.", "Quieter."),
    "mute": ("Ton aus.", "Muted."),
    "unmute": ("Ton an.", "Sound on."),
    "play_pause": ("Okay.", "Okay."),
    "next_track": ("Nächstes Lied.", "Next track."),
    "previous_track": ("Vorheriges Lied.", "Previous track."),
    "lock_screen": ("Ich sperre den Bildschirm.", "Locking the screen."),
}


def _osascript(script: str) -> str:
    return subprocess.run(
        ["osascript", "-e", script], check=True, capture_output=True, text=True, timeout=10
    ).stdout.strip()


def _key(name: str, times: int = 1) -> None:
    from pynput.keyboard import Controller, Key

    keyboard, key = Controller(), getattr(Key, name)
    for _ in range(times):
        keyboard.press(key)
        keyboard.release(key)


def system_action(action: str, level: int | None = None) -> None:
    """Do ``action`` on this computer (``level`` 0–100 for set_volume)."""
    mac = sys.platform == "darwin"
    if action in ("volume_up", "volume_down", "set_volume") and mac:
        current = int(_osascript("output volume of (get volume settings)") or 50)
        target = {"volume_up": current + STEP, "volume_down": current - STEP}.get(
            action, level or 0
        )
        _osascript(f"set volume output volume {max(0, min(100, target))}")
    elif action in ("volume_up", "volume_down"):
        # Windows: one media key step is 2 %.
        _key("media_volume_up" if action == "volume_up" else "media_volume_down", STEP // 2)
    elif action == "set_volume":
        _key("media_volume_down", 50)
        _key("media_volume_up", max(0, min(100, level or 0)) // 2)
    elif action in ("mute", "unmute") and mac:
        _osascript(f"set volume {'with' if action == 'mute' else 'without'} output muted")
    elif action in ("mute", "unmute"):
        _key("media_volume_mute")  # toggles
    elif action == "play_pause":
        _key("media_play_pause")
    elif action == "next_track":
        _key("media_next")
    elif action == "previous_track":
        _key("media_previous")
    elif action == "lock_screen" and mac:
        _osascript(
            'tell application "System Events" to keystroke "q" using {control down, command down}'
        )
    elif action == "lock_screen" and sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined, unused-ignore]
    else:
        raise RuntimeError(f"{action} is not supported here")


class SystemTool:
    def __init__(self, act: Callable[[str, int | None], None] = system_action) -> None:
        self._act = act

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="system_control",
            description=(
                "Volume (up, down, set 'level' 0–100, mute), music (play/pause, next, "
                "previous) or lock the screen."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "level": {"type": "integer"},
                },
                "required": ["action"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = str(arguments.get("action") or "")
        if action not in ACTIONS:
            return ToolResult(f"Unknown action '{action}'.", ok=False)
        level = arguments.get("level")
        level = int(level) if isinstance(level, (int, float)) else None
        if action == "set_volume" and level is None:
            return ToolResult(
                "set_volume needs a level.", ok=False, say=speech("Wie laut?", "How loud?")
            )
        await asyncio.to_thread(self._act, action, level)
        de, en = SAY.get(action, (f"Lautstärke {level} Prozent.", f"Volume {level} percent."))
        return ToolResult(f"Done: {action}.", action=f"System: {action}", say=speech(de, en))
