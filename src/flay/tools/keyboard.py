"""The keyboard: type dictated text, press a shortcut, read what is selected.

Text goes in through the clipboard (one paste instead of key by key: umlauts and other
layouts arrive as they are, and it is instant); the clipboard is restored afterwards.
Only shortcuts from a fixed list can be pressed – Enter is deliberately not among them:
sending (a message, a prompt to Claude) stays with the user.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Callable
from typing import Any

from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech

log = logging.getLogger(__name__)

# Shortcut → (key, with Shift), the modifier being Cmd on macOS and Ctrl elsewhere.
SHORTCUTS: dict[str, tuple[str, bool]] = {
    "save": ("s", False),
    "undo": ("z", False),
    "redo": ("z", True),
    "copy": ("c", False),
    "cut": ("x", False),
    "paste": ("v", False),
    "select_all": ("a", False),
    "find": ("f", False),
    "new_tab": ("t", False),
    "close_tab": ("w", False),
    "reopen_tab": ("t", True),
    "reload": ("r", False),
    "new_window": ("n", False),
    "zoom_in": ("+", False),
    "zoom_out": ("-", False),
}
# Keys pressed alone.
SINGLE_KEYS = ("escape", "tab", "page_down", "page_up")
SPOKEN = {
    "save": ("gespeichert", "saved"),
    "undo": ("rückgängig gemacht", "undone"),
    "redo": ("wiederhergestellt", "redone"),
    "copy": ("kopiert", "copied"),
    "cut": ("ausgeschnitten", "cut"),
    "paste": ("eingefügt", "pasted"),
    "select_all": ("alles markiert", "selected everything"),
    "find": ("die Suche geöffnet", "opened search"),
    "new_tab": ("einen neuen Tab geöffnet", "opened a new tab"),
    "close_tab": ("den Tab geschlossen", "closed the tab"),
    "reopen_tab": ("den Tab wieder geöffnet", "reopened the tab"),
    "reload": ("neu geladen", "reloaded"),
    "new_window": ("ein neues Fenster geöffnet", "opened a new window"),
    "zoom_in": ("vergrößert", "zoomed in"),
    "zoom_out": ("verkleinert", "zoomed out"),
}
# How long the app gets to put the selection on the clipboard / to read the pasted text.
COPY_WAIT = 0.3
PASTE_WAIT = 0.5
MAX_SELECTION_CHARS = 6000


def _modifier() -> Any:
    from pynput.keyboard import Key

    return Key.cmd if sys.platform == "darwin" else Key.ctrl


def press_shortcut(name: str) -> None:
    from pynput.keyboard import Controller, Key

    keyboard = Controller()
    if name in SINGLE_KEYS:
        key = getattr(Key, "esc" if name == "escape" else name)
        keyboard.press(key)
        keyboard.release(key)
        return
    key, shift = SHORTCUTS[name]
    modifiers = [_modifier(), *([Key.shift] if shift else [])]
    for modifier in modifiers:
        keyboard.press(modifier)
    try:
        keyboard.press(key)
        keyboard.release(key)
    finally:
        for modifier in reversed(modifiers):
            keyboard.release(modifier)


def _clipboard() -> tuple[Callable[[], str], Callable[[str], None]]:
    import pyperclip

    return pyperclip.paste, pyperclip.copy


def paste_text(text: str) -> None:
    """Type ``text`` into the focused field via the clipboard, then restore it."""
    read, write = _clipboard()
    before = read()
    write(text)
    try:
        press_shortcut("paste")
        time.sleep(PASTE_WAIT)  # the app reads the clipboard after the key press
    finally:
        write(before)


def selected_text() -> str:
    """What is selected in the front app (copied, then the clipboard restored)."""
    read, write = _clipboard()
    before = read()
    marker = f"⁣flay-{time.monotonic()}"  # invisible, never selected by anyone
    write(marker)
    try:
        press_shortcut("copy")
        deadline = time.monotonic() + COPY_WAIT
        text = read()
        while text == marker and time.monotonic() < deadline:
            time.sleep(0.05)
            text = read()
    finally:
        write(before)
    return "" if text == marker else text


def describe_selection(text: str) -> str:
    """The selection for the model, shortened if long."""
    if not text.strip():
        return "(Nothing is selected in the front app.)"
    shown = text[:MAX_SELECTION_CHARS]
    more = (
        f" … ({len(text) - MAX_SELECTION_CHARS} more characters)" if len(text) > len(shown) else ""
    )
    return f"(Text the user has selected in the front app:\n«{shown}{more}»)"


class KeyboardTool:
    def __init__(
        self,
        paste: Callable[[str], None] = paste_text,
        press: Callable[[str], None] = press_shortcut,
        front_is_own: Callable[[], bool] | None = None,
    ) -> None:
        self._paste = paste
        self._press = press
        self._front_is_own = front_is_own or _front_is_own

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="use_keyboard",
            description=(
                "Type dictated 'text' word for word into the field the user is in, or press "
                "a 'shortcut' in the front app. Cannot press Enter."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Exactly what to type"},
                    "shortcut": {"type": "string", "enum": [*SHORTCUTS, *SINGLE_KEYS]},
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        text = str(arguments.get("text") or "")
        shortcut = str(arguments.get("shortcut") or "")
        if not text and shortcut not in (*SHORTCUTS, *SINGLE_KEYS):
            return ToolResult(
                "Give 'text' to type or a 'shortcut' from the list.",
                ok=False,
                say=speech("Was soll ich tippen?", "What should I type?"),
            )
        if await asyncio.to_thread(self._front_is_own):
            return ToolResult(
                "Refused: Flay's own window is in front.",
                ok=False,
                action="Tastatur: Flays eigenes Fenster ist vorn",
                say=speech(
                    "Vorn ist gerade mein eigenes Fenster. Klick erst in das Feld, in das ich "
                    "schreiben soll.",
                    "My own window is in front. Click into the field I should type in first.",
                ),
            )
        if text:
            await asyncio.to_thread(self._paste, text)
            short = text if len(text) <= 40 else text[:37] + "…"
            return ToolResult(
                f"Typed {len(text)} characters into the front app.",
                action=f"Getippt: „{short}“",
                say=speech("Erledigt.", "Done."),
            )
        await asyncio.to_thread(self._press, shortcut)
        de, en = SPOKEN.get(shortcut, (f"{shortcut} gedrückt", f"pressed {shortcut}"))
        return ToolResult(
            f"Pressed the shortcut '{shortcut}'.",
            action=f"Tastenkürzel: {shortcut}",
            say=speech(f"Ich habe {de}.", f"I {en}."),
        )


def _front_is_own() -> bool:
    from flay.tools.screen import is_own, list_windows

    windows = list_windows()
    return bool(windows) and is_own(windows[0])
