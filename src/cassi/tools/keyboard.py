"""The keyboard: type dictated text, press a shortcut, read what is selected.

Text goes in through the clipboard (one paste instead of key by key: umlauts and other
layouts arrive as they are, and it is instant); the clipboard is restored afterwards.
Only shortcuts from a fixed list can be pressed – Enter is deliberately not among them:
sending (a message, a prompt to Claude) stays with the user.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import sys
import time
from collections.abc import Callable
from typing import Any

from cassi.core.messages import ToolCall
from cassi.llm.base import ToolSpec
from cassi.tools.base import ToolResult, speech
from cassi.tools.virtual_keys import (
    MAX_TIMES,
    DirectMode,
    KeyStore,
    check_keys,
    combos,
    is_global,
    press_sequence,
)

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
SINGLE_KEYS = ("escape", "tab", "page_down", "page_up", "delete_word")
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
    "delete_word": ("das letzte Wort gelöscht", "deleted the last word"),
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
    if name == "delete_word":
        # "Streich das letzte Wort": Option+Backspace on macOS, Ctrl+Backspace elsewhere.
        modifier = Key.alt if sys.platform == "darwin" else Key.ctrl
        with keyboard.pressed(modifier):
            keyboard.press(Key.backspace)
            keyboard.release(Key.backspace)
        return
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


# Names in key combinations ("cmd+shift+n", "enter", "down") → pynput key names.
_KEY_NAMES = {
    "enter": "enter", "return": "enter", "esc": "esc", "escape": "esc", "tab": "tab",
    "space": "space", "backspace": "backspace", "delete": "delete", "up": "up",
    "down": "down", "left": "left", "right": "right", "home": "home", "end": "end",
    "page_up": "page_up", "pageup": "page_up", "page_down": "page_down",
    "pagedown": "page_down", **{f"f{n}": f"f{n}" for n in range(1, 21)},
    # Media and volume (virtual keys may use them; they act on the whole computer).
    "play_pause": "media_play_pause", "next_track": "media_next",
    "previous_track": "media_previous", "volume_up": "media_volume_up",
    "volume_down": "media_volume_down", "mute": "media_volume_mute",
}  # fmt: skip
# Characters that separate keys ("+") or combinations (",") are written as words.
_CHAR_NAMES = {"plus": "+", "comma": ","}
_MODIFIERS = {"shift": "shift", "alt": "alt", "option": "alt", "ctrl": "ctrl", "control": "ctrl"}


def parse_keys(combo: str) -> tuple[list[str], str] | None:
    """ "cmd+shift+n" → (["cmd", "shift"], "n"); None if a part is unknown. "cmd" and "mod"
    are Cmd on macOS and Ctrl elsewhere."""
    parts = [p.strip().lower() for p in combo.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return None
    *mods, key = parts
    modifiers = []
    for mod in mods:
        if mod in ("cmd", "command", "mod", "meta", "win"):
            modifiers.append("cmd" if sys.platform == "darwin" else "ctrl")
        elif mod in _MODIFIERS:
            modifiers.append(_MODIFIERS[mod])
        else:
            return None
    if key not in _KEY_NAMES and key not in _CHAR_NAMES and len(key) != 1:
        return None
    return modifiers, key


def press_keys(combo: str) -> None:
    """Press a key combination like "cmd+shift+n", "enter" or "down"."""
    from pynput.keyboard import Controller, Key, KeyCode

    parsed = parse_keys(combo)
    if parsed is None:
        raise ValueError(f"unknown keys: {combo}")
    mods, name = parsed
    keyboard = Controller()
    key = (
        getattr(Key, _KEY_NAMES[name])
        if name in _KEY_NAMES
        else KeyCode.from_char(_CHAR_NAMES.get(name, name))
    )
    modifiers = [getattr(Key, m) for m in mods]
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
    marker = f"⁣cassi-{time.monotonic()}"  # invisible, never selected by anyone
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


def clipboard_text() -> str:
    """What is on the clipboard right now (text only)."""
    read, _ = _clipboard()
    try:
        return read() or ""
    except Exception as exc:  # no clipboard access (e.g. no display)
        log.info("Zwischenablage nicht lesbar: %s", exc)
        return ""


def describe_clipboard(text: str) -> str:
    """The clipboard for the model, shortened if long."""
    if not text.strip():
        return "(The clipboard holds no text.)"
    shown = text[:MAX_SELECTION_CHARS]
    more = (
        f" … ({len(text) - MAX_SELECTION_CHARS} more characters)" if len(text) > len(shown) else ""
    )
    return f"(Text on the user's clipboard:\n«{shown}{more}»)"


# Dictated punctuation: "Hallo Komma wie geht's Fragezeichen" → "Hallo, wie geht's?".
_PUNCTUATION = [
    (r"\s*\b(?:neuer absatz|new paragraph)\b\s*", "\n\n"),
    (r"\s*\b(?:neue zeile|new line)\b\s*", "\n"),
    (r"\s*\b(?:komma|comma)\b", ","),
    (r"\s*\b(?:fragezeichen|question mark)\b", "?"),
    (r"\s*\b(?:ausrufezeichen|exclamation mark)\b", "!"),
    (r"\s*\b(?:doppelpunkt|colon)\b", ":"),
    (r"\s*\b(?:semikolon|semicolon)\b", ";"),
    (r"\s*\b(?:punkt|full stop|period)\b(?=\s*(?:$|\n|(?-i:[A-ZÄÖÜ])))", "."),
]


def dictated(text: str) -> str:
    """Spoken punctuation and line breaks written as such."""
    import re

    for pattern, replacement in _PUNCTUATION:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # "Hallo, wie" – one space after a mark, none before.
    text = re.sub(r"([,;:!?.])(?=[^\s\d,;:!?.])", r"\1 ", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


class KeyboardTool:
    def __init__(
        self,
        paste: Callable[[str], None] = paste_text,
        press: Callable[[str], None] = press_shortcut,
        front_is_own: Callable[[], bool] | None = None,
        store: KeyStore | None = None,
        press_keys: Callable[[str, int], None] | None = None,
    ) -> None:
        self._paste = paste
        self._press = press
        self._front_is_own = front_is_own or _front_is_own
        self._store = store
        self._press_keys = press_keys or press_sequence
        # Keys waiting for the user's yes: approval code → (keys, times, what they are).
        self._waiting: dict[str, tuple[str, int, str]] = {}

    @property
    def key_names(self) -> list[str]:
        """The virtual keys (for recognizing "Aufnahme starten" without the model)."""
        return [key.name for key in self._store.keys] if self._store else []

    @property
    def key_purposes(self) -> str:
        """The virtual keys with what they control, for deciding which tool a request needs."""
        keys = self._store.keys if self._store else []
        return "; ".join(f"{k.name} ({k.purpose})" if k.purpose else k.name for k in keys)

    @property
    def direct(self) -> DirectMode:
        """Whether Cassi may press keys of its own choosing: never, ask, always."""
        return self._store.direct if self._store else "never"

    @property
    def spec(self) -> ToolSpec:
        properties: dict[str, Any] = {
            "text": {"type": "string", "description": "Exactly what to type"},
            "shortcut": {"type": "string", "enum": [*SHORTCUTS, *SINGLE_KEYS]},
        }
        description = (
            "Type dictated 'text' word for word into the field the user is in, or press "
            "a 'shortcut' in the front app."
        )
        if self._store is not None:
            properties["new_key"] = {
                "type": "string",
                "description": "Name for a new virtual key, e.g. 'OBS Aufnahme starten'",
            }
            properties["purpose"] = {
                "type": "string",
                "description": "With new_key: what it does and in which program",
            }
            description += (
                " Virtual keys: 'new_key' makes a key combination no shortcut uses; the user "
                "sets it up as a hotkey in a program, and pressing it controls that program."
            )
            keys = self._store.keys
            if keys:
                listed = "; ".join(f"{k.name} ({k.purpose})" if k.purpose else k.name for k in keys)
                properties["key"] = {
                    "type": "string",
                    "enum": [k.name for k in keys],
                    "description": f"A virtual key to press: {listed}",
                }
                properties["remove_key"] = {"type": "string", "enum": [k.name for k in keys]}
        if self.direct != "never":
            properties["keys"] = {
                "type": "string",
                "description": "Keys to press when that does what the user wants: a "
                "combination like 'ctrl+shift+t', 'enter', 'down', 'f5', 'alt+tab'; several in "
                "turn separated by ', '",
            }
            description += " 'key' or 'keys' presses keys of your choosing."
        else:
            description += " The user has not allowed pressing other keys (nor virtual ones)."
        if "key" in properties or "keys" in properties:
            properties["times"] = {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TIMES,
                "description": "How often, e.g. 3 for 'dreimal runter'",
            }
        return ToolSpec(
            name="use_keyboard",
            description=description,
            parameters={"type": "object", "properties": properties},
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        text = str(arguments.get("text") or "")
        shortcut = str(arguments.get("shortcut") or "")
        key = str(arguments.get("key") or "")
        keys = str(arguments.get("keys") or "")
        try:
            times = max(1, min(int(arguments.get("times") or 1), MAX_TIMES))
        except (TypeError, ValueError):
            times = 1
        if approval := str(arguments.get("approval") or ""):
            return await self._approved(approval)
        if new_key := str(arguments.get("new_key") or ""):
            return self._create(new_key, str(arguments.get("purpose") or ""))
        if removed := str(arguments.get("remove_key") or ""):
            return self._remove(removed)
        virtual = self._store.get(key) if self._store and key else None
        if key and virtual is None:
            return ToolResult(
                f"There is no virtual key '{key}'. Known: {', '.join(self.key_names)}.",
                ok=False,
                say=speech(f"Eine Taste {key} kenne ich nicht.", f"I don't know a key {key}."),
            )
        if not text and not keys and virtual is None and shortcut not in (*SHORTCUTS, *SINGLE_KEYS):
            return ToolResult(
                "Give 'text' to type or a 'shortcut' from the list.",
                ok=False,
                say=speech("Was soll ich tippen?", "What should I type?"),
            )
        pressed = virtual.keys if virtual else keys
        if pressed and (problem := check_keys(pressed)):
            return ToolResult(
                f"Cannot press '{pressed}': {problem}",
                ok=False,
                say=speech("Diese Tasten kenne ich nicht.", "I don't know these keys."),
            )
        if not (pressed and is_global(pressed)) and await asyncio.to_thread(self._front_is_own):
            return ToolResult(
                "Refused: Cassi's own window is in front.",
                ok=False,
                action="Tastatur: Cassis eigenes Fenster ist vorn",
                say=speech(
                    "Vorn ist gerade mein eigenes Fenster. Klick erst in das Feld, in das ich "
                    "schreiben soll.",
                    "My own window is in front. Click into the field I should type in first.",
                ),
            )
        if pressed:
            label = f"{virtual.name} ({virtual.keys})" if virtual else keys
            return await self._press_allowed(pressed, times, label)
        if text:
            text = dictated(text)
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

    def _create(self, name: str, purpose: str) -> ToolResult:
        assert self._store is not None
        made = self._store.put(name, purpose)
        if isinstance(made, str):
            return ToolResult(
                f"No virtual key made: {made}",
                ok=False,
                say=speech(f"Das ging nicht: {made}", f"That didn't work: {made}"),
            )
        allowed = (
            ""
            if self.direct != "never"
            else " Pressing keys is not allowed yet: the user can allow it in the management "
            "page, tab Tastatur."
        )
        return ToolResult(
            f"Made the virtual key '{made.name}' = {made.keys}. Tell the user to set "
            f"{made.keys} as the hotkey in the program; where its hotkey field waits for a key "
            f"press, you can press it once they say so.{allowed}",
            action=f"Virtuelle Taste angelegt: {made.name} = {made.keys}",
            say=speech(
                f"Ich habe die Taste {made.name} angelegt: {_spoken(made.keys)}. Trag sie im "
                "Programm als Tastenkürzel ein, oder sag Bescheid, wenn das Feld für das "
                "Kürzel bereit ist, dann drücke ich sie.",
                f"I made the key {made.name}: {made.keys}. Set it as the hotkey in the "
                "program, or tell me when its hotkey field is ready and I'll press it.",
            ),
        )

    def _remove(self, name: str) -> ToolResult:
        if self._store is None or not self._store.remove(name):
            return ToolResult(f"There is no virtual key '{name}'.", ok=False)
        return ToolResult(
            f"Removed the virtual key '{name}'.",
            action=f"Virtuelle Taste entfernt: {name}",
            say=speech(f"Die Taste {name} ist entfernt.", f"The key {name} is removed."),
        )

    async def _press_allowed(self, keys: str, times: int, label: str) -> ToolResult:
        """Keys Cassi chose to press (virtual or any): as far as the user allows."""
        mode = self.direct
        if mode == "never":
            return ToolResult(
                "Refused: the user has not allowed Cassi to press keys.",
                ok=False,
                action=f"Tastatur: Tasten drücken nicht erlaubt ({label})",
                say=speech(
                    "Tasten drücken darf ich nicht. Das kannst du in der Verwaltung unter "
                    "Tastatur erlauben.",
                    "I'm not allowed to press keys. You can allow it in the management page "
                    "under Tastatur.",
                ),
            )
        shown = label + (f" ×{times}" if times > 1 else "")
        if mode == "ask":
            # A code only this tool knows: the model cannot approve on the user's behalf.
            code = secrets.token_hex(8)
            self._waiting = {code: (keys, times, shown)}
            return ToolResult(
                f"Nothing pressed yet: asked the user before pressing {shown}.",
                ok=False,
                action=f"Tastatur: Rückfrage vor {shown}",
                say=speech(f"Soll ich {_spoken(keys)} drücken?", f"Shall I press {keys}?"),
                confirm=ToolCall("confirmed", "use_keyboard", {"approval": code}),
            )
        await asyncio.to_thread(self._press_keys, keys, times)
        return _pressed(shown)

    async def _approved(self, code: str) -> ToolResult:
        waiting = self._waiting.pop(code, None)
        if waiting is None:
            return ToolResult("Refused: this approval is unknown or used.", ok=False)
        keys, times, shown = waiting
        await asyncio.to_thread(self._press_keys, keys, times)
        return _pressed(shown)


def _pressed(shown: str) -> ToolResult:
    return ToolResult(
        f"Pressed {shown}.",
        action=f"Tastatur: {shown} gedrückt",
        say=speech("Erledigt.", "Done."),
    )


# How keys are said: "ctrl+shift+t" → "Strg Shift T".
_SPOKEN_KEYS = {
    "ctrl": "Strg", "control": "Strg", "cmd": "Command", "alt": "Alt", "shift": "Shift",
    "enter": "Enter", "esc": "Escape", "escape": "Escape", "space": "Leertaste",
    "backspace": "Rücktaste", "delete": "Entfernen", "up": "Pfeil hoch", "down": "Pfeil runter",
    "left": "Pfeil links", "right": "Pfeil rechts", "tab": "Tab",
}  # fmt: skip


def _spoken(keys: str) -> str:
    said = []
    for combo in combos(keys):
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        said.append(" ".join(_SPOKEN_KEYS.get(p, p.upper() if len(p) <= 3 else p) for p in parts))
    return ", dann ".join(said)


def _front_is_own() -> bool:
    from cassi.tools.screen import is_own, list_windows

    windows = list_windows()
    return bool(windows) and is_own(windows[0])
