"""Virtual keys, and whether the assistant may operate the keyboard at all.

A virtual key is one the assistant makes for itself: a free key combination no keyboard
shortcut uses (``ctrl+alt+shift+f13`` …), with a name and what it is for ("OBS: Aufnahme
starten"). Set up as a hotkey in a program (OBS, a macro tool, the system's shortcuts), it
lets the assistant control that program by pressing the key.

Pressing keys – virtual ones or any others the assistant decides on (enter, ctrl+shift+t) –
is up to the user: never (the default), after asking each time, or always.

Both live in ``keyboard.json`` in the data folder, shown and edited in the management UI
("Tastatur") and read again when it changes, so a running assistant sees changes at once.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

log = logging.getLogger(__name__)

DirectMode = Literal["never", "ask", "always"]
DIRECT_MODES: tuple[DirectMode, ...] = get_args(DirectMode)
DIRECT_LABELS: dict[DirectMode, str] = {
    "never": "nie",
    "ask": "nach Rückfrage",
    "always": "immer",
}
MAX_NAME = 60
MAX_PURPOSE = 200
MAX_KEYS = 120
MAX_TIMES = 20
# Between the combinations of a sequence ("ctrl+k, ctrl+s"): apps need a moment.
SEQUENCE_PAUSE = 0.05
# Keys that act on the whole computer, not on the front window.
GLOBAL_KEYS = frozenset(
    {"play_pause", "next_track", "previous_track", "volume_up", "volume_down", "mute"}
)
# Free keys for virtual keys: F13–F20 exist on no usual keyboard, but programs accept them as
# hotkeys (Windows and macOS). Always with modifiers: macOS uses F14/F15 alone for brightness.
_MODIFIER_SETS = ("ctrl+alt+shift", "ctrl+alt", "ctrl+shift", "alt+shift")
_F_KEYS = tuple(f"f{n}" for n in range(13, 21))


def free_keys() -> Iterator[str]:
    """Key combinations for virtual keys, in the order they are handed out."""
    for modifiers in _MODIFIER_SETS:
        for key in _F_KEYS:
            yield f"{modifiers}+{key}"


@dataclass(frozen=True, slots=True)
class VirtualKey:
    name: str
    keys: str
    # What it is for and where it is set up, e.g. "OBS: Aufnahme starten".
    purpose: str = ""


def combos(keys: str) -> list[str]:
    """ "ctrl+k, ctrl+s" → ["ctrl+k", "ctrl+s"]."""
    return [part.strip() for part in keys.split(",") if part.strip()]


def check_keys(keys: str) -> str | None:
    """Why ``keys`` cannot be pressed, or None if every combination is known."""
    from cassi.tools.keyboard import parse_keys

    parts = combos(keys)
    if not parts:
        return "Keine Tasten angegeben."
    if len(keys) > MAX_KEYS:
        return f"Höchstens {MAX_KEYS} Zeichen."
    unknown = [part for part in parts if parse_keys(part) is None]
    if unknown:
        return f"Unbekannt: {', '.join(unknown)}"
    return None


def is_global(keys: str) -> bool:
    """Only media and volume keys: they work whichever window is in front."""
    parts = combos(keys)
    return bool(parts) and all(part.lower() in GLOBAL_KEYS for part in parts)


def press_sequence(keys: str, times: int = 1) -> None:
    """Press every combination of ``keys`` in turn, ``times`` times."""
    from cassi.tools.keyboard import press_keys

    for round_ in range(max(1, min(times, MAX_TIMES))):
        for index, combo in enumerate(combos(keys)):
            if round_ or index:
                time.sleep(SEQUENCE_PAUSE)
            press_keys(combo)


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def named_key(text: str, names: Sequence[str]) -> str | None:
    """The virtual key a short request names ("Aufnahme starten", "Flay, Aufnahme starten").

    Whole words only, and the request may hold at most three other words: "Erklär mir, wie
    ich die Aufnahme starten kann" is a question, not a key press. The longest name wins.
    """
    said = f" {_normalize(text)} "
    words = len(said.split())
    hits = [
        name
        for name in names
        if (wanted := _normalize(name)) and f" {wanted} " in said
        if words <= len(wanted.split()) + 3
    ]
    return max(hits, key=len) if hits else None


class KeyStore:
    """``keyboard.json``: {"direct": "never", "keys": [{"name", "keys", "purpose"}]}."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stamp: tuple[float, int] | None = None
        self._direct: DirectMode = "never"
        self._keys: list[VirtualKey] = []

    def _load(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            self._stamp, self._direct, self._keys = None, "never", []
            return
        stamp = (stat.st_mtime, stat.st_size)
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Tastatur-Einstellungen nicht lesbar (%s): %s", self.path, exc)
            self._direct, self._keys = "never", []
            return
        direct = raw.get("direct") if isinstance(raw, dict) else None
        self._direct = direct if direct in DIRECT_MODES else "never"
        entries = raw.get("keys") if isinstance(raw, dict) else None
        self._keys = [
            VirtualKey(str(e["name"]), str(e["keys"]), str(e.get("purpose") or ""))
            for e in entries or []
            if isinstance(e, dict) and e.get("name") and e.get("keys")
        ]

    def _save(self) -> None:
        data = {
            "direct": self._direct,
            "keys": [{"name": k.name, "keys": k.keys, "purpose": k.purpose} for k in self._keys],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1, ensure_ascii=False), "utf-8")
        self._stamp = None

    @property
    def direct(self) -> DirectMode:
        self._load()
        return self._direct

    @property
    def keys(self) -> list[VirtualKey]:
        self._load()
        return list(self._keys)

    def get(self, name: str) -> VirtualKey | None:
        wanted = _normalize(name)
        return next((k for k in self.keys if _normalize(k.name) == wanted), None)

    def set_direct(self, mode: DirectMode) -> None:
        self._load()
        self._direct = mode
        self._save()

    def put(self, name: str, purpose: str = "", keys: str = "") -> VirtualKey | str:
        """Add a virtual key or change the one with this name; without ``keys`` it keeps its
        keys, or a new one gets the next free combination. The key, or why not."""
        name, purpose, keys = " ".join(name.split()), " ".join(purpose.split()), keys.strip()
        if not _normalize(name):
            return "Der Name fehlt."
        if len(name) > MAX_NAME:
            return f"Der Name darf höchstens {MAX_NAME} Zeichen haben."
        if len(purpose) > MAX_PURPOSE:
            return f"Der Zweck darf höchstens {MAX_PURPOSE} Zeichen haben."
        self._load()
        wanted = _normalize(name)
        before = next((k for k in self._keys if _normalize(k.name) == wanted), None)
        others = [k for k in self._keys if k is not before]
        if keys:
            if problem := check_keys(keys):
                return problem
            if any(k.keys == keys for k in others):
                return f"{keys} gehört schon zu einer anderen Taste."
        elif before is not None:
            keys = before.keys
        else:
            used = {k.keys for k in others}
            free = next((k for k in free_keys() if k not in used), None)
            if free is None:
                return "Alle freien Tasten sind vergeben; erst eine entfernen."
            keys = free
        key = VirtualKey(name, keys, purpose or (before.purpose if before else ""))
        self._keys = [*others, key]
        self._save()
        return key

    def remove(self, name: str) -> bool:
        self._load()
        wanted = _normalize(name)
        kept = [k for k in self._keys if _normalize(k.name) != wanted]
        if len(kept) == len(self._keys):
            return False
        self._keys = kept
        self._save()
        return True
