"""The reflexes Cassi has: fixed rules, described scenes and learned situations.

Kept in ``reflexes.json`` in the data folder, read again when it changes (the management
UI edits it while Cassi runs). Learned situations carry the picture's impression (a
vector from the encoder), never the picture itself.

Kinds (``Reflex.kind``), cheapest first:
- ``color``: a colour covers part of a region ("die Lebensleiste wird rot")
- ``change``: a region changes ("im Chat kommt etwas Neues")
- ``text``: OCR in a region finds a word, or a number below/above a limit
- ``scene``: the picture fits a description ("ein Ladebildschirm") – the encoder
- ``learned``: the picture looks like a situation seen before – the encoder
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

Kind = Literal["color", "change", "text", "scene", "learned"]
KINDS: tuple[Kind, ...] = ("color", "change", "text", "scene", "learned")
# 0–1000 of the watched picture's width and height: x1, y1, x2, y2 (how Qwen-VL answers).
Box = tuple[int, int, int, int]
MAX_NAME = 60
# Learned situations: strength gained by "gut so", lost by "falsch"; below FORGET gone.
GOOD = 0.25
BAD = 0.5
FORGET = 0.1
# Unused reflexes fade a little every day.
FADE_PER_DAY = 0.97


@dataclass(frozen=True, slots=True)
class Action:
    say: str = ""
    keys: str = ""
    # Wake the thinking model (with the picture) instead of a fixed reaction.
    think: bool = False

    def describe(self) -> str:
        parts = []
        if self.say:
            parts.append(f"sagt „{self.say}“")
        if self.keys:
            parts.append(f"drückt {self.keys}")
        if self.think:
            parts.append("denkt nach")
        return ", ".join(parts) or "nichts"


@dataclass(frozen=True, slots=True)
class Reflex:
    name: str
    kind: Kind
    action: Action
    region: Box | None = None
    # color: rgb [r, g, b], tolerance (0–255), share (0–1); change: threshold (0–255 mean);
    # text: contains / below / above; scene: label, chance (0–1); learned: see ``vector``.
    params: dict[str, Any] = field(default_factory=dict)
    vector: tuple[float, ...] = ()
    # Learned only: how often the thinking model chose this before it became a reflex.
    seen: int = 1
    active: bool = True
    strength: float = 1.0
    uses: int = 0
    last_used: float = 0.0  # epoch seconds
    cooldown: float = 2.0
    source: str = "rule"  # rule, shown, described, practice

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["vector"] = [round(float(v), 5) for v in self.vector]
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Reflex | None:
        try:
            kind = data["kind"]
            if kind not in KINDS:
                return None
            region = data.get("region")
            return cls(
                name=str(data["name"]),
                kind=kind,
                action=Action(
                    **{
                        k: v
                        for k, v in (data.get("action") or {}).items()
                        if k in ("say", "keys", "think")
                    }
                ),  # fmt: skip
                region=tuple(int(v) for v in region) if region else None,  # type: ignore[arg-type]
                params=dict(data.get("params") or {}),
                vector=tuple(float(v) for v in data.get("vector") or ()),
                seen=int(data.get("seen", 1)),
                active=bool(data.get("active", True)),
                strength=float(data.get("strength", 1.0)),
                uses=int(data.get("uses", 0)),
                last_used=float(data.get("last_used", 0.0)),
                cooldown=float(data.get("cooldown", 2.0)),
                source=str(data.get("source", "rule")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def describe(self) -> str:
        what = {
            "color": lambda: f"Farbe {self.params.get('rgb')} in {self.region}",
            "change": lambda: f"Änderung in {self.region or 'ganzem Bild'}",
            "text": lambda: (
                "Text "
                + " ".join(
                    f"{k} {self.params[k]}"
                    for k in ("contains", "below", "above")
                    if k in self.params
                )
            ),  # fmt: skip
            "scene": lambda: f"Szene „{self.params.get('label')}“",
            "learned": lambda: (
                "gelernte Situation" + ("" if self.active else f" (noch in Übung, {self.seen}×)")
            ),
        }[self.kind]()
        return f"{what} → {self.action.describe()}"


def effective_strength(reflex: Reflex, now: float) -> float:
    """A learned reflex's strength, faded a little for every day it was not used."""
    if reflex.source == "rule" or not reflex.last_used:
        return reflex.strength
    days = (now - reflex.last_used) / 86400
    return float(reflex.strength * FADE_PER_DAY ** max(0.0, days - 1))


def _normalize(name: str) -> str:
    return " ".join(re.findall(r"\w+", name.casefold()))


class ReflexStore:
    """``reflexes.json``: {"may_press": false, "stop_at": 0, "reflexes": [...]}."""

    def __init__(self, path: Path, clock: Any = time.time) -> None:
        self.path = path
        self._clock = clock
        self._stamp: tuple[float, int] | None = None
        self._reflexes: list[Reflex] = []
        self._may_press = False
        self._stop_at = 0.0

    def _load(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            self._stamp, self._reflexes, self._may_press, self._stop_at = None, [], False, 0.0
            return
        stamp = (stat.st_mtime, stat.st_size)
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Reflexe nicht lesbar (%s): %s", self.path, exc)
            self._reflexes, self._may_press = [], False
            return
        if not isinstance(raw, dict):
            raw = {}
        self._may_press = bool(raw.get("may_press", False))
        self._stop_at = float(raw.get("stop_at", 0.0) or 0.0)
        loaded = [Reflex.from_json(r) for r in raw.get("reflexes") or [] if isinstance(r, dict)]
        self._reflexes = [r for r in loaded if r is not None]

    def _save(self) -> None:
        data = {
            "may_press": self._may_press,
            "stop_at": self._stop_at,
            "reflexes": [r.to_json() for r in self._reflexes],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written aside and swapped in: the watching thread never reads half a file.
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        temporary.replace(self.path)
        self._stamp = None

    @property
    def reflexes(self) -> list[Reflex]:
        self._load()
        return list(self._reflexes)

    @property
    def may_press(self) -> bool:
        """Whether reflexes may press keys (set by the user in the management UI only)."""
        self._load()
        return self._may_press

    @property
    def stop_at(self) -> float:
        """When the management UI or tray last asked to stop watching (epoch seconds)."""
        self._load()
        return self._stop_at

    def set_may_press(self, allowed: bool) -> None:
        self._load()
        self._may_press = allowed
        self._save()

    def request_stop(self) -> None:
        self._load()
        self._stop_at = self._clock()
        self._save()

    def get(self, name: str) -> Reflex | None:
        wanted = _normalize(name)
        return next((r for r in self.reflexes if _normalize(r.name) == wanted), None)

    def put(self, reflex: Reflex) -> str | None:
        """Add or replace (by name); why not, or None."""
        if not _normalize(reflex.name):
            return "Der Name fehlt."
        if len(reflex.name) > MAX_NAME:
            return f"Der Name darf höchstens {MAX_NAME} Zeichen haben."
        self._load()
        wanted = _normalize(reflex.name)
        self._reflexes = [r for r in self._reflexes if _normalize(r.name) != wanted] + [reflex]
        self._save()
        return None

    def remove(self, name: str) -> bool:
        self._load()
        wanted = _normalize(name)
        kept = [r for r in self._reflexes if _normalize(r.name) != wanted]
        if len(kept) == len(self._reflexes):
            return False
        self._reflexes = kept
        self._save()
        return True

    def used(self, name: str) -> None:
        self._update(name, lambda r: replace(r, uses=r.uses + 1, last_used=self._clock()))

    def judge(self, name: str, good: bool) -> Reflex | None:
        """ "Gut so" strengthens, "falsch" weakens; too weak a reflex is forgotten."""
        found = self.get(name)
        if found is None:
            return None
        strength = min(2.0, found.strength + GOOD) if good else found.strength - BAD
        if strength < FORGET:
            self.remove(name)
            return replace(found, strength=0.0)
        judged = replace(found, strength=strength, active=True if good else found.active)
        self.put(judged)
        return judged

    def fade(self) -> list[str]:
        """Forget learned reflexes that faded away unused (see ``effective_strength``)."""
        self._load()
        now = self._clock()
        gone = [
            r.name
            for r in self._reflexes
            if r.source != "rule" and effective_strength(r, now) < FORGET
        ]
        if gone:
            self._reflexes = [r for r in self._reflexes if r.name not in gone]
            self._save()
        return gone

    def _update(self, name: str, change: Any) -> None:
        found = self.get(name)
        if found is not None:
            self.put(change(found))
