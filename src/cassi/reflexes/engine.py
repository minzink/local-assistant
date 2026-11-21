"""Watching with reflexes – fast reactions modelled on a human's.

A human does not think through every moment: inborn reflexes act in milliseconds, learned
habits react without deliberation, and only what is new or unclear is thought about – and
what was decided then becomes a habit. Cassi watches the same way, in layers:

0. Perception: a picture of the watched window or monitor several times a second; how much
   changed decides what else is looked at.
1. Inborn reflexes: fixed rules (a colour, a change, a word or number via OCR) – no model,
   a few milliseconds.
2. Learned reflexes: the small model (SigLIP's encoder, ~40 ms) turns the picture into an
   impression; one close to a learned situation triggers its reaction at once, and
   described scenes ("a loading screen") are recognized the same way.
3. Thinking: only for what is new (and only with a goal), at most every few seconds, the
   vision model decides with the picture. Its decision is learned: after a few times the
   same in a similar situation it becomes a reflex (layer 2).

A reflex fires when its condition becomes true (not again while it stays true) and then
rests for its cooldown, like a nerve cell. "Gut so" strengthens the last one, "falsch"
weakens it; unused learned reflexes fade. Pressing keys needs the user's leave (management
UI) – without it reflexes only speak. Private windows and a paused screen are not watched.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cassi.reflexes.capture import FrameSource, Target, crop
from cassi.reflexes.encoder import Encoder
from cassi.reflexes.store import Action, Reflex, ReflexStore, effective_strength

log = logging.getLogger(__name__)

THUMBNAIL = (48, 30)
# Descriptions a scene is weighed against (the picture shows the scene, not just a screen).
SCENE_OTHERS = [
    "a screenshot of a computer screen",
    "a screenshot of an app window",
    "a picture",
]
# Thinking only when the picture differs this much from what it last thought about.
NOVELTY = 0.92
STATUS_SECONDS = 2.0
PRIVACY_SECONDS = 1.0
MAX_ACTIONS_PER_SECOND = 4


@dataclass
class Settings:
    fps: float = 8.0
    encoder_hz: float = 4.0
    ocr_hz: float = 2.0
    # A learned situation counts from this similarity of impressions (cosine, 0–1).
    match: float = 0.95
    # Thinking about something new: at most this often (seconds).
    think_seconds: float = 8.0
    # The same decision in a similar situation this often: it becomes a reflex.
    promote_after: int = 2


@dataclass(frozen=True, slots=True)
class Decision:
    """What the thinking model decided for a picture."""

    action: Action
    situation: str = ""


Thinker = Callable[[Image.Image, str, list[str]], Awaitable[Decision | None]]


@dataclass
class _State:
    true: bool = False
    rest_until: float = 0.0
    streak: int = 0


@dataclass
class Status:
    watching: bool = False
    target: str = ""
    goal: str = ""
    started: float = 0.0
    frames: int = 0
    fps: float = 0.0
    fired: int = 0
    thoughts: int = 0
    last: str = ""
    note: str = ""
    recent: list[str] = field(default_factory=list)

    def remember(self, line: str) -> None:
        self.last = line
        self.recent = [f"{time.strftime('%H:%M:%S')} {line}", *self.recent][:8]


def _thumbnail(picture: Image.Image) -> np.ndarray:
    return np.asarray(picture.convert("L").resize(THUMBNAIL), dtype=np.int16)


def _color_share(picture: Image.Image, rgb: list[int], tolerance: float) -> float:
    pixels = np.asarray(picture.convert("RGB").resize((64, 64)), dtype=np.int16)
    distance = np.abs(pixels - np.array(rgb[:3], dtype=np.int16)).max(axis=2)
    return float((distance <= tolerance).mean())


_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def _text_matches(text: str, params: dict[str, Any]) -> bool:
    if (wanted := params.get("contains")) and str(wanted).casefold() not in text.casefold():
        return False
    numbers = [float(n.replace(",", ".")) for n in _NUMBER.findall(text)]
    if "below" in params and not any(n < float(params["below"]) for n in numbers):
        return False
    if "above" in params and not any(n > float(params["above"]) for n in numbers):
        return False
    return any(k in params for k in ("contains", "below", "above"))


class Observer:
    """One watching at a time: started by the observe tool, stopped by voice, tray or UI."""

    def __init__(
        self,
        store: ReflexStore,
        encoder: Encoder,
        say: Callable[[str], None],
        press: Callable[[str], None] | None = None,
        thinker: Thinker | None = None,
        settings: Settings | None = None,
        status_file: Path | None = None,
        may_press_keys: Callable[[], bool] | None = None,
        read_text: Callable[[Image.Image], str] | None = None,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self._say = say
        self._press = press
        self._thinker = thinker
        self.settings = settings or Settings()
        self._status_file = status_file
        self._may_press_keys = may_press_keys or (lambda: store.may_press)
        self._read_text = read_text or _ocr_text
        self.status = Status()
        self._task: asyncio.Task[None] | None = None
        self._thinking: asyncio.Task[None] | None = None
        self._source: FrameSource | None = None
        self._states: dict[str, _State] = {}
        self._previous: np.ndarray | None = None
        self._region_previous: dict[str, np.ndarray] = {}
        self._impression: np.ndarray | None = None
        self._seen: dict[str, bool] = {}
        self._thought_impression: np.ndarray | None = None
        self._last_encoded = 0.0
        self._last_ocr = 0.0
        self._last_think = 0.0
        self._last_privacy = 0.0
        self._private: str | None = None
        self._actions: list[float] = []
        self.last_frame: Image.Image | None = None
        self.last_fired: str | None = None
        self._goal = ""

    # Starting and stopping ------------------------------------------------------------

    @property
    def watching(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, target: Target, goal: str = "") -> None:
        if self.watching:
            self.stop("neu gestartet")
        self._goal = goal
        self._source = FrameSource(target)
        self._states, self._previous, self._impression, self._seen = {}, None, None, {}
        self._region_previous, self._thought_impression = {}, None
        self.status = Status(True, target.describe(), goal, time.time())
        for name in self.store.fade():
            log.info("Reflex verblasst: %s", name)
        self._task = asyncio.get_running_loop().create_task(self._run(self.status.started))
        log.info("Beobachte %s%s", target.describe(), f" – Ziel: {goal}" if goal else "")

    def stop(self, reason: str = "") -> bool:
        was = self.watching
        if self._task is not None:
            self._task.cancel()
        if self._thinking is not None:
            self._thinking.cancel()
        if self._source is not None:
            self._source.close()
        self._task = self._thinking = None
        if was:
            log.info("Beobachten beendet%s", f" ({reason})" if reason else "")
            self.status.watching = False
            self.status.note = reason
            self._write_status()
        return was

    # The loop ---------------------------------------------------------------------------

    async def _run(self, started: float) -> None:
        last_status = 0.0
        try:
            while True:
                begin = time.monotonic()
                if self.store.stop_at > started:
                    self.stop("über die Verwaltung beendet")
                    return
                try:
                    fired, novel = await asyncio.to_thread(self.look, begin)
                except LookupError as exc:  # the window was closed
                    self.status.note = str(exc)
                    fired, novel = [], False
                except Exception as exc:
                    log.debug("Beobachten: %s", exc)
                    self.status.note = str(exc)
                    fired, novel = [], False
                for reflex in fired:
                    await self._fire(reflex)
                if novel and not fired:
                    self._think_soon()
                if begin - last_status >= STATUS_SECONDS:
                    last_status = begin
                    self._write_status()
                elapsed = time.monotonic() - begin
                await asyncio.sleep(max(0.0, 1 / self.settings.fps - elapsed))
        except asyncio.CancelledError:
            raise

    def look(self, now: float) -> tuple[list[Reflex], bool]:
        """One picture through layers 0–2 (blocking): the reflexes that fire, and whether
        the picture is new enough to think about."""
        assert self._source is not None
        if now - self._last_privacy >= PRIVACY_SECONDS:
            self._last_privacy = now
            self._private = self._source.private()
        if self._private:
            self.status.note = f"nicht beobachtet: {self._private}"
            self._previous = None
            return [], False
        picture = self._source.grab()
        self.last_frame = picture
        self.status.frames += 1
        elapsed = time.time() - self.status.started
        self.status.fps = round(self.status.frames / elapsed, 1) if elapsed > 0 else 0.0
        thumbnail = _thumbnail(picture)
        changed = self._previous is None or float(np.abs(thumbnail - self._previous).mean()) >= 1.5
        self._previous = thumbnail

        reflexes = [r for r in self.store.reflexes if r.active or r.kind == "learned"]
        conditions: dict[str, bool] = {}
        for reflex in reflexes:
            if reflex.kind == "color":
                share = _color_share(
                    crop(picture, reflex.region),
                    list(reflex.params.get("rgb", [255, 0, 0])),
                    float(reflex.params.get("tolerance", 40)),
                )
                conditions[reflex.name] = share >= float(reflex.params.get("share", 0.3))
            elif reflex.kind == "change":
                part = _thumbnail(crop(picture, reflex.region))
                before = self._region_previous.get(reflex.name)
                self._region_previous[reflex.name] = part
                conditions[reflex.name] = before is not None and float(
                    np.abs(part - before).mean()
                ) >= float(reflex.params.get("threshold", 8))
        if (
            any(r.kind == "text" for r in reflexes)
            and now - self._last_ocr >= 1 / self.settings.ocr_hz
        ):
            self._last_ocr = now
            for reflex in reflexes:
                if reflex.kind == "text":
                    text = self._read_text(crop(picture, reflex.region))
                    conditions[reflex.name] = _text_matches(text, reflex.params)
        seeing = [r for r in reflexes if r.kind in ("scene", "learned")]
        novel = False
        if (
            (seeing or self._thinker)
            and changed
            and (now - self._last_encoded >= 1 / self.settings.encoder_hz)
        ):
            self._last_encoded = now
            impression = self.encoder.image(picture)
            self._impression = impression
            self._seen = self._recognize(picture, impression, seeing)
            novel = (
                self._thinker is not None
                and bool(self._goal)
                and (
                    self._thought_impression is None
                    or float(impression @ self._thought_impression) < NOVELTY
                )
            )
        # Between encodings what was recognized still holds: the picture has not changed
        # (a scene that appears and stays still gets its second picture).
        names = {r.name for r in seeing}
        conditions.update({n: v for n, v in self._seen.items() if n in names})
        return self._edges(reflexes, conditions, now), novel

    def _recognize(
        self, picture: Image.Image, impression: np.ndarray, seeing: list[Reflex]
    ) -> dict[str, bool]:
        """Layer 2: described scenes and learned situations, by their impression."""
        found: dict[str, bool] = {}
        scored: list[tuple[float, Reflex]] = []
        for reflex in seeing:
            seen = (
                impression
                if reflex.region is None
                else self.encoder.image(crop(picture, reflex.region))
            )
            if reflex.kind == "scene":
                chance = self.encoder.chance(
                    seen, str(reflex.params.get("label", "")), SCENE_OTHERS
                )
                found[reflex.name] = chance >= float(reflex.params.get("chance", 0.5))
            elif reflex.active and reflex.vector:
                similarity = float(np.asarray(reflex.vector, np.float32) @ seen)
                scored.append((similarity, reflex))
        # The closest learned situation only, and only if clearly closer than one asking for
        # a different reaction.
        scored.sort(key=lambda s: -s[0])
        if scored and scored[0][0] >= self.settings.match:
            best = scored[0]
            rival = next((s for s, r in scored[1:] if r.action != best[1].action), 0.0)
            found[best[1].name] = best[0] - rival >= 0.01
        for _, reflex in scored:
            found.setdefault(reflex.name, False)
        return found

    def _edges(
        self, reflexes: list[Reflex], conditions: dict[str, bool], now: float
    ) -> list[Reflex]:
        """Fire when a condition becomes true, then rest for the cooldown."""
        fired = []
        for reflex in reflexes:
            if reflex.name not in conditions:
                continue
            state = self._states.setdefault(reflex.name, _State())
            true = conditions[reflex.name]
            state.streak = state.streak + 1 if true else 0
            # Scenes need two pictures in a row: one odd frame is no scene.
            needed = 2 if reflex.kind == "scene" else 1
            if true and not state.true and state.streak >= needed and now >= state.rest_until:
                state.true = True
                state.rest_until = now + reflex.cooldown
                fired.append(reflex)
            elif not true:
                state.true = False
        return fired

    # Acting -----------------------------------------------------------------------------

    def _allowed_now(self) -> bool:
        """At most a few actions a second, whatever fires."""
        now = time.monotonic()
        self._actions = [t for t in self._actions if now - t < 1.0]
        if len(self._actions) >= MAX_ACTIONS_PER_SECOND:
            return False
        self._actions.append(now)
        return True

    async def _fire(self, reflex: Reflex) -> None:
        if not self._allowed_now():
            return
        self.last_fired = reflex.name
        self.status.fired += 1
        self.status.remember(f"{reflex.name}: {reflex.action.describe()}")
        log.info("Reflex: %s → %s", reflex.name, reflex.action.describe())
        await self._do(reflex.action)
        await asyncio.to_thread(self.store.used, reflex.name)
        if reflex.action.think:
            self._think_soon(force=True)

    async def _do(self, action: Action) -> None:
        if action.keys:
            if self._press is not None and self._may_press_keys():
                await asyncio.to_thread(self._press, action.keys)
            else:
                self.status.note = "Tasten nicht erlaubt (Verwaltung → Reflexe)"
        if action.say:
            self._say(action.say)

    # Thinking (layer 3) -----------------------------------------------------------------

    def _think_soon(self, force: bool = False) -> None:
        now = time.monotonic()
        if self._thinker is None or (self._thinking is not None and not self._thinking.done()):
            return
        if not force and now - self._last_think < self.settings.think_seconds:
            return
        if self.last_frame is None:
            return
        self._last_think = now
        self._thought_impression = self._impression
        picture, impression = self.last_frame, self._impression
        self._thinking = asyncio.get_running_loop().create_task(self._think(picture, impression))

    async def _think(self, picture: Image.Image, impression: np.ndarray | None) -> None:
        assert self._thinker is not None
        known = [r.name for r in self.store.reflexes]
        try:
            decision = await self._thinker(picture, self._goal, known)
        except Exception as exc:
            log.info("Nachdenken fehlgeschlagen: %s", exc)
            return
        self.status.thoughts += 1
        if decision is None or not (decision.action.say or decision.action.keys):
            self.status.remember("nachgedacht: nichts zu tun")
            return
        self.status.remember(f"nachgedacht: {decision.action.describe()}")
        if self._allowed_now():
            await self._do(decision.action)
        if impression is not None:
            await asyncio.to_thread(self.practise, impression, decision)

    def practise(self, impression: np.ndarray, decision: Decision) -> Reflex:
        """Learn from a decision: the same reaction in a similar situation, often enough,
        becomes a reflex (``promote_after``)."""
        for reflex in self.store.reflexes:
            if (
                reflex.kind == "learned"
                and reflex.region is None
                and reflex.action == decision.action
                and reflex.vector
                and float(np.asarray(reflex.vector, np.float32) @ impression) >= self.settings.match
            ):
                seen = reflex.seen + 1
                merged = np.asarray(reflex.vector, np.float32) * reflex.seen + impression
                merged = merged / np.linalg.norm(merged)
                active = reflex.active or seen >= self.settings.promote_after
                updated = replace(
                    reflex, vector=tuple(float(v) for v in merged), seen=seen, active=active
                )
                self.store.put(updated)
                if active and not reflex.active:
                    self.status.remember(f"neuer Reflex: {reflex.name}")
                    log.info("Aus Übung wird ein Reflex: %s", reflex.name)
                return updated
        base = decision.situation.strip() or "Situation"
        name, number = base[:50], 2
        while self.store.get(name) is not None:
            name, number = f"{base[:46]} {number}", number + 1
        learned = Reflex(
            name=name,
            kind="learned",
            action=decision.action,
            vector=tuple(float(v) for v in impression),
            active=self.settings.promote_after <= 1,
            source="practice",
            cooldown=3.0,
        )
        self.store.put(learned)
        return learned

    # Taught by the user -----------------------------------------------------------------

    def impression_now(self, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """The impression of what is shown now (blocking): of the watched picture, or the
        primary monitor when nothing is watched."""
        picture = self.last_frame if self.watching and self.last_frame is not None else None
        if picture is None:
            source = FrameSource(Target())
            if (private := source.private()) is not None:
                raise PermissionError(f"nicht erlaubt: {private}")
            picture = source.grab()
        return self.encoder.image(crop(picture, region))

    def view(self) -> tuple[Image.Image, tuple[float, float, float, float]]:
        """What reflex regions refer to (blocking): the watched picture and where it is on
        the screen (points) – or the primary monitor when nothing is watched."""
        from cassi.tools.screen import monitor_bounds

        if self.watching and self._source is not None and self.last_frame is not None:
            bounds = self._source._bounds()
            area = bounds or monitor_bounds(self._source.target.monitor)
            return self.last_frame, tuple(float(v) for v in area)  # type: ignore[return-value]
        source = FrameSource(Target())
        if (private := source.private()) is not None:
            raise PermissionError(f"nicht erlaubt: {private}")
        return source.grab(), monitor_bounds(1)

    def strength(self, reflex: Reflex) -> float:
        return effective_strength(reflex, time.time())

    def _write_status(self) -> None:
        if self._status_file is None:
            return
        try:
            data = self.status.__dict__ | {"written": time.time()}
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            self._status_file.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
        except OSError as exc:
            log.debug("Reflex-Status nicht schreibbar: %s", exc)


def _ocr_text(picture: Image.Image) -> str:
    from cassi.tools import ocr

    return " ".join(box.text for box in ocr.read_image(picture))


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
