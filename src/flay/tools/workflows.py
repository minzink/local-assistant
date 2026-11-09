"""Workflows shown once and repeated on request.

"Merk dir einen Ablauf: Release bauen" starts recording what the user does: clicks (with
the clicked element's label from accessibility, and its place in the window as a fallback),
keyboard shortcuts and special keys, scrolling. "Ablauf fertig" stores it. "Mach den Ablauf
Release bauen" plays it back: each element is looked up again by its label – windows move,
menus change place – and only without a label by its place in the window.

Typed text is not recorded (it may be a password); a workflow fills in nothing. Steps that
press Enter or click labels like "Löschen" make Flay ask before it plays the workflow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flay.core.messages import ToolCall
from flay.llm.base import ToolSpec
from flay.tools.base import ToolResult, speech
from flay.tools.pointer import (
    Element,
    accessibility_allowed,
    best_matches,
    click,
    element_at,
    element_groups,
    is_risky,
    is_visible,
)
from flay.tools.screen import find_window, is_own, list_windows

log = logging.getLogger(__name__)

MAX_RECORD_SECONDS = 600.0
DOUBLE_CLICK_SECONDS = 0.35
# Waiting for an element to appear while playing (a menu opening, a dialog loading).
FIND_SECONDS = 5.0
MAX_PAUSE = 1.5
MIN_PAUSE = 0.3
_MODIFIERS = {"cmd", "cmd_l", "cmd_r", "ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l", "alt_gr",
              "shift", "shift_l", "shift_r"}  # fmt: skip
_SPECIAL = {
    "enter", "tab", "esc", "up", "down", "left", "right", "page_up", "page_down", "home",
    "end", "delete", "backspace", "space", *(f"f{n}" for n in range(1, 13)),
}  # fmt: skip
_RISKY_KEYS = {"enter", "delete", "backspace"}
_GENERIC = {"ablauf", "der ablauf", "den ablauf", "workflow", "makro", "macro", "aufnahme"}


@dataclass
class Step:
    kind: str  # "click", "keys" or "scroll"
    app: str = ""
    label: str = ""
    role: str = ""
    button: str = "left"
    # Place in the app's window, 0..1 from its top left: the fallback without a label.
    rel: tuple[float, float] | None = None
    keys: list[str] = field(default_factory=list)
    dy: float = 0.0
    # Seconds since the step before (pauses are kept, shortened).
    wait: float = 0.0

    def describe(self) -> str:
        if self.kind == "click":
            what = f"„{self.label}“" if self.label else "eine Stelle"
            return f"Klick auf {what} in {self.app or 'einem Fenster'}"
        if self.kind == "keys":
            return "Tasten " + "+".join(self.keys)
        return f"Scrollen in {self.app or 'einem Fenster'}"


def _key_name(key: Any) -> str | None:
    """A pynput key as a name for playing it back ("cmd", "enter", "s")."""
    name = getattr(key, "name", None)
    if name:
        return str(name)
    char = getattr(key, "char", None)
    if isinstance(char, str) and char:
        if ord(char[0]) < 32:  # Ctrl+letter arrives as a control character on Windows
            return chr(ord(char[0]) + 96)
        return char.lower()
    return None


class Recorder:
    """Listens to the mouse and keyboard until stopped."""

    def __init__(self, ignore_key: str = "") -> None:
        self.steps: list[Step] = []
        self._ignore = ignore_key.lower()
        self._held: set[str] = set()
        self._last = time.monotonic()
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._listeners: list[Any] = []

    def start(self) -> None:
        from pynput import keyboard, mouse

        self._listeners = [
            mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll),
            keyboard.Listener(on_press=self._on_press, on_release=self._on_release),
        ]
        for listener in self._listeners:
            listener.start()

    def stop(self) -> list[Step]:
        for listener in self._listeners:
            listener.stop()
        self._listeners = []
        return self.steps

    @property
    def expired(self) -> bool:
        return time.monotonic() - self._started > MAX_RECORD_SECONDS

    def _add(self, step: Step) -> None:
        with self._lock:
            now = time.monotonic()
            step.wait = round(now - self._last, 2)
            self._last = now
            self.steps.append(step)

    def _on_click(self, x: float, y: float, button: Any, pressed: bool) -> None:
        if not pressed:
            return
        try:
            self._click((float(x), float(y)), "right" if "right" in str(button) else "left")
        except Exception as exc:  # never break the listener
            log.debug("Klick nicht aufgezeichnet: %s", exc)

    def _click(self, point: tuple[float, float], button: str) -> None:
        windows = list_windows()
        window = next(
            (w for w in windows if _inside(point, w.bounds)),
            None,
        )
        if window is not None and is_own(window):
            return  # talking to Flay is not part of the workflow
        previous = self.steps[-1] if self.steps else None
        if (
            previous is not None
            and previous.kind == "click"
            and button == "left"
            and time.monotonic() - self._last < DOUBLE_CLICK_SECONDS
        ):
            previous.button = "double"
            return
        element = element_at(point)
        rel = None
        if window is not None:
            left, top, width, height = window.bounds
            if width and height:
                rel = (round((point[0] - left) / width, 4), round((point[1] - top) / height, 4))
        self._add(
            Step(
                "click",
                app=(element.app if element and element.app else window.app if window else ""),
                label=element.label if element else "",
                role=element.role if element else "",
                button=button,
                rel=rel,
            )
        )

    def _on_scroll(self, x: float, y: float, dx: float, dy: float) -> None:
        previous = self.steps[-1] if self.steps else None
        if previous is not None and previous.kind == "scroll" and time.monotonic() - self._last < 1:
            previous.dy += dy
            self._last = time.monotonic()
            return
        window = next((w for w in list_windows() if _inside((x, y), w.bounds)), None)
        if window is not None and is_own(window):
            return
        self._add(Step("scroll", app=window.app if window else "", dy=dy))

    def _on_press(self, key: Any) -> None:
        name = _key_name(key)
        if name is None or name == self._ignore:
            return
        if name in _MODIFIERS:
            self._held.add(name.split("_")[0])
            return
        command = self._held - {"shift"}
        if command or name in _SPECIAL:
            self._add(Step("keys", keys=[*sorted(self._held), name]))
        # Plain typing is not recorded: it may be a password.

    def _on_release(self, key: Any) -> None:
        name = _key_name(key)
        if name in _MODIFIERS:
            self._held.discard(name.split("_")[0])


def _inside(point: tuple[float, float], bounds: tuple[int, int, int, int]) -> bool:
    left, top, width, height = bounds
    return left <= point[0] <= left + width and top <= point[1] <= top + height


class Store:
    """Workflows by name in a JSON file in Flay's data folder."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, list[Step]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        workflows: dict[str, list[Step]] = {}
        for name, steps in raw.items():
            workflows[name] = [
                Step(**{**step, "rel": tuple(step["rel"]) if step.get("rel") else None})
                for step in steps
            ]
        return workflows

    def save(self, workflows: dict[str, list[Step]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: [asdict(step) for step in steps] for name, steps in workflows.items()}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def match_name(wanted: str, names: list[str]) -> str | None:
    """The stored workflow a spoken name means (case and small differences aside)."""
    from flay.tools.phpstorm import match_names

    found = match_names(wanted, names)
    return found[0] if found else None


def _activate(app: str) -> None:
    if sys.platform != "darwin" or not app:
        return
    from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace

    for running in NSWorkspace.sharedWorkspace().runningApplications():
        if str(running.localizedName() or "") == app:
            running.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            return


def locate(step: Step) -> tuple[float, float] | None:
    """Where to click for a recorded step now: its element by label, else its place."""
    if step.label:
        for group in element_groups(step.app or None):
            matches = best_matches(step.label, "", group)
            same_role = [e for e in matches if e.role == step.role] or matches
            visible: list[Element] = [e for e in same_role if is_visible(e)]
            if visible:
                return visible[0].centre
            if group and group[0].app == step.app:
                break  # the app is there, the element (not yet) visible
    if step.rel is not None:
        window = find_window(step.app, list_windows()) if step.app else None
        if window is not None:
            left, top, width, height = window.bounds
            return left + step.rel[0] * width, top + step.rel[1] * height
    return None


def press_keys(keys: list[str]) -> None:
    from pynput.keyboard import Controller, Key

    keyboard = Controller()
    resolved = [Key[name] if name in Key.__members__ else name for name in keys]
    for key in resolved:
        keyboard.press(key)
    for key in reversed(resolved):
        keyboard.release(key)


def scroll_by(dy: float) -> None:
    from pynput.mouse import Controller

    Controller().scroll(0, dy)


async def play(steps: list[Step]) -> tuple[int, str | None]:
    """Play the steps; how many were done and, if one failed, which."""
    for number, step in enumerate(steps, 1):
        await asyncio.sleep(min(max(step.wait, MIN_PAUSE), MAX_PAUSE))
        if step.kind == "keys":
            await asyncio.to_thread(press_keys, step.keys)
            continue
        await asyncio.to_thread(_activate, step.app)
        if step.kind == "scroll":
            await asyncio.to_thread(scroll_by, step.dy)
            continue
        deadline = time.monotonic() + FIND_SECONDS
        point = None
        while point is None and time.monotonic() < deadline:
            point = await asyncio.to_thread(locate, step)
            if point is None:
                await asyncio.sleep(0.4)
        if point is None:
            return number - 1, step.describe()
        target = next((w for w in list_windows() if _inside(point, w.bounds)), None)
        if target is not None and is_own(target):
            return number - 1, step.describe()
        await asyncio.to_thread(click, point, step.button)
    return len(steps), None


def risky_steps(steps: list[Step]) -> list[str]:
    risky = [s.describe() for s in steps if s.kind == "click" and s.label and is_risky(s.label)]
    risky += [s.describe() for s in steps if s.kind == "keys" and _RISKY_KEYS & set(s.keys)]
    return risky


class WorkflowTool:
    def __init__(self, path: Path, ignore_key: str = "") -> None:
        self._store = Store(path)
        self._ignore = ignore_key
        self._recording: tuple[str, Recorder] | None = None

    @property
    def recording(self) -> str | None:
        """Name of the workflow being recorded, if any."""
        return self._recording[0] if self._recording is not None else None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="workflow",
            description=(
                "Workflows the user shows once and Flay repeats: 'start' recording one (the "
                "user then does it), 'stop' and save it, 'run' a saved one, 'list' them, "
                "'delete' one. 'name' is the workflow's name, e.g. 'Release bauen'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "run", "list", "delete"],
                    },
                    "name": {"type": "string"},
                },
                "required": ["action"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = str(arguments.get("action") or "")
        name = " ".join(str(arguments.get("name") or "").split())
        if action == "start":
            return self._start(name)
        if action == "stop":
            return self._stop(name)
        if action == "list":
            return self._list()
        if action == "delete":
            return self._delete(name)
        if action == "run":
            return await self._run(name, bool(arguments.get("confirmed")), arguments)
        return ToolResult(f"Unknown action '{action}'.", ok=False)

    def _start(self, name: str) -> ToolResult:
        if not accessibility_allowed(prompt=True):
            from flay.tools.pointer import PERMISSION

            return PERMISSION
        if self._recording is not None:
            self._recording[1].stop()
        recorder = Recorder(self._ignore)
        recorder.start()
        self._recording = (name or f"Ablauf {time.strftime('%H:%M')}", recorder)
        shown = self._recording[0]
        return ToolResult(
            f"Recording workflow '{shown}': the user now does it; 'stop' saves it.",
            action=f"Ablauf: Aufnahme „{shown}“ gestartet",
            say=speech(
                f"Ich schaue zu. Zeig mir {shown} und sag Bescheid, wenn du fertig bist. "
                "Getippten Text merke ich mir nicht.",
                f"I'm watching. Show me {shown} and tell me when you're done. I don't record "
                "typed text.",
            ),
        )

    def _stop(self, name: str) -> ToolResult:
        if self._recording is None:
            return ToolResult(
                "Nothing is being recorded.",
                ok=False,
                say=speech("Ich nehme gerade keinen Ablauf auf.", "I'm not recording."),
            )
        recorded, recorder = self._recording
        self._recording = None
        steps = recorder.stop()
        # The last click is usually the one that brings Flay to the front to say "fertig".
        steps = [s for s in steps if not (s.kind == "click" and not s.label and not s.app)]
        if not steps:
            return ToolResult(
                "Nothing was recorded.",
                ok=False,
                say=speech(
                    "Ich habe keine Schritte gesehen; es ist nichts gespeichert.",
                    "I saw no steps; nothing was saved.",
                ),
            )
        # "Ablauf fertig" names no workflow: the name from the start stays.
        final = name if name and name.lower() not in _GENERIC else recorded
        workflows = self._store.load()
        workflows[final] = steps
        self._store.save(workflows)
        unnamed = sum(1 for s in steps if s.kind == "click" and not s.label)
        note = (
            speech(
                f" {unnamed} Klicks ohne Beschriftung finde ich nur über ihre Position.",
                f" {unnamed} clicks without a label I find only by position.",
            )
            if unnamed
            else speech("", "")
        )
        return ToolResult(
            f"Saved workflow '{final}' with {len(steps)} steps: "
            + "; ".join(s.describe() for s in steps),
            action=f"Ablauf „{final}“ gespeichert ({len(steps)} Schritte)",
            say={
                language: text + note[language]
                for language, text in speech(
                    f"Ich habe {final} mit {len(steps)} Schritten gespeichert.",
                    f"I saved {final} with {len(steps)} steps.",
                ).items()
            },
        )

    def _list(self) -> ToolResult:
        names = list(self._store.load())
        if not names:
            return ToolResult(
                "No workflows saved.",
                say=speech("Ich kenne noch keine Abläufe.", "I don't know any workflows yet."),
            )
        listed = ", ".join(names)
        return ToolResult(
            f"Saved workflows: {listed}.",
            say=speech(f"Ich kenne: {listed}.", f"I know: {listed}."),
        )

    def _delete(self, name: str) -> ToolResult:
        workflows = self._store.load()
        found = match_name(name, list(workflows)) if name else None
        if found is None:
            return _unknown(name)
        del workflows[found]
        self._store.save(workflows)
        return ToolResult(
            f"Deleted workflow '{found}'.",
            action=f"Ablauf „{found}“ gelöscht",
            say=speech(f"Ich habe {found} gelöscht.", f"I deleted {found}."),
        )

    async def _run(self, name: str, confirmed: bool, arguments: dict[str, Any]) -> ToolResult:
        workflows = self._store.load()
        found = match_name(name, list(workflows)) if name else None
        if found is None:
            return _unknown(name)
        steps = workflows[found]
        if not confirmed and (risky := risky_steps(steps)):
            return ToolResult(
                f"Asked the user before running '{found}': it has risky steps ({risky}).",
                ok=False,
                say=speech(
                    f"{found} enthält {risky[0]}. Soll ich den Ablauf ausführen?",
                    f"{found} includes {risky[0]}. Should I run it?",
                ),
                confirm=ToolCall("confirmed", "workflow", {**arguments, "confirmed": True}),
            )
        if not accessibility_allowed(prompt=True):
            from flay.tools.pointer import PERMISSION

            return PERMISSION
        done, failed = await play(steps)
        if failed is not None:
            return ToolResult(
                f"Workflow '{found}' stopped at step {done + 1}: {failed} not found.",
                ok=False,
                action=f"Ablauf „{found}“ abgebrochen bei Schritt {done + 1}",
                say=speech(
                    f"Ich habe {done} Schritte gemacht; bei Schritt {done + 1} finde ich "
                    f"{re.sub(r'^Klick auf ', '', failed)} nicht.",
                    f"I did {done} steps; at step {done + 1} I can't find the target.",
                ),
            )
        return ToolResult(
            f"Ran workflow '{found}' ({len(steps)} steps).",
            action=f"Ablauf „{found}“ ausgeführt ({len(steps)} Schritte)",
            say=speech(f"Ich habe {found} ausgeführt.", f"I ran {found}."),
        )


def _unknown(name: str) -> ToolResult:
    return ToolResult(
        f"No workflow named '{name}'.",
        ok=False,
        say=speech(
            f"Einen Ablauf {name} kenne ich nicht.".replace("  ", " "),
            f"I don't know a workflow {name}.".replace("  ", " "),
        ),
    )
