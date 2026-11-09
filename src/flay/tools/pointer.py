"""Acting on what is on the screen: find an element, move the mouse there, click, scroll.

Finding an element, fastest first:

1. Accessibility (macOS): the label and exact frame of the buttons, menus, links … of the
   visible apps, in a fraction of a second. Only for elements that have a label.
2. The vision model: it answers with a point – 0–1000 of the image's width and height – in
   a fresh screenshot. A new image costs ~8 s with Qwen3-VL 8B on an M2 Pro (Ollama turns
   every image into ~1,040 tokens, whatever its size; docs/roadmap.md, phase 12). If the
   element at that point does not carry the label, a second look at a crop around it
   sharpens the point.

Coordinates are screen points: the system of window bounds, mss monitors and the mouse.
On Windows the process is made DPI-aware first, so screenshots, window bounds and the
mouse all use physical pixels.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from flay.core.messages import ImagePart, Message, Role, TextPart, ToolCall
from flay.llm.base import ChatModel, TextDelta, ToolSpec
from flay.tools.base import ToolResult, speech
from flay.tools.screen import (
    ScreenPermissionError,
    Shot,
    Window,
    capture_monitor,
    capture_region,
    capture_window,
    find_window,
    is_own,
    list_windows,
    monitor_bounds,
)

log = logging.getLogger(__name__)

# Elements larger than this are areas (editors, lists), not something to click by name.
MAX_ELEMENT_SIZE = (700, 200)
# Accessibility search budget; a big app tree is cut off rather than waited for.
SEARCH_SECONDS = 1.5
MAX_ELEMENTS = 4000
# The closer look covers this share of the first image around the first guess.
ZOOM_SHARE = 1 / 3


def is_risky(label: str) -> bool:
    """Pressing it deletes, sends, buys or the like: asked for first."""
    return _RISKY.search(label) is not None


# A label counts as matching from this similarity on (difflib ratio).
MIN_LABEL_SIMILARITY = 0.85
# Left clicks on such labels are confirmed first; a right click only opens a menu.
_RISKY = re.compile(
    # Word starts only: "Weitere Informationen" is not "format".
    r"\b(?:lösch|entfern|delete|remove|send|senden|absend|abschick|submit|kauf|buy|bezahl|"
    r"pay|bestell|order|überschreib|overwrite|zurücksetz|reset|deinstall|uninstall|format|"
    r"beend|quit|verwerf|discard|push|merge|deploy|veröffentlich|publish|abonnier|subscribe|"
    r"kündig|unsubscribe|registrier|sign ?up|warenkorb|cart|checkout|zur kasse)",
    re.IGNORECASE,
)
# Words in a description that say which kind of element is meant.
_ROLE_WORDS = {
    "AXMenuBarItem": r"menü|menu",
    "AXMenuItem": r"menü|menu|eintrag|entry|item",
    "AXButton": r"button|knopf|schaltfläche|symbol|icon",
    "AXLink": r"link",
    "AXTab": r"tab|reiter",
    "AXRadioButton": r"tab|reiter|option",
    "AXCheckBox": r"checkbox|haken|kästchen",
    "AXTextField": r"feld|field|eingabe|input|suche|search",
    "AXDockItem": r"dock",
}
_CLICKABLE = frozenset(
    {
        "AXButton", "AXMenuBarItem", "AXMenuItem", "AXMenuButton", "AXPopUpButton",
        "AXCheckBox", "AXRadioButton", "AXTab", "AXLink", "AXDockItem", "AXTextField",
        "AXComboBox", "AXSearchField", "AXStaticText", "AXCell", "AXRow", "AXImage",
    }
)  # fmt: skip

Point = tuple[float, float]
Frame = tuple[float, float, float, float]


def make_dpi_aware() -> None:
    """Windows: one coordinate system (physical pixels) for screenshots and the mouse."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined, unused-ignore]
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined, unused-ignore]
        except (AttributeError, OSError):
            log.debug("DPI-Bewusstsein ließ sich nicht setzen")


make_dpi_aware()


@dataclass(frozen=True, slots=True)
class Element:
    label: str
    role: str
    frame: Frame
    app: str = ""
    # The accessibility object, to check what lies at its centre.
    ref: Any = field(default=None, compare=False, repr=False)

    @property
    def centre(self) -> Point:
        x, y, width, height = self.frame
        return x + width / 2, y + height / 2


def inside(point: Point, frame: Frame) -> bool:
    x, y, width, height = frame
    return x <= point[0] <= x + width and y <= point[1] <= y + height


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def label_score(wanted: str, label: str) -> float:
    """How well an element's label fits the text the user wants clicked (0 = not at all)."""
    a, b = normalize(wanted), normalize(label)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
    if ratio >= MIN_LABEL_SIMILARITY:
        return 0.9 * ratio
    # "Git" in "Git-Menü" or "Commit" in "Commit…" – but not two letters inside a word.
    if len(a) >= 3 and re.search(rf"\b{re.escape(a)}\b", b):
        return 0.7
    return 0.0


def best_matches(wanted: str, description: str, elements: Sequence[Element]) -> list[Element]:
    """The elements whose label fits best; a role named in the description breaks ties."""
    scored = [(label_score(wanted, e.label), e) for e in elements]
    top = max((score for score, _ in scored), default=0.0)
    if top == 0.0:
        return []
    best = [e for score, e in scored if score == top]
    roles = {
        role for role, words in _ROLE_WORDS.items() if re.search(words, description, re.IGNORECASE)
    }
    preferred = [e for e in best if e.role in roles]
    return preferred or best


def named_in(description: str, elements: Sequence[Element]) -> list[Element]:
    """Elements whose label the description contains as words ("UTF-8 unten in der
    Statusleiste" → "UTF-8"); the longest label wins. Labels under 3 letters never count."""
    words = f" {normalize(description)} "
    found = [
        e for e in elements if len(normalize(e.label)) >= 3 and f" {normalize(e.label)} " in words
    ]
    longest = max((len(normalize(e.label)) for e in found), default=0)
    return [e for e in found if len(normalize(e.label)) == longest]


def parse_point(text: str) -> Point | None:
    """The point from a Qwen-VL answer like ``{"point_2d": [214, 12]}`` (0–1000)."""
    match = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    if match is None:
        return None
    x, y = float(match[1]), float(match[2])
    # [0, 0] is how the model says it found nothing.
    if (x, y) == (0, 0) or not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


# macOS accessibility ------------------------------------------------------------------------


def accessibility_allowed(prompt: bool = False) -> bool:
    """macOS only lets apps with the "Accessibility" permission read other apps' controls
    and post mouse events – without it clicks are dropped silently."""
    if sys.platform != "darwin":
        return True
    import ApplicationServices as AX

    if prompt:
        options = {AX.kAXTrustedCheckOptionPrompt: True}
        return bool(AX.AXIsProcessTrustedWithOptions(options))
    return bool(AX.AXIsProcessTrusted())


def _attribute(element: Any, name: str) -> Any:
    import ApplicationServices as AX

    error, value = AX.AXUIElementCopyAttributeValue(element, name, None)
    return value if error == 0 else None


def _frame(element: Any) -> Frame | None:
    import ApplicationServices as AX

    position, size = _attribute(element, "AXPosition"), _attribute(element, "AXSize")
    if position is None or size is None:
        return None
    _, point = AX.AXValueGetValue(position, AX.kAXValueCGPointType, None)
    _, extent = AX.AXValueGetValue(size, AX.kAXValueCGSizeType, None)
    return (float(point.x), float(point.y), float(extent.width), float(extent.height))


def _label(element: Any) -> str:
    for name in ("AXTitle", "AXDescription", "AXValue"):
        value = _attribute(element, name)
        if isinstance(value, str) and value.strip() and len(value) <= 80:
            return value.strip()
    return ""


def apps_on_screen(app: str | None = None) -> list[tuple[int, str]]:
    """Apps with windows on screen, frontmost first, plus the Dock (``app``: only that one)."""
    import Quartz

    options = Quartz.kCGWindowListOptionOnScreenOnly
    seen: dict[int, str] = {}
    for info in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
        owner = str(info.get("kCGWindowOwnerName", ""))
        pid = int(info["kCGWindowOwnerPID"])
        if pid in seen or (app and app.lower() not in owner.lower() and owner != "Dock"):
            continue
        if info.get("kCGWindowLayer") == 0 or owner == "Dock":
            seen[pid] = owner
    return list(seen.items())


def elements_of(pid: int, owner: str) -> list[Element]:
    """Labelled controls of one app, including those of windows lying under others."""
    import ApplicationServices as AX

    deadline = time.monotonic() + SEARCH_SECONDS
    found: list[Element] = []
    stack = [AX.AXUIElementCreateApplication(pid)]
    seen = 0
    while stack and seen < MAX_ELEMENTS and time.monotonic() < deadline:
        element = stack.pop()
        seen += 1
        role = _attribute(element, "AXRole")
        if role in _CLICKABLE and (label := _label(element)):
            frame = _frame(element)
            if (
                frame
                and 4 <= frame[2] <= MAX_ELEMENT_SIZE[0]
                and 4 <= frame[3] <= MAX_ELEMENT_SIZE[1]
            ):
                found.append(Element(label, str(role), frame, owner, element))
        # Closed menus keep their items in the tree; they are not on screen.
        if role != "AXMenu":
            stack.extend(reversed(_attribute(element, "AXChildren") or []))
    if stack:
        log.debug("Bedienungshilfen: %s nach %d Elementen abgebrochen", owner, seen)
    return found


def element_groups(app: str | None = None) -> Iterator[list[Element]]:
    """The labelled controls app by app, frontmost first – a search can stop early."""
    if sys.platform != "darwin" or not accessibility_allowed():
        return
    for pid, owner in apps_on_screen(app):
        yield elements_of(pid, owner)


def element_at(point: Point) -> Element | None:
    """The control at a point of the screen, if accessibility tells."""
    if sys.platform != "darwin" or not accessibility_allowed():
        return None
    import ApplicationServices as AX

    error, element = AX.AXUIElementCopyElementAtPosition(
        AX.AXUIElementCreateSystemWide(), point[0], point[1], None
    )
    if error != 0 or element is None:
        return None
    frame = _frame(element) or (point[0], point[1], 0.0, 0.0)
    role = str(_attribute(element, "AXRole") or "")
    return Element(_label(element), role, frame, _app_of(element), ref=element)


def _app_of(element: Any) -> str:
    """The app an accessibility element belongs to (the menu bar lies in no window)."""
    import ApplicationServices as AX
    from AppKit import NSRunningApplication

    error, pid = AX.AXUIElementGetPid(element, None)
    if error != 0:
        return ""
    running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    return str(running.localizedName() or "") if running is not None else ""


def is_visible(element: Element) -> bool:
    """Nothing lies on top of the element: at its centre accessibility finds it (or a part
    of it). Without accessibility every element counts as visible."""
    if element.ref is None:
        return True
    found = element_at(element.centre)
    for _ in range(8):
        if found is None or found.ref is None:
            return False
        if found.ref == element.ref:
            return True
        parent = _attribute(found.ref, "AXParent")
        found = Element("", "", (0, 0, 0, 0), ref=parent) if parent is not None else None
    return False


# The vision model ---------------------------------------------------------------------------

LOCATE_PROMPT = (
    "Locate {target} in this screenshot. Answer only with JSON: "
    '{{"point_2d": [x, y]}} – the point to click, x and y from 0 to 1000. If it is not '
    'visible, answer {{"point_2d": [0, 0]}}.'
)


class VisionLocator:
    """Asks the vision model where something is."""

    def __init__(self, model: ChatModel, max_size: int = 1280) -> None:
        self._model = model
        self._max_size = max_size

    async def locate(self, shot: Shot, target: str) -> Point | None:
        message = Message(
            Role.USER,
            (ImagePart(shot.png, "image/png"), TextPart(LOCATE_PROMPT.format(target=target))),
        )
        answer = "".join(
            [e.text async for e in self._model.stream([message]) if isinstance(e, TextDelta)]
        )
        point = parse_point(answer)
        if point is None:
            log.info("Bild: „%s“ nicht gefunden (%s)", target, answer.strip()[:80])
            return None
        return shot.to_screen(*point)

    async def refine(self, shot: Shot, guess: Point, target: str) -> Point:
        """A second look at a crop around ``guess``; small targets need it."""
        left, top, width, height = shot.region
        w, h = width * ZOOM_SHARE, height * ZOOM_SHARE
        x = min(max(guess[0] - w / 2, left), left + width - w)
        y = min(max(guess[1] - h / 2, top), top + height - h)
        closer = await asyncio.to_thread(capture_region, (x, y, w, h), self._max_size)
        return await self.locate(closer, target) or guess


# Mouse ----------------------------------------------------------------------------------------


def click(point: Point, button: str) -> None:
    from pynput.mouse import Button, Controller

    mouse = Controller()
    mouse.position = point
    # Let hover effects and the app see the pointer arrive, as with a real mouse.
    time.sleep(0.15)
    which = Button.right if button == "right" else Button.left
    mouse.click(which, 2 if button == "double" else 1)


def scroll(region: Frame, direction: str, pages: float) -> None:
    """Scroll the content under the middle of ``region`` by ``pages`` of its height."""
    from pynput.mouse import Controller

    mouse = Controller()
    left, top, width, height = region
    mouse.position = (left + width / 2, top + height / 2)
    time.sleep(0.1)
    sign = 1 if direction == "up" else -1
    # Most of a page, so the last lines of the previous one are seen again.
    distance = pages * height * 0.85
    # pynput: macOS scrolls 10 points per step, Windows one wheel notch (~100 pixels).
    step_size = 10 if sys.platform == "darwin" else 100
    steps = max(1, round(distance / step_size))
    for chunk in [20] * (steps // 20) + [steps % 20]:
        if chunk:
            mouse.scroll(0, sign * chunk)
            time.sleep(0.02)


# Tools ----------------------------------------------------------------------------------------

_BUTTONS = {"left": "", "right": " mit rechts", "double": " doppelt"}
_BUTTONS_EN = {"left": "", "right": " with the right button", "double": " twice"}
PERMISSION = ToolResult(
    "Not allowed to control the mouse: macOS needs the 'Accessibility' permission for the "
    "app running flay.",
    ok=False,
    action="Maus: keine Berechtigung (Bedienungshilfen)",
    say=speech(
        "Ich darf die Maus nicht steuern. Erlaube es bitte in den Systemeinstellungen unter "
        "Datenschutz und Sicherheit, Bedienungshilfen, und starte mich dann neu.",
        "I'm not allowed to control the mouse. Please allow it in System Settings under "
        "Privacy and Security, Accessibility, and restart me.",
    ),
)
SCREEN_PERMISSION = ToolResult(
    "Not allowed to capture the screen.",
    ok=False,
    action="Bildschirm: keine Berechtigung für Bildschirmaufnahmen",
    say=speech(
        "Ich darf den Bildschirm nicht aufnehmen. Erlaube es bitte in den Systemeinstellungen "
        "unter Bildschirmaufnahme.",
        "I'm not allowed to capture the screen. Please allow it in System Settings under "
        "Screen Recording.",
    ),
)


def _window_or_none(query: str | None) -> tuple[Window | None, list[Window]]:
    windows = list_windows()
    return (find_window(query, windows) if query else None), windows


def window_at(point: Point, windows: Sequence[Window]) -> Window | None:
    """The topmost window at a point (``list_windows`` is frontmost first)."""
    return next((w for w in windows if inside(point, w.bounds)), None)


class ClickTool:
    def __init__(
        self,
        locator: VisionLocator | None,
        monitor: int = 1,
        max_size: int = 1280,
        refine: bool = True,
        elements: Callable[[str | None], Iterable[list[Element]]] = element_groups,
    ) -> None:
        self._locator = locator
        self._monitor = monitor
        self._max_size = max_size
        self._refine = refine
        self._elements = elements

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="click_on_screen",
            description=(
                "Click something visible on the screen: a button, menu, link, tab, icon. Put "
                "its label exactly as shown into 'text' (e.g. 'Git', 'UTF-8'); describe what "
                "and where it is in 'description' only if it has no text or to tell several "
                "apart. button: 'left' (default), 'right' (context menu) or 'double'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The element's visible text or label, e.g. 'Git'",
                    },
                    "description": {
                        "type": "string",
                        "description": "What and where it is, e.g. 'bell icon at the top right'",
                    },
                    "button": {"type": "string", "enum": ["left", "right", "double"]},
                    "window": {
                        "type": "string",
                        "description": "App or window name to look in; omit for the whole screen",
                    },
                },
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        text = str(arguments.get("text") or "").strip()
        description = str(arguments.get("description") or "").strip()
        button = str(arguments.get("button") or "left")
        if button not in _BUTTONS:
            button = "left"
        if not text and not description:
            return ToolResult(
                "Nothing to click was named.",
                ok=False,
                say=speech("Worauf soll ich klicken?", "What should I click?"),
            )
        if not accessibility_allowed(prompt=True):
            return PERMISSION
        start = time.monotonic()
        window, windows = await asyncio.to_thread(_window_or_none, arguments.get("window"))
        try:
            found = await self._find(text, description, button, window)
        except ScreenPermissionError:
            return SCREEN_PERMISSION
        if isinstance(found, ToolResult):
            return found
        point, label, how = found
        target = window_at(point, windows)
        if target is not None and is_own(target):
            return ToolResult(
                "Refused: the point is in Flay's own window.",
                ok=False,
                action="Klick abgelehnt: Flays eigenes Fenster",
                say=speech(
                    "Das liegt in meinem eigenen Fenster, da klicke ich nicht.",
                    "That's in my own window; I won't click there.",
                ),
            )
        named = label or text or description
        if (
            button != "right"
            and not arguments.get("confirmed")
            and _RISKY.search(f"{named} {text}")
        ):
            return ToolResult(
                f"Nothing was clicked yet: asked the user to confirm clicking '{named}'.",
                ok=False,
                action=f"Klick: Rückfrage vor „{named}“",
                say=speech(
                    f"Soll ich wirklich auf {named} klicken?", f"Should I really click {named}?"
                ),
                confirm=ToolCall("confirmed", "click_on_screen", {**arguments, "confirmed": True}),
            )
        await asyncio.to_thread(click, point, button)
        seconds = time.monotonic() - start
        return ToolResult(
            f"Clicked{_BUTTONS_EN[button]} on '{named}' at {point[0]:.0f},{point[1]:.0f}.",
            action=(
                f"Klick{_BUTTONS[button]} auf „{named}“ bei {point[0]:.0f},{point[1]:.0f} "
                f"({how}, {seconds:.1f} s)"
            ),
            say=speech(
                f"Ich habe{_BUTTONS[button]} auf {named} geklickt.",
                f"I clicked {named}{_BUTTONS_EN[button]}.",
            ),
        )

    async def _find(
        self, text: str, description: str, button: str, window: Window | None
    ) -> tuple[Point, str, str] | ToolResult:
        """Where to click, the element's label (if known) and how it was found."""
        candidates: list[Element] = []
        groups = iter(self._elements(window.app if window else None))
        # App by app, frontmost first: the first app showing the label decides.
        while not candidates and (group := await asyncio.to_thread(next, groups, None)) is not None:
            # Models sometimes put the label into the description only.
            matches = (
                best_matches(text, description, group) if text else named_in(description, group)
            )
            candidates = [e for e in matches if await asyncio.to_thread(is_visible, e)]
        if len(candidates) == 1:
            element = candidates[0]
            log.info(
                "Bedienungshilfen: „%s“ → %s „%s“", text or description, element.role, element.label
            )
            return element.centre, element.label, "Bedienungshilfen"
        if self._locator is None:
            return self._not_found(text or description)
        shot = await asyncio.to_thread(self._capture, window)
        target = _describe(text, description)
        point = await self._locator.locate(shot, target)
        if point is None:
            return self._not_found(text or description)
        if candidates:
            # Several elements carry the label: the picture decides between them.
            chosen = next((e for e in candidates if inside(point, e.frame)), None)
            if chosen is not None:
                return chosen.centre, chosen.label, "Bild + Bedienungshilfen"
        under = await asyncio.to_thread(element_at, point)
        if text and under is not None and label_score(text, under.label) > 0:
            return point, under.label, "Bild"
        if self._refine:
            point = await self._locator.refine(shot, point, target)
            under = await asyncio.to_thread(element_at, point)
        label = under.label if under is not None and text and label_score(text, under.label) else ""
        return point, label, "Bild, nachgeschärft" if self._refine else "Bild"

    def _capture(self, window: Window | None) -> Shot:
        if window is not None:
            return capture_window(window, self._max_size)
        return capture_monitor(self._monitor, self._max_size)

    @staticmethod
    def _not_found(what: str) -> ToolResult:
        return ToolResult(
            f"'{what}' is not visible on the screen; nothing was clicked.",
            ok=False,
            action=f"Klick: „{what}“ nicht gefunden",
            say=speech(
                f"Ich finde {what} gerade nicht auf dem Bildschirm.",
                f"I can't find {what} on the screen right now.",
            ),
        )


def _describe(text: str, description: str) -> str:
    if text and description:
        return f'"{text}" ({description})'
    return f'"{text}"' if text else description


class ScrollTool:
    def __init__(self, monitor: int = 1, max_size: int = 1280) -> None:
        self._monitor = monitor
        self._max_size = max_size

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="scroll_screen",
            description=(
                "Scroll the content of a window (or of the screen) up or down by pages. Set "
                "'look' to true to get a screenshot afterwards, e.g. to read on."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["down", "up"]},
                    "pages": {"type": "number", "description": "How far, in screen pages (1)"},
                    "window": {
                        "type": "string",
                        "description": "'active' or an app/window name; omit for the active one",
                    },
                    "look": {"type": "boolean"},
                },
                "required": ["direction"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        direction = "up" if str(arguments.get("direction")) == "up" else "down"
        try:
            pages = min(max(float(arguments.get("pages") or 1), 0.1), 10)
        except (TypeError, ValueError):
            pages = 1.0
        if not accessibility_allowed(prompt=True):
            return PERMISSION
        query = str(arguments.get("window") or "active")
        window, _ = await asyncio.to_thread(_window_or_none, query)
        region: Frame | None = window.bounds if window else None
        if region is None:
            region = await asyncio.to_thread(monitor_bounds, self._monitor)
        await asyncio.to_thread(scroll, region, direction, pages)
        where = window.app if window else "Bildschirm"
        words = ("nach oben", "up") if direction == "up" else ("nach unten", "down")
        action = f"Gescrollt: {where}, {pages:g} Seite(n) {words[0]}"
        if not arguments.get("look"):
            return ToolResult(
                f"Scrolled {words[1]} by {pages:g} page(s) in {where}.",
                action=action,
                say=speech(f"Ich habe {words[0]} gescrollt.", f"I scrolled {words[1]}."),
            )
        await asyncio.sleep(0.4)  # smooth scrolling ends
        try:
            shot = await asyncio.to_thread(self._capture, window)
        except ScreenPermissionError:
            return SCREEN_PERMISSION
        return ToolResult(
            f"Scrolled {words[1]}; the screenshot after scrolling is attached.",
            images=(ImagePart(shot.png, "image/png"),),
            action=action,
        )

    def _capture(self, window: Window | None) -> Shot:
        if window is not None:
            return capture_window(window, self._max_size)
        return capture_monitor(self._monitor, self._max_size)
