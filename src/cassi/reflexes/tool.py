"""The observe tool: watching with reflexes, set up and taught by voice.

"Schau beim Spiel zu und sag Bescheid, wenn meine Lebensleiste rot wird", "Wenn das hier
kommt, drück die Leertaste", "Gut so", "Hör auf zu beobachten". Regions come as 0–1000 of
the watched picture (x1, y1, x2, y2), as the vision model gives them after looking.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from typing import Any

from PIL import Image

from cassi.core.messages import ImagePart, Message, Role, TextPart
from cassi.llm.base import StructuredChat, ToolSpec
from cassi.reflexes.capture import Target
from cassi.reflexes.engine import Decision, Observer, Thinker
from cassi.reflexes.store import Action, Box, Reflex
from cassi.tools.base import ToolResult, speech

log = logging.getLogger(__name__)

ACTIONS = ("start", "stop", "rule", "scene", "teach", "feedback", "list", "remove")
RULE_KINDS = ("color", "change", "text")
COLORS = {
    "red": (220, 40, 40), "rot": (220, 40, 40), "green": (40, 190, 60), "grün": (40, 190, 60),
    "blue": (40, 90, 220), "blau": (40, 90, 220), "yellow": (240, 210, 40),
    "gelb": (240, 210, 40), "orange": (245, 140, 30), "white": (245, 245, 245),
    "weiß": (245, 245, 245), "black": (10, 10, 10), "schwarz": (10, 10, 10),
    "gray": (128, 128, 128), "grau": (128, 128, 128),
}  # fmt: skip
THINK_SIZE = 640
LOCATE_SIZE = 1024
# Around the mouse pointer ("das hier"): this many screen points wide and high.
POINTER_BOX = (240, 90)


# "Wenn das hier rot wird": the place is where the mouse points.
_HERE = re.compile(
    r"^\W*(?:das|dies\w*|da|hier|this|that|here)(?:\s+(?:hier|da|here|one))?\W*$", re.I
)


def flatten(arguments: dict[str, Any]) -> dict[str, Any]:
    """Arguments as the spec wants them, also when a model nests them ("reaction": {"say":
    …}, "rule": {…}) or names the place "das hier"."""
    flat = {k: v for k, v in arguments.items() if not isinstance(v, dict)}
    for value in arguments.values():
        if isinstance(value, dict):
            for key, inner in value.items():
                flat.setdefault(key, inner)
    where = str(flat.get("where") or "")
    if where and _HERE.match(where):
        flat.pop("where")
        flat["at_pointer"] = True
    return flat


# Spoken key names → key combinations ("drück Strg S" → "ctrl+s").
_KEY_WORDS = {
    "enter": "enter", "eingabe": "enter", "eingabetaste": "enter", "return": "enter",
    "leertaste": "space", "leerzeichen": "space", "space": "space", "escape": "esc",
    "esc": "esc", "tab": "tab", "tabulator": "tab", "entf": "delete", "entfernen": "delete",
    "rücktaste": "backspace", "backspace": "backspace", "strg": "ctrl", "control": "ctrl",
    "ctrl": "ctrl", "steuerung": "ctrl", "alt": "alt", "shift": "shift", "umschalt": "shift",
    "hoch": "up", "oben": "up", "runter": "down", "unten": "down", "links": "left",
    "rechts": "right", "pos1": "home", "ende": "end",
}  # fmt: skip
_PRESS_WORDS = re.compile(r"\bdrück\w*\s+(?:auf\s+)?(?:die\s+)?(?:tasten?\s+)?(.+?)\W*$", re.I)
_FILLER = {"pfeil", "pfeiltaste", "taste", "und", "plus", "dann", "die", "bitte", "mal", "+"}


def keys_from_words(text: str) -> str:
    """ "Wenn das kommt, drück Enter" → "enter"; "" if no keys are named."""
    found = _PRESS_WORDS.search(text)
    if found is None:
        return ""
    parts = []
    for word in re.split(r"[\s+,-]+", found.group(1).casefold()):
        if not word or word in _FILLER or re.fullmatch(r"\w+mal|\d+x?", word):
            continue  # "dreimal": how often is the model's 'times'
        if word in _KEY_WORDS:
            parts.append(_KEY_WORDS[word])
        elif re.fullmatch(r"f\d{1,2}|[a-z0-9]", word):
            parts.append(word)
        else:
            return ""  # something that is no key: rather nothing than a wrong key
    if not parts:
        return ""
    from cassi.tools.virtual_keys import check_keys

    keys = "+".join(parts)
    return keys if check_keys(keys) is None else ""


def default_name(arguments: dict[str, Any]) -> str:
    for key in ("name", "contains", "label", "where"):
        if value := str(arguments.get(key) or "").strip():
            return value[:50]
    goal = str(arguments.get("goal") or "").strip()
    if goal and len(goal) <= 30:  # a word put in the goal ("Fehler")
        return goal
    if keys := str(arguments.get("keys") or "").strip():
        return f"{keys} drücken"
    if say := str(arguments.get("say") or "").strip():
        return say[:50]
    return "Reflex"


def parse_color(value: str) -> tuple[int, int, int] | None:
    value = value.strip().casefold()
    if value in COLORS:
        return COLORS[value]
    hexed = value.lstrip("#")
    if len(hexed) == 6:
        try:
            return int(hexed[0:2], 16), int(hexed[2:4], 16), int(hexed[4:6], 16)
        except ValueError:
            return None
    return None


def parse_region(value: Any) -> Box | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (max(0, min(1000, int(v))) for v in value)
    except (TypeError, ValueError):
        return None
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return x1, y1, x2, y2


def pointer_region(
    area: tuple[float, float, float, float],
    pointer: tuple[float, float],
    size: tuple[float, float] = POINTER_BOX,
) -> Box | None:
    """The region around the mouse pointer, in 0–1000 of ``area`` (screen points)."""
    left, top, width, height = area
    x, y = pointer
    if not width or not height or not (left <= x <= left + width and top <= y <= top + height):
        return None
    half_w, half_h = size[0] / 2, size[1] / 2
    return parse_region(
        [
            round((x - half_w - left) / width * 1000),
            round((y - half_h - top) / height * 1000),
            round((x + half_w - left) / width * 1000),
            round((y + half_h - top) / height * 1000),
        ]
    )


def text_region(picture: Image.Image, wanted: str) -> Box | None:
    """Where ``wanted`` is written (OCR), widened to the right where a value usually is
    ("HP" → "HP 45/100")."""
    from cassi.tools import ocr

    if not wanted.strip() or not ocr.available():
        return None
    boxes = ocr.read_image(picture)
    found = ocr.find_text(wanted, boxes) or [
        b for b in boxes if wanted.casefold() in b.text.casefold()
    ]
    if not found:
        return None
    x, y, w, h = found[0].frame
    width, height = picture.size
    return parse_region(
        [
            round((x - h * 0.5) / width * 1000),
            round((y - h * 0.5) / height * 1000),
            round((x + w + max(w * 2, h * 4)) / width * 1000),
            round((y + h * 1.5) / height * 1000),
        ]
    )


async def model_region(llm: StructuredChat, picture: Image.Image, wanted: str) -> Box | None:
    """Where the vision model sees ``wanted`` – approximate, so it is said back."""
    small = picture.copy()
    small.thumbnail((LOCATE_SIZE, LOCATE_SIZE))
    buffer = io.BytesIO()
    small.convert("RGB").save(buffer, "JPEG", quality=85)
    schema = {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "box": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["found", "box"],
        "additionalProperties": False,
    }
    message = Message(
        Role.USER,
        (
            TextPart(
                f"Find {wanted} in the picture. 'box': [x1, y1, x2, y2] in 0-1000 of the "
                "picture's width and height, tight around it. found=false if it is not there."
            ),
            ImagePart(buffer.getvalue(), "image/jpeg"),
        ),
    )
    answer = await llm.complete_json([message], schema, "location")
    region = parse_region(answer.get("box"))
    if not answer.get("found") or region is None or region == (0, 0, 1000, 1000):
        return None
    return region


def area_words(box: Box) -> str:
    """ "oben links", "unten in der Mitte" – where a region is, for saying it back."""
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    vertical = "oben" if y < 333 else "unten" if y > 666 else "mittig"
    horizontal = "links" if x < 333 else "rechts" if x > 666 else "in der Mitte"
    if vertical == "mittig" and horizontal == "in der Mitte":
        return "in der Mitte"
    return f"{vertical} {horizontal}"


def make_thinker(llm: StructuredChat, may_press: Any) -> Thinker:
    """Layer 3: the vision model decides, with the picture, what to do now."""

    async def think(picture: Image.Image, goal: str, known: list[str]) -> Decision | None:
        pressing = bool(may_press())
        kinds = ["none", "say", "keys"] if pressing else ["none", "say"]
        schema = {
            "type": "object",
            "properties": {
                "situation": {"type": "string"},
                "action": {"type": "string", "enum": kinds},
                "say": {"type": "string"},
                "keys": {"type": "string"},
            },
            "required": ["situation", "action", "say", "keys"],
            "additionalProperties": False,
        }
        small = picture.copy()
        small.thumbnail((THINK_SIZE, THINK_SIZE))
        buffer = io.BytesIO()
        small.convert("RGB").save(buffer, "JPEG", quality=80)
        system = (
            "You are the quick-reacting part of a voice assistant that watches the user's "
            f"screen. The user's goal: {goal}\nLook at the picture and decide what to do right "
            "now. 'situation': 2–5 words naming what is seen. 'action': 'none' if nothing is "
            "needed, 'say' for one short sentence to the user in German"
            + (", 'keys' to press keys (e.g. 'space', 'ctrl+s')" if pressing else "")
            + ". Most moments need nothing."
            + (f" Already known reactions: {', '.join(known[:20])}." if known else "")
        )
        message = Message(
            Role.USER,
            (
                TextPart("What now?"),
                ImagePart(buffer.getvalue(), "image/jpeg"),
            ),
        )
        answer = await llm.complete_json(
            [Message.from_text(Role.SYSTEM, system), message], schema, "reaction"
        )
        kind = str(answer.get("action") or "none")
        say = str(answer.get("say") or "").strip() if kind == "say" else ""
        keys = str(answer.get("keys") or "").strip() if kind == "keys" and pressing else ""
        if keys:
            from cassi.tools.virtual_keys import check_keys

            if check_keys(keys) is not None:
                keys = ""
        return Decision(Action(say=say[:200], keys=keys), str(answer.get("situation") or ""))

    return think


class ObserveTool:
    def __init__(self, observer: Observer, locator: StructuredChat | None = None) -> None:
        self.observer = observer
        # Finds what the user names ("die Lebensleiste") when neither the mouse nor OCR does.
        self._locator = locator

    async def _region(self, arguments: dict[str, Any]) -> tuple[Box | None, str]:
        """Where to look: a given region, the mouse ("das hier"), text OCR finds, or the
        vision model; and how it was found, to be said back."""
        if (given := parse_region(arguments.get("region"))) is not None:
            return given, ""
        where = str(arguments.get("where") or "").strip()
        if not where and not arguments.get("at_pointer"):
            return None, ""
        try:
            picture, area = await asyncio.to_thread(self.observer.view)
        except Exception as exc:
            log.info("Bereich nicht bestimmbar: %s", exc)
            return None, ""
        if arguments.get("at_pointer"):
            from cassi.tools.screen import pointer_position

            box = pointer_region(area, await asyncio.to_thread(pointer_position))
            if box is not None:
                return box, "um den Mauszeiger"
        if where:
            box = await asyncio.to_thread(text_region, picture, where)
            if box is not None:
                return box, f"beim Text „{where}“"
            if self._locator is not None:
                try:
                    box = await model_region(self._locator, picture, where)
                except Exception as exc:
                    log.info("Bildmodell fand %s nicht: %s", where, exc)
                    box = None
                if box is not None:
                    return box, area_words(box)
        return None, ""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="observe",
            description=(
                "Watch the screen (a game, a video, a window) and react fast with reflexes. "
                "'start' watches 'target' (an app/window name, or empty for the screen) with "
                "a 'goal'; 'stop' ends it. Reflexes: 'rule' (kind color/change/text at a "
                "place), 'scene' (a 'label' in English describing a picture the user names, "
                "e.g. 'a night sky'), 'teach' when the user means what is shown now ('dieser "
                "Bildschirm', 'wenn das kommt'). Each has a short German 'name' and a reaction: "
                "'say' (a sentence), 'keys' for 'drück …' (e.g. 'enter'), or 'think'. All "
                "arguments flat. 'feedback' with 'good' judges the last reflex; 'list', "
                "'remove'. Where to look: 'at_pointer' when the "
                "user says 'das hier' (the mouse shows it), else 'where' naming it ('the health "
                "bar', or a label written there like 'HP'), or 'region' [x1, y1, x2, y2] in "
                "0–1000 of the watched picture."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(ACTIONS)},
                    "target": {"type": "string"},
                    "goal": {"type": "string"},
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": list(RULE_KINDS)},
                    "region": {"type": "array", "items": {"type": "integer"}},
                    "where": {"type": "string"},
                    "at_pointer": {"type": "boolean"},
                    "color": {"type": "string", "description": "e.g. 'red' or '#ff0000'"},
                    "contains": {"type": "string"},
                    "below": {"type": "number"},
                    "above": {"type": "number"},
                    "label": {"type": "string"},
                    "say": {"type": "string"},
                    "keys": {"type": "string"},
                    "think": {"type": "boolean"},
                    "good": {"type": "boolean"},
                },
                "required": ["action"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        arguments = flatten(arguments)
        action = str(arguments.get("action") or "")
        handler = getattr(self, f"_{action}", None) if action in ACTIONS else None
        if handler is None:
            return ToolResult(f"Unknown action '{action}'.", ok=False)
        result: ToolResult = await handler(arguments)
        return result

    async def _start(self, arguments: dict[str, Any]) -> ToolResult:
        target = str(arguments.get("target") or "").strip()
        goal = str(arguments.get("goal") or "").strip()
        if not self.observer.encoder.available and any(
            r.kind in ("scene", "learned") for r in self.observer.store.reflexes
        ):
            log.warning("Reflex-Modell fehlt – 'cassi setup' ausführen; Szenen bleiben aus")
        self.observer.start(Target(window=target or None), goal)
        also = ""
        if arguments.get("kind") in RULE_KINDS:
            added = await self._rule(arguments)
            also = f" {added.text}"
        elif arguments.get("label"):
            added = await self._scene(arguments)
            also = f" {added.text}"
        where = target or "den Bildschirm"
        rules = len([r for r in self.observer.store.reflexes if r.active])
        return ToolResult(
            f"Watching {where}"
            + (f" with the goal: {goal}" if goal else "")
            + f". {rules} active reflexes.{also}",
            action=f"Beobachten: {where}",
            say=speech(f"Ich beobachte {where}.", f"I'm watching {target or 'the screen'}."),
        )

    async def _stop(self, arguments: dict[str, Any]) -> ToolResult:
        was = self.observer.stop("auf Wunsch")
        return ToolResult(
            "Stopped watching." if was else "Nothing was being watched.",
            action="Beobachten beendet" if was else None,
            say=speech("Ich beobachte nicht mehr.", "I've stopped watching.")
            if was
            else speech("Ich habe gerade nichts beobachtet.", "I wasn't watching anything."),
        )

    def _reaction(self, arguments: dict[str, Any]) -> Action | str:
        say = str(arguments.get("say") or "").strip()[:200]
        keys = str(arguments.get("keys") or "").strip()
        if not keys and (spoken := keys_from_words(say)):
            # The model put "… drück Enter" where the saying goes: a key, not a sentence.
            keys, say = spoken, ""
            arguments["keys"] = keys
            arguments["say"] = ""
        think = bool(arguments.get("think"))
        if keys:
            from cassi.tools.virtual_keys import check_keys

            if problem := check_keys(keys):
                return f"keys: {problem}"
        if not (say or keys or think):
            # "Sag Bescheid, wenn …": say what was seen.
            wanted = str(arguments.get("contains") or arguments.get("goal") or "").strip()
            say = f"{wanted} ist da." if wanted else f"Achtung: {default_name(arguments)}."
        return Action(say=say, keys=keys, think=think)

    def _saved(self, reflex: Reflex, found: str = "") -> ToolResult:
        if problem := self.observer.store.put(reflex):
            return ToolResult(f"Not saved: {problem}", ok=False)
        note = ""
        if reflex.action.keys and not self.observer.store.may_press:
            note = (
                " Pressing keys is not allowed for reflexes yet: the user allows it in the "
                "management page, tab Reflexe."
            )
        looking = f" Ich achte auf den Bereich {found}." if found else ""
        return ToolResult(
            f"Reflex '{reflex.name}' saved: {reflex.describe()}"
            + (f", region {reflex.region} found {found}" if found else "")
            + f".{note}",
            action=f"Reflex: {reflex.name} – {reflex.describe()}",
            say=speech(
                f"Gemerkt: {reflex.name}.{looking}"
                + (" Tasten darf ich dafür noch nicht drücken." if note else ""),
                f"Got it: {reflex.name}.",
            ),  # fmt: skip
        )

    async def _rule(self, arguments: dict[str, Any]) -> ToolResult:
        reaction = self._reaction(arguments)
        if isinstance(reaction, str):
            return ToolResult(reaction, ok=False)
        name = default_name(arguments)
        kind = str(arguments.get("kind") or "")
        if kind not in RULE_KINDS:
            return ToolResult("'kind' must be color, change or text.", ok=False)
        region, found = await self._region(arguments)
        if region is None and (arguments.get("where") or arguments.get("at_pointer")):
            return ToolResult(
                f"Could not find {arguments.get('where') or 'the place'} on the screen.",
                ok=False,
                say=speech(
                    "Die Stelle finde ich nicht. Zeig sie mir mit der Maus und sag: das hier.",
                    "I can't find that spot. Point at it with the mouse and say: this here.",
                ),
            )
        params: dict[str, Any] = {}
        if kind == "color":
            rgb = parse_color(str(arguments.get("color") or ""))
            if rgb is None or region is None:
                return ToolResult("A color rule needs 'color' and a 'region'.", ok=False)
            params = {"rgb": list(rgb), "tolerance": 50, "share": 0.3}
        elif kind == "change":
            params = {"threshold": 8}
        else:
            for key in ("contains", "below", "above"):
                if arguments.get(key) not in (None, ""):
                    params[key] = arguments[key]
            if not params and arguments.get("goal"):
                params["contains"] = str(arguments["goal"])  # the word went to the goal
            if not params or region is None:
                return ToolResult(
                    "A text rule needs a 'region' and 'contains', 'below' or 'above'.", ok=False
                )
        reflex = Reflex(name, kind, reaction, region, params)  # type: ignore[arg-type]
        return self._saved(reflex, found)

    async def _scene(self, arguments: dict[str, Any]) -> ToolResult:
        reaction = self._reaction(arguments)
        if isinstance(reaction, str):
            return ToolResult(reaction, ok=False)
        name = default_name(arguments)
        label = str(arguments.get("label") or "").strip()
        if not label:
            return ToolResult(
                "A scene needs a 'label' (English, e.g. 'a game over screen').", ok=False
            )
        region, found = await self._region(arguments)
        params = {"label": label, "chance": 0.5}
        reflex = Reflex(name, "scene", reaction, region, params, source="described")
        return self._saved(reflex, found)

    async def _teach(self, arguments: dict[str, Any]) -> ToolResult:
        reaction = self._reaction(arguments)
        if isinstance(reaction, str):
            return ToolResult(reaction, ok=False)
        name = default_name(arguments)
        region, found = await self._region(arguments)
        try:
            impression = await asyncio.to_thread(self.observer.impression_now, region)
        except Exception as exc:
            return ToolResult(
                f"Could not see the screen: {exc}",
                ok=False,
                say=speech(
                    "Ich kann den Bildschirm gerade nicht sehen.", "I can't see the screen."
                ),
            )
        reflex = Reflex(
            name, "learned", reaction, region, vector=tuple(float(v) for v in impression),
            source="shown", cooldown=3.0,
        )  # fmt: skip
        return self._saved(reflex, found)

    async def _feedback(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments.get("name") or "") or self.observer.last_fired or ""
        good = bool(arguments.get("good", True))
        judged = self.observer.store.judge(name, good) if name else None
        if judged is None:
            return ToolResult(
                "No reflex to judge.",
                ok=False,
                say=speech("Welchen Reflex meinst du?", "Which reflex do you mean?"),
            )
        if judged.strength <= 0:
            return ToolResult(
                f"Reflex '{name}' forgotten.",
                action=f"Reflex vergessen: {name}",
                say=speech(f"Okay, {name} lasse ich künftig.", f"Okay, I'll drop {name}."),
            )
        return ToolResult(
            f"Reflex '{name}' {'strengthened' if good else 'weakened'} ({judged.strength:.2f}).",
            action=f"Reflex {'gestärkt' if good else 'geschwächt'}: {name}",
            say=speech(
                "Gut, merke ich mir." if good else "Okay, mache ich seltener.",
                "Good, noted." if good else "Okay, less of that.",
            ),  # fmt: skip
        )

    async def _list(self, arguments: dict[str, Any]) -> ToolResult:
        reflexes = self.observer.store.reflexes
        if not reflexes:
            return ToolResult(
                "No reflexes yet.",
                say=speech("Ich habe noch keine Reflexe.", "I have no reflexes yet."),
            )
        lines = [f"- {r.name}: {r.describe()}" for r in reflexes]
        return ToolResult("Reflexes:\n" + "\n".join(lines))

    async def _remove(self, arguments: dict[str, Any]) -> ToolResult:
        name = str(arguments.get("name") or "")
        if not self.observer.store.remove(name):
            return ToolResult(f"There is no reflex '{name}'.", ok=False)
        return ToolResult(
            f"Removed reflex '{name}'.",
            action=f"Reflex entfernt: {name}",
            say=speech(f"{name} ist entfernt.", f"{name} is removed."),
        )
