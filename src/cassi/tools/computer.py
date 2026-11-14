"""operate_computer: Cassi finds its own way through an app to reach a goal.

Instead of one tool per task (layouts, tabs, settings, …) the model gets a few basic moves
and looks after each one: click a control or a text, type, press keys, scroll, open or
arrange apps, wait, take a closer look, watch for something in the background. Every round
it sees the front window – its labelled controls (accessibility / UI Automation) and the
text on it (OCR) – and what it did so far, and picks the next move.

Safety:
- Clicks on labels like "Löschen", "Senden", "Kaufen" and pressing Enter (which sends
  messages) need the user's yes, unless the goal asks for exactly that.
- Passwords and payment data are never typed; private windows are never looked at
  (cassi.tools.screen.PRIVACY).
- After max_steps Cassi stops and says how far it got.

Solutions that reached their goal are kept (data_dir/solutions.json) and shown to the model
for similar goals: the second time it knows the way.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from cassi.core.messages import Message, Role, TextPart, ToolCall
from cassi.llm.base import ChatModel, StructuredChat, TextDelta, ToolSpec
from cassi.tools.base import ToolRegistry, ToolResult, speech

log = logging.getLogger(__name__)

MOVES = (
    "click", "double_click", "right_click", "type", "key", "scroll", "app", "wait", "look",
    "watch", "done", "ask", "fail",
)  # fmt: skip
MAX_LISTED = 40
# The same move proposed this often in a row: the model is stuck, Cassi stops.
MAX_REPEATS = 2
# Cosine similarity from which an earlier goal counts as the same task (qwen3-embedding:
# rephrasings 0.70–0.78, unrelated goals up to 0.49).
MIN_MEANING = 0.65
WAIT_LIMIT = 10.0
WATCH_EVERY = 3.0
WATCH_MINUTES = 30
# Goals in which Enter may send or confirm something: there Cassi asks first. Elsewhere
# (opening a settings pane, searching) Enter only submits a search or a field.
_SENDING_GOAL = re.compile(
    r"\b(?:schreib\w*|schick\w*|send\w*|antwort\w*|nachricht\w*|mail\w*|post\w*|"
    r"kommentar\w*|bestätig\w*|lösch\w*|kauf\w*|bezahl\w*|bestell\w*|überweis\w*|"
    r"write|reply|message|comment|confirm|delete|buy|pay|order)\b",
    re.IGNORECASE,
)
_SECRET = re.compile(
    r"\b(?:passwor[dt]\w*|kennwort\w*|pin|tan|iban|kreditkart\w*|credit card|cvc|cvv)\b",
    re.IGNORECASE,
)

SYSTEM = """You operate this computer for the user, one move at a time, until the goal is \
reached. Each turn you get the front window, its controls, the text shown on it, and the \
moves so far with their results. Pick the single next move.

Moves:
- app: windows and apps – target = app name, text = open, switch, close, move (text \
"move:<monitor>", e.g. "move:2", "move:left"), maximize, minimize, side_by_side ("side_by_side:\
<other app>"). Always use this for opening, switching, moving or arranging windows.
- click / double_click / right_click: target = the exact label of a control or a text on \
the screen as listed – only the label, e.g. "Refactor", never a description. Menu bar \
entries are among the texts.
- type: text = what to type into the focused field (click the field first). Typed text \
needs no Enter; press Enter only to submit a search.
- key: target = keys, e.g. "enter", "escape", "tab", "down", "cmd+f", "cmd+shift+n" \
("cmd" is Ctrl on Windows)
- scroll: target = "up" or "down"
- wait: text = seconds (at most 10), when something is loading
- look: text = a question about the screen; a screenshot is looked at (slow – only when \
the lists are not enough, e.g. for icons or pictures)
- watch: target = a text; text = "appear" or "disappear". Watching continues in the \
background and the user is told when it happens – for "tell me when …"
- done: text = what was achieved, one short sentence in {language}
- ask: text = a question to the user, in {language} (goal unclear, or a decision needed)
- fail: text = why it cannot be done, in {language}

Rules:
- Do only what the goal asks. Never delete, send, buy, confirm or close anything the goal \
does not ask for.
- Never type passwords, PINs or payment data.
- If a move had no effect, try another way instead of repeating it.
- A control that is not visible may be further down: scroll, or type its name into the \
app's search field.
- When the goal is reached, say done right away. After typing the asked text, the goal of
  typing it is reached.
- "thought": at most 12 words."""


@dataclass(frozen=True, slots=True)
class Observation:
    app: str
    title: str
    windows: str
    controls: tuple[str, ...]
    texts: tuple[str, ...]
    note: str = ""

    def describe(self) -> str:
        lines = [f"Front window: {self.app}" + (f" – {self.title}" if self.title else "")]
        if self.windows:
            lines.append(self.windows)
        if self.controls:
            lines.append("Controls: " + "; ".join(self.controls))
        if self.texts:
            lines.append("Text on screen: " + " | ".join(self.texts))
        if self.note:
            lines.append(self.note)
        return "\n".join(lines)


Reader = Callable[[tuple[float, float, float, float]], Sequence[Any]]


def _ranked(entries: list[str], goal: str) -> list[str]:
    """Entries sharing a word with the goal first ("Ton" for "öffne den Bereich Ton"), so a
    long sidebar does not push them past MAX_LISTED."""
    words = {w for w in re.findall(r"[\wäöüß]{3,}", goal.casefold())}

    def related(entry: str) -> bool:
        return any(w in entry.casefold() for w in words)

    return [e for e in entries if related(e)] + [e for e in entries if not related(e)]


def observe(read_text: Reader | None, goal: str = "") -> Observation:
    """What the front window shows right now (blocking: run in a thread)."""
    from cassi.tools.pointer import element_groups
    from cassi.tools.screen import (
        PrivateScreenError,
        describe_windows,
        is_own,
        is_private,
        list_windows,
        monitor_regions,
    )

    windows = [w for w in list_windows() if not is_own(w)]
    if not windows:
        return Observation("", "", "", (), (), "No window is open.")
    front = windows[0]
    # Private windows by app name only: their titles may name an account or a bank.
    shown = [replace(w, title="") if is_private(w) else w for w in windows]
    overview = describe_windows(shown, monitor_regions()) or ""
    if is_private(front):
        return Observation(front.app, "", overview, (), (), "The front window is private.")

    def read_controls() -> list[str]:
        controls: list[str] = []
        try:
            group = next(iter(element_groups(front.app)), [])
            for element in group:
                if element.label:
                    role = element.role.removeprefix("AX").lower()
                    entry = f"{element.label} ({role})"
                    if entry not in controls:
                        controls.append(entry)
        except Exception as exc:
            log.info("Bedienelemente nicht lesbar: %s", exc)
        return _ranked(controls, goal)[:MAX_LISTED]

    def read_texts() -> tuple[list[str], str]:
        texts: list[str] = []
        if read_text is None:
            return texts, ""
        try:
            for box in read_text(front.bounds):
                wanted = " " in box.text or len(box.text) > 2
                if wanted and box.text not in texts:
                    texts.append(box.text)
        except PrivateScreenError as exc:
            return texts, f"(No text read: {exc})"
        except Exception as exc:
            log.info("Text nicht lesbar: %s", exc)
        return _ranked(texts, goal)[:MAX_LISTED], ""

    # Accessibility (~1 s for a big window) and OCR (~0.3 s) at the same time.
    with ThreadPoolExecutor(max_workers=1) as pool:
        reading = pool.submit(read_texts)
        controls = read_controls()
        texts, note = reading.result()
    return Observation(front.app, front.title, overview, tuple(controls), tuple(texts), note)


@dataclass
class Solution:
    goal: str
    moves: list[str] = field(default_factory=list)
    used: int = 1


class Solutions:
    """Ways that reached a goal, for similar goals later."""

    def __init__(self, path: Path | None, embedder: Any = None) -> None:
        self._path = path
        self._items: list[Solution] | None = None
        # The memory's embedding model: "Dunkelmodus an" finds "dunkles Design einschalten".
        self._embedder = embedder
        self._vectors: dict[str, Any] = {}

    def _load(self) -> list[Solution]:
        if self._items is None:
            self._items = []
            if self._path is not None and self._path.is_file():
                try:
                    raw = json.loads(self._path.read_text("utf-8"))
                    self._items = [Solution(**item) for item in raw]
                except (ValueError, TypeError) as exc:
                    log.warning("Lösungswege nicht lesbar: %s", exc)
        return self._items

    async def find(self, goal: str) -> Solution | None:
        """The way that reached the goal most like ``goal`` – by meaning if an embedding
        model is there, else by spelling."""
        items = self._load()
        if not items:
            return None
        if self._embedder is not None:
            try:
                return await self._by_meaning(goal, items)
            except Exception as exc:  # embedding server away: spelling still works
                log.debug("Lösungswege ohne Embeddings: %s", exc)
        return self.similar(goal)

    async def _by_meaning(self, goal: str, items: list[Solution]) -> Solution | None:
        import numpy as np

        missing = [s.goal for s in items if s.goal not in self._vectors]
        if missing:
            for text, vector in zip(missing, await self._embedder.embed(missing), strict=True):
                self._vectors[text] = vector
        wanted = (await self._embedder.embed([goal], query=True))[0]
        scores = [float(np.dot(wanted, self._vectors[s.goal])) for s in items]
        best = max(range(len(items)), key=scores.__getitem__)
        log.debug("Lösungsweg „%s“: %.2f", items[best].goal, scores[best])
        return items[best] if scores[best] >= MIN_MEANING else None

    def similar(self, goal: str, minimum: float = 0.6) -> Solution | None:
        wanted = goal.casefold()
        scored = [
            (difflib.SequenceMatcher(None, wanted, s.goal.casefold()).ratio(), s)
            for s in self._load()
        ]
        best = max(scored, key=lambda pair: pair[0], default=None)
        return best[1] if best is not None and best[0] >= minimum else None

    def add(self, goal: str, moves: list[str]) -> None:
        items = self._load()
        same = next((s for s in items if s.goal.casefold() == goal.casefold()), None)
        if same is not None:
            same.moves, same.used = moves, same.used + 1
        else:
            items.append(Solution(goal, moves))
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [{"goal": s.goal, "moves": s.moves, "used": s.used} for s in items[-200:]]
            self._path.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")


class OperateComputerTool:
    def __init__(
        self,
        llm: ChatModel,
        tools: Callable[[], ToolRegistry],
        read_text: Callable[[tuple[float, float, float, float]], Sequence[Any]] | None = None,
        max_steps: int = 12,
        solutions: Solutions | None = None,
        announce: Callable[[Mapping[Any, str]], None] | None = None,
        press: Callable[[str], None] | None = None,
        look: Callable[[], Observation] | None = None,
        snapshot: Callable[[], Any] | None = None,
    ) -> None:
        # A small picture of the screen, to wait after a move until it stops changing.
        self._snapshot = snapshot
        self._llm = llm
        self._tools = tools
        self._read_text = read_text
        self._max_steps = max_steps
        self._solutions = solutions or Solutions(None)
        self._announce = announce
        self._press = press
        # Tests hand in a fixed observation.
        self._seen = look
        self._watches: set[asyncio.Task[None]] = set()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="operate_computer",
            description=(
                "Do a task in the interface of apps on this computer, step by step (looks, "
                "clicks, types, presses keys, scrolls, arranges windows, waits). 'goal': the "
                "task as one complete sentence in the user's words, e.g. 'Schalte in den "
                "Systemeinstellungen den Dunkelmodus ein'."
            ),
            parameters={
                "type": "object",
                "properties": {"goal": {"type": "string", "description": "The task"}},
                "required": ["goal"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        goal = str(arguments.get("goal") or "").strip()
        if not goal:
            return ToolResult(
                "No goal given.", ok=False, say=speech("Was soll ich tun?", "What should I do?")
            )
        if not isinstance(self._llm, StructuredChat):
            return ToolResult("The model cannot answer in JSON.", ok=False)
        allowed = str(arguments.get("allow") or "")
        language = "German" if _german(goal) else "English"
        # The same system prompt for every task and step: the server keeps it processed.
        system = SYSTEM.replace("{language}", language)
        known = await self._solutions.find(goal)
        hint = (
            f"A similar goal ('{known.goal}') was reached before with these moves: "
            + " → ".join(known.moves)
            + "\n\n"
            if known is not None
            else ""
        )
        moves: list[str] = []
        history: list[str] = []
        repeats = 0
        start = time.monotonic()
        for step in range(1, self._max_steps + 1):
            seen = (
                await asyncio.to_thread(self._seen)
                if self._seen is not None
                else await asyncio.to_thread(observe, self._read_text, goal)
            )
            prompt = (
                f"Goal: {goal}\n\n{hint}{seen.describe()}\n\nMoves so far:\n"
                + ("\n".join(history) if history else "(none)")
                + "\n\nNext move?"
            )
            try:
                move = await self._llm.complete_json(
                    [Message.from_text(Role.SYSTEM, system), Message.from_text(Role.USER, prompt)],
                    _SCHEMA,
                    "move",
                )
            except Exception as exc:
                log.warning("Rechner bedienen: keine Antwort des Modells: %s", exc)
                break
            kind = str(move.get("move") or "")
            target = str(move.get("target") or "").strip()
            text = str(move.get("text") or "").strip()
            log.info("Rechner bedienen %d: %s „%s“ %s", step, kind, target, text)
            if kind == "done":
                if moves:
                    self._solutions.add(goal, moves)
                return _finished(text or "Erledigt.", goal, moves, start)
            if kind in ("ask", "fail"):
                return ToolResult(
                    f"{'Question' if kind == 'ask' else 'Not possible'}: {text}",
                    ok=kind == "ask",
                    action=f"Rechner bedienen: {text}",
                    say={lang: text for lang in _LANGUAGES},
                )
            described = f"{kind} {target}".strip() + (f" '{text}'" if text else "")
            if moves and described == moves[-1] and kind not in ("wait", "scroll"):
                # Small models repeat a move that worked ("type 'Hallo'" six times).
                repeats += 1
                if repeats >= MAX_REPEATS:
                    break  # stuck: say how far it got
                history.append(
                    f"{step}. {described} → not done again: it already happened in the "
                    "previous move. If the goal is reached, say done."
                )
                continue
            repeats = 0
            before = await asyncio.to_thread(self._snap) if kind in _SETTLING else None
            outcome = await self._move(kind, target, text, goal, allowed)
            if before is not None and not isinstance(outcome, ToolResult):
                from cassi.tools.pointer import settle

                await asyncio.to_thread(settle, before, self._snap)
            if isinstance(outcome, ToolResult):
                return outcome  # needs the user's yes first
            log.debug("Rechner bedienen %d → %s", step, outcome)
            history.append(f"{step}. {described} → {outcome}")
            moves.append(described)
            if kind == "type" and _typed_as_asked(goal, text, outcome):
                # "Schreib …: Hallo von Cassi" is done once exactly that is typed – no Enter.
                self._solutions.add(goal, moves)
                return _finished(text, goal, moves, start)
        seconds = time.monotonic() - start
        done = "; ".join(moves[-4:]) or "nothing"
        return ToolResult(
            f"Stopped after {len(moves)} moves without reaching the goal. Last moves: {done}.",
            ok=False,
            action=f"Rechner bedienen: abgebrochen nach {len(moves)} Schritten ({seconds:.0f} s)",
            say=speech(
                f"Ich bin nach {len(moves)} Schritten nicht ans Ziel gekommen und habe aufgehört.",
                f"I didn't reach the goal after {len(moves)} steps and stopped.",
            ),
        )

    async def _move(
        self, kind: str, target: str, text: str, goal: str, allowed: str
    ) -> str | ToolResult:
        """Carry out one move; what happened, for the model – or a question to the user."""
        registry = self._tools()
        if kind in ("click", "double_click", "right_click"):
            target = _label(target)
            if not target:
                return "no target given"
            button = {"click": "left", "double_click": "double", "right_click": "right"}[kind]
            arguments: dict[str, Any] = {"text": target, "button": button}
            if allowed and allowed.casefold() == target.casefold():
                arguments["confirmed"] = True
            result = await registry.execute(ToolCall("move", "click_on_screen", arguments))
            if result.confirm is not None:
                return _ask_first(goal, target, result)
            return result.text
        if kind == "type":
            if _SECRET.search(goal) or _SECRET.search(text):
                return "refused: passwords and payment data are never typed"
            result = await registry.execute(ToolCall("move", "use_keyboard", {"text": text}))
            return result.text
        if kind == "key":
            return await self._keys(target or text, goal, allowed)
        if kind == "scroll":
            direction = "up" if "up" in (target or text).lower() else "down"
            result = await registry.execute(
                ToolCall("move", "scroll_screen", {"direction": direction})
            )
            return result.text
        if kind == "app":
            from cassi.tools.apps import ACTIONS

            action, _, extra = (text or "switch").partition(":")
            action = action.strip().lower() or "switch"
            if action not in ACTIONS:
                return (
                    f"'{action}' is no app move (only {', '.join(ACTIONS)}); inside the app "
                    "use click or key, e.g. key cmd+n for a new document"
                )
            arguments = {"action": action, "app": target}
            if extra and action == "side_by_side":
                arguments["other_app"] = extra.strip()
            elif extra and action == "move":
                arguments["monitor"] = extra.strip()
            result = await registry.execute(ToolCall("move", "control_window", arguments))
            return result.text
        if kind == "wait":
            seconds = min(WAIT_LIMIT, _number(text, 2.0))
            await asyncio.sleep(seconds)
            return f"waited {seconds:.0f} s"
        if kind == "look":
            return await self._look(text or goal)
        if kind == "watch":
            return self._watch(target, (text or "appear").lower().startswith("dis"), goal)
        return f"unknown move '{kind}'"

    def _snap(self) -> Any:
        if self._snapshot is None:
            return None
        try:
            return self._snapshot()
        except Exception:  # no screen recording permission: no waiting either
            return None

    async def _keys(self, combo: str, goal: str, allowed: str) -> str | ToolResult:
        from cassi.tools.keyboard import parse_keys, press_keys

        parsed = parse_keys(combo)
        if parsed is None:
            return f"unknown keys '{combo}'"
        mods, key = parsed
        sends = key == "enter" and not mods
        if sends and allowed != "enter" and _SENDING_GOAL.search(goal):
            # Enter sends messages and confirms dialogs: the user decides.
            return ToolResult(
                "Nothing pressed yet: asked the user before pressing Enter.",
                ok=False,
                action="Rechner bedienen: Rückfrage vor Enter",
                say=speech("Soll ich Enter drücken?", "Shall I press Enter?"),
                confirm=ToolCall("confirmed", "operate_computer", {"goal": goal, "allow": "enter"}),
            )
        await asyncio.to_thread(self._press or press_keys, combo)
        return f"pressed {combo}"

    async def _look(self, question: str) -> str:
        result = await self._tools().execute(ToolCall("move", "look_at_screen", {}))
        if not result.images:
            return result.text
        message = Message(
            Role.USER,
            (TextPart(f"{question}\nAnswer briefly; quote labels exactly."), *result.images),
        )
        parts = [e.text async for e in self._llm.stream([message]) if isinstance(e, TextDelta)]
        return "".join(parts).strip()[:800]

    def _watch(self, text: str, gone: bool, goal: str) -> str:
        if not text:
            return "no text to watch for"
        if self._read_text is None or self._announce is None:
            return "watching is not available on this computer"
        task = asyncio.create_task(self._watching(text, gone, goal))
        self._watches.add(task)
        task.add_done_callback(self._watches.discard)
        what = "disappears" if gone else "appears"
        return f"watching in the background until '{text}' {what}; the user will be told"

    async def _watching(self, text: str, gone: bool, goal: str) -> None:
        from cassi.tools.ocr import find_text
        from cassi.tools.screen import monitor_regions

        assert self._read_text is not None and self._announce is not None
        deadline = time.monotonic() + WATCH_MINUTES * 60
        while time.monotonic() < deadline:
            await asyncio.sleep(WATCH_EVERY)
            found = False
            try:
                for region in await asyncio.to_thread(monitor_regions):
                    boxes = await asyncio.to_thread(self._read_text, region)
                    if find_text(text, boxes):
                        found = True
                        break
            except Exception as exc:  # private window, no permission: try again later
                log.debug("Beobachten: %s", exc)
                continue
            if found != gone:
                log.info("Beobachtet: „%s“ %s", text, "ist weg" if gone else "ist da")
                state = ("ist verschwunden", "is gone") if gone else ("ist da", "is there")
                self._announce(speech(f"„{text}“ {state[0]}.", f"'{text}' {state[1]}."))
                return
        self._announce(
            speech(
                f"Ich habe {WATCH_MINUTES} Minuten auf „{text}“ gewartet und höre jetzt auf.",
                f"I waited {WATCH_MINUTES} minutes for '{text}' and stopped watching.",
            )
        )


# Moves after which the screen changes on its own for a moment (clicks wait themselves).
_SETTLING = frozenset({"type", "key", "scroll", "app"})


def screen_snapshot() -> Any:
    """The monitor under the mouse, small and gray (see cassi.tools.pointer.settle)."""
    from cassi.tools.pointer import _snapshot
    from cassi.tools.screen import pointer_position

    return _snapshot(pointer_position())


_SCHEMA = {
    "type": "object",
    "properties": {
        # First the reasoning in a few words: deciding the move straight away made small
        # models repeat the previous one.
        "thought": {"type": "string"},
        "move": {"type": "string", "enum": list(MOVES)},
        "target": {"type": "string"},
        "text": {"type": "string"},
    },
    "required": ["thought", "move", "target", "text"],
    "additionalProperties": False,
}


def _ask_first(goal: str, target: str, result: ToolResult) -> ToolResult:
    return ToolResult(
        f"Nothing clicked yet: asked the user before clicking '{target}'.",
        ok=False,
        action=result.action,
        say=result.say,
        confirm=ToolCall("confirmed", "operate_computer", {"goal": goal, "allow": target}),
    )


def _finished(summary: str, goal: str, moves: list[str], start: float) -> ToolResult:
    seconds = time.monotonic() - start
    return ToolResult(
        f"Done: {summary} (moves: {' → '.join(moves) or 'none'})",
        action=f"Rechner bedienen: {goal} – {len(moves)} Schritte, {seconds:.0f} s",
        say={lang: summary for lang in _LANGUAGES},
    )


_DICTATING = re.compile(r"\b(?:schreib\w*|tipp\w*|type|write|trag\w*)\b", re.IGNORECASE)


def _typed_as_asked(goal: str, text: str, outcome: str) -> bool:
    return (
        outcome.startswith("Typed")
        and len(text) >= 3
        and text.casefold() in goal.casefold()
        and _DICTATING.search(goal) is not None
    )


def _label(target: str) -> str:
    """The label in what the model wrote: "Sound (button) or Ton (button) in System
    Settings menu." → "Sound"."""
    # Labels rarely contain parentheses; a role or explanation after one is not the label.
    label = target.strip().split("(", 1)[0].strip() or target.strip()
    label = re.split(r"\s+(?:or|oder)\s+", label, maxsplit=1)[0]
    # "… in System Settings menu": where it is, not part of the label ("Save in Folder" is).
    label = re.sub(
        r"\s+(?:in|im|from)\s+(?:the\s+|der\s+|den\s+)?[\w ]*?"
        r"(?:menu|menü|window|fenster|app|settings|einstellungen|toolbar|sidebar|seitenleiste)\.?$",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\s*\([^)]*\)\s*", " ", label).strip(" .'\"")
    return label or target.strip()


def _number(text: str, default: float) -> float:
    found = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(found.group().replace(",", ".")) if found else default


def _german(text: str) -> bool:
    return bool(re.search(r"[äöüß]|\b(?:der|die|das|und|ein|eine|den|mit|auf|im)\b", text, re.I))


def _languages() -> tuple[Any, ...]:
    from cassi.core.language import Language

    return tuple(Language)


_LANGUAGES = _languages()
