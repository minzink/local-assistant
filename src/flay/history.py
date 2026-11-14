"""A log of requests for `flay history` and `flay stats`: what was asked, what Flay did, how
long it took, what failed.

Kept on this computer only (data_dir/history.jsonl), for ``privacy.history_days`` days;
``privacy.history_days: 0`` keeps none. It is built from the events the assistant publishes
anyway, so it changes nothing in how a request is handled.

Failed requests can be printed as test cases for eval/cases.yaml (``flay history
--eval``): what goes wrong in daily use becomes a case that `flay eval` checks.
"""

from __future__ import annotations

import difflib
import json
import logging
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flay.core.events import (
    AssistantMessage,
    AssistantState,
    Event,
    EventBus,
    StateChanged,
    ToolUsed,
    TurnProfiled,
    UserMessage,
)

log = logging.getLogger(__name__)

MAX_REPLY_CHARS = 300
REPEAT_SIMILARITY = 0.6
MAX_RESULT_CHARS = 200
# Typed text and form values may be private: the log keeps only how long they were.
_PRIVATE_ARGUMENTS = {"use_keyboard": ("text",), "fill_web_form": ("fields",)}


def _arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    hidden = _PRIVATE_ARGUMENTS.get(tool, ())
    return {
        key: f"({len(str(value))} Zeichen)" if key in hidden and value else value
        for key, value in arguments.items()
    }


@dataclass
class Turn:
    time: str
    text: str
    language: str
    source: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    reply: str = ""
    seconds: float | None = None
    interrupted: bool = False
    # Where the time went (flay.perf): phases, moments, model calls.
    perf: dict[str, Any] | None = None

    @property
    def failed(self) -> bool:
        return any(not t.get("ok", True) for t in self.tools)


class TurnLog:
    def __init__(self, path: Path, days: int, clock: Any = time.monotonic) -> None:
        self._path = path
        self._days = days
        self._clock = clock
        self._turn: Turn | None = None
        self._started = 0.0
        self._pruned = False
        self._last_failed: str | None = None
        # Told when a request fails again in other words (the user is stuck).
        self.on_repeated_failure: Callable[[str], None] | None = None

    async def run(self, bus: EventBus) -> None:
        async with bus.subscribe() as events:
            while True:
                self.handle(await events.get())

    def handle(self, event: Event) -> None:
        if isinstance(event, UserMessage):
            self._flush()
            self._turn = Turn(
                datetime.now().isoformat(timespec="seconds"),
                event.text,
                event.language.value,
                event.source,
            )
            self._started = self._clock()
        elif self._turn is None:
            return
        elif isinstance(event, ToolUsed):
            self._turn.tools.append(
                {
                    "name": event.name,
                    "arguments": _arguments(event.name, event.arguments),
                    "ok": event.ok,
                    "result": event.result[:MAX_RESULT_CHARS],
                }
            )
        elif isinstance(event, AssistantMessage):
            self._turn.reply = event.text[:MAX_REPLY_CHARS]
            self._turn.interrupted = event.interrupted
        elif isinstance(event, StateChanged) and event.state is AssistantState.IDLE:
            if self._turn.seconds is None:
                self._turn.seconds = round(self._clock() - self._started, 2)
        elif isinstance(event, TurnProfiled):
            # Comes last: the turn is complete.
            self._turn.perf = event.profile
            if self._turn.seconds is None:
                self._turn.seconds = round(self._clock() - self._started, 2)
            self._flush()

    def _flush(self) -> None:
        turn, self._turn = self._turn, None
        if turn is None:
            return
        self._notice_repeats(turn)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not self._pruned:
                self._prune()
            with self._path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Verlauf nicht geschrieben: %s", exc)

    def _notice_repeats(self, turn: Turn) -> None:
        if not turn.failed:
            self._last_failed = None
            return
        before, self._last_failed = self._last_failed, turn.text
        if before is None or self.on_repeated_failure is None:
            return
        similar = difflib.SequenceMatcher(None, before.casefold(), turn.text.casefold()).ratio()
        if similar >= REPEAT_SIMILARITY:
            self._last_failed = None  # one hint per stuck request
            self.on_repeated_failure(turn.text)

    def _prune(self) -> None:
        """Drop entries older than ``days`` (once per start)."""
        self._pruned = True
        if not self._path.is_file():
            return
        cutoff = (datetime.now() - timedelta(days=self._days)).isoformat()
        lines = self._path.read_text("utf-8").splitlines()
        kept = [line for line in lines if _time(line) >= cutoff]
        if len(kept) < len(lines):
            self._path.write_text("".join(f"{line}\n" for line in kept), "utf-8")


def _time(line: str) -> str:
    try:
        return str(json.loads(line).get("time", ""))
    except ValueError:
        return ""


def read(path: Path, days: float | None = None) -> Iterator[Turn]:
    if not path.is_file():
        return
    cutoff = (datetime.now() - timedelta(days=days)).isoformat() if days else ""
    for line in path.read_text("utf-8").splitlines():
        try:
            raw = json.loads(line)
            turn = Turn(**raw)
        except (ValueError, TypeError):
            continue
        if turn.time >= cutoff:
            yield turn


def format_turn(turn: Turn) -> str:
    when = turn.time.replace("T", " ")[:16]
    took = f"{turn.seconds:.1f} s" if turn.seconds is not None else "–"
    tools = ", ".join(f"{t['name']}{'' if t.get('ok', True) else ' ✗'}" for t in turn.tools)
    mark = "✗" if turn.failed else "✓"
    lines = [f"{mark} {when}  {took:>7}  {turn.text}"]
    if tools:
        lines.append(f"      Werkzeuge: {tools}")
    if turn.reply:
        lines.append(f"      Flay: {turn.reply}")
    return "\n".join(lines)


def stats(turns: list[Turn]) -> list[str]:
    """Counts, failures and durations, one line each."""
    if not turns:
        return ["Noch keine Anfragen im Verlauf."]
    durations = [t.seconds for t in turns if t.seconds is not None]
    used = Counter(tool["name"] for t in turns for tool in t.tools)
    failed = Counter(tool["name"] for t in turns for tool in t.tools if not tool.get("ok", True))
    lines = [
        f"Anfragen: {len(turns)} (davon gesprochen: {sum(t.source == 'voice' for t in turns)})",
        f"Mit Aktion: {sum(bool(t.tools) for t in turns)}, "
        f"mit Fehler: {sum(t.failed for t in turns)}, "
        f"unterbrochen: {sum(t.interrupted for t in turns)}",
    ]
    if durations:
        lines.append(
            f"Dauer bis fertig: Median {statistics.median(durations):.1f} s, "
            f"langsamste {max(durations):.1f} s"
        )
    if used:
        lines.append("Werkzeuge (Aufrufe / Fehler):")
        lines += [f"  {name:<18} {count:>4} / {failed[name]}" for name, count in used.most_common()]
    return lines


def review_hint(path: Path, days: float = 7) -> str | None:
    """A line for the start of Flay when requests failed lately (``flay history --eval``)."""
    failed = [t for t in read(path, days) if t.failed]
    if len(failed) < MIN_FAILURES_FOR_HINT:
        return None
    return (
        f"{len(failed)} Anfragen sind in den letzten {days:.0f} Tagen fehlgeschlagen – "
        "`flay history --failed` zeigt sie, `flay history --eval` macht Testfälle daraus."
    )


MIN_FAILURES_FOR_HINT = 3


def perf_stats(profiles: list[dict[str, Any]]) -> list[str]:
    """Medians and slow ends of phases, moments and model calls over many turns."""
    if not profiles:
        return ["Keine Messprofile im Verlauf (erst ab dieser Version aufgezeichnet)."]

    def row(name: str, values: list[float]) -> str:
        values = sorted(values)
        p90 = values[min(len(values) - 1, int(len(values) * 0.9))]
        return (
            f"  {name:<24} {statistics.median(values):6.2f} s  p90 {p90:6.2f} s  ({len(values)}×)"
        )

    lines = [f"Messprofile: {len(profiles)} Anfragen", "Zeitpunkte ab Ende der Aufnahme:"]
    for mark in ("stt", "route", "tool", "llm_first_token", "first_sentence", "first_audio",
                 "done", "end"):  # fmt: skip
        values = [p["marks"][mark] for p in profiles if mark in p.get("marks", {})]
        if values:
            lines.append(row(mark, values))
    phases: dict[str, list[float]] = {}
    for p in profiles:
        for name, seconds in p.get("phases", {}).items():
            phases.setdefault(name, []).append(seconds)
    if phases:
        lines.append("Phasen (Dauer):")
        for name in sorted(phases, key=lambda n: -statistics.median(phases[n])):
            lines.append(row(name, phases[name]))
    calls: dict[str, list[dict[str, Any]]] = {}
    for p in profiles:
        for call in p.get("calls", []):
            calls.setdefault(call["purpose"], []).append(call)
    if calls:
        lines.append("Modellaufrufe (Dauer, erstes Token, Token ein→aus, Lesetempo):")
        for purpose, found in sorted(calls.items(), key=lambda kv: -len(kv[1])):
            seconds = statistics.median(c["seconds"] for c in found)
            firsts = [c["first_token"] for c in found if c.get("first_token")]
            first = f"{statistics.median(firsts):.2f} s" if firsts else "–"
            prompt = [c["prompt_tokens"] for c in found if c.get("prompt_tokens")]
            output = [c["output_tokens"] for c in found if c.get("output_tokens")]
            rates = [
                c["prompt_tokens"] / c["first_token"]
                for c in found
                if c.get("prompt_tokens") and c.get("first_token")
            ]
            tokens = (
                f"{statistics.median(prompt):.0f}→{statistics.median(output):.0f}"
                if prompt and output
                else "–"
            )
            rate = f"{statistics.median(rates):.0f} tok/s" if rates else "–"
            lines.append(
                f"  {purpose:<24} {seconds:6.2f} s  erstes {first:>7}  {tokens:>11}  "
                f"{rate:>10}  ({len(found)}×)"
            )
        per_turn = [len(p.get("calls", [])) for p in profiles]
        lines.append(f"Modellaufrufe je Anfrage: Median {statistics.median(per_turn):.0f}, "
                     f"höchstens {max(per_turn)}")  # fmt: skip
    return lines


def eval_cases(turns: list[Turn]) -> str:
    """Failed requests as cases for eval/cases.yaml – the expectation still to be filled in."""
    entries = []
    for turn in turns:
        if not turn.failed:
            continue
        tried = ", ".join(t["name"] for t in turn.tools)
        entries.append(
            f"  # {turn.time}: {tried} schlug fehl – erwartetes Werkzeug eintragen\n"
            f"  - say: {json.dumps(turn.text, ensure_ascii=False)}\n"
            f"    expect: {{tool: TODO}}"
        )
    return "\n".join(entries) or "  # keine fehlgeschlagenen Anfragen"
