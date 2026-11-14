"""Shell commands the user listed in the config, run by name: "Lass die Tests laufen".

Nothing else runs: the model only picks a name from ``tools.commands``. A command that
takes longer than a few seconds keeps running in the background, and Cassi says when it
is done (with the outcome) through the announcements.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cassi.llm.base import ToolSpec
from cassi.notifications import Announcements
from cassi.tools.base import ToolResult, speech
from cassi.tools.phpstorm import match_names

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 3000
# The answer waits this long; a longer run is reported when it ends.
WAIT_SECONDS = 15.0
_FAILED_TESTS = re.compile(r"(\d+)\s+(?:failed|failures?|errors?|fehlgeschlagen)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    command: str
    cwd: Path | None = None
    timeout: float = 600.0


@dataclass(frozen=True, slots=True)
class Outcome:
    code: int | None  # None: timed out
    output: str


async def execute(command: Command) -> Outcome:
    process = await asyncio.create_subprocess_shell(
        command.command,
        cwd=command.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), command.timeout)
    except TimeoutError:
        process.kill()
        return Outcome(None, "")
    return Outcome(process.returncode, stdout.decode(errors="replace"))


def summary(outcome: Outcome) -> tuple[str, str]:
    """A spoken one-liner (de, en) for a finished command."""
    if outcome.code is None:
        return "ist nicht rechtzeitig fertig geworden", "did not finish in time"
    if outcome.code == 0:
        return "ist erfolgreich durchgelaufen", "finished successfully"
    if match := _FAILED_TESTS.search(outcome.output):
        return f"ist fertig, {match[1]} fehlgeschlagen", f"finished, {match[1]} failed"
    return f"ist mit Fehler {outcome.code} beendet", f"failed with code {outcome.code}"


class RunCommandTool:
    def __init__(
        self,
        commands: Mapping[str, Command],
        announcements: Announcements | None = None,
        wait_seconds: float = WAIT_SECONDS,
    ) -> None:
        self._commands = dict(commands)
        self._announcements = announcements
        self._wait = wait_seconds
        self._running: dict[str, asyncio.Task[Outcome]] = {}

    @property
    def names(self) -> list[str]:
        return list(self._commands)

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="run_command",
            description="Run one of the user's prepared commands by 'name': "
            + ", ".join(self._commands)
            + ". Nothing else can be run.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "enum": list(self._commands)}},
                "required": ["name"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        wanted = str(arguments.get("name") or "")
        names = match_names(wanted, list(self._commands)) if wanted else []
        if len(names) != 1:
            listed = ", ".join(self._commands)
            return ToolResult(
                f"Which command? Prepared: {listed}.",
                ok=False,
                say=speech(
                    f"Welchen Befehl? Es gibt {listed}.", f"Which command? There is {listed}."
                ),
            )
        command = self._commands[names[0]]
        if (task := self._running.get(command.name)) is not None and not task.done():
            return ToolResult(
                f"'{command.name}' is still running.",
                ok=False,
                say=speech(f"{command.name} läuft noch.", f"{command.name} is still running."),
            )
        log.info("Befehl „%s“: %s", command.name, command.command)
        task = asyncio.create_task(execute(command))
        self._running[command.name] = task
        done, _ = await asyncio.wait([task], timeout=self._wait)
        if not done:
            task.add_done_callback(lambda t: self._report(command, t))
            return ToolResult(
                f"'{command.name}' is running; the user will be told when it is done.",
                action=f"Befehl gestartet: {command.name}",
                say=speech(
                    f"{command.name} läuft. Ich sage Bescheid, wenn es fertig ist.",
                    f"{command.name} is running. I'll tell you when it's done.",
                ),
            )
        outcome = task.result()
        de, en = summary(outcome)
        return ToolResult(
            f"'{command.name}' {en}. Output (end):\n{outcome.output[-MAX_OUTPUT_CHARS:]}\n"
            "Summarize the outcome in one or two sentences.",
            ok=outcome.code == 0,
            action=f"Befehl {command.name}: {de}",
        )

    def _report(self, command: Command, task: asyncio.Task[Outcome]) -> None:
        if task.cancelled() or self._announcements is None:
            return
        if (error := task.exception()) is not None:
            self._announcements.say(
                f"{command.name} ist abgebrochen.", f"{command.name} broke off."
            )
            log.warning("Befehl %s: %s", command.name, error)
            return
        de, en = summary(task.result())
        log.info("Befehl %s: %s", command.name, de)
        self._announcements.say(f"{command.name} {de}.", f"{command.name} {en}.")
