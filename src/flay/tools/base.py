"""Tools the assistant may call. Only tools registered here exist for the model."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from flay.core.language import Language
from flay.core.messages import ImagePart, ToolCall
from flay.llm.base import ToolSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolResult:
    # Short factual outcome for the model, e.g. "Opened project 'flay'".
    text: str
    # Images (screenshots) the model should look at next.
    images: tuple[ImagePart, ...] = ()
    ok: bool = True
    # What happened on the computer, for the user (logs, UI), e.g. "PhpStorm: Projekt … geöffnet".
    action: str | None = None
    # What Flay says about it, word for word. Set for actions: the model would otherwise
    # paraphrase the outcome – sometimes wrongly – and needs another round to do so.
    say: Mapping[Language, str] | None = None
    # Nothing done yet, the user is asked first ("Soll ich wirklich …?"): this call runs if
    # they say yes.
    confirm: ToolCall | None = None


def speech(de: str, en: str) -> dict[Language, str]:
    return {Language.DE: de, Language.EN: en}


def spoken(name: str) -> str:
    """A project or folder name as TTS should read it: "project-symfony-kx-ewo" → "project
    symfony kx ewo"."""
    return re.sub(r"[-_.]+", " ", name).strip()


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def run(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(f"Unknown tool '{call.name}'.", ok=False)
        log.debug("Werkzeug %s(%s)", call.name, call.arguments)
        try:
            result = await tool.run(call.arguments)
        except Exception as exc:
            log.warning("%s fehlgeschlagen: %s", call.name, exc)
            return ToolResult(
                f"The tool failed: {exc}",
                ok=False,
                action=f"Fehler: {exc}",
                say=speech(
                    "Das hat leider nicht geklappt, Details stehen im Log.",
                    "Sorry, that did not work; the details are in the log.",
                ),
            )
        if result.action:
            log.info(result.action)
        elif not result.ok:
            log.warning("%s: %s", call.name, result.text)
        return result
