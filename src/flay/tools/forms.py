"""Fill in a form on a web page with what the user dictated – never send it.

"Trag auf dieser Seite bei Name Max Mustermann und bei E-Mail max@example.org ein": the
page in the browser (the open one, or a link opened in the background) is read through
accessibility, each named field is matched by its label, and Flay asks first what it is
going to enter. Only after a "ja" does it fill the fields; then it brings the browser to
the front so the user can check and send the form. Password fields are never filled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from flay.core.messages import ToolCall
from flay.llm.base import ToolSpec
from flay.tools import browser_text
from flay.tools.base import ToolResult, speech
from flay.tools.pages import Input, Page
from flay.tools.pointer import label_score
from flay.tools.web import find_url, front_browser, short

log = logging.getLogger(__name__)

# How long the page read for the question is reused for the "ja".
REUSE_SECONDS = 120.0


def match_fields(
    page: Page, wanted: list[tuple[str, str]]
) -> tuple[list[tuple[Input, str, str]], list[str]]:
    """The page's field for each (field name, value), and the names not found.

    A field is used once; names are matched like click targets (exact, similar, contained).
    """
    found: list[tuple[Input, str, str]] = []
    missing: list[str] = []
    used: set[int] = set()
    for name, value in wanted:
        scored = [
            (label_score(name, field.label), index)
            for index, field in enumerate(page.inputs)
            if index not in used and field.label
        ]
        best = max(scored, default=(0.0, -1))
        if best[0] <= 0:
            missing.append(name)
            continue
        used.add(best[1])
        found.append((page.inputs[best[1]], name, value))
    return found, missing


def _fields(arguments: dict[str, Any]) -> list[tuple[str, str]]:
    wanted = []
    for item in arguments.get("fields") or []:
        if isinstance(item, dict):
            name = " ".join(str(item.get("field") or "").split())
            value = str(item.get("value") or "").strip()
            if name and value:
                wanted.append((name, value))
    return wanted


class FillWebFormTool:
    def __init__(self) -> None:
        self._asked: tuple[float, Page] | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fill_web_form",
            description=(
                "Fill in fields of a form on a web page with the values the user dictates "
                "(name, e-mail, message …). It asks the user before filling and never sends "
                "the form. Without 'url' the page open in the browser."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "description": "The field's label, e.g. 'E-Mail'",
                                },
                                "value": {"type": "string"},
                            },
                            "required": ["field", "value"],
                        },
                    },
                },
                "required": ["fields"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        wanted = _fields(arguments)
        if not wanted:
            return ToolResult(
                "No field and value given.",
                ok=False,
                say=speech(
                    "Was soll ich in welches Feld eintragen?",
                    "What should I enter into which field?",
                ),
            )
        given = str(arguments.get("url") or "").strip()
        url = find_url(given) if given else None
        page = self._reuse(url) if arguments.get("confirmed") else None
        if page is None:
            page = await self._page(url)
        if page is None:
            return ToolResult(
                "No page with a form could be read in the browser.",
                ok=False,
                say=speech(
                    "Ich finde gerade keine Seite mit einem Formular im Browser.",
                    "I can't find a page with a form in the browser right now.",
                ),
            )
        found, missing = match_fields(page, wanted)
        secret = [name for field, name, _ in found if field.secret]
        found = [item for item in found if not item[0].secret]
        notes = _notes(missing, secret)
        if not found:
            return ToolResult(
                f"None of the fields was found on {page.url}: {', '.join(missing)}.",
                ok=False,
                say=speech(
                    f"Das Formular hat kein passendes Feld. {notes['de']}".strip(),
                    f"The form has no matching field. {notes['en']}".strip(),
                ),
            )
        listed_de = ", ".join(f"bei {field.label} „{value}“" for field, _, value in found)
        listed_en = ", ".join(f"'{value}' into {field.label}" for field, _, value in found)
        if not arguments.get("confirmed"):
            self._asked = (time.monotonic(), page)
            return ToolResult(
                f"Asked the user to confirm filling in {listed_en} on {page.url}.",
                ok=False,
                action=f"Formular: Rückfrage ({page.url})",
                say=speech(
                    f"Soll ich auf {short(page.url)} {listed_de} eintragen? "
                    f"Abschicken musst du selbst. {notes['de']}".strip(),
                    f"Should I enter {listed_en} on {short(page.url)}? You send it "
                    f"yourself. {notes['en']}".strip(),
                ),
                confirm=ToolCall("confirmed", "fill_web_form", {**arguments, "confirmed": True}),
            )
        self._asked = None
        filled = [field.label for field, _, value in found if await browser_text.fill(field, value)]
        if not filled:
            return ToolResult("Filling in failed.", ok=False, say=_FAILED)
        await asyncio.to_thread(browser_text.activate_default)  # the user checks and sends
        names = ", ".join(filled)
        return ToolResult(
            f"Filled in {names} on {page.url}; not sent. The browser is in front.",
            action=f"Formular ausgefüllt: {names} ({page.url}), nicht abgeschickt",
            say=speech(
                f"Ich habe {names} ausgefüllt. Prüf es und schick es selbst ab.",
                f"I filled in {names}. Check it and send it yourself.",
            ),
        )

    def _reuse(self, url: str | None) -> Page | None:
        if self._asked is None or time.monotonic() - self._asked[0] > REUSE_SECONDS:
            return None
        page = self._asked[1]
        return page if url is None or host_matches(page.url, url) else None

    async def _page(self, url: str | None) -> Page | None:
        if url is not None:
            return await browser_text.read_url(url)
        front = await asyncio.to_thread(front_browser)
        return await browser_text.read_front(front.app) if front is not None else None


def host_matches(a: str, b: str) -> bool:
    from flay.tools.pages import host

    return host(a) == host(b)


def _notes(missing: list[str], secret: list[str]) -> dict[str, str]:
    de, en = [], []
    if missing:
        de.append(f"Nicht gefunden: {', '.join(missing)}.")
        en.append(f"Not found: {', '.join(missing)}.")
    if secret:
        de.append("Passwörter trage ich nicht ein.")
        en.append("I don't enter passwords.")
    return {"de": " ".join(de), "en": " ".join(en)}


_FAILED = speech(
    "Das Ausfüllen hat nicht geklappt; die Seite nimmt keine Eingaben an.",
    "Filling in didn't work; the page doesn't take the input.",
)
