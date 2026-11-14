"""Routines: a plan with a name, set up by voice.

"Merk dir als Feierabend: starte die Tests, dann zeig den Git-Status, dann schließ Slack"
keeps the steps under "Feierabend"; "Feierabend" or "Mach Feierabend" later runs them as a
plan (cassi.plans) – with the same question first if a step reaches far. Nothing new to
learn for Cassi: the steps are ordinary requests.

Kept in data_dir/routines.json. Recorded clicks are a different thing (cassi.tools.workflows).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_DEFINE = re.compile(
    r"^\W*(?:merk dir|speicher\w*|leg\w* an|remember)\b.*?\b(?:als|as)\s+"
    r"(?:routine\s+|ablauf\s+)?[„\"']?(?P<name>[\wäöüß -]{2,40}?)[“\"']?\s*:\s*(?P<steps>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_RUN = re.compile(
    r"^\W*(?:mach|starte?|führ\w*(?: aus)?|run|start|do)?\s*(?:die |den |das )?"
    r"(?:routine\s+)?(?P<name>[\wäöüß -]{2,40}?)(?:[- ]routine)?(?:\s+aus)?\W*$",
    re.IGNORECASE,
)
_FORGET = re.compile(
    r"^\W*(?:vergiss|lösch\w*|forget|delete)\s+(?:die |den )?(?:routine\s+)?"
    r"(?P<name>[\wäöüß -]{2,40}?)(?:[- ]routine)?\W*$",
    re.IGNORECASE,
)


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9äöüß]+", "", name.casefold())


class Routines:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._items: dict[str, tuple[str, str]] = {}  # key → (name, steps)
        if path is not None and path.is_file():
            try:
                raw = json.loads(path.read_text("utf-8"))
                self._items = {_key(n): (n, s) for n, s in raw.items()}
            except (ValueError, AttributeError) as exc:
                log.warning("Routinen nicht lesbar: %s", exc)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self._items.values()]

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(self._items.values())
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")

    def define(self, text: str) -> tuple[str, str] | None:
        """(name, steps) if ``text`` sets up a routine; it is stored."""
        found = _DEFINE.match(text.strip())
        if found is None:
            return None
        name, steps = found["name"].strip(), found["steps"].strip()
        self._items[_key(name)] = (name, steps)
        self._save()
        log.info("Routine „%s“ gespeichert: %s", name, steps)
        return name, steps

    def requested(self, text: str) -> tuple[str, str] | None:
        """(name, steps) if ``text`` asks to run a stored routine."""
        found = _RUN.match(text.strip())
        if found is None:
            return None
        return self._items.get(_key(found["name"]))

    def forget(self, text: str) -> str | None:
        found = _FORGET.match(text.strip())
        if found is None:
            return None
        item = self._items.pop(_key(found["name"]), None)
        if item is None:
            return None
        self._save()
        return item[0]
