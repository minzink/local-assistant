"""Adapters: packages that build on Cassi and personalize it, e.g. Flay.

Cassi works on its own; an adapter is installed next to it and registers itself as an entry
point in the group ``cassi.adapters``::

    [project.entry-points."cassi.adapters"]
    flay = "flay.adapter:FlayAdapter"

The entry point names a subclass of ``Adapter`` (or a function returning an instance). Its
hooks are called by the CLI:

- ``configure`` right after the configuration is loaded, for every command: change the
  settings (model, persona, ...). No I/O here – it also runs for ``cassi devices``.
- ``prepare`` before the LLM is used (``download=False``: ``run``, ``eval``, ``bench``,
  ``eval-computer``) and in ``cassi setup`` (``download=True``, the only time an adapter may
  download anything). Returns lines to print; raises ``AdapterError`` if Cassi cannot start.
- ``manage_tabs`` adds tabs to the management UI (``cassi manage``): information and actions,
  described as data (``cassi.manage.Tab``) – no HTML.
- ``uninstall`` when the user uninstalls: remove what the adapter created outside its folder
  (``models``: its models too). Returns lines to show.
- ``turn_note`` for every request while Cassi runs: a short note that goes with the user's
  message (like the time and what is open on the screen), e.g. a mood that changes over the
  day. Not into the system prompt: that stays the same, so the server reuses its work.
- ``resolve_model`` when the server does not know the model Cassi uses (deleted, or replaced
  by a newer one, e.g. Flay's after its Modelfile changed): the name to use instead (created
  if need be, without download), or None if the model is none of the adapter's.
- ``run`` in the background while Cassi runs (``Running``: the language model, finished
  conversations, waiting until Cassi is idle, replacing the system prompt) – e.g. Flay
  learning about its relationship with the user. Errors are logged, never stop Cassi.

``adapters`` in the config chooses which installed adapters are used (default: all).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from cassi.config import Settings
    from cassi.llm.base import ChatModel
    from cassi.manage import Tab

log = logging.getLogger(__name__)

GROUP = "cassi.adapters"


class AdapterError(RuntimeError):
    """An adapter cannot do its part; the message says what to do (German, for the user)."""


@dataclass(frozen=True)
class Running:
    """What an adapter's ``run`` may use while Cassi runs."""

    settings: Settings
    # The conversation model; one request at a time on local servers – use it when idle.
    llm: ChatModel
    # Summaries of finished conversations after a point in time: (written at, text), oldest
    # first. Empty without persistent memory.
    episodes: Callable[[float], list[tuple[float, str]]]
    # Until no answer is being given or spoken.
    wait_until_idle: Callable[[], Awaitable[None]]
    # Replace the system prompt from the next request on (e.g. after learning something).
    set_system_prompt: Callable[[str], None]


class Adapter:
    """Base class with hooks that change nothing; adapters override what they need."""

    name = "adapter"

    def configure(self, settings: Settings) -> Settings:
        return settings

    def prepare(self, settings: Settings, download: bool) -> list[str]:
        return []

    def manage_tabs(self, settings: Settings) -> list[Tab]:
        return []

    def uninstall(self, settings: Settings, models: bool) -> list[str]:
        return []

    def turn_note(self, settings: Settings, now: datetime) -> str | None:
        return None

    def resolve_model(self, settings: Settings, missing: str) -> str | None:
        return None

    async def run(self, running: Running) -> None:
        return None


def installed() -> dict[str, Callable[[], Callable[[], Adapter]]]:
    """The registered adapters by entry point name (not loaded yet)."""
    return {point.name: point.load for point in entry_points(group=GROUP)}


def load(selected: Sequence[str] | None = None) -> list[Adapter]:
    """The adapters to use: ``selected`` by name, or all installed ones if None."""
    available = installed()
    names = sorted(available) if selected is None else list(selected)
    missing = [name for name in names if name not in available]
    if missing:
        raise AdapterError(
            f"Adapter nicht installiert: {', '.join(missing)} "
            f"(installiert: {', '.join(sorted(available)) or 'keiner'})"
        )
    adapters = []
    for name in names:
        try:
            factory = available[name]()
            adapter = factory()
        except Exception as exc:  # a broken adapter must not look like a missing one
            raise AdapterError(f"Adapter {name} lässt sich nicht laden: {exc}") from exc
        if not isinstance(adapter, Adapter):
            raise AdapterError(f"Adapter {name} ist keine Unterklasse von cassi.adapters.Adapter")
        log.info("Adapter: %s", name)
        adapters.append(adapter)
    return adapters


def configure(adapters: Sequence[Adapter], settings: Settings) -> Settings:
    for adapter in adapters:
        settings = adapter.configure(settings)
    return settings


def prepare(adapters: Sequence[Adapter], settings: Settings, download: bool) -> list[str]:
    lines: list[str] = []
    for adapter in adapters:
        lines += adapter.prepare(settings, download)
    return lines
