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

``adapters`` in the config chooses which installed adapters are used (default: all).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cassi.config import Settings
    from cassi.manage import Tab

log = logging.getLogger(__name__)

GROUP = "cassi.adapters"


class AdapterError(RuntimeError):
    """An adapter cannot do its part; the message says what to do (German, for the user)."""


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
