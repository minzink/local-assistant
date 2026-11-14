"""Windows: the labelled controls of the visible windows through UI Automation.

The counterpart of the macOS accessibility search in ``cassi.tools.pointer``: buttons, menu
items, links, tabs … with their name and screen rectangle, window by window, frontmost
first. Clicking a labelled element then needs no screenshot and no vision model.

Roles are mapped to the macOS names, so the role words in a description ("der Button …",
"das Menü …") work the same on both systems.

Not yet tried on a real Windows machine: any failure is logged and the click falls back
to the vision model, as before.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from typing import Any

log = logging.getLogger(__name__)

# UI Automation control type ids → the macOS role names pointer.py works with.
ROLES = {
    50000: "AXButton",
    50002: "AXCheckBox",
    50003: "AXComboBox",
    50004: "AXTextField",
    50005: "AXLink",
    50007: "AXCell",
    50011: "AXMenuItem",
    50013: "AXRadioButton",
    50019: "AXTab",
    50024: "AXRow",
    50031: "AXMenuButton",
}
_DESCENDANTS = 4
_CONTROL_TYPE = 30003
# Big windows (an IDE) are cut off rather than waited for.
SEARCH_SECONDS = 1.5
MAX_ELEMENTS = 3000
MAX_SIZE = (700, 200)


def _condition(uia: Any) -> Any:
    """Any of the clickable control types."""
    conditions = [uia.CreatePropertyCondition(_CONTROL_TYPE, role) for role in ROLES]
    condition = conditions[0]
    for other in conditions[1:]:
        condition = uia.CreateOrCondition(condition, other)
    return condition


def _rect(element: Any) -> tuple[float, float, float, float] | None:
    rect = element.CurrentBoundingRectangle
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < 4 or height < 4 or width > MAX_SIZE[0] or height > MAX_SIZE[1]:
        return None
    return (float(rect.left), float(rect.top), float(width), float(height))


def elements_of_window(handle: int, app: str, unlabeled: bool = False) -> list[Any]:
    """The labelled clickable controls of one top-level window (``unlabeled``: also those
    without a name, e.g. icons)."""
    from cassi.tools.browser_uia import _uia
    from cassi.tools.pointer import Element

    _, uia = _uia()
    window = uia.ElementFromHandle(handle)
    found = window.FindAll(_DESCENDANTS, _condition(uia))
    deadline = time.monotonic() + SEARCH_SECONDS
    elements = []
    for index in range(min(found.Length, MAX_ELEMENTS)):
        if time.monotonic() > deadline:
            log.debug("UI Automation: %s nach %d Elementen abgebrochen", app, index)
            break
        element = found.GetElement(index)
        try:
            if element.CurrentIsOffscreen:
                continue
            name = str(element.CurrentName or "").strip()
            frame = _rect(element)
            role = ROLES.get(int(element.CurrentControlType), "")
        except Exception:  # the element went away meanwhile
            continue
        if (name or unlabeled) and len(name) <= 80 and frame is not None:
            elements.append(Element(name, role, frame, app))
    return elements


def element_groups(app: str | None = None, unlabeled: bool = False) -> Iterator[list[Any]]:
    """The labelled controls window by window, frontmost first (``app``: only its windows)."""
    if sys.platform != "win32":
        return
    from cassi.tools.screen import is_own, list_windows

    try:
        windows = [w for w in list_windows() if not is_own(w)]
    except Exception as exc:
        log.info("Fenster nicht lesbar: %s", exc)
        return
    for window in windows:
        if app and app.casefold() not in f"{window.app} {window.title}".casefold():
            continue
        try:
            yield elements_of_window(window.id, window.app, unlabeled)
        except Exception as exc:  # COM or UI Automation not usable
            log.info("UI Automation für %s nicht verfügbar: %s", window.app, exc)
            return
