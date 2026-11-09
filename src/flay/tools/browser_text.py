"""Web pages from the user's browser through the macOS accessibility tree.

The browser hands out the page as a tree: text, headings, links with their address, buttons.
Reading it takes ~0.2 s for a whole page, and pressing a link (``AXPress``) needs neither the
mouse nor the window in front, so the user can keep working. Pages are opened with
``open -g``: the browser loads them without coming to the front.

Firefox and Chromium build the tree of the page only when an assistive app asks for it
(``AXEnhancedUserInterface`` / ``AXManualAccessibility``).

macOS only; elsewhere every function reports "not available" and Flay reads over HTTP or
from screenshots (Windows: UI Automation would be the counterpart).
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from flay.tools.pages import (
    MIN_TEXT,
    Input,
    Link,
    Page,
    SearchForm,
    same_document,
    short_label,
)

log = logging.getLogger(__name__)

LOAD_SECONDS = 15.0
POLL_SECONDS = 0.4
MAX_ELEMENTS = 8000
WALK_SECONDS = 2.5
_CHROMIUM = re.compile(r"chrom|edge|brave|vivaldi|opera|thebrowser", re.IGNORECASE)
_SEARCH = re.compile(r"search|such|finden|find\b|\bgo\b|\blos\b", re.IGNORECASE)
_BLOCKS = {"AXHeading", "AXGroup", "AXList", "AXRow", "AXTable", "AXParagraph"}
_enabled: set[int] = set()


def available() -> bool:
    if sys.platform != "darwin":
        return False
    from flay.tools.pointer import accessibility_allowed

    return accessibility_allowed()


def _attribute(element: Any, name: str) -> Any:
    import ApplicationServices as AX

    error, value = AX.AXUIElementCopyAttributeValue(element, name, None)
    return value if error == 0 else None


def default_browser() -> tuple[str, str] | None:
    """Bundle id and name of the app that opens https links."""
    from AppKit import NSURL, NSBundle, NSWorkspace

    app = NSWorkspace.sharedWorkspace().URLForApplicationToOpenURL_(
        NSURL.URLWithString_("https://example.org")
    )
    bundle = NSBundle.bundleWithURL_(app) if app is not None else None
    if bundle is None or bundle.bundleIdentifier() is None:
        return None
    name = bundle.objectForInfoDictionaryKey_("CFBundleName") or bundle.bundleIdentifier()
    return str(bundle.bundleIdentifier()), str(name)


def _pid(bundle_id: str | None = None, name: str | None = None) -> int | None:
    from AppKit import NSRunningApplication, NSWorkspace

    if bundle_id:
        apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)
    else:
        apps = [
            a
            for a in NSWorkspace.sharedWorkspace().runningApplications()
            if name and str(a.localizedName() or "").lower() == name.lower()
        ]
    return int(apps[0].processIdentifier()) if apps else None


def _application(pid: int) -> Any:
    import ApplicationServices as AX

    app = AX.AXUIElementCreateApplication(pid)
    if pid not in _enabled:
        from AppKit import NSRunningApplication

        running = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        bundle = str(running.bundleIdentifier() or "") if running is not None else ""
        AX.AXUIElementSetAttributeValue(app, "AXManualAccessibility", True)
        if not _CHROMIUM.search(bundle):
            # Firefox; Chromium misplaces windows with this one set.
            AX.AXUIElementSetAttributeValue(app, "AXEnhancedUserInterface", True)
        _enabled.add(pid)
    return app


def _web_areas(element: Any, found: list[Any], depth: int = 0) -> list[Any]:
    if depth > 30:
        return found
    if _attribute(element, "AXRole") == "AXWebArea":
        found.append(element)
        return found
    for child in _attribute(element, "AXChildren") or []:
        _web_areas(child, found, depth + 1)
    return found


def web_areas(pid: int) -> list[tuple[Any, str]]:
    """The pages the browser shows (one per window: its active tab), with their address."""
    app = _application(pid)
    areas: list[tuple[Any, str]] = []
    for window in _attribute(app, "AXWindows") or []:
        for area in _web_areas(window, []):
            areas.append((area, str(_attribute(area, "AXURL") or "")))
    return areas


def _focused_area(pid: int) -> Any:
    app = _application(pid)
    for window in (_attribute(app, "AXFocusedWindow"), *(_attribute(app, "AXWindows") or [])):
        if window is not None and (areas := _web_areas(window, [])):
            return areas[0]
    return None


def read_area(area: Any) -> Page:
    """Text, links and buttons of a page in the browser."""
    text: list[str] = []
    main: list[str] = []
    links: list[Link] = []
    fields: list[tuple[Any, str]] = []
    inputs: list[Input] = []
    url = str(_attribute(area, "AXURL") or "")
    deadline = time.monotonic() + WALK_SECONDS
    count = 0

    def write(part: str, in_main: bool) -> None:
        text.append(part)
        if in_main:
            main.append(part)

    def walk(element: Any, in_main: bool, depth: int) -> None:
        nonlocal count
        count += 1
        if count > MAX_ELEMENTS or depth > 60 or time.monotonic() > deadline:
            return
        role = str(_attribute(element, "AXRole") or "")
        subrole = str(_attribute(element, "AXSubrole") or "")
        in_main = in_main or subrole in ("AXLandmarkMain", "AXDocumentArticle")
        if role in _BLOCKS:
            write("\n", in_main)
        if role == "AXHeading":
            write("## ", in_main)
        if role == "AXStaticText":
            value = _attribute(element, "AXValue")
            if isinstance(value, str) and value.strip():
                write(value + " ", in_main)
        elif role in ("AXTextField", "AXSearchField", "AXComboBox", "AXTextArea"):
            fields.append((element, f"{subrole} {_field_label(element)}"))
            inputs.append(Input(_field_label(element), subrole == "AXSecureTextField", element))
        elif role in ("AXLink", "AXButton"):
            label = _label(element)
            target = str(_attribute(element, "AXURL") or "")
            if label and (role == "AXButton" or target.startswith("http")):
                if role == "AXButton":
                    links.append(Link(label, "", "button", in_main, element))
                elif not same_document(target, url):
                    links.append(Link(label, target, "link", in_main, element))
            if role == "AXButton" and label:
                write(f"[{label}] ", in_main)
        for child in _attribute(element, "AXChildren") or []:
            walk(child, in_main, depth + 1)
        if role in _BLOCKS:
            write("\n", in_main)

    walk(area, False, 0)
    main_text = _tidy(main)
    page_text = main_text if len(main_text) >= MIN_TEXT else _tidy(text)
    has_main = len(main_text) >= MIN_TEXT
    unique: dict[str, Link] = {}
    for link in links:
        key = link.url or f"button:{link.text}"
        unique.setdefault(key, link if has_main else Link(*_fields(link, main=True)))
    title = str(_attribute(area, "AXTitle") or _attribute(area, "AXDescription") or "")
    return Page(
        url,
        title,
        page_text,
        tuple(unique.values()),
        "browser",
        _search(fields, links),
        tuple(inputs),
    )


def _field_label(element: Any) -> str:
    """What a form field is called: its label, else its placeholder or help text."""
    for name in ("AXTitle", "AXDescription", "AXPlaceholderValue", "AXHelp"):
        value = _attribute(element, name)
        if isinstance(value, str) and value.strip():
            return short_label(value.strip().rstrip(":*").strip())
    return ""


def _search(fields: list[tuple[Any, str]], links: list[Link]) -> SearchForm | None:
    """The page's search field and the button that starts the search, if it has both."""
    field = next((f for f, label in fields if _SEARCH.search(label)), None)
    if field is None and len(fields) == 1:
        field = fields[0][0]  # the only field on the page, e.g. docs search
    button = next((b.ref for b in links if b.kind == "button" and _SEARCH.search(b.text)), None)
    if field is None or button is None:
        return None
    return SearchForm(ref=field, button=button)


def _fields(link: Link, main: bool) -> tuple[Any, ...]:
    return (link.text, link.url, link.kind, main, link.ref)


def _label(element: Any) -> str:
    for name in ("AXTitle", "AXDescription"):
        value = _attribute(element, name)
        if isinstance(value, str) and value.strip():
            return short_label(value)
    parts: list[str] = []
    for child in _attribute(element, "AXChildren") or []:
        value = _attribute(child, "AXValue") or _attribute(child, "AXTitle")
        if isinstance(value, str):
            parts.append(value)
    return short_label(" ".join(parts))


def _tidy(parts: list[str]) -> str:
    joined = "".join(parts)
    joined = re.sub(r"[ \t]+", " ", joined)
    joined = re.sub(r" ?\n ?", "\n", joined)
    return re.sub(r"\n{2,}", "\n", joined).strip()


def _busy(area: Any) -> bool:
    return bool(_attribute(area, "AXElementBusy"))


def activate_default() -> None:
    from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication

    if (browser := default_browser()) is None:
        return
    for app in NSRunningApplication.runningApplicationsWithBundleIdentifier_(browser[0]):
        app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)


async def open_in_background(url: str) -> None:
    """Open ``url`` in the default browser without bringing it to the front."""
    process = await asyncio.create_subprocess_exec(
        "open", "-g", url, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    await process.wait()


async def _settle(
    pid: int,
    matches: Callable[[str], bool],
    seconds: float = LOAD_SECONDS,
    differs_from: str | None = None,
) -> Page | None:
    """Wait for a page ``matches(url)`` accepts to finish loading; the page.

    ``differs_from``: the text before an action – the same text again is not the result.
    """
    deadline = time.monotonic() + seconds
    last: Page | None = None
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_SECONDS)
        areas = await asyncio.to_thread(web_areas, pid)
        area = next((a for a, url in areas if matches(url)), None)
        if area is None or await asyncio.to_thread(_busy, area):
            continue
        page = await asyncio.to_thread(read_area, area)
        if differs_from is not None and page.text == differs_from:
            continue
        # Nothing changed since the last look: loaded (a form may have little text).
        if last is not None and page.text == last.text and (not page.thin or page.inputs):
            return page
        last = page
    return last


async def read_url(url: str) -> Page | None:
    """Open ``url`` in the default browser in the background and read it."""
    if not available() or (browser := await asyncio.to_thread(default_browser)) is None:
        return None
    bundle_id, name = browser
    pid = await asyncio.to_thread(_pid, bundle_id)
    before = {u for _, u in await asyncio.to_thread(web_areas, pid)} if pid else set()
    await open_in_background(url)
    for _ in range(int(LOAD_SECONDS / POLL_SECONDS)):
        if (pid := pid or await asyncio.to_thread(_pid, bundle_id)) is not None:
            break
        await asyncio.sleep(POLL_SECONDS)  # the browser is starting
    if pid is None:
        log.info("Browser %s nicht gefunden", name)
        return None

    def matches(found: str) -> bool:
        # The address, or a tab that was not there before (redirects change the address).
        return same_document(found, url) or (bool(found) and found not in before)

    page = await _settle(pid, matches)
    log.info("Webseite im Browser (%s, Hintergrund): %s", name, page.url if page else "–")
    return page


async def read_front(app: str) -> Page | None:
    """The page shown in the front window of the browser ``app`` ("diese Seite")."""
    if not available() or (pid := await asyncio.to_thread(_pid, None, app)) is None:
        return None
    area = await asyncio.to_thread(_focused_area, pid)
    if area is None:
        return None
    return await asyncio.to_thread(read_area, area)


async def press(link: Link, page: Page) -> Page | None:
    """Press a link or button of ``page`` without the mouse; the page afterwards."""
    if link.ref is None or not available():
        return None
    import ApplicationServices as AX

    error, pid = AX.AXUIElementGetPid(link.ref, None)
    if error != 0:
        return None
    if AX.AXUIElementPerformAction(link.ref, "AXPress") != 0:
        return None
    if link.url:
        return await _settle(pid, lambda found: same_document(found, link.url))
    # A button: the same page changes (a tab, "show more"), or another one loads.
    before = {u for _, u in await asyncio.to_thread(web_areas, pid)} - {page.url}

    def same_tab(found: str) -> bool:
        return same_document(found, page.url) or (bool(found) and found not in before)

    return await _settle(pid, same_tab, LOAD_SECONDS / 2, differs_from=page.text)


async def search(page: Page, query: str) -> Page | None:
    """Type ``query`` into the page's search field and start the search, in the background."""
    form = page.search
    if form is None or form.ref is None or form.button is None or not available():
        return None
    import ApplicationServices as AX

    error, pid = AX.AXUIElementGetPid(form.ref, None)
    if error != 0 or AX.AXUIElementSetAttributeValue(form.ref, "AXValue", query) != 0:
        return None
    if AX.AXUIElementPerformAction(form.button, "AXPress") != 0:
        return None
    before = {u for _, u in await asyncio.to_thread(web_areas, pid)} - {page.url}

    def same_tab(found: str) -> bool:
        return same_document(found, page.url) or (bool(found) and found not in before)

    return await _settle(pid, same_tab, LOAD_SECONDS / 2, differs_from=page.text)


def _close_tab(page: Page) -> bool:
    """Close the browser tab showing ``page`` – only if it is the selected tab of a window
    whose page is still ``page`` (the user may have switched tabs since)."""
    pid = None
    if (browser := default_browser()) is not None:
        pid = _pid(browser[0])
    if pid is None:
        return False
    import ApplicationServices as AX

    app = _application(pid)
    for window in _attribute(app, "AXWindows") or []:
        areas = _web_areas(window, [])
        if not areas or not same_document(str(_attribute(areas[0], "AXURL") or ""), page.url):
            continue
        tab = _selected_tab(window)
        if tab is None:
            return False
        for child in _attribute(tab, "AXChildren") or []:
            if _attribute(child, "AXRole") == "AXButton":
                return bool(AX.AXUIElementPerformAction(child, "AXPress") == 0)
    return False


def _selected_tab(element: Any, depth: int = 0) -> Any:
    if depth > 12 or _attribute(element, "AXRole") == "AXWebArea":
        return None
    if _attribute(element, "AXSubrole") == "AXTabButton" and _attribute(element, "AXValue"):
        return element
    for child in _attribute(element, "AXChildren") or []:
        if (found := _selected_tab(child, depth + 1)) is not None:
            return found
    return None


async def close_tab(page: Page) -> bool:
    """Close the tab Flay opened for ``page`` in the background."""
    if not available():
        return False
    try:
        return await asyncio.to_thread(_close_tab, page)
    except Exception as exc:  # the browser's tree is not what was expected
        log.debug("Tab nicht geschlossen: %s", exc)
        return False


def _fill(field: Input, value: str) -> bool:
    import ApplicationServices as AX

    if field.ref is None or field.secret:
        return False
    if AX.AXUIElementSetAttributeValue(field.ref, "AXValue", value) != 0:
        return False
    # Some pages only take the value with the field focused once.
    AX.AXUIElementSetAttributeValue(field.ref, "AXFocused", True)
    return True


async def fill(field: Input, value: str) -> bool:
    """Type ``value`` into a form field of a page in the browser; nothing is sent."""
    if not available():
        return False
    try:
        return await asyncio.to_thread(_fill, field, value)
    except Exception as exc:
        log.debug("Feld nicht ausgefüllt: %s", exc)
        return False


# What the rest of Flay uses; on Windows these come from UI Automation instead. A plain
# list of names, not imports: a linter once removed the "unused" re-imports.
PLATFORM_FUNCTIONS = (
    "activate_default",
    "available",
    "close_tab",
    "fill",
    "press",
    "read_front",
    "read_url",
    "search",
)

if sys.platform == "win32":
    from flay.tools import browser_uia

    for _name in PLATFORM_FUNCTIONS:
        globals()[_name] = getattr(browser_uia, _name)
