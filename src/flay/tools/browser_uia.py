"""Web pages from the user's browser on Windows through UI Automation.

The counterpart of the macOS accessibility path in ``flay.tools.browser_text``, with the same
functions: Chrome, Edge and Firefox hand out the page as UI Automation elements (text,
hyperlinks with their address, buttons, edit fields); links and buttons are pressed with the
Invoke pattern, without the mouse and without bringing the browser to the front.

Not yet tried on a real Windows machine: every failure is logged and reported as "not
available", so Flay falls back to reading directly over HTTP or from screenshots.
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
POLL_SECONDS = 0.5
MAX_ELEMENTS = 8000

# UI Automation ids (UIAutomationClient.h).
_CHILDREN, _DESCENDANTS = 2, 4
_PROCESS_ID, _CONTROL_TYPE, _NAME, _LOCALIZED_TYPE = 30002, 30003, 30005, 30004
_VALUE, _LEGACY_VALUE, _IS_SELECTED, _IS_PASSWORD = 30045, 30093, 30079, 30019
_BUTTON, _EDIT, _HYPERLINK, _TAB_ITEM, _TEXT, _DOCUMENT = 50000, 50004, 50005, 50019, 50020, 50030
_INVOKE, _VALUE_PATTERN = 10000, 10002
_SW_SHOWNOACTIVATE = 4

# Default browser (ProgId of the https handler) → its program.
_PROG_IDS = (
    ("ChromeHTML", "chrome.exe"),
    ("MSEdgeHTM", "msedge.exe"),
    ("FirefoxURL", "firefox.exe"),
    ("BraveHTML", "brave.exe"),
    ("Opera", "opera.exe"),
    ("VivaldiHTM", "vivaldi.exe"),
)
_HEADING = re.compile(r"heading|überschrift", re.IGNORECASE)
_SEARCH = re.compile(r"search|such|finden|find\b|\bgo\b|\blos\b", re.IGNORECASE)
_CLOSE = re.compile(r"close|schlie(?:ß|ss)en", re.IGNORECASE)

_automation: tuple[Any, Any] | None = None


def program_for(prog_id: str) -> str | None:
    """The browser program for the ProgId Windows opens https links with."""
    return next((exe for prefix, exe in _PROG_IDS if prog_id.startswith(prefix)), None)


def _uia() -> tuple[Any, Any]:
    """The UIAutomationCore type library and an IUIAutomation, COM set up for this thread."""
    global _automation
    import comtypes
    import comtypes.client

    comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    if _automation is None:
        module = comtypes.client.GetModule("UIAutomationCore.dll")
        client = comtypes.client.CreateObject(module.CUIAutomation, interface=module.IUIAutomation)
        _automation = (module, client)
    return _automation


def available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        _uia()
    except Exception as exc:  # comtypes missing or COM not usable
        log.info("UI Automation nicht verfügbar: %s", exc)
        return False
    return True


def default_program() -> str | None:
    if sys.platform != "win32":
        return None
    import winreg

    key = r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
            prog_id = str(winreg.QueryValueEx(handle, "ProgId")[0])
    except OSError:
        return None
    return program_for(prog_id)


def _process_name(pid: int) -> str:
    if sys.platform != "win32":
        return ""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(1024)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return buffer.value.rsplit("\\", 1)[-1].lower()
    finally:
        kernel.CloseHandle(handle)


def _windows(program: str) -> list[Any]:
    """Top-level windows of the browser ``program``, frontmost first."""
    _, uia = _uia()
    root = uia.GetRootElement()
    found = root.FindAll(_CHILDREN, uia.CreateTrueCondition())
    windows = []
    names: dict[int, str] = {}
    for index in range(found.Length):
        window = found.GetElement(index)
        pid = int(window.CurrentProcessId)
        if pid not in names:
            names[pid] = _process_name(pid)
        if names[pid] == program:
            windows.append(window)
    return windows


def _document(window: Any) -> Any:
    _, uia = _uia()
    condition = uia.CreatePropertyCondition(_CONTROL_TYPE, _DOCUMENT)
    return window.FindFirst(_DESCENDANTS, condition)


def _url(element: Any) -> str:
    for prop in (_VALUE, _LEGACY_VALUE):
        try:
            value = element.GetCurrentPropertyValue(prop)
        except Exception:
            continue
        if isinstance(value, str) and value.startswith("http"):
            return value
    return ""


def _documents(program: str) -> list[tuple[Any, str]]:
    return [
        (document, _url(document))
        for window in _windows(program)
        if (document := _document(window)) is not None
    ]


def read_document(document: Any) -> Page:
    """Text, links and buttons of the page a browser document shows."""
    module, uia = _uia()
    cache = uia.CreateCacheRequest()
    for prop in (_CONTROL_TYPE, _NAME, _LOCALIZED_TYPE, _VALUE, _LEGACY_VALUE, _IS_PASSWORD):
        cache.AddProperty(prop)
    # One call for the whole page: asking element by element costs a round trip each.
    found = document.FindAllBuildCache(_DESCENDANTS, uia.CreateTrueCondition(), cache)
    url = _url(document)
    text: list[str] = []
    links: list[Link] = []
    fields: list[tuple[Any, str]] = []
    inputs: list[Input] = []
    for index in range(min(found.Length, MAX_ELEMENTS)):
        element = found.GetElement(index)
        kind = element.GetCachedPropertyValue(_CONTROL_TYPE)
        name = " ".join(str(element.GetCachedPropertyValue(_NAME) or "").split())
        if kind == _TEXT and name:
            heading = _HEADING.search(str(element.GetCachedPropertyValue(_LOCALIZED_TYPE) or ""))
            text.append(f"\n## {name}\n" if heading else name + " ")
        elif kind == _HYPERLINK and name:
            target = ""
            for prop in (_VALUE, _LEGACY_VALUE):
                value = element.GetCachedPropertyValue(prop)
                if isinstance(value, str) and value.startswith("http"):
                    target = value
                    break
            if target and not same_document(target, url):
                links.append(Link(short_label(name), target, "link", True, element))
        elif kind == _BUTTON and name:
            links.append(Link(short_label(name), "", "button", True, element))
            text.append(f"[{name}] ")
        elif kind == _EDIT:
            fields.append((element, name))
            secret = bool(element.GetCachedPropertyValue(_IS_PASSWORD))
            inputs.append(Input(short_label(name.rstrip(":*").strip()), secret, element))
    unique: dict[str, Link] = {}
    for link in links:
        unique.setdefault(link.url or f"button:{link.text}", link)
    body = re.sub(r"\n{2,}", "\n", re.sub(r"[ \t]+", " ", "".join(text))).strip()
    title = str(document.CurrentName or "")
    return Page(
        url, title, body, tuple(unique.values()), "browser", _search(fields, links), tuple(inputs)
    )


def _search(fields: list[tuple[Any, str]], links: list[Link]) -> SearchForm | None:
    field = next((f for f, label in fields if _SEARCH.search(label)), None)
    if field is None and len(fields) == 1:
        field = fields[0][0]
    button = next((b.ref for b in links if b.kind == "button" and _SEARCH.search(b.text)), None)
    if field is None or button is None:
        return None
    return SearchForm(ref=field, button=button)


def _invoke(element: Any) -> bool:
    module, _ = _uia()
    try:
        pattern = element.GetCurrentPattern(_INVOKE)
        pattern.QueryInterface(module.IUIAutomationInvokePattern).Invoke()
    except Exception as exc:
        log.debug("Invoke fehlgeschlagen: %s", exc)
        return False
    return True


def _set_value(element: Any, value: str) -> bool:
    module, _ = _uia()
    try:
        pattern = element.GetCurrentPattern(_VALUE_PATTERN)
        pattern.QueryInterface(module.IUIAutomationValuePattern).SetValue(value)
    except Exception as exc:
        log.debug("Wert nicht gesetzt: %s", exc)
        return False
    return True


def _open(url: str) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    # Not activated: the browser loads the page behind the user's window.
    ctypes.windll.shell32.ShellExecuteW(None, "open", url, None, None, _SW_SHOWNOACTIVATE)


async def _run(function: Callable[..., Any], *args: Any) -> Any:
    try:
        return await asyncio.to_thread(function, *args)
    except Exception as exc:  # the browser's tree is not what was expected
        log.info("UI Automation: %s fehlgeschlagen: %s", function.__name__, exc)
        return None


async def _settle(
    program: str, matches: Callable[[str], bool], seconds: float, differs_from: str | None = None
) -> Page | None:
    deadline = time.monotonic() + seconds
    last: Page | None = None
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_SECONDS)
        documents = await _run(_documents, program) or []
        document = next((d for d, url in documents if matches(url)), None)
        if document is None:
            continue
        page: Page | None = await _run(read_document, document)
        if page is None or (differs_from is not None and page.text == differs_from):
            continue
        loaded = len(page.text) >= MIN_TEXT or bool(page.inputs)
        if last is not None and page.text == last.text and loaded:
            return page
        last = page
    return last


async def read_url(url: str) -> Page | None:
    if not available() or (program := default_program()) is None:
        return None
    before = {u for _, u in await _run(_documents, program) or []}
    await asyncio.to_thread(_open, url)

    def matches(found: str) -> bool:
        return same_document(found, url) or (bool(found) and found not in before)

    return await _settle(program, matches, LOAD_SECONDS)


async def read_front(app: str) -> Page | None:
    if not available():
        return None
    program = app.lower() if app.lower().endswith(".exe") else f"{app.lower()}.exe"
    windows = await _run(_windows, program) or []
    for window in windows:
        if (document := await _run(_document, window)) is not None:
            page: Page | None = await _run(read_document, document)
            return page
    return None


async def press(link: Link, page: Page) -> Page | None:
    if link.ref is None or not available() or (program := default_program()) is None:
        return None
    before = {u for _, u in await _run(_documents, program) or []} - {page.url}
    if not await _run(_invoke, link.ref):
        return None
    if link.url:
        return await _settle(program, lambda found: same_document(found, link.url), LOAD_SECONDS)
    return await _settle(
        program,
        lambda found: same_document(found, page.url) or (bool(found) and found not in before),
        LOAD_SECONDS / 2,
        differs_from=page.text,
    )


async def search(page: Page, query: str) -> Page | None:
    form = page.search
    if form is None or form.ref is None or form.button is None or not available():
        return None
    if (program := default_program()) is None or not await _run(_set_value, form.ref, query):
        return None
    if not await _run(_invoke, form.button):
        return None
    return await _settle(
        program, lambda found: bool(found), LOAD_SECONDS / 2, differs_from=page.text
    )


async def fill(field: Input, value: str) -> bool:
    if field.ref is None or field.secret or not available():
        return False
    return bool(await _run(_set_value, field.ref, value))


def _close(page: Page, program: str) -> bool:
    _, uia = _uia()
    tabs = uia.CreatePropertyCondition(_CONTROL_TYPE, _TAB_ITEM)
    buttons = uia.CreatePropertyCondition(_CONTROL_TYPE, _BUTTON)
    for window in _windows(program):
        document = _document(window)
        if document is None or not same_document(_url(document), page.url):
            continue
        found = window.FindAll(_DESCENDANTS, tabs)
        for index in range(found.Length):
            tab = found.GetElement(index)
            if not tab.GetCurrentPropertyValue(_IS_SELECTED):
                continue
            close = tab.FindAll(_DESCENDANTS, buttons)
            for number in range(close.Length):
                button = close.GetElement(number)
                if _CLOSE.search(str(button.CurrentName or "")):
                    return _invoke(button)
        return False
    return False


async def close_tab(page: Page) -> bool:
    if not available() or (program := default_program()) is None:
        return False
    return bool(await _run(_close, page, program))


def activate_default() -> None:
    """Bring the browser's window to the front (for screenshots)."""
    if (program := default_program()) is None:
        return
    for window in _windows(program)[:1]:
        try:
            window.SetFocus()
        except Exception as exc:
            log.debug("Browser nicht nach vorn geholt: %s", exc)
