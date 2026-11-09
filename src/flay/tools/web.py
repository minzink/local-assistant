"""Web pages: open a link in the default browser and read the page from screenshots.

Reading scrolls through the page screen by screen, so the model sees what the user would
see – also pages that need a login in the user's browser. Each screenshot costs the model
~8 s on an M2 Pro (see flay.tools.pointer), so the number of screens is limited.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import webbrowser
from typing import Any

from flay.core.language import Language
from flay.core.messages import ImagePart
from flay.llm.base import ToolSpec
from flay.tools import browser_text
from flay.tools.base import ToolResult, speech
from flay.tools.pages import Link, Page, describe, fetch_page
from flay.tools.pointer import PERMISSION, SCREEN_PERMISSION, accessibility_allowed, scroll
from flay.tools.screen import (
    ScreenPermissionError,
    Window,
    capture_window,
    find_window,
    list_windows,
)

log = logging.getLogger(__name__)

# Top-level domains a spoken or typed address without "https://" may end in.
_TLDS = (
    "com|de|org|net|io|dev|ai|eu|info|app|co|uk|at|ch|nl|fr|it|es|gov|edu|me|tv|news|"
    "online|shop|cloud|tech|so|sh|ly|gg"
)
_URL = re.compile(r"https?://[^\s<>\"'„“”]+", re.IGNORECASE)
_DOMAIN = re.compile(
    rf"(?<![\w@./-])((?:www\.)?(?:[a-z0-9-]+\.)+(?:{_TLDS}))(/[^\s<>\"'„“”]*)?(?![\w-])",
    re.IGNORECASE,
)
# "heise punkt de", "google dot com": addresses as speech recognition writes them.
_SPOKEN_DOT = re.compile(r"(?<=\w)\s+(?:punkt|dot)\s+(?=\w)", re.IGNORECASE)
_SPOKEN_SLASH = re.compile(r"(?<=\w)\s+(?:slash|schrägstrich)\s+(?=\w)", re.IGNORECASE)
_CLIPBOARD = re.compile(
    r"zwischenablage|clipboard|kopiert\w*\s+(?:link|url|adresse)|copied\s+(?:link|url)",
    re.IGNORECASE,
)
# How long a page may take to settle after opening, and the pause between checks.
LOAD_SECONDS = 12.0
SETTLE_SECONDS = 0.8
# Apps that show web pages: "this page" means the page open in one of them.
_BROWSERS = re.compile(
    r"safari|firefox|chrome|chromium|edge|brave|\barc\b|opera|vivaldi|\bzen\b|orion|"
    r"librewolf|waterfox|floorp|duckduckgo",
    re.IGNORECASE,
)
# Mean grey-level difference (0–255) of two downscaled screenshots below which they count
# as the same screen: the end of a page, or a page done loading.
SAME_SCREEN_DIFFERENCE = 1.0


def find_url(text: str) -> str | None:
    """The first web address in ``text``: typed, with or without https://, or spoken."""
    if match := _URL.search(text):
        return match.group(0).rstrip(".,;:!?)]}")
    for candidate in (text, _SPOKEN_SLASH.sub("/", _SPOKEN_DOT.sub(".", text))):
        if match := _DOMAIN.search(candidate):
            return "https://" + match.group(1).lower() + (match.group(2) or "").rstrip(".,;:!?)")
    return None


def find_urls(text: str) -> list[str]:
    """Every web address in ``text`` in the order they are named, each once."""
    for candidate in (text, _SPOKEN_SLASH.sub("/", _SPOKEN_DOT.sub(".", text))):
        found: list[tuple[int, str]] = []
        taken: list[tuple[int, int]] = []
        for match in _URL.finditer(candidate):
            found.append((match.start(), match.group(0).rstrip(".,;:!?)]}")))
            taken.append(match.span())
        for match in _DOMAIN.finditer(candidate):
            if any(start <= match.start() < end for start, end in taken):
                continue  # the host of a full address
            address = match.group(1).lower() + (match.group(2) or "").rstrip(".,;:!?)")
            found.append((match.start(), "https://" + address))
        urls = list(dict.fromkeys(url for _, url in sorted(found)))
        if urls:
            return urls
    return []


def last_url(text: str) -> str | None:
    """The last web address in ``text`` (e.g. Claude's latest answer)."""
    urls = [m.group(0).rstrip(".,;:!?)]}") for m in _URL.finditer(text)]
    return urls[-1] if urls else None


def wants_clipboard(text: str) -> bool:
    return bool(_CLIPBOARD.search(text))


def clipboard_url() -> str | None:
    import pyperclip

    try:
        return find_url(str(pyperclip.paste() or ""))
    except pyperclip.PyperclipException as exc:
        log.debug("Zwischenablage nicht lesbar: %s", exc)
        return None


def short(url: str) -> str:
    """What to say for a URL: the host without www."""
    host = re.sub(r"^https?://", "", url).split("/")[0]
    return host.removeprefix("www.")


def same_screen(a: bytes, b: bytes) -> bool:
    """Two screenshots show the same thing – a blinking cursor or a spinner aside."""
    import io

    from PIL import Image, ImageChops, ImageStat

    def small(png: bytes) -> Image.Image:
        return Image.open(io.BytesIO(png)).convert("L").resize((160, 100))

    difference = ImageChops.difference(small(a), small(b))
    return ImageStat.Stat(difference).mean[0] < SAME_SCREEN_DIFFERENCE


def _front_window() -> Window | None:
    return find_window(None, list_windows())


def is_browser(window: Window) -> bool:
    return bool(_BROWSERS.search(window.app))


def front_browser() -> Window | None:
    """The frontmost window if it is a browser (Flay's own window aside)."""
    window = _front_window()
    return window if window is not None and is_browser(window) else None


async def settled(
    before: Window | None = None, seconds: float = LOAD_SECONDS, max_size: int = 640
) -> Window | None:
    """Wait until the front window stops changing (a page loaded); that window.

    ``before`` is the window that showed before an action: as long as it is still in front
    unchanged, the new page has not come up yet.
    """
    deadline = time.monotonic() + seconds
    window: Window | None = None
    previous: bytes | None = None
    while time.monotonic() < deadline:
        await asyncio.sleep(SETTLE_SECONDS)
        window = await asyncio.to_thread(_front_window)
        if window is None or (before is not None and window == before):
            continue  # the browser is not in front yet, or still shows the old page
        png = (await asyncio.to_thread(capture_window, window, max_size)).png
        if previous is not None and same_screen(png, previous):
            break  # nothing changed since the last look: loaded
        previous = png
    return window


# Links listed with a page; the ones matching the request come first.
MAX_LINKS = 40


async def show_browser() -> Window | None:
    """Bring the default browser to the front (its page is loaded already)."""
    before = await asyncio.to_thread(_front_window)
    await asyncio.to_thread(browser_text.activate_default)
    return await settled(before, seconds=LOAD_SECONDS / 2)


async def open_page(url: str) -> Window | None:
    """Open ``url`` and wait until its window stops changing; the window, if one shows."""
    before = await asyncio.to_thread(_front_window)
    await asyncio.to_thread(webbrowser.open, url)
    return await settled(before)


async def look_at_page(max_size: int = 1280) -> ImagePart | None:
    """A screenshot of the front window once it has settled, e.g. after a click."""
    window = await settled(seconds=LOAD_SECONDS / 2)
    if window is None:
        return None
    shot = await asyncio.to_thread(capture_window, window, max_size)
    return ImagePart(shot.png, "image/png")


class OpenUrlTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="open_url",
            description="Open a web address (link) in the user's browser.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        )

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        url = find_url(str(arguments.get("url") or ""))
        if url is None:
            return _no_url()
        await asyncio.to_thread(webbrowser.open, url)
        return ToolResult(
            f"Opened {url} in the browser.",
            action=f"Browser: {url} geöffnet",
            say=speech(f"Ich habe {short(url)} geöffnet.", f"I opened {short(url)}."),
        )


class ReadWebpageTool:
    """Reads a page as text if it can, from screenshots if it must.

    In this order: directly over HTTP (invisible, fast), from the user's browser in the
    background through accessibility (logins, JavaScript), from screenshots (browser in
    front). "Diese Seite" starts with the browser, where the user's page is.
    """

    def __init__(
        self,
        max_pages: int = 3,
        max_size: int = 1280,
        direct: bool = True,
        browser: bool = True,
        max_chars: int = 8000,
        close_tabs: bool = True,
    ) -> None:
        self._max_pages = max_pages
        self._max_size = max_size
        self._direct = direct
        self._browser = browser
        self._close_tabs = close_tabs
        self.max_chars = max_chars
        # What was read last: further steps on the page start from it. ``page`` if it was
        # read as text, ``window`` if from screenshots.
        self.page: Page | None = None
        self.window: Window | None = None
        # Opened in the background while trying to read its text.
        self._opened: str | None = None
        # Tabs Flay opened in the background, by the page they show now; closed by finish().
        self._tabs: list[Page] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_webpage",
            description=(
                "Read a web page: its text and links (or screenshots, if it has little "
                "text). Use it to summarize or answer questions about a page. Without 'url' "
                "it reads the page open in the browser now."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "pages": {
                        "type": "integer",
                        "description": f"Screens to read, at most {self._max_pages}",
                    },
                },
            },
        )

    def announcement(self, arguments: dict[str, Any]) -> dict[Language, str]:
        if not arguments.get("url"):
            return speech("Ich lese die Seite.", "I'm reading the page.")
        return speech("Ich öffne die Seite und lese sie.", "I'm opening the page and reading it.")

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        given = str(arguments.get("url") or "").strip()
        url = find_url(given) if given else None
        if given and url is None:
            return _no_url()
        self.page, self.window, self._opened = None, None, None
        start = time.monotonic()
        page = await self._read_text(url)
        if page is not None:
            self.page = page
            seconds = time.monotonic() - start
            how = "direkt" if page.source == "http" else "aus dem Browser"
            return ToolResult(
                describe(page, "", self.max_chars, MAX_LINKS)[0],
                action=(
                    f"Webseite gelesen: {page.url} (Text {how}, {len(page.text)} Zeichen, "
                    f"{seconds:.1f} s)"
                ),
            )
        return await self._read_screens(url, arguments, start)

    async def _read_text(self, url: str | None) -> Page | None:
        if url is None:
            # The user's page: only the browser has it (with its login).
            front = await asyncio.to_thread(front_browser)
            if front is None or not self._browser:
                return None
            page = await browser_text.read_front(front.app)
            if page is not None and page.thin and self._direct and page.url:
                page = await fetch_page(page.url)
            return page if page is not None and not page.thin else None
        if self._direct and (page := await fetch_page(url)) is not None and not page.thin:
            return page
        if self._browser and (page := await self._in_background(url)) is not None:
            self._opened = url
            if not page.thin:
                return page
        return None

    async def _in_background(self, url: str) -> Page | None:
        page = await browser_text.read_url(url)
        if page is not None:
            self._tabs.append(page)
        return page

    def _moved(self, before: Page, after: Page) -> None:
        """A tab Flay opened shows another page now."""
        self._tabs = [after if tab is before else tab for tab in self._tabs]

    async def load(self, url: str) -> Page | None:
        """Another page as text, without changing what was read last (to compare pages)."""
        if self._direct and (page := await fetch_page(url)) is not None and not page.thin:
            return page
        if self._browser and (page := await self._in_background(url)) is not None:
            return None if page.thin else page
        return None

    async def search(self, query: str) -> Page | None:
        """Search with the page's own search: a GET form directly, else its search field in
        the browser. Nothing else is ever filled in."""
        current = self.page
        if current is None or current.search is None:
            return None
        page: Page | None = None
        if current.source == "browser" and current.search.ref is not None:
            page = await browser_text.search(current, query)
            if page is not None:
                self._moved(current, page)
        elif (url := current.search.url(query)) is not None:
            if self._direct:
                page = await fetch_page(url)
            if page is not None and not _mentions(page, query):
                page = None  # an empty results page: JavaScript fills it in
            if (page is None or page.thin) and self._browser:
                page = await self._in_background(url)
        if page is None or page.thin:
            return None
        self.page = page
        return page

    async def finish(self) -> None:
        """Close the tabs Flay opened in the background for this request."""
        tabs, self._tabs = self._tabs, []
        if not self._close_tabs:
            return
        for tab in tabs:
            if await browser_text.close_tab(tab):
                log.info("Tab geschlossen: %s", tab.url)

    async def follow(self, link: Link) -> Page | None:
        """Open a link or press a button of the page read last; the page it leads to.

        A page from the user's browser stays there (logins); one read over HTTP is
        followed over HTTP.
        """
        current = self.page
        if current is None:
            return None
        page: Page | None = None
        if current.source == "browser" and link.ref is not None:
            page = await browser_text.press(link, current)
            if page is not None:
                self._moved(current, page)
        elif link.url and current.source == "http" and self._direct:
            page = await fetch_page(link.url)
            if (page is None or page.thin) and self._browser:
                page = await self._in_background(link.url)
        elif link.url and self._browser:
            page = await self._in_background(link.url)
        if page is None or page.thin:
            return None
        self.page = page
        return page

    async def _read_screens(
        self, url: str | None, arguments: dict[str, Any], start: float
    ) -> ToolResult:
        try:
            pages = min(max(int(arguments.get("pages") or self._max_pages), 1), self._max_pages)
        except (TypeError, ValueError):
            pages = self._max_pages
        if pages > 1 and not accessibility_allowed(prompt=True):
            return PERMISSION
        try:
            if url and self._opened == url:
                # Already open in the background from trying to read its text; now the user
                # sees it, so it stays open.
                self._tabs.clear()
                window = await show_browser()
            else:
                window = await open_page(url) if url else await asyncio.to_thread(front_browser)
            self.window = window
            if window is None:
                return _no_window(url)
            shots = [await asyncio.to_thread(capture_window, window, self._max_size)]
            while len(shots) < pages:
                await asyncio.to_thread(scroll, window.bounds, "down", 1)
                await asyncio.sleep(SETTLE_SECONDS)
                shot = await asyncio.to_thread(capture_window, window, self._max_size)
                if same_screen(shot.png, shots[-1].png):
                    break  # the end of the page
                shots.append(shot)
        except ScreenPermissionError:
            return SCREEN_PERMISSION
        seconds = time.monotonic() - start
        page = url or f"the page open in {window.app} ('{window.title}')"
        where = "from the top" if url else "from where it was scrolled to"
        return ToolResult(
            f"Read {page}: {len(shots)} screenshot(s) of the page {where}, attached in "
            "order. Answer from what they show.",
            images=tuple(ImagePart(shot.png, "image/png") for shot in shots),
            action=(
                f"Webseite gelesen: {url or window.title} ({len(shots)} Bildschirm(e), "
                f"{seconds:.1f} s)"
            ),
        )


def _mentions(page: Page, query: str) -> bool:
    """Results for ``query`` name at least one of its words (beyond the search field)."""
    words = [w for w in re.findall(r"\w{3,}", query.lower())]
    text = page.text.lower()
    return not words or any(text.count(word) > 0 for word in words)


def _no_window(url: str | None) -> ToolResult:
    if url is None:
        return ToolResult(
            "No browser window is in front.",
            ok=False,
            say=speech(
                "Ich sehe gerade keine Webseite. Welchen Link soll ich öffnen?",
                "I can't see a web page right now. Which link should I open?",
            ),
        )
    return ToolResult(
        f"Opened {url}, but no browser window came to the front.",
        ok=False,
        action=f"Browser: {url} geöffnet, aber kein Fenster gefunden",
        say=speech(
            "Ich habe die Seite geöffnet, sehe aber kein Browserfenster.",
            "I opened the page but can't see a browser window.",
        ),
    )


def _no_url() -> ToolResult:
    return ToolResult(
        "No web address given.",
        ok=False,
        say=speech("Welchen Link soll ich öffnen?", "Which link should I open?"),
    )
