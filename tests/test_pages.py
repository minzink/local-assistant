from typing import Any

import pytest
from aiohttp import web as aioweb
from aiohttp.test_utils import TestServer

from flay.tools import browser_text, pages
from flay.tools import web as web_module
from flay.tools.base import ToolResult
from flay.tools.pages import (
    Link,
    Page,
    SearchForm,
    describe,
    fetch_page,
    parse_html,
    ranked_links,
    sections,
)
from flay.tools.web import ReadWebpageTool

DOCS = (
    """<!doctype html><html><head><title>uv – Introduction</title>
<script>var tracking = "do not read";</script><style>body{}</style></head>
<body>
<header><a href="/">uv</a> <a href="/search">Search</a></header>
<nav><ul><li><a href="getting-started/installation/">Installation</a></li>
<li><a href="guides/">Guides</a></li></ul></nav>
<main>
<h1>Introduction</h1>
<p>An extremely fast Python package and project manager, written in Rust.</p>
<p hidden>Hidden text</p><div aria-hidden="true">Also hidden</div>
<h2 id="installation">Installation</h2>
<p>Install uv with our standalone installers:</p>
<pre>curl -LsSf https://astral.sh/uv/install.sh | sh</pre>
<p>See the <a href="#installation">section above</a> or the
<a href="https://docs.astral.sh/uv/reference/">reference</a>.</p>
<ul><li>Fast</li><li>Compatible with pip</li></ul>
<p>"""
    + "More text about projects and scripts. " * 10
    + """</p>
<a href="mailto:hi@example.org">Mail</a> <a href="javascript:void(0)">Menu</a>
</main>
<footer><a href="/privacy">Privacy</a></footer>
</body></html>"""
)


def test_html_is_read_as_main_text_with_links() -> None:
    page = parse_html(DOCS, "https://docs.astral.sh/uv/")

    assert page.title == "uv – Introduction"
    assert "# Introduction" in page.text and "## Installation" in page.text
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in page.text
    assert "- Compatible with pip" in page.text
    for missing in ("tracking", "Hidden text", "Also hidden", "Privacy", "Guides"):
        assert missing not in page.text  # scripts, hidden parts and navigation
    urls = {link.text: (link.url, link.main) for link in page.links}
    assert urls["Installation"] == (
        "https://docs.astral.sh/uv/getting-started/installation/",
        False,
    )
    assert urls["reference"] == ("https://docs.astral.sh/uv/reference/", True)
    # Sections of the same page, mail and script links are no places to go.
    assert {"section above", "Mail", "Menu"}.isdisjoint(urls)
    assert not page.thin


def test_javascript_shells_are_thin() -> None:
    shell = "<html><body><div id=root></div><noscript>Please enable JavaScript.</noscript>"
    assert parse_html(shell, "https://app.example.org/").thin
    assert Page("u", "", "You need to enable JavaScript to run this app.", (), "http").thin


def test_links_matching_the_request_come_first_with_stable_numbers() -> None:
    links = tuple(Link(f"Seite {i}", f"https://x.org/{i}") for i in range(1, 30)) + (
        Link("Installation", "https://x.org/install"),
    )
    page = Page("https://x.org/", "X", "text", links, "http")

    shown = ranked_links(page, "Schick Claude die Installationsschritte", limit=6)

    assert (30, links[29]) in shown and len(shown) == 6
    assert [n for n, _ in shown] == sorted(n for n, _ in shown)
    assert "[30] Installation" in describe(page, "Installationsschritte", 1000, 6)[0]


GUIDE = "\n".join(
    [
        "# uv",
        "An extremely fast package manager. " * 20,
        "## Highlights",
        "Fast and compatible. " * 30,
        "## Projects",
        "Manage projects with a lockfile. " * 30,
        "## Installation",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "## Scripts",
        "Run single-file scripts. " * 30,
    ]
)


def test_long_pages_are_split_at_headings() -> None:
    parts = sections(GUIDE, 1000)

    assert [p.split("\n", 1)[0] for p in parts][:2] == ["# uv", "## Highlights"]
    assert all(len(p) <= 1000 for p in parts)
    assert len(sections("line\n" * 1000, 2000)) == 3


def test_the_parts_about_the_request_come_first() -> None:
    page = Page("https://docs.astral.sh/uv/", "uv", GUIDE, (), "http")

    text, shown, more = describe(page, "Schick Claude die Installationsschritte", 1500, 10)

    assert "install.sh" in text and more
    assert "Not shown yet:" in text and "## Projects" in text.split("Not shown yet:")[1]
    # Reading on brings the rest, nothing twice.
    rest, everything, more = describe(page, "Installationsschritte", 1500, 10, shown)
    assert "install.sh" not in rest.split("Text:")[1] and shown < everything
    # About the whole page: in order, from the top.
    whole, _, _ = describe(page, "Fasse die Seite zusammen", 1500, 10, whole=True)
    assert whole.split("Text:\n")[1].startswith("# uv")


SEARCHES = (
    """<html><body><main>"""
    + "<p>Docs text.</p>" * 80
    + """
<form action="/login" method="post"><input name="user"><input name="pass" type="password">
</form>
<form class="contact"><input name="name"><input name="mail"></form>
<form role="search" action="search.html"><input type="text" name="q">
<input type="hidden" name="check_keywords" value="yes"></form>
</main></body></html>"""
)


def test_only_a_get_search_form_is_used() -> None:
    page = parse_html(SEARCHES, "https://docs.python.org/3/index.html")

    assert page.search is not None
    assert page.search.url("asyncio tasks") == (
        "https://docs.python.org/3/search.html?check_keywords=yes&q=asyncio+tasks"
    )
    assert "has a search" in describe(page, "", 8000, 10)[0]
    login_only = SEARCHES.replace('role="search" action="search.html"', 'method="post"')
    assert parse_html(login_only, "https://x.org/").search is None


async def test_pages_are_fetched_directly() -> None:
    async def docs(request: aioweb.Request) -> aioweb.Response:
        return aioweb.Response(text=DOCS, content_type="text/html")

    async def moved(request: aioweb.Request) -> aioweb.Response:
        raise aioweb.HTTPFound("/uv/")

    async def pdf(request: aioweb.Request) -> aioweb.Response:
        return aioweb.Response(body=b"%PDF", content_type="application/pdf")

    app = aioweb.Application()
    app.add_routes([aioweb.get("/uv/", docs), aioweb.get("/old", moved), aioweb.get("/a.pdf", pdf)])
    async with TestServer(app) as server:
        page = await fetch_page(str(server.make_url("/old")))
        assert page is not None and page.url.endswith("/uv/") and page.source == "http"
        install = next(link for link in page.links if link.text == "Installation")
        assert install.url == str(server.make_url("/uv/getting-started/installation/"))
        assert await fetch_page(str(server.make_url("/a.pdf"))) is None
        assert await fetch_page(str(server.make_url("/missing"))) is None


GOOD = Page(
    "https://x.org/", "X", "Docs " * 100, (Link("Installation", "https://x.org/i"),), "http"
)
THIN = Page("https://x.org/", "X", "Loading…", (), "http")


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Fake HTTP, browser and screenshots; records what was used."""
    used: dict[str, list[Any]] = {"http": [], "browser": [], "press": [], "screens": []}
    results: dict[str, Page | None] = {"http": GOOD, "browser": GOOD}

    async def fetch(url: str, language: str = "de") -> Page | None:
        used["http"].append(url)
        return results["http"]

    async def read_url(url: str) -> Page | None:
        used["browser"].append(url)
        return results["browser"]

    async def press(link: Link, page: Page) -> Page | None:
        used["press"].append(link.text)
        return GOOD

    async def screens(self: ReadWebpageTool, url: str | None, *args: Any) -> ToolResult:
        used["screens"].append(url)
        return ToolResult("screenshots")

    monkeypatch.setattr(web_module, "fetch_page", fetch)
    monkeypatch.setattr(browser_text, "read_url", read_url)
    monkeypatch.setattr(browser_text, "press", press)
    monkeypatch.setattr(ReadWebpageTool, "_read_screens", screens)
    used["results"] = [results]
    return used


async def test_direct_reading_comes_first(sources: dict[str, list[Any]]) -> None:
    tool = ReadWebpageTool()

    result = await tool.run({"url": "https://x.org/"})

    assert sources["http"] == ["https://x.org/"] and sources["browser"] == []
    assert tool.page == GOOD and "[1] Installation" in result.text
    assert not result.images


async def test_thin_pages_are_read_in_the_browser_then_from_screenshots(
    sources: dict[str, list[Any]],
) -> None:
    sources["results"][0]["http"] = THIN
    tool = ReadWebpageTool()

    await tool.run({"url": "https://x.org/"})

    assert sources["browser"] == ["https://x.org/"] and tool.page == GOOD

    sources["results"][0]["browser"] = THIN
    await tool.run({"url": "https://x.org/"})

    assert sources["screens"] == ["https://x.org/"] and tool.page is None


async def test_without_direct_reading_only_the_browser_loads(
    sources: dict[str, list[Any]],
) -> None:
    await ReadWebpageTool(direct=False).run({"url": "https://x.org/"})

    assert sources["http"] == [] and sources["browser"] == ["https://x.org/"]


async def test_links_are_followed_where_the_page_came_from(
    sources: dict[str, list[Any]],
) -> None:
    tool = ReadWebpageTool()
    tool.page = GOOD
    assert await tool.follow(GOOD.links[0]) == GOOD
    assert sources["http"] == ["https://x.org/i"] and sources["press"] == []

    # The user's browser (logged in): pressed there, not loaded without the login.
    button = Link("Mehr anzeigen", "", "button", ref=object())
    tool.page = Page("https://x.org/", "X", "Docs " * 100, (button,), "browser")
    assert await tool.follow(button) == GOOD
    assert sources["press"] == ["Mehr anzeigen"] and len(sources["http"]) == 1


def test_browser_text_is_unavailable_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(browser_text.sys, "platform", "win32")
    assert not browser_text.available()
    assert pages.MIN_TEXT > 0


def test_long_link_texts_are_shortened() -> None:
    from flay.tools.pages import short_label

    card = "uv An extremely fast Python package manager, written in Rust, with many features"
    assert short_label(card) == "uv An extremely fast Python package manager, written in …"
    assert short_label("Installation") == "Installation"


async def test_searching_uses_the_form_then_the_browser(sources: dict[str, list[Any]]) -> None:
    tool = ReadWebpageTool()
    form = SearchForm("https://docs.python.org/3/search.html", (), "q")
    tool.page = Page("https://docs.python.org/3/", "Docs", "Docs " * 100, (), "http", form)
    # Results built by JavaScript: the page itself has text, but none about the words.
    sources["results"][0]["http"] = Page("u", "Search", "Searching for words. " * 30, (), "http")

    assert await tool.search("asyncio") == GOOD

    assert sources["http"] == ["https://docs.python.org/3/search.html?q=asyncio"]
    assert sources["browser"] == ["https://docs.python.org/3/search.html?q=asyncio"]


async def test_tabs_opened_in_the_background_are_closed(
    sources: dict[str, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[str] = []

    async def close(page: Page) -> bool:
        closed.append(page.url)
        return True

    monkeypatch.setattr(browser_text, "close_tab", close)
    sources["results"][0]["http"] = THIN
    tool = ReadWebpageTool()
    await tool.run({"url": "https://x.org/"})  # read in a background tab

    await tool.finish()

    assert closed == ["https://x.org/"]
    await tool.finish()
    assert closed == ["https://x.org/"]  # once

    closed.clear()
    sources["results"][0]["http"] = GOOD
    await tool.run({"url": "https://x.org/"})  # read directly: no tab
    await tool.finish()
    assert closed == []
    await ReadWebpageTool(close_tabs=False).finish()


def test_windows_has_every_browser_function() -> None:
    import inspect

    from flay.tools import browser_uia

    for name in browser_text.PLATFORM_FUNCTIONS:
        mac, windows = getattr(browser_text, name), getattr(browser_uia, name)
        assert inspect.signature(mac).parameters.keys() == (
            inspect.signature(windows).parameters.keys()
        ), name
    # Everything web.py and forms use from browser_text is swapped on Windows.
    import re
    from pathlib import Path

    used = set()
    for module in ("web.py", "forms.py"):
        path = Path(browser_text.__file__).with_name(module)
        if path.exists():
            used |= set(re.findall(r"browser_text\.(\w+)\(", path.read_text()))
    assert used <= set(browser_text.PLATFORM_FUNCTIONS), used - set(browser_text.PLATFORM_FUNCTIONS)
