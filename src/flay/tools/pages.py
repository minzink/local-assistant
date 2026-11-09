"""Web pages as text: what a page says and where its links lead.

Reading the text is far faster and more exact than screenshots (a whole docs page is
~1500–1800 tokens, three screenshots ~3200 tokens plus the image encoder, and commands come
out letter for letter). Pages come from two places:

- directly over HTTP (``fetch_page``): invisible, ~0.2 s, but without the user's logins
  and empty for pages that JavaScript builds;
- from the user's browser through the accessibility tree (``flay.tools.browser_text``):
  logins and JavaScript included.

Screenshots stay the fallback for pages with little text (images, canvas).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import urldefrag, urljoin, urlsplit

log = logging.getLogger(__name__)

# Less text than this is not the page (a JavaScript shell, a cookie wall, an error).
MIN_TEXT = 300
FETCH_SECONDS = 8.0
MAX_BYTES = 5_000_000
# A desktop browser's Accept headers; some servers answer anything else with a stub page.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:140.0) Gecko/20100101 "
    "Firefox/140.0",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
}
_JS_ONLY = re.compile(
    r"(?:enable|activate) javascript|javascript (?:is )?(?:required|disabled)|"
    r"javascript aktivieren",
    re.IGNORECASE,
)

Source = Literal["http", "browser", "file"]


@dataclass(frozen=True, slots=True)
class Link:
    text: str
    url: str = ""
    # "button": no address, only pressing it in the browser does something.
    kind: Literal["link", "button"] = "link"
    # In the page's main content (not navigation).
    main: bool = True
    # The browser's accessibility element, to press it without the mouse.
    ref: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class SearchForm:
    """A page's search: a GET form (``action`` + ``params`` + ``name`` of the words' field)
    or, in the browser, the search field and its button to fill in and press."""

    action: str = ""
    params: tuple[tuple[str, str], ...] = ()
    name: str = ""
    ref: Any = field(default=None, compare=False, repr=False)
    button: Any = field(default=None, compare=False, repr=False)

    def url(self, query: str) -> str | None:
        if not self.action or not self.name:
            return None
        from urllib.parse import urlencode

        separator = "&" if "?" in self.action else "?"
        return self.action + separator + urlencode([*self.params, (self.name, query)])


@dataclass(frozen=True, slots=True)
class Input:
    """A field of a form in the browser, to fill in what the user dictated."""

    label: str
    # Passwords are never filled in.
    secret: bool = False
    ref: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    links: tuple[Link, ...]
    source: Source
    search: SearchForm | None = None
    inputs: tuple[Input, ...] = ()

    @property
    def thin(self) -> bool:
        return len(self.text) < MIN_TEXT or (
            len(self.text) < 1000 and _JS_ONLY.search(self.text) is not None
        )


def same_document(a: str, b: str) -> bool:
    return urldefrag(a).url.rstrip("/") == urldefrag(b).url.rstrip("/")


def host(url: str) -> str:
    return (urlsplit(url).hostname or "").removeprefix("www.")


def sections(text: str, size: int) -> list[str]:
    """The text in parts: one per heading, long ones cut at line ends to at most ``size``."""
    parts: list[str] = []
    for part in re.split(r"(?m)^(?=#{1,4} )", text):
        part = part.strip()
        while len(part) > size:
            cut = part.rfind("\n", 0, size)
            cut = cut if cut > size // 2 else size
            parts.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            parts.append(part)
    return parts


# Words of a request that say nothing about the page's content.
_FILLER = {
    "claude", "schick", "schicke", "gib", "gebe", "übergib", "aufgabe", "seite", "webseite",
    "link", "bitte", "mir", "sag", "lies", "lese", "öffne", "send", "give", "tell", "page",
    "please", "read", "open", "diese", "dieser", "dieses", "this", "that", "von", "from",
    "und", "and", "was", "what", "wie", "how", "über", "about", "geht", "steht", "punkt", "dot",
}  # fmt: skip


def request_words(request: str) -> set[str]:
    """Stems (first five letters) of the request's meaningful words."""
    return {
        word[:5]
        for word in re.findall(r"\w{4,}", request.lower())
        if word not in _FILLER and not word.isdigit()
    }


def pick(
    parts: list[str], request: str, budget: int, shown: frozenset[int], prefer: str = ""
) -> list[int]:
    """Parts to show next: the one titled ``prefer``, those with the request's words, the
    rest in order."""
    words = request_words(request)
    wanted = " ".join(prefer.lower().lstrip("#").split())

    def score(index: int) -> int:
        lower = parts[index].lower()
        heading = lower.split("\n", 1)[0]
        named = 1000 if wanted and wanted in " ".join(heading.lstrip("#").split()) else 0
        return named + sum(3 if word in heading else 1 for word in words if word in lower)

    left = [i for i in range(len(parts)) if i not in shown]
    matching = sorted((i for i in left if score(i) > 0), key=lambda i: (-score(i), i))
    order = matching + [i for i in left if i not in matching]
    chosen: list[int] = []
    size = 0
    for index in order:
        if chosen and size + len(parts[index]) > budget:
            continue
        chosen.append(index)
        size += len(parts[index])
    return sorted(chosen)


def ranked_links(page: Page, request: str, limit: int) -> list[tuple[int, Link]]:
    """Up to ``limit`` links with their numbers: the request's words first, then in order.

    Numbers are positions in ``page.links``, so they stay the same whatever is shown.
    """
    words = {w for w in re.findall(r"\w{3,}", request.lower())}

    def score(link: Link) -> int:
        label = set(re.findall(r"\w{3,}", link.text.lower()))
        return sum(1 for w in words for t in label if t.startswith(w[:5]) or w.startswith(t[:5]))

    numbered = list(enumerate(page.links, 1))
    relevant = [item for item in numbered if score(item[1]) > 0]
    relevant.sort(key=lambda item: -score(item[1]))
    chosen = relevant[: limit // 2]
    taken = {number for number, _ in chosen}
    rest = [item for item in numbered if item[0] not in taken]
    # Links in the content before navigation.
    rest.sort(key=lambda item: not item[1].main)
    chosen += rest[: limit - len(chosen)]
    return sorted(chosen)


def _about(parts: list[str], chosen: list[int], request: str) -> bool:
    words = request_words(request)
    return any(word in parts[i].lower() for i in chosen for word in words)


def describe(
    page: Page,
    request: str,
    max_chars: int,
    max_links: int,
    shown: frozenset[int] = frozenset(),
    whole: bool = False,
    prefer: str = "",
) -> tuple[str, frozenset[int], bool]:
    """The page as the model reads it: text, then numbered links and buttons.

    A long page is shown in parts: first those about the request (all in order if
    ``whole``), with the headings of the rest; ``shown`` are the parts seen already.
    Returns the text, all parts shown now and whether parts are left.
    """
    parts = sections(page.text, max_chars)
    request_for_parts = "" if whole else request
    chosen = pick(parts, request_for_parts, max_chars, shown, prefer)
    now = shown | frozenset(chosen)
    left = [i for i in range(len(parts)) if i not in now]
    where = {"browser": "the user's browser", "http": "the web", "file": "a local file"}[
        page.source
    ]
    lines = [f"Page: {page.title or page.url} ({page.url}), read as text from {where}."]
    if left or shown:
        lines.append(
            f"(A long page: parts {len(now)} of {len(parts)}"
            + (
                ", the ones about the request first"
                if _about(parts, chosen, request_for_parts)
                else ""
            )
            + "; read_more shows more.)"
        )
    lines += ["Text:", "\n…\n".join(parts[i] for i in chosen)]
    if left:
        headings = [parts[i].split("\n", 1)[0][:80] for i in left if parts[i].startswith("#")]
        if headings:
            lines.append("Not shown yet: " + " · ".join(headings[:30]))
    links = ranked_links(page, request, max_links)
    if links:
        # Texts only: the addresses would double the list's tokens and tell the model little.
        lines.append("Links and buttons:")
        for number, link in links:
            lines.append(f"[{number}] {link.text}" + ("" if link.url else " (button)"))
    if page.search is not None:
        lines.append("The page has a search (search_page).")
    return "\n".join(lines), now, bool(left)


# --- HTTP ----------------------------------------------------------------------------------

_SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head", "canvas", "select"}
_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
    "track", "wbr",
}  # fmt: skip
_BLOCK = {
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "tr", "section", "article",
    "dt", "dd", "table", "ul", "ol", "nav", "header", "footer", "main", "aside", "blockquote",
    "figure", "figcaption", "details", "summary", "form", "fieldset", "br", "hr",
}  # fmt: skip
_OUTSIDE_MAIN = {"nav", "header", "footer", "aside"}


class _Extractor(HTMLParser):
    """Visible text and links of an HTML page; the main content is kept separately."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.hidden_at: int | None = None
        self.main_at: int | None = None
        self.outside_at: int | None = None
        self.pre = 0
        self.all: list[str] = []
        self.main: list[str] = []
        self.title: list[str] = []
        self.in_title = False
        self.base: str | None = None
        self.links: list[tuple[str, str, bool]] = []
        self.href: str | None = None
        self.label = ""
        self.link_text: list[str] = []
        # The open <form>: action, whether it looks like a search, inputs (name, type, value).
        self.form: tuple[str, bool, list[tuple[str, str, str]]] | None = None
        self.searches: list[tuple[int, str, tuple[tuple[str, str], ...], str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k: v or "" for k, v in attrs}
        if tag == "title":
            self.in_title = True
        if tag == "base" and values.get("href"):
            self.base = values["href"]
        if tag == "form":
            method = values.get("method", "get").lower()
            marks = " ".join(values.get(k, "") for k in ("role", "class", "id", "action"))
            looks = bool(re.search(r"search|such", marks, re.IGNORECASE))
            self.form = (values.get("action", ""), looks, []) if method == "get" else None
        if tag == "input" and self.form is not None and values.get("name"):
            self.form[2].append(
                (values["name"], values.get("type", "text").lower(), values.get("value", ""))
            )
        if tag in _VOID:
            if tag in ("br", "hr"):
                self._write("\n")
            return
        self.stack.append(tag)
        depth = len(self.stack)
        hidden = (
            tag in _SKIP
            or "hidden" in values
            or values.get("aria-hidden") == "true"
            or re.search(r"display\s*:\s*none", values.get("style", "")) is not None
        )
        if hidden and self.hidden_at is None:
            self.hidden_at = depth
        if self.main_at is None and (tag in ("main", "article") or values.get("role") == "main"):
            self.main_at = depth
        if self.outside_at is None and tag in _OUTSIDE_MAIN:
            self.outside_at = depth
        if tag == "pre":
            self.pre += 1
        if tag in _BLOCK:
            self._write("\n")
        if tag in ("h1", "h2", "h3", "h4"):
            self._write("#" * int(tag[1]) + " ")
        elif tag == "li":
            self._write("- ")
        if tag == "a" and values.get("href"):
            self.href = values["href"]
            self.label = values.get("aria-label") or values.get("title") or ""
            self.link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "form" and self.form is not None:
            self._end_form(*self.form)
            self.form = None
        if tag in _VOID or tag not in self.stack:
            return
        while self.stack:
            open_tag = self.stack.pop()
            depth = len(self.stack) + 1
            if self.hidden_at is not None and depth <= self.hidden_at:
                self.hidden_at = None
            if self.main_at is not None and depth <= self.main_at:
                self.main_at = None
            if self.outside_at is not None and depth <= self.outside_at:
                self.outside_at = None
            if open_tag == "pre":
                self.pre = max(0, self.pre - 1)
            if open_tag == "a" and self.href is not None:
                text = " ".join("".join(self.link_text).split()) or self.label
                if text and self.hidden_at is None:
                    self.links.append((text, self.href, self.main_at is not None))
                self.href = None
            if open_tag in _BLOCK:
                self._write("\n")
            if open_tag == tag:
                break

    def _end_form(self, action: str, looks: bool, inputs: list[tuple[str, str, str]]) -> None:
        """Keep a GET form with one field to type into, scored by how much it is a search."""
        typed = [(n, t) for n, t, _ in inputs if t in ("text", "search")]
        if len(typed) != 1:
            return  # no field, or a form with several (login, contact): never filled in
        name, kind = typed[0]
        score = (
            2 * (kind == "search")
            + 2 * (name.lower() in ("q", "query", "search", "s", "suche", "term", "keyword"))
            + looks
        )
        if score:
            hidden = tuple((n, v) for n, t, v in inputs if t == "hidden")
            self.searches.append((score, action, hidden, name))

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
            return
        if self.hidden_at is not None:
            return
        if not self.pre:
            data = re.sub(r"\s+", " ", data)
        self._write(data)
        if self.href is not None:
            self.link_text.append(data)

    def _write(self, text: str) -> None:
        if self.hidden_at is not None:
            return
        self.all.append(text)
        if self.main_at is not None and self.outside_at is None:
            self.main.append(text)


def _tidy(parts: list[str]) -> str:
    text = "".join(parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def short_label(label: str, limit: int = 60) -> str:
    """A link's text as said and listed: cards and logos carry whole paragraphs."""
    label = " ".join(label.split())
    return label if len(label) <= limit else label[: label.rfind(" ", 0, limit) or limit] + " …"


def parse_html(html: str, url: str) -> Page:
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    main = _tidy(parser.main)
    text = main if len(main) >= MIN_TEXT else _tidy(parser.all)
    base = urljoin(url, parser.base) if parser.base else url
    links: list[Link] = []
    seen: set[str] = set()
    for label, href, in_main in parser.links:
        target = urljoin(base, href.strip())
        if urlsplit(target).scheme not in ("http", "https") or same_document(target, url):
            continue  # mailto:, javascript:, a section of this page
        key = urldefrag(target).url
        if key in seen:
            continue
        seen.add(key)
        links.append(Link(short_label(label), target, main=in_main or len(main) < MIN_TEXT))
    title = " ".join("".join(parser.title).split())
    search = None
    if parser.searches:
        _, action, hidden, name = max(parser.searches, key=lambda found: found[0])
        search = SearchForm(urljoin(base, action) if action else url.split("?")[0], hidden, name)
    return Page(url, title, text, tuple(links), "http", search)


async def fetch_page(url: str, language: str = "de") -> Page | None:
    """The page at ``url`` over HTTP, or None if it is not an HTML or text page."""
    import aiohttp

    headers = {**_HEADERS, "Accept-Language": f"{language},en;q=0.8"}
    timeout = aiohttp.ClientTimeout(total=FETCH_SECONDS)
    try:
        async with (
            aiohttp.ClientSession(headers=headers, timeout=timeout) as session,
            session.get(url, allow_redirects=True) as response,
        ):
            kind = response.content_type or ""
            if response.status >= 400 or not (kind.startswith("text/") or "html" in kind):
                log.info("Webseite direkt: %s → %s %s", url, response.status, kind)
                return None
            # read(n) returns what has arrived so far, not the whole page.
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.content.iter_chunked(65536):
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_BYTES:
                    break
            body = b"".join(chunks)
            final = str(response.url)
            charset = response.charset or "utf-8"
    except (aiohttp.ClientError, TimeoutError, UnicodeError, ValueError) as exc:
        log.info("Webseite direkt nicht ladbar: %s (%s)", url, exc or type(exc).__name__)
        return None
    html = body.decode(charset, errors="replace")
    if kind == "text/plain":
        return Page(final, "", html.strip(), (), "http")
    return replace(parse_html(html, final), url=final)
