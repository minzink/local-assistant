"""Tasks on a web page: read, follow links, scroll and click until the information shows.

"Schick Claude die Installationsschritte von <Link> als Aufgabe" needs more than one
screenshot: the steps may sit further down, behind a tab or a link, or under a cookie
banner. The model sees the page, picks one action at a time (scroll or click, in the
browser window only) and gets a fresh screenshot after it; when it has what the request is
about, it writes the result: the message for Claude, or the answer for the user.

Pages read as text (``browse_text``) list their links and buttons by number; the model picks
one and Flay follows it over HTTP or presses it in the background browser – no mouse, no
screen change. Pages with little text are worked on from screenshots (``browse``): every new
screenshot costs the model ~8 s on an M2 Pro. Either way the steps are limited and each one
is announced, and the conversation only grows at the end, so what the model has already
read stays in the server's prompt cache.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from flay.core.language import Language
from flay.core.messages import ImagePart, Message, Role, TextPart, ToolCall
from flay.llm.base import ChatModel, TextDelta, ToolSpec
from flay.prompt import LANGUAGE_NAMES
from flay.tools.base import ToolResult, speech
from flay.tools.pages import Link, Page, describe
from flay.tools.pointer import is_risky
from flay.tools.web import same_screen

log = logging.getLogger(__name__)

# Actions on the page; anything else (opening projects, Claude, …) is not offered here.
BROWSE_TOOLS = ("click_on_screen", "scroll_screen")
MAX_STEPS = 4
# Clicks that only start something new (a search, a login) instead of leading to more of the
# page; the small model tried "Google Suche" and "Auf gut Glück" to "read" Google.
_NO_CLICK = re.compile(
    r"\b(?:such\w*|search\w*|glück|lucky|logo|log ?in|anmeld\w*|einlogg\w*|sign ?in|"
    r"abmeld\w*|ausloggen|log ?out|sign ?out)\b",
    re.IGNORECASE,
)

# Requests about the page itself: nothing to look for elsewhere. The small model opened
# "Gmail" to summarize Google, so these get no actions at all.
_WHOLE_PAGE = re.compile(
    r"zusammen\w*|inhalt\w*|überblick|übersicht|worum geht|was steht|was ist auf|"
    r"summar\w*|content|overview|what(?:'s| is) on|tl;?dr|\blinks\b",
    re.IGNORECASE,
)


def about_whole_page(request: str) -> bool:
    """The request is about the page as a whole (summary, content, its links)."""
    return _WHOLE_PAGE.search(request) is not None


_INSTRUCTIONS = """\
You work in the user's web browser for them. You see screenshots of a web page in order; \
later ones show the page after your actions. First make sure the screenshots show what the \
user's request needs; you will be asked to write the result afterwards.
Usually they already do: then reply only DONE. Act only if what the request asks for is \
clearly missing: scroll down (scroll_screen) for the rest of the page, or click the link, \
tab or button whose name leads to it (click_on_screen, its visible text in 'text'), e.g. \
"Installation" for installation steps. One action per turn; you get a new screenshot \
after it. If a cookie or consent banner covers the page, click its button to reject or to \
allow only necessary cookies. Never click search buttons or logos, never log in, fill in \
forms, buy, subscribe or send anything. If nothing on the page leads further, reply DONE."""

_WRITE = """\
Now write {result} Do not answer DONE, call no tools and do not repeat the request. Only \
use what the page shows; if something the user asked for is not on the page, say so \
briefly.{rules}"""

_FOR_CLAUDE = """\
the message for Claude Code, an AI coding agent, in {language}: plain text, no markdown, \
no line breaks. If the user wants Claude to do something, write it as a task in the \
imperative, starting with what to do (e.g. "Installiere …"), with every detail from the \
page Claude needs (commands, versions, names, settings, steps, copied exactly). If the \
user only passes the page on, summarize what it shows, at most six sentences, without \
remarks about tasks or what is missing."""

_FOR_USER = """\
your spoken answer to the user in {language}: one to three short sentences, plain text, \
no markdown."""


_TEXT_INSTRUCTIONS = """\
You work on web pages for the user. You get a page as text with its links and buttons, \
numbered. First make sure you have what the user's request needs; you will be asked to \
write the result afterwards.
Usually the page already has it: then reply only DONE. Act only if what the request asks \
for is clearly missing: open the link or press the button that leads to it (open_link with \
its number), e.g. "Installation" for installation steps; read the part of a long page \
that has it (read_more with its heading from "Not shown yet", e.g. "Inheritance" for \
"Vererbung"); or, if the page has a search and nothing \
links to it, search the page (search_page). One action per turn; you get the new text \
after it. Never open login, logout, shop or form links. If nothing leads further, reply \
DONE."""

_OPEN_LINK = ToolSpec(
    name="open_link",
    description="Open a link or press a button of the page by its number; returns the new page.",
    parameters={
        "type": "object",
        "properties": {"number": {"type": "integer", "description": "The number in [ ]"}},
        "required": ["number"],
    },
)
_READ_MORE = ToolSpec(
    name="read_more",
    description=(
        "Read more of a long page. 'part': a heading from 'Not shown yet' to read that part "
        "first (in the page's language); without it the next parts in order."
    ),
    parameters={"type": "object", "properties": {"part": {"type": "string"}}},
)
_SEARCH_PAGE = ToolSpec(
    name="search_page",
    description="Search with the page's own search field; returns the results page.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "What to search for"}},
        "required": ["text"],
    },
)


class PageReader(Protocol):
    """The page read last and how to follow its links (``ReadWebpageTool``)."""

    page: Page | None
    max_chars: int

    async def follow(self, link: Link) -> Page | None: ...

    async def search(self, query: str) -> Page | None: ...


@dataclass(frozen=True, slots=True)
class Finding:
    text: str
    steps: int = 0
    # The page the result comes from, if links were followed.
    url: str | None = None


async def browse_text(
    llm: ChatModel,
    request: str,
    reader: PageReader,
    language: Language,
    say: Callable[[str], None],
    *,
    for_claude: bool,
    rules: str = "",
    max_steps: int = MAX_STEPS,
    max_links: int = 40,
    others: Sequence[Page] = (),
) -> Finding:
    """Like ``browse``, on a page read as text: follow links instead of clicking.

    ``others``: further pages the request names ("vergleiche …"), shown after the first;
    only the first one's links can be followed.
    """
    page = reader.page
    assert page is not None
    start = page.url
    whole = about_whole_page(request) or bool(others)
    budget = reader.max_chars // (1 + len(others))
    shown_text, shown, more = describe(page, request, budget, max_links, whole=whole)
    first = [f"Request: {request}", shown_text]
    for number, other in enumerate(others, 2):
        first.append(f"Page {number}:\n" + describe(other, request, budget, 0, whole=True)[0])
    conversation = [
        Message.from_text(Role.SYSTEM, _TEXT_INSTRUCTIONS),
        Message.from_text(Role.USER, "\n".join(first), language),
    ]
    steps = 0
    text = ""
    if whole:
        max_steps = 0
    # The same tools in every turn, the writing one too: Ollama renders them at the start of
    # the prompt, and any change there makes the model read the whole conversation again
    # (15–25 s per turn with a docs page in a test).
    tools = [_OPEN_LINK, _READ_MORE, _SEARCH_PAGE]
    while steps < max_steps:
        text, calls = await _turn(llm, conversation, tools)
        if not calls:
            break
        call = calls[0]
        steps += 1
        conversation.append(
            Message(Role.ASSISTANT, (TextPart(text),), language, tool_calls=(call,))
        )
        text = ""
        opened: Page | None = None
        if call.name == "read_more" and not more:
            result = "The text ends here; there is nothing more to read on this page."
        elif call.name == "read_more":
            part = str(call.arguments.get("part") or "").strip()
            said = speech(f"Ich lese {part}.", f"I'm reading {part}.") if part else None
            say((said or speech("Ich lese weiter.", "I'm reading on."))[language])
            result, shown, more = describe(page, request, budget, max_links, shown, prefer=part)
        elif call.name == "open_link":
            result, opened = await _open_link(call, page, reader, language, say)
        elif call.name == "search_page":
            result, opened = await _search(call, page, reader, language, say)
        else:
            result = f"'{call.name}' is not available here."
        if opened is not None:
            page = opened
            result, shown, more = describe(page, request, budget, max_links)
        conversation.append(Message(Role.TOOL, (TextPart(result),), tool_call_id=call.id))
        log.info("Webseite (Text), Schritt %d: %s %s", steps, call.name, call.arguments)
    written = await _write(llm, conversation, text, language, for_claude, rules, tools)
    return Finding(written, steps, page.url if page.url != start else None)


async def _search(
    call: ToolCall,
    page: Page,
    reader: PageReader,
    language: Language,
    say: Callable[[str], None],
) -> tuple[str, Page | None]:
    query = " ".join(str(call.arguments.get("text") or "").split())[:200]
    if not query:
        return "Say what to search for.", None
    if page.search is None:
        return "This page has no search. Reply DONE or open a link.", None
    say(speech(f"Ich suche nach {query}.", f"I'm searching for {query}.")[language])
    found = await reader.search(query)
    if found is None:
        return f"The search for '{query}' brought no readable results.", None
    return "", found


async def _open_link(
    call: ToolCall,
    page: Page,
    reader: PageReader,
    language: Language,
    say: Callable[[str], None],
) -> tuple[str, Page | None]:
    """What to tell the model, and the new page if the link was followed."""
    try:
        given = call.arguments.get("number")
        if isinstance(given, list):  # "[6]" as the model copies it from the list
            given = given[0] if given else 0
        number = int(str(given or 0).strip("[] "))
    except (TypeError, ValueError):
        return "Give the number of a link from the list.", None
    if not 1 <= number <= len(page.links):
        return f"There is no link {number}.", None
    link = page.links[number - 1]
    if _NO_CLICK.search(link.text) or is_risky(link.text):
        return (
            f"Not opened: '{link.text}' does not lead to content (search, login, shop or "
            "sending). Reply DONE or pick another one.",
            None,
        )
    if link.kind == "button":
        say(speech(f"Ich drücke auf {link.text}.", f"I'm pressing {link.text}.")[language])
    else:
        say(speech(f"Ich öffne {link.text}.", f"I'm opening {link.text}.")[language])
    opened = await reader.follow(link)
    if opened is None:
        return f"'{link.text}' could not be opened or shows hardly any text.", None
    return "", opened


async def _write(
    llm: ChatModel,
    conversation: list[Message],
    text: str,
    language: Language,
    for_claude: bool,
    rules: str,
    tools: Sequence[ToolSpec] = (),
) -> str:
    """The second turn: the result, written from everything read.

    ``tools``: the ones the deciding turns had, so the prompt start stays cached; calls are
    ignored.
    """
    if text:
        conversation.append(Message(Role.ASSISTANT, (TextPart(text),), language))
    result = (_FOR_CLAUDE if for_claude else _FOR_USER).format(language=LANGUAGE_NAMES[language])
    conversation.append(
        Message.from_text(Role.USER, _WRITE.format(result=result, rules=rules), language)
    )
    written, _ = await _turn(llm, conversation, tools, decide=False)
    if not _clean(written, for_claude) and tools:
        # It called a tool or said DONE again: once more without any.
        written, _ = await _turn(llm, conversation, [], decide=False)
    return _clean(written, for_claude)


def _clean(written: str, for_claude: bool) -> str:
    """For Claude: "Schick Claude die Installationsschritte …" is the request echoed, not
    the task. Spoken: markdown marks would be read out or garble the voice."""
    written = _DONE.sub("", " ".join(written.split()))
    return _ECHO.sub("", written, count=1) if for_claude else _MARKS.sub("", written)


async def browse(
    llm: ChatModel,
    request: str,
    page: str,
    seen: ToolResult,
    language: Language,
    tools: Sequence[ToolSpec],
    act: Callable[[ToolCall], Awaitable[ToolResult]],
    say: Callable[[str], None],
    *,
    for_claude: bool,
    look: Callable[[], Awaitable[ImagePart | None]] | None = None,
    window: str | None = None,
    rules: str = "",
    max_steps: int = MAX_STEPS,
) -> Finding:
    """Work on the page until the request can be answered; what the model wrote.

    ``seen`` is what read_webpage returned (its screenshots). ``act`` runs a tool call;
    ``look`` takes a screenshot of the page after a click (scrolling brings its own);
    ``window`` keeps clicks and scrolling in the browser window.

    Looking and writing are two turns of one conversation: asked in one go, the small model
    gave up too early ("not on the page") or clicked around. The writing turn only adds a
    message at the end, so the screenshots stay in the prompt cache.
    """
    offered = [spec for spec in tools if spec.name in BROWSE_TOOLS]
    conversation = [
        Message.from_text(Role.SYSTEM, _INSTRUCTIONS),
        Message(
            Role.USER,
            (TextPart(f"Request: {request}\nPage: {page}\n{seen.text}"), *seen.images),
            language,
        ),
    ]
    steps = 0
    last = seen.images[-1] if seen.images else None
    if about_whole_page(request):
        # Screens further down are still worth a look (read_webpage stops after a few).
        offered = [spec for spec in offered if spec.name == "scroll_screen"]
    while offered and steps < max_steps:
        text, calls = await _turn(llm, conversation, offered)
        if not calls:
            break
        # One action at a time: each one changes what the next has to look at.
        call = _in_window(calls[0], window)
        steps += 1
        if announcement := _announce(call, language):
            say(announcement)
        conversation.append(
            Message(Role.ASSISTANT, (TextPart(text),), language, tool_calls=(call,))
        )
        if call.name not in {spec.name for spec in offered}:
            outcome = ToolResult(f"'{call.name}' is not available here.", ok=False)
        elif call.name == "click_on_screen" and _NO_CLICK.search(_target(call)):
            outcome = ToolResult(
                "Not clicked: searching or logging in is not reading the page. Reply DONE or "
                "pick a link that leads to the requested content.",
                ok=False,
            )
        else:
            outcome = await act(call)
        if outcome.confirm is not None:
            outcome = ToolResult(
                "Not clicked: this needs the user's consent. Do not click it; go on without "
                "it or finish.",
                ok=False,
            )
        conversation.append(Message(Role.TOOL, (TextPart(outcome.text),), tool_call_id=call.id))
        images = list(outcome.images)
        if call.name == "scroll_screen" and images and _unchanged(last, images[-1]):
            # Said instead of shown: the same picture again makes the model scroll on.
            conversation[-1] = Message(
                Role.TOOL,
                (TextPart("The page did not move: this is its end. Reply DONE."),),
                tool_call_id=call.id,
            )
            images = []
        clicked = call.name == "click_on_screen" and outcome.ok
        if clicked and look is not None and (after := await look()) is not None:
            images.append(after)
        if images:
            last = images[-1]
            # Tool messages carry text only in the chat API; images follow as a user turn.
            conversation.append(
                Message(Role.USER, (TextPart(f"The page after {call.name}:"), *images))
            )
        log.info("Webseite, Schritt %d: %s %s → %s", steps, call.name, call.arguments, outcome.ok)
    else:
        text = ""
    written = await _write(llm, conversation, text, language, for_claude, rules, offered)
    return Finding(written, steps)


# In a deciding turn, this much text without an action means "done": the model has started
# to write the answer, which the writing turn does properly. Stopping saves its whole
# generation (17 s of 24 s for one page in a test).
_DECIDED = 24


_DONE = re.compile(r"^DONE\b\W*")
_MARKS = re.compile(r"[`*_#]+")
_ECHO = re.compile(r"^[^.!?]*\bclaude\b[^.!?]*[.!?]\s*", re.IGNORECASE)


async def _turn(
    llm: ChatModel,
    conversation: list[Message],
    tools: Sequence[ToolSpec],
    decide: bool = True,
) -> tuple[str, list[ToolCall]]:
    text: list[str] = []
    calls: list[ToolCall] = []
    async for event in llm.stream(conversation, tools):
        if isinstance(event, ToolCall):
            calls.append(event)
        elif isinstance(event, TextDelta):
            text.append(event.text)
            if decide and tools and not calls and sum(map(len, text)) > _DECIDED:
                return "DONE", []  # leaving the loop cancels the generation
    return "".join(text), calls


def _in_window(call: ToolCall, window: str | None) -> ToolCall:
    arguments = dict(call.arguments)
    if window:
        arguments["window"] = window
    if call.name == "scroll_screen":
        arguments["look"] = True
        arguments.setdefault("direction", "down")
    return ToolCall(call.id, call.name, arguments)


def _target(call: ToolCall) -> str:
    return f"{call.arguments.get('text') or ''} {call.arguments.get('description') or ''}"


def _unchanged(before: ImagePart | None, after: ImagePart) -> bool:
    if before is None:
        return False
    try:
        return same_screen(before.data, after.data)
    except Exception:  # not a picture (simulated tools)
        return False


def _announce(call: ToolCall, language: Language) -> str | None:
    if call.name == "click_on_screen":
        what = str(call.arguments.get("text") or call.arguments.get("description") or "").strip()
        said = (
            speech(f"Ich klicke auf {what}.", f"I'm clicking {what}.")
            if what
            else speech("Ich klicke.", "I'm clicking.")
        )
    elif call.name == "scroll_screen":
        said = speech("Ich scrolle weiter.", "I'm scrolling on.")
    else:
        return None
    return said[language]
