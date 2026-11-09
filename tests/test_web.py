import io
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from PIL import Image

from flay.assistant import Assistant, web_intent
from flay.config import Settings
from flay.core.messages import ImagePart, Message, Modality, Role, ToolCall
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.tools import web
from flay.tools.base import ToolRegistry, ToolResult, speech
from flay.tools.screen import Window
from flay.tools.web import find_url, last_url, same_screen, short
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Öffne diesen Link https://www.google.com/ fasse ihn zusammen", "https://www.google.com/"),
        ("Lies mal heise punkt de", "https://heise.de"),
        ("open python dot org", "https://python.org"),
        ("Öffne www.python.org/downloads.", "https://www.python.org/downloads"),
        ("Schau dir docs.python.org/3/tutorial an", "https://docs.python.org/3/tutorial"),
        ("Öffne den Link aus der Zwischenablage", None),
        ("Schreib an max@example.com", None),
        ("Öffne das Projekt flay", None),
    ],
)
def test_find_url(text: str, expected: str | None) -> None:
    assert find_url(text) == expected


def test_last_url_and_short_name() -> None:
    answer = "Docs: https://a.example/x and then https://b.example/y."

    assert last_url(answer) == "https://b.example/y"
    assert short("https://www.google.com/search?q=1") == "google.com"


def png(shade: int, dot: bool = False) -> bytes:
    image = Image.new("RGB", (400, 300), (shade, shade, shade))
    if dot:  # a blinking text cursor
        image.putpixel((200, 150), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_same_screen_ignores_a_blinking_cursor() -> None:
    assert same_screen(png(40), png(40, dot=True))
    assert not same_screen(png(40), png(200))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Öffne diesen Link https://www.google.com/ und gib es an Claude", "read_webpage"),
        ("Fasse mir heise punkt de zusammen", "read_webpage"),
        ("Lies den Link aus der Zwischenablage", "read_webpage"),
        ("Öffne google punkt de", "open_url"),
        ("Öffne den Link aus der Zwischenablage", "open_url"),
        ("Wie öffne ich einen Link in PhpStorm?", None),
        ("Öffne das Projekt flay", None),
        ("Schick Claude die Installationsschritte von dieser Seite als Aufgabe", "read_webpage"),
        ("Was steht auf dieser Webseite?", "read_webpage"),
        ("Öffne heise punkt de und klick auf Impressum", "read_webpage"),
        ("Klick auf den Link Impressum", None),
        ("Scroll auf dieser Seite runter", None),
        ("Blätter eine Seite weiter", None),
    ],
)
def test_web_intent(text: str, expected: str | None) -> None:
    assert web_intent(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Schick Claude die Schritte von dieser Seite", True),
        ("Übergib Claude die Links von google.com", True),
        ("Leite den Inhalt von heise.de an Claude weiter", True),
        ("Gib es an Claude", True),
        ("Ist Claude mit der Aufgabe fertig?", False),
        ("Schick mir die Seite", False),
    ],
)
def test_handing_over_to_claude(text: str, expected: bool) -> None:
    from flay.assistant import claude_intent, to_claude

    assert to_claude(text) is expected
    # Elsewhere "schick" stays with the model, which knows what "das Gleiche" was.
    assert claude_intent("Schick das Gleiche auch an Claude") is None


class PageTool:
    def __init__(self, name: str, result: ToolResult) -> None:
        self.name = name
        self.result = result
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.name, {"type": "object", "properties": {}})

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append(arguments)
        return self.result


class SummaryLLM:
    """Writes a summary of whatever it is shown."""

    def __init__(self) -> None:
        self.prompts: list[Sequence[Message]] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(frozenset({Modality.TEXT, Modality.IMAGE}))

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append(messages)
        yield TextDelta("Die Seite ist eine Suchmaschine.")


SHOT = ImagePart(b"png", "image/png")


def web_assistant(llm: SummaryLLM, *tools: PageTool) -> Assistant:
    return Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry(tools),
    )


async def test_page_is_read_summarized_and_handed_to_claude() -> None:
    reader = PageTool("read_webpage", ToolResult("Read it.", images=(SHOT, SHOT)))
    claude = PageTool(
        "send_to_claude", ToolResult("Typed.", say=speech("Steht in Claude.", "In Claude."))
    )
    llm = SummaryLLM()
    tts = FakeTTS()
    assistant = web_assistant(llm, reader, claude)
    assistant._tts = tts  # type: ignore[assignment]

    await assistant.handle_text(
        "Öffne diesen Link https://www.google.com/ fasse den Inhalt zusammen und gib es an Claude."
    )

    assert reader.calls == [{"url": "https://www.google.com/"}]
    # One model call (nothing to scroll or click with): the summary, from both screenshots.
    assert len(llm.prompts) == 1
    shown = [part for message in llm.prompts[0] for part in message.parts]
    assert [part for part in shown if part == SHOT] == [SHOT, SHOT]
    assert claude.calls == [
        {"instruction": "Die Seite ist eine Suchmaschine. (Quelle: https://www.google.com/)"}
    ]
    # Said first, while reading; then what the Claude tool reports.
    assert [text for text, _ in tts.spoken] == [
        "Ich öffne die Seite und lese sie.",
        "Steht in Claude.",
    ]


async def test_read_page_is_answered_by_the_model() -> None:
    reader = PageTool("read_webpage", ToolResult("Read it.", images=(SHOT,)))
    llm = SummaryLLM()
    assistant = web_assistant(llm, reader)

    await assistant.handle_text("Lies heise punkt de und sag mir, was da steht.")

    assert reader.calls == [{"url": "https://heise.de"}]
    shown = [part for message in llm.prompts[-1] for part in message.parts]
    assert SHOT in shown and llm.prompts[-1][-1].role is Role.USER
    assert assistant.history[-1].text == (
        "Ich öffne die Seite und lese sie. Die Seite ist eine Suchmaschine."
    )


async def test_link_from_the_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "clipboard_url", lambda: "https://example.org/a")
    monkeypatch.setattr("flay.assistant.clipboard_url", lambda: "https://example.org/a")
    opener = PageTool("open_url", ToolResult("Opened.", say=speech("Offen.", "Open.")))
    assistant = web_assistant(SummaryLLM(), opener)

    await assistant.handle_text("Öffne den Link aus der Zwischenablage")

    assert opener.calls == [{"url": "https://example.org/a"}]
    assert assistant.history[-1].text == "Offen."


async def test_missing_link_is_asked_for() -> None:
    opener = PageTool("open_url", ToolResult("Opened."))
    llm = SummaryLLM()
    assistant = web_assistant(llm, opener)

    await assistant.handle_text("Öffne den Link")

    assert opener.calls == [] and llm.prompts == []
    assert assistant.history[-1].text.startswith("Welchen Link soll ich öffnen?")


class ScriptedLLM:
    """Answers each call with the next scripted reply: a tool call or text."""

    def __init__(self, *replies: ToolCall | str) -> None:
        self.replies = list(replies)
        self.prompts: list[Sequence[Message]] = []
        self.offered: list[list[str]] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(frozenset({Modality.TEXT, Modality.IMAGE}))

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        self.prompts.append(list(messages))
        self.offered.append([spec.name for spec in tools])
        reply = self.replies.pop(0)
        if isinstance(reply, ToolCall):
            yield reply
        else:
            yield TextDelta(reply)


BROWSER = Window(4, "Firefox", "Getting started – Example Docs", (0, 30, 1500, 900))
AFTER_CLICK = ImagePart(b"after-click", "image/png")


class Reader(PageTool):
    """read_webpage that remembers the browser window, like the real tool."""

    window = BROWSER


def page_assistant(llm: ScriptedLLM, *tools: PageTool) -> tuple[Assistant, FakeTTS]:
    tts = FakeTTS()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        llm,
        tts,
        FakeOutput(),
        tools=ToolRegistry(tools),
    )
    return assistant, tts


async def test_task_on_a_page_scrolls_clicks_and_goes_to_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def look(max_size: int) -> ImagePart:
        return AFTER_CLICK

    monkeypatch.setattr("flay.assistant.look_at_page", look)
    reader = Reader("read_webpage", ToolResult("Read it.", images=(SHOT,)))
    scroller = PageTool("scroll_screen", ToolResult("Scrolled.", images=(SHOT,)))
    clicker = PageTool("click_on_screen", ToolResult("Clicked."))
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))
    llm = ScriptedLLM(
        ToolCall("1", "scroll_screen", {"direction": "down"}),
        ToolCall("2", "click_on_screen", {"text": "Installation"}),
        "DONE",
        "Installiere das Paket mit pip install example==2.1 und lege example.toml an.",
    )
    assistant, tts = page_assistant(llm, reader, scroller, clicker, claude)

    await assistant.handle_text(
        "Schick Claude die Installationsschritte von https://example.org/docs als Aufgabe"
    )

    # Only actions on the page are offered, and they stay in the browser window.
    assert llm.offered[0] == ["scroll_screen", "click_on_screen"]
    assert scroller.calls == [{"direction": "down", "window": "Firefox", "look": True}]
    assert clicker.calls == [{"text": "Installation", "window": "Firefox"}]
    # The screenshot after the click is shown to the model; then it writes, without tools.
    assert AFTER_CLICK in llm.prompts[2][-1].parts
    # The same tools as before (the prompt start stays cached), but asked to write.
    assert llm.offered[3] == llm.offered[0] and "Claude Code" in llm.prompts[3][-1].text
    assert claude.calls == [
        {
            "instruction": "Installiere das Paket mit pip install example==2.1 und lege "
            "example.toml an. (Quelle: https://example.org/docs)"
        }
    ]
    assert [text for text, _ in tts.spoken] == [
        "Ich öffne die Seite und lese sie.",
        "Ich scrolle weiter.",
        "Ich klicke auf Installation.",
        "In Claude.",
    ]


async def test_steps_are_limited() -> None:
    reader = PageTool("read_webpage", ToolResult("Read it.", images=(SHOT,)))
    scroller = PageTool("scroll_screen", ToolResult("Scrolled.", images=(SHOT,)))
    settings = Settings.model_validate({"tools": {"web": {"max_steps": 2}}})
    llm = ScriptedLLM(
        ToolCall("1", "scroll_screen", {"direction": "down"}),
        ToolCall("2", "scroll_screen", {"direction": "down"}),
        "Ganz unten steht das Impressum.",
    )
    assistant = Assistant(
        settings,
        FakeSource(),
        FakeSTT("x"),
        llm,
        FakeTTS(),
        FakeOutput(),
        tools=ToolRegistry([reader, scroller]),
    )

    await assistant.handle_text("Lies heise punkt de und sag mir, was ganz unten steht")

    assert len(scroller.calls) == 2 and len(llm.prompts) == 3
    assert "Now write" in llm.prompts[-1][-1].text
    assert assistant.history[-1].text.endswith("Ganz unten steht das Impressum.")


async def test_nothing_written_sends_nothing_to_claude() -> None:
    reader = PageTool("read_webpage", ToolResult("Read it.", images=(SHOT,)))
    claude = PageTool("send_to_claude", ToolResult("Typed."))
    llm = ScriptedLLM("")  # no actions on the page offered: it only writes
    assistant, _ = page_assistant(llm, reader, claude)

    await assistant.handle_text("Gib Claude die Installationsschritte von example punkt org")

    assert claude.calls == []
    assert assistant.history[-1].text.endswith("Ich konnte auf der Seite nichts lesen.")


async def test_risky_click_on_a_page_is_not_left_pending() -> None:
    from tests.test_pointer import AskingTool

    reader = PageTool("read_webpage", ToolResult("Read it.", images=(SHOT,)))
    llm = ScriptedLLM(
        ToolCall("1", "click_on_screen", {"text": "Löschen"}),
        "DONE",
        "Ich habe nichts gelöscht.",
    )
    assistant, _ = page_assistant(llm, reader, AskingTool())  # type: ignore[arg-type]

    await assistant.handle_text("Öffne example punkt org und klick auf Löschen")

    assert assistant.pending_confirmation is None
    tool_message = llm.prompts[1][-1]
    assert tool_message.role is Role.TOOL and "consent" in tool_message.text


async def test_the_open_page_is_read_without_a_link(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "front_browser", lambda: BROWSER)
    reader = Reader("read_webpage", ToolResult("Read it.", images=(SHOT,)))
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))
    llm = ScriptedLLM("Die Seite erklärt die Installation.")
    assistant, tts = page_assistant(llm, reader, claude)

    await assistant.handle_text("Schick Claude die Infos von dieser Seite")

    assert reader.calls == [{}]
    assert claude.calls == [
        {
            "instruction": "Die Seite erklärt die Installation. "
            "(Quelle: Getting started – Example Docs)"
        }
    ]
    assert tts.spoken[0][0] == "Ich lese die Seite."


async def test_this_page_without_a_browser_is_not_a_web_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web, "front_browser", lambda: None)
    reader = PageTool("read_webpage", ToolResult("Read it."))
    llm = ScriptedLLM("Dazu sehe ich nichts.")
    assistant, _ = page_assistant(llm, reader)

    await assistant.handle_text("Was steht auf dieser Seite?")

    assert reader.calls == []


async def test_search_buttons_are_not_clicked_and_the_page_end_is_said() -> None:
    page = ImagePart(png(40), "image/png")
    reader = PageTool("read_webpage", ToolResult("Read it.", images=(page,)))
    # At the end of the page scrolling shows the same picture again.
    scroller = PageTool("scroll_screen", ToolResult("Scrolled.", images=(page,)))
    clicker = PageTool("click_on_screen", ToolResult("Clicked."))
    llm = ScriptedLLM(
        ToolCall("1", "click_on_screen", {"text": "Google Suche"}),
        ToolCall("2", "scroll_screen", {"direction": "down"}),
        "DONE",
        "Eine Suchseite.",
    )
    assistant, _ = page_assistant(llm, reader, scroller, clicker)

    await assistant.handle_text("Lies google punkt com und sag mir, was drauf ist")

    assert clicker.calls == []
    assert "Not clicked" in llm.prompts[1][-1].text
    end = llm.prompts[2][-1]
    assert end.role is Role.TOOL and "end" in end.text and page not in end.parts
    assert assistant.history[-1].text.endswith("Eine Suchseite.")


class TextReader(PageTool):
    """read_webpage that read the page as text, like the real tool for most pages."""

    max_chars = 12000

    def __init__(self, page: Any, next_page: Any = None) -> None:
        super().__init__("read_webpage", ToolResult("Page text."))
        self.page = page
        self.next_page = next_page
        self.followed: list[str] = []

    async def follow(self, link: Any) -> Any:
        self.followed.append(link.text)
        self.page = self.next_page
        return self.next_page

    async def search(self, query: str) -> Any:
        self.followed.append(f"search: {query}")
        self.page = self.next_page
        return self.next_page

    async def load(self, url: str) -> Any:
        from flay.tools.pages import Page

        return Page(url, url, f"Text of {url}. " * 40, (), "http")


def docs_pages() -> tuple[Any, Any]:
    from flay.tools.pages import Link, Page

    start = Page(
        "https://docs.astral.sh/uv/",
        "uv",
        "An extremely fast Python package manager. " * 20,
        (
            Link("Log in", "https://docs.astral.sh/login"),
            Link("Installation", "https://docs.astral.sh/uv/getting-started/installation/"),
        ),
        "http",
    )
    install = Page(
        "https://docs.astral.sh/uv/getting-started/installation/",
        "Installing uv",
        "curl -LsSf https://astral.sh/uv/install.sh | sh " * 20,
        (),
        "http",
    )
    return start, install


async def test_task_on_a_text_page_follows_a_link_and_goes_to_claude() -> None:
    start, install = docs_pages()
    reader = TextReader(start, install)
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))
    llm = ScriptedLLM(
        ToolCall("1", "open_link", {"number": 1}),  # login: refused
        ToolCall("2", "open_link", {"number": 2}),
        "DONE",
        "Installiere uv mit curl -LsSf https://astral.sh/uv/install.sh | sh.",
    )
    assistant, tts = page_assistant(llm, reader, claude)

    await assistant.handle_text(
        "Schick Claude die Installationsschritte von docs punkt astral punkt sh als Aufgabe"
    )

    assert reader.followed == ["Installation"]
    first = llm.prompts[0][-1].text
    assert "[1] Log in\n[2] Installation" in first
    assert "Not opened" in llm.prompts[1][-1].text
    # The new page's text is what the model reads next; no screenshots anywhere.
    assert "install.sh" in llm.prompts[2][-1].text
    assert not any(isinstance(p, ImagePart) for m in llm.prompts[-1] for p in m.parts)
    assert claude.calls == [
        {
            "instruction": "Installiere uv mit curl -LsSf https://astral.sh/uv/install.sh | sh. "
            "(Quelle: https://docs.astral.sh/uv/getting-started/installation/)"
        }
    ]
    assert [text for text, _ in tts.spoken] == [
        "Ich öffne die Seite und lese sie.",
        "Ich öffne Installation.",
        "In Claude.",
    ]


async def test_long_text_is_read_on_request() -> None:
    from flay.tools.pages import Page

    lines = [f"## Abschnitt {i}\n" + "Zeile. " * 50 for i in range(40)]
    long = Page("https://x.org/", "X", "\n".join(lines), (), "http")
    reader = TextReader(long)
    reader.max_chars = 2000
    llm = ScriptedLLM(ToolCall("1", "read_more", {}), "DONE", "Ganz viele Zeilen.")
    assistant, _ = page_assistant(llm, reader)

    await assistant.handle_text("Lies x punkt org und sag mir, was drauf steht")

    assert llm.offered[0] == ["open_link", "read_more", "search_page"]
    first, more = llm.prompts[0][-1].text, llm.prompts[1][-1].text
    assert "Not shown yet:" in first and "Abschnitt 30" in first
    assert "## Abschnitt 5\n" in more and "## Abschnitt 5\n" not in first
    assert assistant.history[-1].text.endswith("Ganz viele Zeilen.")


@pytest.mark.parametrize(
    ("request_text", "whole"),
    [
        ("Fasse mir heise punkt de zusammen", True),
        ("Schick Claude den Inhalt dieser Webseite", True),
        ("Übergib Claude die Links von google.com", True),
        ("Summarize docs.python.org for me", True),
        ("Schick Claude die Installationsschritte von uv als Aufgabe", False),
        ("Lies heise punkt de und sag mir, was es Neues gibt", False),
    ],
)
def test_whole_page_requests_need_no_navigation(request_text: str, whole: bool) -> None:
    from flay.browsing import about_whole_page

    assert about_whole_page(request_text) is whole


async def test_an_answer_in_the_deciding_turn_counts_as_done() -> None:
    start, _ = docs_pages()
    llm = ScriptedLLM("uv ist ein schneller Paketmanager für Python, geschrieben in Rust.", "Kurz.")
    assistant, _ = page_assistant(llm, TextReader(start))

    await assistant.handle_text("Lies docs punkt astral punkt sh und sag mir, was uv ist")

    # The answer is written once more, in the writing turn, not followed by more steps.
    assert len(llm.prompts) == 2 and "Now write" in llm.prompts[1][-1].text
    assert llm.prompts[1][-2].text == "DONE"
    assert assistant.history[-1].text.endswith("Kurz.")


async def test_done_is_never_the_result() -> None:
    start, _ = docs_pages()
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))
    llm = ScriptedLLM("DONE", "uv ist ein Paketmanager.")
    assistant, _ = page_assistant(llm, TextReader(start), claude)

    await assistant.handle_text("Fasse docs punkt astral punkt sh zusammen und gib es an Claude")

    # Whole page: no deciding turn; the writing turn said DONE and is asked again, without
    # tools.
    assert llm.offered[-1] == []
    assert claude.calls[0]["instruction"].startswith("uv ist ein Paketmanager.")


async def test_the_request_echoed_to_claude_is_dropped() -> None:
    start, _ = docs_pages()
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))
    llm = ScriptedLLM(
        "DONE",
        "Schick Claude die Installationsschritte von uv. Installiere uv mit pip install uv.",
    )
    assistant, _ = page_assistant(llm, TextReader(start), claude)

    await assistant.handle_text(
        "Schick Claude die Installationsschritte von docs punkt astral punkt sh"
    )

    assert claude.calls[0]["instruction"].startswith("Installiere uv mit pip install uv.")


def test_all_links_in_a_request_are_found() -> None:
    from flay.tools.web import find_urls

    assert find_urls("Vergleiche python punkt org und ruby-lang punkt org") == [
        "https://python.org",
        "https://ruby-lang.org",
    ]
    assert find_urls("https://a.org/x, docs.b.com/y und nochmal https://a.org/x") == [
        "https://a.org/x",
        "https://docs.b.com/y",
    ]
    assert web_intent("Vergleiche python punkt org und ruby-lang punkt org") == "read_webpage"


async def test_pages_are_compared() -> None:
    start, _ = docs_pages()
    reader = TextReader(start)
    llm = ScriptedLLM("Beide sind Paketmanager.")
    assistant, _ = page_assistant(llm, reader)

    await assistant.handle_text("Vergleiche docs punkt astral punkt sh und pip punkt pypa punkt io")

    first = llm.prompts[0][1].text  # after the system prompt
    assert "Page 2:" in first and "Text of https://pip.pypa.io" in first
    # Comparing is about the pages as they are: no links followed.
    assert reader.followed == [] and len(llm.prompts) == 1
    assert assistant.history[-1].text.endswith("Beide sind Paketmanager.")


async def test_the_page_is_searched_when_nothing_links_there() -> None:
    from flay.tools.pages import Page, SearchForm

    docs = Page(
        "https://docs.python.org/3/",
        "Python",
        "Python documentation. " * 30,
        (),
        "http",
        SearchForm("https://docs.python.org/3/search.html", (), "q"),
    )
    results = Page(
        "https://docs.python.org/3/search.html?q=asyncio", "Search", "asyncio " * 50, (), "http"
    )
    reader = TextReader(docs, results)
    llm = ScriptedLLM(
        ToolCall("1", "search_page", {"text": "asyncio"}),
        "DONE",
        "asyncio ist für Nebenläufigkeit.",
    )
    assistant, tts = page_assistant(llm, reader)

    await assistant.handle_text("Lies docs punkt python punkt org und sag mir, wofür asyncio ist")

    assert "has a search" in llm.prompts[0][-1].text
    assert reader.followed == ["search: asyncio"]
    assert "Ich suche nach asyncio." in [text for text, _ in tts.spoken]


def test_windows_default_browser_programs() -> None:
    from flay.tools.browser_uia import available, program_for

    assert program_for("ChromeHTML") == "chrome.exe"
    assert program_for("FirefoxURL-308046B0AF4A39CB") == "firefox.exe"
    assert program_for("MSEdgeHTM") == "msedge.exe"
    assert program_for("SomethingElse") is None
    assert not available()  # not on Windows


async def test_a_named_part_of_a_long_page_is_read_and_list_numbers_work() -> None:
    from flay.tools.pages import Page

    lines = [f"## Part {i}\n" + "Text. " * 60 for i in range(30)]
    lines.append("## Inheritance\nclass B(A)\n" + "Inherit. " * 60)
    long = Page("https://x.org/", "X", "\n".join(lines), (), "http")
    reader = TextReader(long)
    reader.max_chars = 2000
    llm = ScriptedLLM(ToolCall("1", "read_more", {"part": "Inheritance"}), "DONE", "B erbt von A.")
    assistant, tts = page_assistant(llm, reader)

    await assistant.handle_text("Lies x punkt org und sag mir, was über Vererbung steht")

    assert "Inheritance" in llm.prompts[0][-1].text.split("Not shown yet:")[1]
    assert "class B(A)" in llm.prompts[1][-1].text
    assert "Ich lese Inheritance." in [text for text, _ in tts.spoken]


async def test_a_link_number_given_as_a_list_is_opened() -> None:
    start, install = docs_pages()
    reader = TextReader(start, install)
    llm = ScriptedLLM(ToolCall("1", "open_link", {"number": [2]}), "DONE", "Mit curl.")
    assistant, _ = page_assistant(llm, reader)

    await assistant.handle_text(
        "Lies docs punkt astral punkt sh und sag mir, wie man uv installiert"
    )

    assert reader.followed == ["Installation"]


async def test_an_echo_is_dropped_after_a_second_try_too() -> None:
    start, _ = docs_pages()
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))
    llm = ScriptedLLM(
        ToolCall("1", "read_more", {}),  # the writing turn calls a tool: asked again
        "Fasse die Seite zusammen und gib es Claude. uv ist ein Paketmanager.",
    )
    assistant, _ = page_assistant(llm, TextReader(start), claude)

    await assistant.handle_text("Fasse docs punkt astral punkt sh zusammen und gib es Claude")

    assert claude.calls[0]["instruction"].startswith("uv ist ein Paketmanager.")
