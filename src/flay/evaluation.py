"""`flay eval`: how often the LLM picks the right action, and how fast it answers.

Every case goes through the real assistant – routing, prompt, tool loop, the check for
made-up claims – against the configured LLM server. Only the tools and what is open on
the computer are simulated (from the case file), so nothing happens on the computer and
the results are the same on macOS and Windows.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from flay.assistant import Assistant
from flay.audio.base import AudioBuffer
from flay.config import Settings
from flay.core.events import AssistantDelta, Event, EventBus, ToolUsed
from flay.core.language import Language
from flay.core.messages import Message, Role, ToolCall, ToolOutcome
from flay.core.text_language import detect_text_language
from flay.llm.base import ChatModel, ToolSpec
from flay.memory.base import Memory, ShortTermMemory
from flay.memory.persistent import PersistentMemory
from flay.memory.store import MemoryStore
from flay.stt.base import Transcript
from flay.stt.vocabulary import Vocabulary
from flay.tools.base import ToolRegistry, ToolResult, speech
from flay.tools.pages import Link, Page, SearchForm
from flay.tools.phpstorm import Project, match_projects
from flay.tools.phpstorm_mcp import describe

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "eval" / "cases.yaml"

# Arguments that name a project, per tool.
PROJECT_ARGUMENT = {
    "open_project": "name",
    "close_project": "name",
    "send_to_claude": "project",
    "claude_status": "project",
}
TEXT_ARGUMENT = {
    "send_to_claude": "instruction",
    "remember": "fact",
    "forget": "fact",
    "click_on_screen": "text",
    "read_webpage": "url",
    "open_url": "url",
    "fill_web_form": "fields",
    "read_file": "name",
    "workflow": "name",
}
# Tools that only read; the model answers from what they return.
READING_TOOLS = ("claude_status", "look_at_screen", "read_webpage")


@dataclass
class CaseResult:
    say: str
    expected: str
    tools: list[dict[str, Any]]
    reply: str
    ok: bool
    problem: str
    # Seconds from the request to the first reply text / to the first tool call.
    first_text: float | None
    first_tool: float | None
    total: float
    # Flay asked to confirm first ("Meinst du …?").
    confirmed: bool = False


@dataclass
class Report:
    label: str
    model: str
    base_url: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(r.ok for r in self.results) / len(self.results) if self.results else 0.0

    def summary(self) -> dict[str, Any]:
        def median(values: list[float | None]) -> float | None:
            present = [v for v in values if v is not None]
            return round(statistics.median(present), 2) if present else None

        actions = [r for r in self.results if r.expected != "none"]
        chats = [r for r in self.results if r.expected == "none"]
        return {
            "label": self.label,
            "model": self.model,
            "runs": len(self.results),
            "accuracy": round(self.accuracy, 3),
            "actions_ok": f"{sum(r.ok for r in actions)}/{len(actions)}",
            # Failed actions where Flay asked back instead of doing something wrong.
            "asked_instead": sum(not r.ok and not r.tools and "?" in r.reply for r in actions),
            "wrong_action": sum(not r.ok and bool(r.tools) for r in self.results),
            "confirmed_first": sum(r.confirmed for r in self.results),
            "chat_ok": f"{sum(r.ok for r in chats)}/{len(chats)}",
            "first_tool_median_s": median([r.first_tool for r in actions]),
            "first_text_median_s": median([r.first_text for r in chats]),
            "total_median_s": median([r.total for r in self.results]),
        }


class _TimedEvents(EventBus):
    """Remembers when the reply's first text and the first tool call happened."""

    def __init__(self) -> None:
        super().__init__()
        self.start = time.perf_counter()
        self.first_text: float | None = None
        self.first_tool: float | None = None

    def publish(self, event: Event) -> None:
        now = time.perf_counter() - self.start
        if isinstance(event, AssistantDelta) and event.text.strip() and self.first_text is None:
            self.first_text = now
        if isinstance(event, ToolUsed) and self.first_tool is None:
            self.first_tool = now
        super().publish(event)


class _SimulatedTool:
    """A tool with the real spec that only records its calls."""

    def __init__(self, spec: ToolSpec, calls: list[dict[str, Any]]) -> None:
        self._spec = spec
        self._calls = calls

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self._calls.append({"tool": self._spec.name, **arguments})
        if self._spec.name == "read_webpage":
            return ToolResult(
                "(simulated) The page shows a search engine: a search field, the buttons "
                "'Google Suche' and 'Auf gut Glück' and links to privacy and terms."
            )
        if self._spec.name in ("scroll_screen", "click_on_screen"):
            # On the simulated page there is nothing more to find.
            return ToolResult("(simulated) Done; the page shows the same as before.")
        if self._spec.name in READING_TOOLS:
            # Reading tools: the model answers from the result, as in real use.
            return ToolResult("(simulated) Claude is idle and waiting for input.")
        return ToolResult("(simulated) Done.", say=speech("Erledigt.", "Done."))


class _SimulatedReader(_SimulatedTool):
    """read_webpage as in real use for most pages: the page as text with numbered links."""

    max_chars = 12000

    def __init__(self, spec: ToolSpec, calls: list[dict[str, Any]]) -> None:
        super().__init__(spec, calls)
        self.page: Page | None = None

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        from flay.tools.pages import describe

        self._calls.append({"tool": self._spec.name, **arguments})
        url = str(arguments.get("url") or "https://www.google.com/")
        self.page = _SIMULATED_PAGE if "google." in url else _simulated_page(url)
        return ToolResult(describe(self.page, "", self.max_chars, 40)[0])

    async def load(self, url: str) -> Page | None:
        self._calls.append({"tool": "load_page", "url": url})
        return _simulated_page(url)

    async def search(self, query: str) -> Page | None:
        self._calls.append({"tool": "search_page", "text": query})
        assert self.page is not None
        self.page = Page(
            self.page.url,
            "Search results",
            f"(simulated) Search results for '{query}': " + f"{query} explained. " * 20,
            (),
            "http",
        )
        return self.page

    async def follow(self, link: Link) -> Page | None:
        self._calls.append({"tool": "open_link", "text": link.text})
        self.page = Page(
            link.url, link.text, f"(simulated) The page '{link.text}' of Google. " * 10, (), "http"
        )
        return self.page


class _SimulatedFile(_SimulatedTool):
    """read_file with a found file: a project README as text."""

    max_chars = 8000

    def __init__(self, spec: ToolSpec, calls: list[dict[str, Any]]) -> None:
        super().__init__(spec, calls)
        self.page: Page | None = None
        self.path: Path | None = None

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        self._calls.append({"tool": self._spec.name, **arguments})
        name = str(arguments.get("name") or "README")
        self.path = Path("/simulated") / str(arguments.get("where") or "project") / name
        self.page = Page(
            self.path.as_uri(),
            name,
            f"# {name}\n(simulated) Installation: run uv sync, then flay setup. Usage: flay run. "
            * 5,
            (),
            "file",
        )
        return ToolResult(f"Read {self.path}.")

    async def follow(self, link: Link) -> Page | None:
        return None

    async def search(self, query: str) -> Page | None:
        return None


def _simulated_page(url: str) -> Page:
    from flay.tools.pages import host

    site = host(url) or url
    text = (
        f"# {site}\n(simulated) The start page of {site}: an introduction to what the site "
        "offers, the latest news with short teasers, and a menu of sections.\n"
    ) + f"More introductory text of {site}. " * 15
    links = tuple(
        Link(label, f"https://{site}/{path}")
        for label, path in (
            ("News", "news"),
            ("Installation", "install"),
            ("Tutorial", "tutorial"),
            ("Anmelden", "login"),
            ("Datenschutz", "privacy"),
        )
    )
    search = SearchForm(f"https://{site}/search", (), "q")
    return Page(url, site, text, links, "http", search)


_SIMULATED_PAGE = Page(
    "https://www.google.com/",
    "Google",
    "Google\nSuchfeld\n[Google Suche] [Auf gut Glück!]\nGoogle gibt es auch auf: English\n"
    "Werbeprogramme · Unternehmensangebote · Über Google · Google.de\n"
    "© 2026 · Datenschutzerklärung · Nutzungsbedingungen\n" + "Suche im Web. " * 25,
    (
        Link("Gmail", "https://mail.google.com/"),
        Link("Bilder", "https://www.google.com/imghp"),
        Link("Anmelden", "https://accounts.google.com/"),
        Link("Über Google", "https://about.google/"),
        Link("Datenschutzerklärung", "https://policies.google.com/privacy"),
        Link("Nutzungsbedingungen", "https://policies.google.com/terms"),
    ),
    "http",
)


def _tool_specs() -> list[ToolSpec]:
    """The specs of all tools, exactly as the model sees them in real use."""
    from flay.tools.claude_code import ClaudeStatusTool, SendToClaudeTool
    from flay.tools.files import ReadFileTool
    from flay.tools.forms import FillWebFormTool
    from flay.tools.memory import ForgetTool, RememberTool
    from flay.tools.phpstorm import CloseProjectTool, OpenProjectTool
    from flay.tools.pointer import ClickTool, ScrollTool
    from flay.tools.screen import LookAtScreenTool
    from flay.tools.web import OpenUrlTool, ReadWebpageTool
    from flay.tools.workflows import WorkflowTool

    return [
        OpenProjectTool().spec,
        CloseProjectTool(None).spec,  # type: ignore[arg-type]
        SendToClaudeTool(tmux=None).spec,
        ClaudeStatusTool().spec,
        LookAtScreenTool().spec,
        ClickTool(None).spec,
        ScrollTool().spec,
        OpenUrlTool().spec,
        FillWebFormTool().spec,
        ReadFileTool().spec,
        WorkflowTool(Path("workflows.json")).spec,
        ReadWebpageTool().spec,
        RememberTool(None).spec,  # type: ignore[arg-type]
        ForgetTool(None).spec,  # type: ignore[arg-type]
    ]


class _Silent:
    """Source, speech recognition and audio output that are never used."""

    async def utterances(self) -> AsyncIterator[AudioBuffer]:
        return
        yield

    async def transcribe(
        self, audio: AudioBuffer, language_hint: Language | None = None
    ) -> Transcript:
        return Transcript("", Language.DE)

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
        return
        yield

    async def play(self, chunks: AsyncIterator[AudioBuffer]) -> bool:
        return True

    def stop(self) -> None:
        pass


def _state_line(state: dict[str, Any]) -> str:
    parts = []
    if projects := state.get("open_projects"):
        parts.append(f"Open in PhpStorm: {', '.join(projects)}.")
    if sessions := state.get("claude_sessions"):
        described = [
            f"{s['project']} ({'shown in PhpStorm' if s.get('shown') else 'in the background'})"
            for s in sessions
        ]
        parts.append(f"Claude Code running for: {', '.join(described)}.")
    return " ".join(parts)


def _judge_answer(expect: dict[str, Any], reply: str) -> str:
    answers = expect.get("answer")
    # Whole words only: "C" must not be found in "ich", "5" not in "15".
    words = [rf"(?<!\w){re.escape(answer)}(?!\w)" for answer in answers or []]
    if words and not any(re.search(word, reply, re.IGNORECASE) for word in words):
        return f"Antwort ohne {answers}: „{reply[:120]}“"
    return ""


def _judge(
    case: dict[str, Any], calls: list[dict[str, Any]], projects: list[Project], reply: str = ""
) -> str:
    """Empty if the calls are what the case expects, otherwise what is wrong."""
    expect = case["expect"]
    allowed = expect["tool"] if isinstance(expect["tool"], list) else [expect["tool"]]
    first = calls[0] if calls else None
    tool = first["tool"] if first else "none"
    if tool not in allowed:
        return f"Werkzeug {tool} statt {'/'.join(allowed)}"
    if first is None:
        return _judge_answer(expect, reply)
    wanted = expect.get("project")
    argument = PROJECT_ARGUMENT.get(tool)
    if wanted and argument:
        named = str(first.get(argument) or "")
        found = {p.name for p in match_projects(named, projects)} if named else set()
        if not found or not found <= set(wanted):
            return f"Projekt „{named}“ passt nicht zu {wanted}"
    if words := expect.get("contains"):
        text = str(first.get(TEXT_ARGUMENT.get(tool, ""), "")).lower()
        if not any(word.lower() in text for word in words):
            return f"Text „{text}“ enthält keins von {words}"
    if (then := expect.get("then")) and not any(call["tool"] == then for call in calls[1:]):
        return f"danach kein {then}"
    if wrong := [c["tool"] for c in calls if c["tool"] in expect.get("never", ())]:
        return f"außerdem {wrong[0]}"
    return ""


async def run_case(
    settings: Settings,
    llm: ChatModel,
    case: dict[str, Any],
    state: dict[str, Any],
    memory: Memory | None = None,
) -> CaseResult:
    calls: list[dict[str, Any]] = []
    tools = ToolRegistry(
        {"read_webpage": _SimulatedReader, "read_file": _SimulatedFile}.get(
            spec.name, _SimulatedTool
        )(spec, calls)
        for spec in _tool_specs()
    )
    memory = memory or ShortTermMemory()
    for turn in case.get("history", []):
        language = detect_text_language(turn["user"], Language.DE)
        # Actions Flay carried out in that turn, kept with the reply as in a real conversation.
        actions = tuple(
            ToolOutcome(
                ToolCall(f"h{index}", action["tool"], dict(action.get("arguments") or {})),
                str(action.get("result", turn["assistant"])),
            )
            for index, action in enumerate(turn.get("actions", []))
        )
        memory.record(
            Message.from_text(Role.USER, turn["user"], language),
            replace(
                Message.from_text(Role.ASSISTANT, turn["assistant"], language), actions=actions
            ),
        )
    recent = state.get("recent_projects", [])
    projects = [Project(name, Path("/") / name) for name in recent]
    line = _state_line(state)

    async def current_state() -> str | None:
        return line or None

    silent = _Silent()
    events = _TimedEvents()
    assistant = Assistant(
        settings,
        silent,
        silent,
        llm,
        silent,
        silent,
        memory=memory,
        tools=tools,
        events=events,
        state=current_state,
        projects=lambda: recent,
    )
    assistant.set_muted(speaker=True)
    # The same correction as for spoken input.
    text = Vocabulary(lambda: recent).correct(case["say"])
    events.start = time.perf_counter()
    await assistant.handle_text(text)
    total = time.perf_counter() - events.start
    confirmed = assistant.pending_confirmation is not None
    if confirmed:
        # Flay asked "Meinst du …?" – answer yes and judge what it does then.
        await assistant.handle_text("Ja.")
    reply = assistant.history[-1].text if assistant.history else ""
    problem = _judge(case, calls, projects, reply)
    expected = case["expect"]["tool"]
    return CaseResult(
        say=case["say"],
        expected=expected if isinstance(expected, str) else "/".join(expected),
        tools=calls,
        reply=reply,
        ok=not problem,
        problem=problem,
        first_text=events.first_text,
        first_tool=events.first_tool,
        total=total,
        confirmed=confirmed,
    )


async def seed_examples(settings: Settings, llm: ChatModel, path: Path, database: Path) -> None:
    """A memory database holding the examples of ``path``, embedded once."""
    from flay.memory.embeddings import create_embedder

    store = MemoryStore(database)
    memory = PersistentMemory(store, llm, settings.memory, embedder=create_embedder(settings))
    for example in yaml.safe_load(path.read_text("utf-8"))["examples"]:
        call = ToolCall("seed", example["tool"], dict(example.get("arguments") or {}))
        memory.learn(example["say"], call)
    await memory.warm_up()  # embeds them
    await memory.close()


def memory_from(settings: Settings, llm: ChatModel, template: Path, copy: Path) -> Memory:
    """A fresh copy of the seeded memory for one case (cases must not learn from each other)."""
    from flay.memory.embeddings import create_embedder

    shutil.copyfile(template, copy)
    quiet = settings.memory.model_copy(
        update={"summarize": False, "extract_facts": False, "idle_seconds": 3600}
    )
    return PersistentMemory(MemoryStore(copy), llm, quiet, embedder=create_embedder(settings))


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = yaml.safe_load(path.read_text("utf-8"))
    return data.get("state", {}), data["cases"]


async def evaluate(
    settings: Settings,
    llm: ChatModel,
    cases_file: Path,
    repeat: int,
    label: str,
    only: Sequence[int] = (),
    progress: Any = print,
    examples: Path | None = None,
) -> Report:
    state, cases = load_cases(cases_file)
    report = Report(label, settings.llm.model, settings.llm.base_url)
    workdir = Path(tempfile.mkdtemp(prefix="flay-eval-"))
    template = workdir / "examples.sqlite3"
    if examples is not None:
        await seed_examples(settings, llm, examples, template)
    # The first request loads the model; it would distort the first case's timing.
    await run_case(settings, llm, {"say": "Hallo", "expect": {"tool": "none"}}, state)
    for round_ in range(repeat):
        for index, case in enumerate(cases, start=1):
            if only and index not in only:
                continue
            memory = (
                memory_from(settings, llm, template, workdir / "case.sqlite3")
                if examples is not None
                else None
            )
            try:
                result = await run_case(settings, llm, case, state, memory)
            except Exception as exc:  # a crash is a failed case, not the end of the run
                result = CaseResult(
                    say=case["say"],
                    expected=str(case["expect"]["tool"]),
                    tools=[],
                    reply="",
                    ok=False,
                    problem=f"Fehler: {describe(exc)[:200]}",
                    first_text=None,
                    first_tool=None,
                    total=0.0,
                )
            finally:
                if memory is not None:
                    await memory.close()
            report.results.append(result)
            mark = "✓" if result.ok else "✗"
            timing = result.first_tool if result.expected != "none" else result.first_text
            shown = f"{timing:.2f}s" if timing is not None else "  –  "
            progress(
                f"{mark} [{round_ + 1}/{repeat} #{index:02}] {shown}  {case['say'][:60]}"
                + (f"\n      → {result.problem}" if result.problem else "")
            )
    return report


def save(report: Report, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}-{report.label}.json"
    payload = {"summary": report.summary(), "results": [asdict(r) for r in report.results]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    return path
