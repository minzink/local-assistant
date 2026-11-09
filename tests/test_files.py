import os
import time
import zipfile
from pathlib import Path

import pytest

from flay.assistant import Assistant, file_intent
from flay.config import Settings
from flay.tools.base import ToolRegistry, ToolResult, speech
from flay.tools.files import ReadFileTool, find_file, read_text, readable
from flay.tools.phpstorm import Project
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_tools import PickingLLM
from tests.test_web import PageTool, ScriptedLLM


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home with a project, a desktop and downloads."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    project = tmp_path / "code" / "flay"
    (project / "docs").mkdir(parents=True)
    (project / "node_modules" / "x").mkdir(parents=True)
    (project / "README.md").write_text("# Flay\nInstall with uv sync.\n")
    (project / "docs" / "roadmap.md").write_text("# Roadmap\n## Phase 12\nWeb pages.\n")
    (project / ".ai-instruct").write_text("Rules.")
    (project / "src.py").write_text("print('training')")
    (project / "node_modules" / "x" / "README.md").write_text("not this one")
    (tmp_path / "Desktop").mkdir()
    (tmp_path / "Desktop" / "Rechnung März.txt").write_text("Betrag: 120 Euro")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    for index, name in enumerate(("alt.pdf", "neu.pdf")):
        (downloads / name).write_bytes(b"%PDF")
        stamp = time.time() - 100 + index * 50
        os.utime(downloads / name, (stamp, stamp))
    return tmp_path


def projects(home: Path) -> list[Project]:
    return [Project("flay", home / "code" / "flay"), Project("other", home / "code" / "other")]


def test_files_are_found_by_name_and_place(home: Path) -> None:
    known = projects(home)

    assert find_file("README", "flay", known)[0] == home / "code" / "flay" / "README.md"
    assert find_file("ai instruct", "flay", known)[0] == home / "code" / "flay" / ".ai-instruct"
    assert find_file("roadmap", "", known)[0] == home / "code" / "flay" / "docs" / "roadmap.md"
    assert find_file("Rechnung", "Schreibtisch", known)[0] == (
        home / "Desktop" / "Rechnung März.txt"
    )
    # "die neueste PDF" in the downloads: the newest, whatever its name.
    assert find_file("neueste pdf", "Downloads", known)[0] == home / "Downloads" / "neu.pdf"
    # A project named among the words is the place; words match at the start of name parts.
    assert find_file("flay readme", "", known)[0] == home / "code" / "flay" / "README.md"
    assert find_file("ai", "flay", known)[0] != home / "code" / "flay" / "src.py"
    assert find_file("gibt es nicht", "flay", known)[0] is None


def test_documents_are_read_as_text(tmp_path: Path) -> None:
    docx = tmp_path / "Brief.docx"
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr(
            "word/document.xml",
            "<w:document><w:p><w:r><w:t>Hallo</w:t></w:r></w:p>"
            "<w:p><w:r><w:t xml:space='preserve'>Welt </w:t></w:r><w:r><w:t>!</w:t></w:r>"
            "</w:p></w:document>",
        )
    latin = tmp_path / "alt.txt"
    latin.write_bytes("Größe".encode("cp1252"))

    assert read_text(docx) == "Hallo\nWelt !"
    assert read_text(latin) == "Größe"
    assert readable(Path("README")) and readable(Path(".ai-instruct"))
    assert not readable(Path("photo.jpg"))


async def test_the_tool_reads_the_file(home: Path) -> None:
    tool = ReadFileTool(lambda: projects(home))

    result = await tool.run({"name": "README", "where": "flay"})

    assert result.ok and tool.page is not None and "uv sync" in tool.page.text
    missing = await ReadFileTool(lambda: projects(home)).run({"name": "Steuer", "where": "flay"})
    assert not missing.ok and missing.say is not None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Lies die README von flay und sag mir, wie man es installiert", True),
        ("Was steht im PDF auf dem Schreibtisch?", True),
        ("Fasse die neueste Datei in Downloads zusammen und gib es Claude", True),
        ("Öffne die Datei assistant.py", False),
        ("Lies heise punkt de", False),
        ("Wie lese ich eine Datei in Python?", False),
        ("Fasse die Roadmap im Projekt flay zusammen", True),
        ("Öffne das Projekt flay", False),
        ("Sagt Claude im Projekt Flay, er soll die Readme zusammenfassen", False),
        ("Tell Claude in flay to update the changelog", False),
        ("Schick das Gleiche auch an Claude im Projekt umx", False),
    ],
)
def test_file_intent(text: str, expected: bool) -> None:
    assert file_intent(text) is expected


async def test_a_file_is_read_and_handed_to_claude(home: Path) -> None:
    reader = ReadFileTool(lambda: projects(home))
    claude = PageTool("send_to_claude", ToolResult("Typed.", say=speech("In Claude.", "-")))

    llm = ScriptedLLM("Die Roadmap beschreibt Phase 12: Webseiten.")
    router = PickingLLM("read_file", {"name": "Roadmap", "where": "flay"})

    class Both:
        capabilities = llm.capabilities

        def stream(self, messages, tools=()):  # type: ignore[no-untyped-def]
            names = [spec.name for spec in tools]
            return (
                router.stream(messages, tools)
                if names == ["read_file"]
                else llm.stream(messages, tools)
            )

    tts = FakeTTS()
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        Both(),  # type: ignore[arg-type]
        tts,
        FakeOutput(),
        tools=ToolRegistry([reader, claude]),
    )

    await assistant.handle_text("Fasse die Roadmap im Projekt flay zusammen und gib es Claude")

    roadmap = home / "code" / "flay" / "docs" / "roadmap.md"
    assert claude.calls == [
        {"instruction": f"Die Roadmap beschreibt Phase 12: Webseiten. (Quelle: {roadmap})"}
    ]
    assert [text for text, _ in tts.spoken][:2] == ["Ich suche die Datei.", "Ich lese roadmap.md."]
