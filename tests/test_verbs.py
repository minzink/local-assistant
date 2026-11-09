"""Verbs as speech recognition hears them ("She" for "Schließe") and spelled project names."""

from pathlib import Path

import pytest

from flay.config import MemorySettings
from flay.core.messages import ToolCall
from flay.memory.base import VERB
from flay.memory.persistent import PersistentMemory
from flay.memory.store import MemoryStore
from flay.tools.base import ToolRegistry
from flay.tools.phpstorm import Project, match_projects
from flay.verbs import candidates, learned_verb
from tests.test_examples import EagerLLM
from tests.test_recall import Clock, ScriptedLLM, TopicEmbedder
from tests.test_tools import SayingCloseTool

NAMES = [
    "flay", "project-symfony-kx-ewo", "component-symfony-kx-command",
    "project-symfony-kx-nvs", "project-symfony-kx-kpfb", "project-symfony-kx-kpf",
    "project-irm-kpf", "project-symfony-kx-forms", "lib-react-kx-forms-app",
]  # fmt: skip
PROJECTS = [Project(name, Path("/") / name) for name in NAMES]


@pytest.fixture
def memory(tmp_path: Path) -> PersistentMemory:
    return PersistentMemory(
        MemoryStore(tmp_path / "m.sqlite3"), ScriptedLLM(), MemorySettings(), Clock(),
        TopicEmbedder(),
    )  # fmt: skip


def best(heard: str) -> list[str]:
    return [p.name for p in match_projects(heard, PROJECTS)]


def test_spelled_names_find_the_project_that_fits_best() -> None:
    assert best("K K P F B") == ["project-symfony-kx-kpfb"]  # not kx-kpf
    assert best("K K P F") == ["project-symfony-kx-kpf"]  # the X swallowed; not irm-kpf
    assert best("K NVS") == ["project-symfony-kx-nvs"]
    # Really ambiguous stays ambiguous: Flay asks.
    assert len(best("KPF")) == 3
    assert len(best("KX Forms")) == 2


def test_the_garbled_first_word_of_a_short_request_is_learned_as_verb() -> None:
    assert learned_verb("She K Command.", "close_project", "component-symfony-kx-command") == (
        "she"
    )
    assert learned_verb("Slie K X E V O", "close_project", "project-symfony-kx-ewo") == "slie"
    # Known verbs, common words, the name itself, long sentences: nothing to learn.
    assert learned_verb("Schließe KX-Command.", "close_project", "kx-command") is None
    assert learned_verb("Mach mal Moewe.", "open_project", "moewe") is None  # see below
    assert learned_verb("Moewe bitte öffnen.", "open_project", "moewe") is None
    assert learned_verb("She K NVS", "send_to_claude", "kx-nvs") is None
    long = "Gestern habe ich lange überlegt, ob ich KX Command zumache."
    assert learned_verb(long, "close_project", "component-symfony-kx-command") is None


def test_split_verbs_are_learned_with_their_particle() -> None:
    assert candidates("Mach K NVS zu") == [("mach … zu", "k nvs"), ("mach", "k nvs zu")]
    assert learned_verb("Mach mal Moewe auf.", "open_project", "moewe") == "mach … auf"
    kx_command = "component-symfony-kx-command"
    assert learned_verb("Mach KX Command zu.", "close_project", kx_command) == "mach … zu"
    kx_forms = "project-symfony-kx-forms"
    assert learned_verb("Für KX Forms aus.", "open_project", kx_forms) == "für … aus"
    # No particle at the end: the first word alone.
    assert learned_verb("Für das Projekt Simphony Kx Forms.", "open_project", kx_forms) == "für"
    # A function word at the start is no verb, particle or not.
    assert learned_verb("Und jetzt KX Command zu.", "close_project", kx_command) is None


def assistant(memory: PersistentMemory, llm: EagerLLM, tool: SayingCloseTool):  # type: ignore[no-untyped-def]
    from flay.assistant import Assistant
    from flay.config import Settings
    from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS

    return Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(),  # type: ignore[arg-type]
        memory=memory, tools=ToolRegistry([tool]), projects=lambda: NAMES,
    )  # fmt: skip


async def test_a_learned_verb_works_for_every_project(memory: PersistentMemory) -> None:
    memory.learn("She K Command.", ToolCall("x", "close_project", {"name": "KX Command"}))
    assert memory.verb("she") == "close_project"
    tool = SayingCloseTool()
    llm = EagerLLM("close_project", {"name": "K NVS"})
    flay = assistant(memory, llm, tool)

    await flay.handle_text("She K NVS")

    # Recognized as a close request without the model guessing; routed written out.
    assert llm.prompts[0][1][0].name == "close_project"
    assert "Schließe k nvs" in llm.prompts[0][0][-1].text
    assert tool.calls or flay.history[-1].text.startswith("Meinst du")
    # Not a verb without a project after it.
    assert flay._learned_verb("She said hello") is None
    await memory.close()


async def test_a_contradicted_verb_is_forgotten(memory: PersistentMemory) -> None:
    call = ToolCall("x", "close_project", {"name": "component-symfony-kx-command"})
    memory.learn("She K Command.", call)

    memory.unlearn("She K Command.", call)

    assert memory.verb("she") is None
    assert memory._store.facts(kind=VERB) == []
    await memory.close()


async def test_mach_zu_and_mach_auf_are_different_verbs(memory: PersistentMemory) -> None:
    memory.learn("Mach KX Command zu.", ToolCall("x", "close_project", {"name": "KX Command"}))
    flay = assistant(memory, EagerLLM("close_project", {"name": "K NVS"}), SayingCloseTool())

    assert flay._learned_verb("Mach K NVS zu") == ("close_project", "Schließe k nvs")
    # "mach … auf" was never learned, and "mach" alone is no verb.
    assert flay._learned_verb("Mach K NVS auf") is None
    assert memory.verb("mach") is None
    await memory.close()


class SelectiveLLM(EagerLLM):
    """Understands only requests containing ``understood``; asks back otherwise."""

    def __init__(self, understood: str) -> None:
        super().__init__("close_project", {"name": "flay"})
        self.understood = understood

    async def stream(self, messages, tools=()):  # type: ignore[no-untyped-def,override]
        from flay.core.messages import Role
        from flay.llm.base import TextDelta

        self.prompts.append((list(messages), list(tools)))
        last = messages[-1]
        if tools and last.role is Role.USER and self.understood in last.text:
            yield self.call
        else:
            yield TextDelta("Was meinst du?")


async def test_a_request_not_understood_is_learned_with_the_retry(
    memory: PersistentMemory,
) -> None:
    tool = SayingCloseTool()
    flay = assistant(memory, SelectiveLLM("Schließe"), tool)  # type: ignore[arg-type]

    await flay.handle_text("Sli flay")  # not understood: Flay asks back
    await flay.handle_text("Schließe flay")  # said again, clearly: done
    await flay.handle_text("Danke.")

    assert tool.calls == [{"name": "flay"}]
    learned = sorted(f.text for f in memory._store.facts(kind="example"))
    assert learned == ["Schließe flay", "Sli flay"]
    assert memory.verb("sli") == "close_project"
    await memory.close()


async def test_a_contradicted_retry_teaches_nothing(memory: PersistentMemory) -> None:
    flay = assistant(memory, SelectiveLLM("Schließe"), SayingCloseTool())  # type: ignore[arg-type]

    await flay.handle_text("Sli flay")
    await flay.handle_text("Schließe flay")
    await flay.handle_text("Nein, falsch.")

    assert memory._store.facts(kind="example") == []
    assert memory.verb("sli") is None
    await memory.close()


async def test_an_unrelated_request_before_is_not_learned(memory: PersistentMemory) -> None:
    flay = assistant(memory, SelectiveLLM("Schließe"), SayingCloseTool())  # type: ignore[arg-type]

    await flay.handle_text("Wie geht es dir?")  # no project: not the same request
    await flay.handle_text("Schließe flay")
    await flay.handle_text("Danke.")

    assert [f.text for f in memory._store.facts(kind="example")] == ["Schließe flay"]
    await memory.close()
