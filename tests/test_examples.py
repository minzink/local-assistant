"""Examples from the conversation: requests as heard and the action they meant."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from flay.assistant import Assistant
from flay.config import MemorySettings, Settings
from flay.core.messages import Message, Role, ToolCall
from flay.llm.base import LLMEvent, ToolSpec
from flay.memory.base import EXAMPLE, MISTAKE
from flay.memory.persistent import PersistentMemory
from flay.memory.store import MemoryStore
from flay.prompt import format_examples
from flay.tools.base import ToolRegistry
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_recall import Clock, ScriptedLLM, TopicEmbedder
from tests.test_tools import PickingLLM, RoutingLLM, SayingCloseTool

CLOSE = ToolCall("x", "close_project", {"name": "component-symfony-kx-command"})


@pytest.fixture
def memory(tmp_path: Path) -> PersistentMemory:
    return PersistentMemory(
        MemoryStore(tmp_path / "m.sqlite3"), ScriptedLLM(), MemorySettings(), Clock(),
        TopicEmbedder(),
    )  # fmt: skip


async def test_a_request_heard_once_is_only_a_hint(memory: PersistentMemory) -> None:
    memory.learn("She K Command.", CLOSE)

    assert memory.known_action("she k command") is None
    await asyncio.sleep(0.05)  # embedded in the background
    (example,) = await memory.examples("She K Command")
    assert example.text == "She K Command."
    # Examples are no knowledge: recall leaves them out, and vice versa.
    assert await memory.recall("She K Command") == ()
    await memory.close()


async def test_confirmed_or_repeated_request_is_known(memory: PersistentMemory) -> None:
    memory.learn("Slie K X E V O", ToolCall("x", "close_project", {"name": "kx-ewo"}), True)
    memory.learn("She K Command.", CLOSE)
    memory.learn("She K Command!", CLOSE)

    confirmed = memory.known_action("slie k x e v o")
    repeated = memory.known_action("She K Command")

    assert confirmed is not None and confirmed[1] is True
    assert repeated is not None and repeated[0].name == "close_project" and not repeated[1]
    # Meaning something else the next time starts counting anew.
    memory.learn("She K Command", ToolCall("x", "open_project", {"name": "moewe"}))
    assert memory.known_action("She K Command") is None
    await memory.close()


async def test_unlearn_removes_the_example(memory: PersistentMemory) -> None:
    memory.learn("She K Command.", CLOSE, confirmed=True)

    memory.unlearn("she k command")

    assert memory.known_action("She K Command.") is None
    await memory.close()


async def test_rejected_action_is_kept_as_mistake_for_training_only(
    memory: PersistentMemory,
) -> None:
    memory.unlearn("She K Command.", CLOSE)
    memory.unlearn("she k command", CLOSE)
    await asyncio.sleep(0.05)

    (mistake,) = memory._store.facts(kind=MISTAKE)
    assert mistake.text == "She K Command." and '"times": 2' in (mistake.data or "")
    # Never shown to the model, never acted on, not something to "forget".
    assert memory.known_action("She K Command") is None
    assert await memory.examples("She K Command") == ()
    assert await memory.recall("She K Command") == ()
    assert memory.facts() == []
    assert memory._store.unembedded("any") == []
    counts = memory.training_counts()
    assert (counts.examples, counts.mistakes) == (0, 1)
    await memory.close()


def test_examples_are_shown_with_the_action_they_meant() -> None:
    from flay.memory.base import Fact

    example = Fact(
        1, "She K Command.", kind=EXAMPLE,
        data='{"tool": "close_project", "arguments": {"name": "kx-command"}}',
    )  # fmt: skip

    assert format_examples([example]) == (
        "Earlier requests that sounded like this, and what the user meant:\n"
        "- \"She K Command.\" → close_project(name='kx-command')"
    )


def assistant(memory: PersistentMemory, llm: object, tool: SayingCloseTool) -> Assistant:
    return Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(),  # type: ignore[arg-type]
        memory=memory, tools=ToolRegistry([tool]),
        projects=lambda: ["flay", "component-symfony-kx-command", "project-symfony-kx-ewo"],
    )  # fmt: skip


class EagerLLM(PickingLLM):
    """Acts on each new request, also with earlier actions in the conversation."""

    acts_on_every_request = False

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[LLMEvent]:
        if self.acts_on_every_request and tools and messages[-1].role is Role.USER:
            self.prompts.append((list(messages), list(tools)))
            yield self.call
            return
        async for event in super().stream(messages, tools):
            yield event


async def test_action_becomes_an_example_unless_the_next_request_objects(
    memory: PersistentMemory,
) -> None:
    tool = SayingCloseTool()
    llm = EagerLLM("close_project", {"name": "component-symfony-kx-command"})
    flay = assistant(memory, llm, tool)

    await flay.handle_text("Mach KX Command zu.")
    await flay.handle_text("Danke.")  # no objection: learned
    llm.acts_on_every_request = True
    await flay.handle_text("Mach bitte KX Command zu.")
    await flay.handle_text("Nein, falsch.")  # objection: not learned, kept as mistake

    learned = [f.text for f in memory._store.facts(kind=EXAMPLE)]
    assert learned == ["Mach KX Command zu."]
    rejected = [f.text for f in memory._store.facts(kind=MISTAKE)]
    assert rejected == ["Mach bitte KX Command zu."]
    await memory.close()


async def test_confirmed_request_is_carried_out_directly_next_time(
    memory: PersistentMemory,
) -> None:
    tool = SayingCloseTool()
    llm = PickingLLM("close_project", {"name": "KEWO"})
    flay = assistant(memory, llm, tool)

    await flay.handle_text("Slie K X E V O")  # only roughly named: Flay asks first
    assert flay.history[-1].text.startswith("Meinst du")
    await flay.handle_text("Ja.")
    assert tool.calls == [{"name": "project-symfony-kx-ewo"}]

    asked = len(llm.prompts)
    await flay.handle_text("Slie K X E V O")

    # Known and confirmed: done at once, no model call, no question.
    assert len(tool.calls) == 2
    assert len(llm.prompts) == asked
    assert flay.history[-1].text == "flay ist geschlossen."
    await memory.close()


async def test_examples_reach_the_weighing_step(memory: PersistentMemory) -> None:
    memory.learn("Mach KX Command zu.", CLOSE)
    await asyncio.sleep(0.05)
    llm = RoutingLLM()
    flay = assistant(memory, llm, SayingCloseTool())

    await flay.handle_text("Schließe das Projekt KX Command.")

    router_prompt = llm.prompts[0][0][0].text
    assert '"Mach KX Command zu." → close_project' in router_prompt
    await memory.close()


def test_correction_wording() -> None:
    from flay.assistant import _CORRECTION

    assert _CORRECTION.search("Nein, das meinte ich nicht.")
    assert _CORRECTION.search("Falsch!")
    assert not _CORRECTION.search("Danke dir.")
    _ = Message, Role  # used by the helpers above


async def test_an_action_done_earlier_can_be_asked_for_again() -> None:
    from flay.memory.base import ShortTermMemory

    tool = SayingCloseTool()
    llm = EagerLLM("close_project", {"name": "flay"})
    llm.acts_on_every_request = True
    flay = assistant(ShortTermMemory(), llm, tool)  # type: ignore[arg-type]

    await flay.handle_text("Mach flay zu.")
    await flay.handle_text("Mach flay zu.")

    assert tool.calls == [{"name": "flay"}, {"name": "flay"}]


def test_stories_and_how_tos_get_no_command_examples() -> None:
    from flay.assistant import wants_examples

    assert wants_examples("She K Command.")
    assert wants_examples("Mach mal Moewe auf.")
    assert not wants_examples("Ich habe das Projekt flay gestern geöffnet, es lief gut.")
    assert not wants_examples("Ich hab KX Command vorhin geschlossen.")
    assert not wants_examples("I opened flay yesterday.")
    assert not wants_examples("Wie schließe ich in PhpStorm ein Projekt?")
    # A story with a request at its end is a request.
    assert wants_examples("Ich habe flay geöffnet, schließ es jetzt wieder.")
    assert wants_examples("Ich habe flay geöffnet, mach es wieder zu.")


async def test_a_story_does_not_act(memory: PersistentMemory) -> None:
    tool = SayingCloseTool()
    llm = EagerLLM("close_project", {"name": "component-symfony-kx-command"})
    llm.acts_on_every_request = True
    flay = assistant(memory, llm, tool)

    await flay.handle_text("Ich hab KX Command vorhin geschlossen, war eh fertig.")

    assert tool.calls == []
    assert llm.prompts and all(not tools for _, tools in llm.prompts)
    await memory.close()
