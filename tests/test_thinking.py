from collections.abc import AsyncIterator, Sequence

import pytest

from flay.assistant import Assistant, wants_thinking
from flay.config import Settings
from flay.core.messages import Message
from flay.llm.base import LLMEvent, ModelCapabilities, TextDelta, ToolSpec
from flay.llm.openai_compatible import ThinkFilter
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Denk mal genau darüber nach, wie wir das Caching bauen.", True),
        ("Überleg dir, wie man das testen kann.", True),
        ("Erklär das Schritt für Schritt.", True),
        ("Think it through: why does the build fail?", True),
        ("Öffne das Projekt flay.", False),
        ("Wie spät ist es?", False),
    ],
)
def test_wants_thinking(text: str, expected: bool) -> None:
    assert wants_thinking(text) is expected


@pytest.mark.parametrize(
    ("pieces", "shown"),
    [
        (["<think>Hm, also</think>Ja."], "Ja."),
        (["<thi", "nk>geteilt</th", "ink>Antwort"], "Antwort"),
        (["a < b"], "a < b"),
        (["Ende <"], "Ende <"),
    ],
)
def test_think_filter_hides_thinking(pieces: list[str], shown: str) -> None:
    hidden = ThinkFilter()

    assert "".join(hidden.feed(p) for p in pieces) + hidden.flush() == shown


class ThinkingLLM:
    def __init__(self) -> None:
        self.thought: list[bool] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec] = (), think: bool = False
    ) -> AsyncIterator[LLMEvent]:
        self.thought.append(think)
        yield TextDelta("Erst das Interface, dann die Tests.")


def assistant(llm: ThinkingLLM, tts: FakeTTS, enabled: bool) -> Assistant:
    settings = Settings.model_validate({"llm": {"thinking": {"enabled": enabled}}})
    return Assistant(settings, FakeSource(), FakeSTT("x"), llm, tts, FakeOutput())


async def test_thinking_is_announced_and_requested() -> None:
    llm, tts = ThinkingLLM(), FakeTTS()

    await assistant(llm, tts, enabled=True).handle_text("Denk mal nach, wie wir das bauen.")

    assert llm.thought == [True]
    assert [text for text, _ in tts.spoken] == [
        "Moment, ich überlege kurz.",
        "Erst das Interface,",
        "dann die Tests.",
    ]


async def test_no_thinking_unless_enabled_and_asked() -> None:
    llm, tts = ThinkingLLM(), FakeTTS()

    await assistant(llm, tts, enabled=False).handle_text("Denk mal nach, wie wir das bauen.")
    await assistant(llm, tts, enabled=True).handle_text("Wie bauen wir das?")

    assert llm.thought == [False, False]


async def test_normal_model_is_reloaded_after_a_thinking_answer() -> None:
    import asyncio

    class WarmingLLM(ThinkingLLM):
        warmed = 0

        async def warm(self, messages: Sequence[Message], tools: Sequence[ToolSpec] = ()) -> None:
            self.warmed += 1

    llm = WarmingLLM()
    settings = Settings.model_validate(
        {"llm": {"thinking": {"enabled": True, "model": "qwen3-vl:8b-thinking"}}}
    )
    flay = Assistant(settings, FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput())

    await flay.handle_text("Denk mal nach, wie wir das bauen.")
    await asyncio.sleep(0)

    assert llm.warmed == 1
