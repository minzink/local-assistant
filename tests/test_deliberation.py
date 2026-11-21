import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cassi.core.messages import Message, Role
from cassi.deliberation import Notes, deliberate, schema_with
from cassi.llm.base import Reasoning


class Chat:
    """Answers in JSON; records what it was asked."""

    def __init__(self, answer: dict[str, Any], thoughts: Reasoning | None = None) -> None:
        self.answer = answer
        self.thoughts = thoughts
        self.asked: list[str] = []
        self.schemas: list[dict[str, Any]] = []
        self.reasoned: list[float] = []

    async def complete_json(
        self,
        messages: Sequence[Message],
        schema: dict[str, Any],
        name: str = "answer",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.asked.append("\n".join(m.text for m in messages))
        self.schemas.append(schema)
        return dict(self.answer)


class ThinkingChat(Chat):
    async def reason(
        self, messages: Sequence[Message], seconds: float, model: str | None = None
    ) -> Reasoning:
        self.reasoned.append(seconds)
        assert self.thoughts is not None
        return self.thoughts


QUESTION = [Message.from_text(Role.USER, "Was nun?")]
SCHEMA = {"type": "object", "properties": {"choice": {"type": "string"}}}


def test_analysis_comes_before_the_answer_and_notes_last() -> None:
    schema = schema_with(SCHEMA, Notes(limit=3))
    assert list(schema["properties"]) == ["analysis", "choice", "notes"]
    assert schema["required"] == ["analysis", "choice", "notes"]
    assert schema["properties"]["notes"]["maxItems"] == 3
    assert list(schema_with(SCHEMA, None)["properties"]) == ["analysis", "choice"]


def test_notes_are_a_short_term_memory_across_decisions(tmp_path: Path) -> None:
    notes = Notes.load(tmp_path / "notes.json", limit=2)
    chat = Chat({"analysis": "a", "choice": "x", "notes": ["Plan: Turm", "Plan: Turm", "b", "c"]})
    decision = asyncio.run(deliberate(chat, QUESTION, SCHEMA, notes=notes))
    assert decision.answer == {"choice": "x"} and decision.analysis == "a"
    assert "noch keine" in chat.asked[0]
    assert decision.notes == ("Plan: Turm", "b")  # doubles dropped, at most the limit

    # Next decision (also after a restart): the notes come back.
    again = Notes.load(tmp_path / "notes.json", limit=2)
    asyncio.run(deliberate(chat, QUESTION, SCHEMA, notes=again))
    assert "- Plan: Turm" in chat.asked[1]
    again.clear()
    assert Notes.load(tmp_path / "notes.json").items == []


def test_thorough_thinks_first_then_decides_with_the_thoughts() -> None:
    chat = ThinkingChat(
        {"analysis": "", "choice": "y"}, Reasoning("Wenn ich x nehme, dann …", "", False)
    )
    decision = asyncio.run(deliberate(chat, QUESTION, SCHEMA, depth="gründlich", think_seconds=5))
    assert chat.reasoned == [5] and decision.answer == {"choice": "y"}
    assert "Wenn ich x nehme" in chat.asked[0] and "abgebrochen" in chat.asked[0]
    assert decision.thoughts.startswith("Wenn") and decision.finished is False


def test_thorough_without_a_thinking_model_decides_briefly() -> None:
    chat = Chat({"analysis": "kurz", "choice": "z"})
    decision = asyncio.run(deliberate(chat, QUESTION, SCHEMA, depth="gründlich"))
    assert decision.answer == {"choice": "z"} and decision.thoughts == ""


def _chunk(delta: dict[str, object]) -> Any:
    from openai.types.chat import ChatCompletionChunk

    return ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def _server(chunks: list[Any], pause: float = 0.0) -> tuple[Any, dict[str, Any]]:
    from cassi.llm.base import ModelCapabilities, ThinkingOptions
    from cassi.llm.openai_compatible import OpenAICompatibleChat

    class Stream:
        async def __aenter__(self) -> "Stream":
            return self

        async def __aexit__(self, *exc: object) -> None:
            pass

        def __aiter__(self):  # type: ignore[no-untyped-def]
            async def gen():  # type: ignore[no-untyped-def]
                for c in chunks:
                    await asyncio.sleep(pause)
                    yield c

            return gen()

    sent: dict[str, Any] = {}

    async def create(**kwargs: object) -> Stream:
        sent.update(kwargs)
        return Stream()

    chat = OpenAICompatibleChat(
        "http://x/v1", "m", ModelCapabilities(), thinking=ThinkingOptions(model="m-thinking")
    )
    chat._client.chat.completions.create = create  # type: ignore[method-assign, assignment]
    return chat, sent


def test_a_thinking_models_reasoning_is_collected() -> None:
    chat, sent = _server(
        [_chunk({"reasoning": "Erst "}), _chunk({"reasoning": "das."}), _chunk({"content": "x"})]
    )
    reasoning = asyncio.run(chat.reason(QUESTION, 5))
    assert sent["model"] == "m-thinking"
    assert (reasoning.thoughts, reasoning.answer, reasoning.finished) == ("Erst das.", "x", True)

    chat, _ = _server([_chunk({"content": "<think>hm</think>y"})])
    assert asyncio.run(chat.reason(QUESTION, 5)).thoughts == "hm"


def test_thinking_stops_when_the_time_is_up() -> None:
    chat, _ = _server([_chunk({"reasoning": "a"})] * 50, pause=0.02)
    reasoning = asyncio.run(chat.reason(QUESTION, 0.2))
    assert reasoning.finished is False and 3 <= len(reasoning.thoughts) < 50


def test_a_long_thinking_budget_leaves_room_for_the_question() -> None:
    chat, sent = _server([_chunk({"reasoning": "a"})])
    chat._thinking = type(chat._thinking)(model="m-thinking", max_tokens=8000)
    chat._context_tokens = 8192
    asyncio.run(chat.reason(QUESTION, 5))
    assert sent["max_tokens"] == 4096
