from __future__ import annotations

import io

from PIL import Image

from cassi.core.messages import ImagePart, Message, Role, TextPart, ToolCall
from cassi.llm import budget


def text(role: Role, chars: int, tool_call_id: str | None = None) -> Message:
    return Message(role, (TextPart("x" * chars),), tool_call_id=tool_call_id)


def total(messages: list[Message], fixed: int = 0) -> int:
    return fixed + sum(budget.message_tokens(m) for m in messages)


def test_a_request_that_fits_is_unchanged() -> None:
    messages = [text(Role.SYSTEM, 300), text(Role.USER, 30)]
    assert budget.fit(messages, 1000) == messages


def test_oldest_messages_go_first_system_and_request_stay() -> None:
    system, request = text(Role.SYSTEM, 300), text(Role.USER, 30)
    old = [text(Role.USER, 3000), text(Role.ASSISTANT, 3000)]
    recent = [text(Role.USER, 300), text(Role.ASSISTANT, 300)]
    kept = budget.fit([system, *old, *recent, request], 300)
    assert kept[0] == system and kept[-1] == request
    assert total(kept) <= 300


def test_earlier_messages_are_condensed_before_they_go() -> None:
    system, request = text(Role.SYSTEM, 300), text(Role.USER, 30)
    old = [text(Role.USER, 3000), text(Role.ASSISTANT, 3000)]
    kept = budget.fit([system, *old, request], 400)
    # Both earlier messages are still there, shortened to their start.
    assert len(kept) == 4
    assert all(m.text.endswith(budget.SHORTENED) for m in kept[1:3])


def test_condense_replaces_pictures() -> None:
    message = Message(Role.USER, (TextPart("Schau"), ImagePart(b"png", "image/png")))
    assert budget.condense(message).text == "Schau[Bild]"


def test_tool_results_go_with_their_call() -> None:
    call = Message(Role.ASSISTANT, (TextPart(""),), tool_calls=(ToolCall("c1", "read_file", {}),))
    history = [text(Role.USER, 60), call, text(Role.TOOL, 3000, "c1"), text(Role.ASSISTANT, 60)]
    request = text(Role.USER, 30)
    condensed = budget.fit([text(Role.SYSTEM, 30), *history, request], 200)
    (result,) = [m for m in condensed if m.role is Role.TOOL]
    assert result.text.endswith(budget.SHORTENED)  # shortened, not gone
    assert condensed[condensed.index(result) - 1] == call
    dropped = budget.fit([text(Role.SYSTEM, 30), *history, request], 60)
    # Not even condensed it fits: the result goes together with its call.
    assert all(m.role is not Role.TOOL for m in dropped)
    assert dropped[-1] == request


def test_long_result_of_the_current_turn_is_shortened() -> None:
    page = text(Role.TOOL, 30000, "c1")
    messages = [text(Role.SYSTEM, 300), text(Role.USER, 30), page]
    kept = budget.fit(messages, 4000, fixed=1000)
    assert total(kept, 1000) <= 4000
    assert kept[-1].text.endswith(budget.SHORTENED)
    assert kept[-1].tool_call_id == "c1"


def test_screenshot_counts_by_its_size() -> None:
    data = io.BytesIO()
    Image.new("RGB", (1280, 720)).save(data, "PNG")
    part = ImagePart(data.getvalue(), "image/png")
    assert 1150 < budget.image_tokens(part) < 1250
    assert budget.image_tokens(ImagePart(b"broken", "image/png")) == budget.UNKNOWN_IMAGE_TOKENS
    small = io.BytesIO()
    Image.new("RGB", (768, 480)).save(small, "PNG")
    # Ollama scales it up: measured 1083 tokens.
    assert budget.image_tokens(ImagePart(small.getvalue(), "image/png")) >= 1083
