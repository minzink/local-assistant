"""Keeps a request within the model's context, so the server never rejects it.

Ollama answers a request that is too long with an error ("request (4603 tokens) exceeds the
available context size (4096 tokens)") – Cassi then said nothing at all. Before sending,
the request is estimated and, if needed, shortened: first the oldest messages of the
conversation, then long tool results. The system prompt and the current request with its
tool rounds always stay.

Tokens are estimated without the model's tokenizer, on the safe side: German text has
about 3.5–4 characters per token with Qwen, JSON fewer; 3 per token leaves room.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Sequence
from dataclasses import replace

from cassi.core.messages import AudioPart, ImagePart, Message, Role, TextPart

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3
# Chat template around each message (role markers, separators).
MESSAGE_OVERHEAD = 8
# Qwen-VL: one token per 28×28 pixels – but Ollama scales small images up: a 768×480
# crop took 1083 tokens, as many as a whole 1280×720 screen. Hence a floor.
PIXELS_PER_TOKEN = 28 * 28
MIN_IMAGE_TOKENS = 1200
UNKNOWN_IMAGE_TOKENS = 1300
AUDIO_TOKENS = 1000
SHORTENED = " … (gekürzt)"
# A tool result is never cut below this, or it would say nothing anymore.
MIN_RESULT_CHARS = 600


def text_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def image_tokens(part: ImagePart) -> int:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(part.data)) as image:
            width, height = image.size
    except Exception:
        return UNKNOWN_IMAGE_TOKENS
    return max(MIN_IMAGE_TOKENS, width * height // PIXELS_PER_TOKEN + 4)


def message_tokens(message: Message) -> int:
    tokens = MESSAGE_OVERHEAD
    for part in message.parts:
        if isinstance(part, TextPart):
            tokens += text_tokens(part.text)
        elif isinstance(part, ImagePart):
            tokens += image_tokens(part)
        elif isinstance(part, AudioPart):
            tokens += AUDIO_TOKENS
    for call in message.tool_calls:
        tokens += text_tokens(call.name + json.dumps(call.arguments, ensure_ascii=False))
    return tokens


def tools_tokens(tools: Sequence[dict[str, object]]) -> int:
    return text_tokens(json.dumps(list(tools), ensure_ascii=False)) if tools else 0


# An earlier message keeps this much of its text when the request has to be shortened.
CONDENSED_CHARS = 240


def condense(message: Message) -> Message:
    """An earlier message in short: its start, pictures replaced by a note."""
    parts: list[TextPart] = []
    changed = False
    for part in message.parts:
        if isinstance(part, TextPart):
            if len(part.text) > CONDENSED_CHARS:
                parts.append(TextPart(part.text[:CONDENSED_CHARS] + SHORTENED))
                changed = True
            else:
                parts.append(part)
        else:
            parts.append(TextPart("[Bild]" if isinstance(part, ImagePart) else "[Audio]"))
            changed = True
    return replace(message, parts=tuple(parts)) if changed else message


def fit(messages: Sequence[Message], budget: int, fixed: int = 0) -> list[Message]:
    """``messages`` shortened so they and ``fixed`` tokens (tool specs) fit ``budget``."""
    kept = list(messages)
    sizes = [message_tokens(m) for m in kept]
    if fixed + sum(sizes) <= budget:
        return kept
    before = fixed + sum(sizes)
    # Everything from the last request on belongs to the current turn.
    current = max((i for i, m in enumerate(kept) if m.role is Role.USER), default=len(kept))
    first = 1 if kept and kept[0].role is Role.SYSTEM else 0
    # First the earlier messages are condensed, oldest first: their gist stays (the user
    # may refer back to it), long texts and pictures go.
    condensed = 0
    for i in range(first, current):
        if fixed + sum(sizes) <= budget:
            break
        short = condense(kept[i])
        if short is not kept[i]:
            kept[i], sizes[i] = short, message_tokens(short)
            condensed += 1
    dropped = 0
    while fixed + sum(sizes) > budget and first < current:
        del kept[first], sizes[first]
        current -= 1
        dropped += 1
        # A tool result without the call it answers is rejected by chat templates.
        while first < current and kept[first].role is Role.TOOL:
            del kept[first], sizes[first]
            current -= 1
            dropped += 1
    shortened = 0
    while fixed + sum(sizes) > budget:
        # Then the longest tool result of the current turn (a web page, a file).
        results = [i for i in range(current, len(kept)) if kept[i].role is Role.TOOL]
        longest = max(results, key=lambda i: len(kept[i].text), default=None)
        if longest is None or len(kept[longest].text) <= MIN_RESULT_CHARS:
            break
        text = kept[longest].text
        excess = (fixed + sum(sizes) - budget) * CHARS_PER_TOKEN + len(SHORTENED)
        keep = max(MIN_RESULT_CHARS, len(text) - excess)
        kept[longest] = replace(kept[longest], parts=(TextPart(text[:keep] + SHORTENED),))
        sizes[longest] = message_tokens(kept[longest])
        shortened += 1
    after = fixed + sum(sizes)
    log.info(
        "Anfrage gekürzt: ~%d → ~%d Token (Budget %d; %d ältere Nachrichten verdichtet, %d "
        "weggelassen, %d Ergebnisse gekürzt)",
        before,
        after,
        budget,
        condensed,
        dropped,
        shortened,
    )
    if after > budget:
        log.warning("Anfrage passt auch gekürzt nicht in den Kontext (~%d Token)", after)
    return kept
