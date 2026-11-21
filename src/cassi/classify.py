"""Which computer action a request asks for, decided by the model.

The fixed patterns (cassi.intents) recognize the usual phrasings in no time; "Wirf Spotify
auf den zweiten Bildschirm" or "Stell den Dunkelmodus an" slip through them. The computer
tools are not offered in free conversation (the model took stories for commands), so such
requests are classified first: one short JSON answer (~0.4 s with qwen3-vl:8b on an M2
Pro), only for requests that mention the computer or start like an order.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence

from cassi.core.messages import Message, Role
from cassi.llm.base import StructuredChat

log = logging.getLogger(__name__)

# The actions the model may pick, described for it.
ACTIONS = {
    "control_window": "open, switch to, move, arrange, minimize or close an app's window",
    "use_keyboard": (
        "type dictated text, press a shortcut or keys, or make a virtual key (a hotkey to "
        "set up in a program so the assistant can control it)"
    ),
    "system_control": "volume, mute, music playback, lock the screen",
    "timer": "a timer or reminder at a time or after a duration ('in 20 Minuten')",
    "find_file": "find or open a document in the user's folders",
    "calendar": "the user's appointments and calendar",
    "adjust_settings": (
        "how the assistant itself speaks (speed, voice, answer length) or whether it may "
        "look at the screen – never the computer's or an app's settings"
    ),
    "operate_computer": (
        "any other task done in an app's interface: clicking through menus, changing the "
        "computer's or an app's settings (dark mode, Wi-Fi …), filling in fields, switching "
        "tabs, or watching the screen to tell the user when something is done or shows up"
    ),
}
NONE = "none"
TIMEOUT_SECONDS = 4.0
_SYSTEM = (
    "Decide which computer action a spoken request asks the assistant to do now. Answer "
    "with JSON. Questions about knowledge, stories about the past and conversation are "
    "'none'.\nActions:\n"
)


async def classify(
    llm: StructuredChat,
    text: str,
    available: Sequence[str],
    extra: Mapping[str, str] | None = None,
) -> str | None:
    """The action ``text`` asks for among ``available``, or None. ``extra``: more about an
    action as things are now (the virtual keys and what they control)."""
    actions = [name for name in ACTIONS if name in available]
    if not actions:
        return None
    more = extra or {}
    described = "\n".join(
        f"- {name}: {ACTIONS[name]}" + (f"; {more[name]}" if more.get(name) else "")
        for name in actions
    )
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": [*actions, NONE]}},
        "required": ["action"],
        "additionalProperties": False,
    }
    messages = [
        Message.from_text(Role.SYSTEM, _SYSTEM + described + f"\n- {NONE}: anything else"),
        Message.from_text(Role.USER, text),
    ]
    try:
        answer = await asyncio.wait_for(
            llm.complete_json(messages, schema, "action"), TIMEOUT_SECONDS
        )
    except (TimeoutError, Exception) as exc:
        log.info("Einordnung fehlgeschlagen: %s", exc)
        return None
    action = str(answer.get("action") or NONE)
    if action == "adjust_settings" and "operate_computer" in actions:
        from cassi.intents import about_flay_settings

        if not about_flay_settings(text):
            # "Stell in den Systemeinstellungen den Dunkelmodus an": the computer's settings.
            action = "operate_computer"
    log.info("Einordnung: „%s“ → %s", text, action)
    return action if action in actions else None
