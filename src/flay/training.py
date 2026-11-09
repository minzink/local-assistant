"""`flay train-export`: training data for a LoRA adapter from the learned examples.

Each row is what Flay sends the model for a request – its system prompt with the tools,
the request with the usual hint – and the answer it should give: the tool call, written
the way Qwen3-VL writes tool calls itself (``<tool_call>{…}</tool_call>``). A few plain
questions with a short answer keep the adapter from turning everything into an action.
Requests whose action the user rejected ("nein, falsch") teach asking back instead – unless
the same request was later learned with the right action, which then is the answer.

The format is the chat format of mlx-vlm's SFT training (``{"messages": [...]}``); the
tools are written into the system prompt as Qwen's chat template does, since the
trainer is not told about tools separately.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from flay.config import Settings
from flay.core.language import Language
from flay.core.messages import Message, Role
from flay.core.text_language import detect_text_language
from flay.llm.base import ToolSpec
from flay.llm.openai_compatible import to_api_tool
from flay.memory.base import EXAMPLE, MISTAKE, MemoryContext
from flay.memory.store import MemoryStore
from flay.prompt import build_messages

# Qwen's chat template puts the tools into the system prompt with this wording.
QWEN_TOOLS = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within \
<tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

# Plain questions: the right answer is words, not an action.
CONVERSATION = [
    ("Kannst du mich hören?", "Ja, ich höre dich gut."),
    ("Danke, das war's.", "Gern geschehen!"),
    ("Wie geht es dir?", "Mir geht es gut, danke. Was kann ich für dich tun?"),
    ("Was ist der Unterschied zwischen einem Interface und einer abstrakten Klasse?",
     "Ein Interface legt nur fest, welche Methoden es gibt; eine abstrakte Klasse kann "
     "schon Code mitbringen."),
    ("Ich habe das Projekt gestern lange debuggt.", "Klingt anstrengend – hat es sich gelohnt?"),
    ("Can you hear me?", "Yes, I can hear you."),
]  # fmt: skip


@dataclass(frozen=True)
class Example:
    request: str
    tool: str
    arguments: dict[str, Any]


# Answer to a request whose guessed action was wrong: ask what was meant.
ASK_BACK = {
    Language.DE: "Was meinst du mit „{request}“?",
    Language.EN: 'What do you mean by "{request}"?',
}


@dataclass(frozen=True)
class Counts:
    examples: int
    confirmed: int
    mistakes: int


def counts(store: MemoryStore) -> Counts:
    """What has been collected for training so far."""
    examples = store.facts(kind=EXAMPLE)
    confirmed = sum(bool(f.data and json.loads(f.data).get("confirmed")) for f in examples)
    return Counts(len(examples), confirmed, len(store.facts(kind=MISTAKE)))


def from_memory(store: MemoryStore, kind: str = EXAMPLE) -> list[Example]:
    """Learned examples – or, with ``kind=MISTAKE``, the actions the user rejected."""
    examples = []
    for fact in store.facts(kind=kind):
        if fact.data:
            data = json.loads(fact.data)
            examples.append(Example(fact.text, data["tool"], dict(data.get("arguments") or {})))
    return examples


def from_file(path: Path) -> list[Example]:
    items = yaml.safe_load(path.read_text("utf-8"))["examples"]
    return [Example(i["say"], i["tool"], dict(i.get("arguments") or {})) for i in items]


def _compact(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def ask_back(request: str) -> str:
    language = detect_text_language(request, Language.DE)
    return ASK_BACK.get(language, ASK_BACK[Language.DE]).format(request=request.strip(" .!?"))


def tool_call_text(tool: str, arguments: dict[str, Any]) -> str:
    call = json.dumps({"name": tool, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>\n{call}\n</tool_call>"


def row(
    settings: Settings,
    specs: Sequence[ToolSpec],
    request: str,
    answer: str,
    state: str | None,
    now: datetime,
) -> dict[str, Any]:
    language = detect_text_language(request, Language.DE)
    messages = build_messages(
        settings.persona,
        MemoryContext(),
        Message.from_text(Role.USER, request, language),
        language,
        now,
        tools=True,
        state=state,
    )
    tools = "\n".join(json.dumps(to_api_tool(spec), ensure_ascii=False) for spec in specs)
    system = messages[0].text + QWEN_TOOLS.format(tools=tools)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": messages[-1].text},
            {"role": "assistant", "content": answer},
        ]
    }


def export(
    settings: Settings,
    specs: Sequence[ToolSpec],
    examples: Sequence[Example],
    out: Path,
    state: str | None = None,
    mistakes: Sequence[Example] = (),
    extra: Sequence[dict[str, Any]] = (),
    validation_share: float = 0.2,
    seed: int = 7,
) -> tuple[int, int]:
    """Write ``train.jsonl`` and ``valid.jsonl`` into ``out``; returns both row counts."""
    now = datetime.now().astimezone()
    rows = [
        row(settings, specs, e.request, tool_call_text(e.tool, e.arguments), state, now)
        for e in examples
    ]
    learned = {_compact(e.request) for e in examples}
    rows += [
        row(settings, specs, m.request, ask_back(m.request), state, now)
        for m in mistakes
        if _compact(m.request) not in learned
    ]
    rows += [row(settings, specs, q, a, state, now) for q, a in CONVERSATION]
    rows += list(extra)
    random.Random(seed).shuffle(rows)
    cut = max(1, round(len(rows) * validation_share))
    valid, train = rows[:cut], rows[cut:]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", train), ("valid.jsonl", valid)):
        lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        (out / name).write_text(lines, "utf-8")
    return len(train), len(valid)
