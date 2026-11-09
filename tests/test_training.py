import json
from pathlib import Path

from flay.config import Settings
from flay.evaluation import _tool_specs
from flay.training import Example, export, tool_call_text


def test_tool_call_is_written_like_qwen_writes_it() -> None:
    assert tool_call_text("close_project", {"name": "flay"}) == (
        '<tool_call>\n{"name": "close_project", "arguments": {"name": "flay"}}\n</tool_call>'
    )


def test_export_writes_chat_rows_with_tools_and_a_held_out_part(tmp_path: Path) -> None:
    examples = [
        Example("She K Command.", "close_project", {"name": "component-symfony-kx-command"}),
        Example("Mach mal Moewe auf.", "open_project", {"name": "moewe"}),
    ]

    train, valid = export(Settings(), _tool_specs(), examples, tmp_path, "Open in PhpStorm: flay.")

    rows = [
        json.loads(line)
        for name in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / name).read_text().splitlines()
    ]
    assert train + valid == len(rows) == 2 + 6  # examples + plain questions
    assert valid >= 1
    by_request = {r["messages"][1]["content"].split("\n")[0]: r for r in rows}
    she = by_request["She K Command."]["messages"]
    assert "<tools>" in she[0]["content"] and '"name": "close_project"' in she[0]["content"]
    assert "Open in PhpStorm: flay." in she[1]["content"]
    assert she[2]["content"].startswith("<tool_call>")
    assert "<tool_call>" not in by_request["Kannst du mich hören?"]["messages"][2]["content"]


def test_rejected_actions_teach_asking_back_unless_learned_later(tmp_path: Path) -> None:
    examples = [Example("She K Command.", "close_project", {"name": "kx-command"})]
    mistakes = [
        Example("Mach das zu.", "close_project", {"name": "flay"}),
        Example("she k command", "close_project", {"name": "flay"}),  # learned right later
    ]

    export(Settings(), _tool_specs(), examples, tmp_path, None, mistakes)

    rows = [
        json.loads(line)
        for name in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / name).read_text().splitlines()
    ]
    answers = {
        r["messages"][1]["content"].split("\n")[0]: r["messages"][2]["content"] for r in rows
    }
    assert answers["Mach das zu."] == "Was meinst du mit „Mach das zu“?"
    assert "she k command" not in answers
    assert len(rows) == 1 + 1 + 6
