import json
from pathlib import Path

import pytest

from cassi.app import App
from cassi.config import Settings
from cassi.direct import direct_arguments
from cassi.intents import computer_intent
from cassi.keyboard_tab import keyboard_tab
from cassi.manage import Option
from cassi.tools.keyboard import KeyboardTool, parse_keys
from cassi.tools.virtual_keys import (
    KeyStore,
    VirtualKey,
    check_keys,
    free_keys,
    is_global,
    named_key,
)


def make_tool(
    tmp_path: Path, direct: str = "never", front_is_own: bool = False
) -> tuple[KeyboardTool, KeyStore, list[tuple[str, int]]]:
    store = KeyStore(tmp_path / "keyboard.json")
    store.set_direct(direct)  # type: ignore[arg-type]
    pressed: list[tuple[str, int]] = []
    tool = KeyboardTool(
        front_is_own=lambda: front_is_own,
        store=store,
        press_keys=lambda keys, times: pressed.append((keys, times)),
    )
    return tool, store, pressed


def test_new_keys_get_free_combinations(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "keyboard.json")
    assert store.direct == "never" and store.keys == []

    first = store.put("OBS Aufnahme starten", "OBS: Aufnahme starten")
    second = store.put("OBS Aufnahme stoppen")
    assert isinstance(first, VirtualKey) and isinstance(second, VirtualKey)
    assert (first.keys, second.keys) == ("ctrl+alt+shift+f13", "ctrl+alt+shift+f14")
    # Every free combination can be pressed; none is plain F14/F15 (macOS brightness).
    assert all(check_keys(k) is None and "+" in k for k in free_keys())

    # Changed by name: keeps its keys unless new ones are given.
    changed = store.put("obs aufnahme starten", "OBS: Aufnahme an")
    assert isinstance(changed, VirtualKey) and changed.keys == "ctrl+alt+shift+f13"
    assert store.put("Andere", keys="ctrl+alt+shift+f14") == (
        "ctrl+alt+shift+f14 gehört schon zu einer anderen Taste."
    )
    assert store.put("", "x") == "Der Name fehlt."
    assert "Unbekannt" in str(store.put("X", keys="ctrl+hyper"))

    # Removed: its combination is free again.
    assert store.remove("OBS Aufnahme starten") and not store.remove("OBS Aufnahme starten")
    third = store.put("Neu")
    assert isinstance(third, VirtualKey) and third.keys == "ctrl+alt+shift+f13"


def test_running_assistant_sees_changes(tmp_path: Path) -> None:
    store = KeyStore(tmp_path / "keyboard.json")
    assert store.direct == "never"
    KeyStore(store.path).set_direct("ask")  # the management UI
    assert str(store.direct) == "ask"


def test_broken_file_allows_nothing(tmp_path: Path) -> None:
    (tmp_path / "keyboard.json").write_text("{kaputt")
    store = KeyStore(tmp_path / "keyboard.json")
    assert store.direct == "never" and store.keys == []


def test_key_syntax() -> None:
    assert check_keys("ctrl+k, ctrl+s") is None
    assert check_keys("ctrl+plus") is None and parse_keys("ctrl+plus") is not None
    assert check_keys("play_pause") is None and check_keys("ctrl+alt+f20") is None
    assert check_keys("") is not None
    assert is_global("play_pause, next_track") and not is_global("play_pause, enter")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Aufnahme starten", "Aufnahme starten"),
        ("Flay, Aufnahme starten bitte", "Aufnahme starten"),
        ("Aufnahme", None),
        ("Erklär mir bitte, wie ich in OBS die Aufnahme starten kann", None),
        ("Aufnahme starten mit Ton", "Aufnahme starten mit Ton"),  # the longer name wins
    ],
)
def test_named_key(text: str, expected: str | None) -> None:
    assert named_key(text, ["Aufnahme starten", "Aufnahme starten mit Ton"]) == expected


def test_recognized_without_the_model() -> None:
    keys = ["Aufnahme starten"]
    assert computer_intent("Aufnahme starten", keys=keys) == "use_keyboard"
    assert direct_arguments("use_keyboard", "Aufnahme starten", keys=keys) == {
        "key": "Aufnahme starten"
    }
    assert direct_arguments("use_keyboard", "dreimal Aufnahme starten", keys=keys) is None
    assert computer_intent("Drück Enter") == "use_keyboard"
    assert computer_intent("Drück auf Weiter") != "use_keyboard"
    assert (
        computer_intent("Leg eine Taste an, mit der du in OBS die Aufnahme startest")
        == "use_keyboard"
    )
    assert computer_intent("Mach dir einen Hotkey für die Aufnahme") == "use_keyboard"


async def test_the_assistant_makes_a_virtual_key(tmp_path: Path) -> None:
    tool, store, pressed = make_tool(tmp_path)
    assert "new_key" in tool.spec.parameters["properties"]

    result = await tool.run({"new_key": "OBS Aufnahme starten", "purpose": "OBS: Aufnahme"})

    assert result.ok and pressed == []
    assert "ctrl+alt+shift+f13" in result.text
    assert "Strg Alt Shift F13" in (result.say or {}).values().__iter__().__next__()
    assert "not allowed yet" in result.text  # pressing is still "never"
    key = tool.spec.parameters["properties"]["key"]
    assert key["enum"] == ["OBS Aufnahme starten"] and "OBS: Aufnahme" in key["description"]
    assert tool.key_purposes == "OBS Aufnahme starten (OBS: Aufnahme)"

    assert (await tool.run({"remove_key": "OBS Aufnahme starten"})).ok
    assert store.keys == []


async def test_pressing_refused_unless_allowed(tmp_path: Path) -> None:
    tool, store, pressed = make_tool(tmp_path, "never")
    store.put("Aufnahme")

    assert not (await tool.run({"key": "Aufnahme"})).ok
    assert not (await tool.run({"keys": "enter"})).ok
    assert pressed == []
    assert "keys" not in tool.spec.parameters["properties"]
    # Dictation and the fixed shortcuts stay as they were.
    assert (
        await KeyboardTool(press=lambda _: None, front_is_own=lambda: False).run(
            {"shortcut": "save"}
        )
    ).ok


async def test_pressing_after_the_users_yes(tmp_path: Path) -> None:
    tool, store, pressed = make_tool(tmp_path, "ask")
    store.put("Aufnahme")
    assert "keys" in tool.spec.parameters["properties"]

    asked = await tool.run({"key": "Aufnahme", "times": 2})
    assert asked.confirm is not None and pressed == []

    # The model cannot approve by itself.
    assert not (await tool.run({"approval": "erfunden"})).ok
    assert (await tool.run(asked.confirm.arguments)).ok
    assert pressed == [("ctrl+alt+shift+f13", 2)]
    assert not (await tool.run(asked.confirm.arguments)).ok  # used up


async def test_pressing_always(tmp_path: Path) -> None:
    tool, store, pressed = make_tool(tmp_path, "always")
    store.put("Aufnahme")
    assert (await tool.run({"keys": "enter"})).ok
    assert (await tool.run({"key": "aufnahme"})).ok
    assert not (await tool.run({"keys": "ctrl+hyper"})).ok
    assert not (await tool.run({"key": "Unbekannt"})).ok
    assert pressed == [("enter", 1), ("ctrl+alt+shift+f13", 1)]


async def test_media_keys_work_with_own_window_in_front(tmp_path: Path) -> None:
    tool, store, pressed = make_tool(tmp_path, "always", front_is_own=True)
    store.put("Musik", keys="play_pause")
    store.put("Aufnahme")
    assert (await tool.run({"key": "Musik"})).ok
    assert not (await tool.run({"key": "Aufnahme"})).ok
    assert pressed == [("play_pause", 1)]


def test_option_values() -> None:
    text = Option("name", "Name", "", kind="text")
    choice = Option(
        "mode", "Modus", "never", kind="choice", choices=[("never", "nie"), ("ask", "frag")]
    )
    assert text.value("  Folie ") == "Folie" and text.value(None) == ""
    assert choice.value("ask") == "ask" and choice.value("hack") == "never"
    assert Option("x", "X").value(None) is False


def test_keyboard_tab(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    tab = keyboard_tab(App("flay", None, None), lambda: settings)
    actions = {a.id: a for a in tab.actions}
    lines: list[str] = []

    assert actions["key-save"].run(lines.append, {"name": "Aufnahme", "purpose": "OBS"})
    assert "ctrl+alt+shift+f13" in lines[0]
    assert actions["key-save"].run(lines.append, {"name": "Folie", "keys": "right"})
    assert not actions["key-save"].run(lines.append, {"name": "Kaputt", "keys": "ctrl+hyper"})
    assert actions["key-direct"].run(lines.append, {"mode": "ask"})

    pressing, keys = tab.sections()
    assert pressing.rows[0].value == "nach Rückfrage"
    assert [(r.label, r.value) for r in keys.rows] == [
        ("Aufnahme", "ctrl+alt+shift+f13 – OBS"),
        ("Folie", "right"),
    ]
    saved = json.loads((tmp_path / "keyboard.json").read_text())
    assert saved["direct"] == "ask" and len(saved["keys"]) == 2
    assert actions["key-remove"].run(lines.append, {"name": "folie"})
