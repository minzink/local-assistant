"""One table for all fixed patterns (flay.intents): which tool a request is routed to
without asking the model. Overlaps between the patterns show up here first – the rows
marked "früher falsch" once went to the wrong tool."""

from __future__ import annotations

import pytest

from flay.intents import computer_intent, file_intent, tool_intent, web_intent
from flay.tools.apps import App

APPS = [App(n) for n in ("Firefox", "Slack", "PhpStorm", "Terminal", "Spotify")]


def route(text: str) -> str | None:
    """The order the assistant checks them in (Assistant._intent and _answer)."""
    for found in (tool_intent(text), computer_intent(text, APPS, ["flay", "moewe"])):
        if found is not None:
            return found
    if file_intent(text):
        return "read_file"
    return web_intent(text)


@pytest.mark.parametrize(
    ("said", "tool"),
    [
        # Projects and Claude
        ("Öffne das Projekt flay", "open_project"),
        ("Schließe das Projekt Moewe", "close_project"),
        ("Sag Claude, er soll Tests schreiben", "send_to_claude"),
        ("Ist Claude fertig?", "claude_status"),
        ("Kopiere den Text in Claude", "send_to_claude"),
        # Memory
        ("Merk dir, dass ich Tee mag", "remember"),
        ("Speichere dir, dass ich Tee mag", "remember"),  # früher falsch: Cmd+S
        ("Vergiss, dass ich Tee mag", "forget"),
        # Web and files
        ("Öffne heise.de", "open_url"),
        ("Öffne google.com und sag mir, was du siehst", "read_webpage"),  # früher falsch
        ("Lies die README von flay", "read_file"),
        # Apps and windows
        ("Öffne Firefox", "control_window"),
        ("Schieb Firefox auf den linken Monitor", "control_window"),  # früher falsch: Link
        ("Mach Slack zu", "control_window"),
        ("Mach in Slack einen neuen Kanal namens test auf", None),  # früher falsch: Tests
        ("Was ist in Firefox offen?", None),  # a question, not a switch
        # Keyboard and system
        ("Schreib: Viele Grüße", "use_keyboard"),
        ("Speichern", "use_keyboard"),
        ("Streich das letzte Wort", "use_keyboard"),
        ("Mach lauter", "system_control"),
        ("Sperr den Bildschirm", "system_control"),
        ("Erinnere mich in 20 Minuten an den Tee", "timer"),
        # PhpStorm (no coding: that is Claude's)
        ("Öffne die Datei UserController", "phpstorm"),
        ("Starte die Tests", "phpstorm"),
        ("Was ist im Git geändert?", "phpstorm"),
        ("Öffne die letzte PDF aus Downloads", "find_file"),
        # Flay's own settings vs. the computer's
        ("Sprich langsamer", "adjust_settings"),
        ("Antworte ab jetzt kürzer", "adjust_settings"),
        ("Schau nicht mehr auf meinen Bildschirm", "adjust_settings"),
        ("Stell in den Systemeinstellungen den Dunkelmodus an", None),  # the model classifies
        # Stories and how-to questions ask for nothing
        ("Ich habe gestern flay geöffnet", None),
        ("Wie öffne ich in Firefox einen neuen Tab?", None),
        ("Erzähl mir einen Witz", None),
    ],
)
def test_route(said: str, tool: str | None) -> None:
    assert route(said) == tool


PROJECTS = [
    "flay", "umx", "project-symfony-kx-ewo", "component-symfony-kx-command",
    "project-symfony-kx-forms", "lib-react-kx-forms-app", "stadt-wolkenthal", "moewe",
]  # fmt: skip


@pytest.mark.parametrize(
    ("tool", "said", "arguments"),
    [
        ("open_project", "Öffne das Projekt flay", {"name": "flay"}),
        ("open_project", "Öffne das Projekt Stadt Wolkenthal", {"name": "stadt-wolkenthal"}),
        ("open_project", "Öffne in PHP Storm das KX Forms Projekt.", None),  # two fit: ask
        ("close_project", "Schließe K X E W O.", {"name": "project-symfony-kx-ewo"}),
        ("control_window", "Öffne Firefox", {"action": "open", "app": "Firefox"}),
        ("control_window", "Schieb Firefox auf den linken Monitor",
         {"action": "move", "app": "Firefox", "monitor": "left"}),
        ("control_window", "Leg PhpStorm und Firefox nebeneinander",
         {"action": "side_by_side", "app": "PhpStorm", "other_app": "Firefox"}),
        ("control_window", "Mach Slack zu", {"action": "close", "app": "Slack"}),
        ("control_window", "Hol mir mal den Browser nach vorne.", None),  # which app?
        ("control_window", "Schieb Firefox irgendwohin", None),  # where?
        ("use_keyboard", "Speichern.", {"shortcut": "save"}),
        ("use_keyboard", "Schreib: Viele Grüße", {"text": "Viele Grüße"}),
        ("use_keyboard", "Schreib dem Max, dass ich später komme", None),  # to be phrased
        ("system_control", "Mach mal etwas lauter.", {"action": "volume_up"}),
        ("system_control", "Lautstärke auf 30", {"action": "set_volume", "level": 30}),
        ("timer", "Erinnere mich in 20 Minuten an den Tee.",
         {"action": "set", "minutes": 20, "text": "Tee"}),
        ("timer", "Erinnere mich um halb drei an den Tee", None),  # a time to work out
        ("adjust_settings", "Sprich bitte etwas langsamer.", {"speed": "slower"}),
        ("calendar", "Hab ich morgen Termine?", {"when": "tomorrow"}),
        ("calendar", "Was steht am Freitag an?", None),
        ("claude_status", "Ist Claude fertig?", {}),
        ("claude_status", "Was macht Claude im Projekt umx?", {"project": "umx"}),
        ("send_to_claude", "Sag Claude, er soll Tests schreiben", None),  # always the model
        ("phpstorm", "Was ist im Git geändert?", {"action": "git_status"}),
        ("phpstorm", "Welche Fehler zeigt PhpStorm in der Datei?", {"action": "problems"}),
        ("phpstorm", "Öffne die Datei UserController.", None),  # a file to find: the model
        ("workflow", "Ablauf fertig.", {"action": "stop"}),
        ("workflow", "Mach den Ablauf Release bauen.", None),
    ],
)  # fmt: skip
def test_direct_arguments(tool: str, said: str, arguments: dict[str, object] | None) -> None:
    from flay.direct import direct_arguments

    assert direct_arguments(tool, said, PROJECTS, APPS) == arguments
