from __future__ import annotations

import pytest

from cassi.intents import computer_intent, mentions_selection
from cassi.tools.apps import App

APPS = [App("Firefox"), App("Slack"), App("Visual Studio Code"), App("PhpStorm")]
PROJECTS = ["flay", "shop"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Öffne Firefox", "control_window"),
        ("Wechsel zu Slack", "control_window"),
        ("Schieb Firefox auf den linken Monitor", "control_window"),
        ("Leg Firefox und Slack nebeneinander", "control_window"),
        ("Mach Slack zu", "control_window"),
        ("Schließ Slack", "control_window"),
        ("Öffne das Projekt flay", None),
        ("Öffne flay", None),
        ("Öffne die Datei UserController", "phpstorm"),
        ("Zeig mir die Klasse OrderService in PhpStorm", "phpstorm"),
        ("Welche Fehler zeigt PhpStorm in der Datei?", "phpstorm"),
        ("Was ist im Git geändert?", "phpstorm"),
        ("Starte die Tests", "phpstorm"),
        ("Öffne die letzte PDF aus Downloads", "find_file"),
        ("Wo liegt die Rechnung von Oktober?", "find_file"),
        ("Erinnere mich in 20 Minuten an den Tee", "timer"),
        ("Stell einen Timer auf 5 Minuten", "timer"),
        ("Erinnerst du dich an gestern?", None),
        ("Mach lauter", "system_control"),
        ("Ton aus", "system_control"),
        ("Nächstes Lied", "system_control"),
        ("Sperr den Bildschirm", "system_control"),
        ("Schreib: Viele Grüße, Timo", "use_keyboard"),
        ("Schreib Claude, dass die Tests fehlen", None),
        ("Speichern", "use_keyboard"),
        ("Mach das rückgängig", "use_keyboard"),
        ("Speichere, dass ich Tee mag", None),
        ("Wie spät ist es?", None),
        ("Zeig mir, was in Firefox offen ist", None),
        ("Wie öffne ich in Firefox einen neuen Tab?", None),
    ],
)
def test_computer_intent(text: str, expected: str | None) -> None:
    assert computer_intent(text, APPS, PROJECTS) == expected


def test_named_commands_win() -> None:
    assert computer_intent("Lass die Tests laufen", APPS, PROJECTS, ["tests"]) == "run_command"
    assert computer_intent("Starte deploy", APPS, PROJECTS, ["deploy"]) == "run_command"


def test_mentions_selection() -> None:
    assert mentions_selection("Übersetz das Markierte")
    assert mentions_selection("Explain the selected text")
    assert not mentions_selection("Erklär mir das")
