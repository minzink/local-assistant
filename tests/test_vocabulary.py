import pytest

from flay.stt.vocabulary import Vocabulary, is_noise

PROJECTS = ["stadt-wolkenthal", "moewe", "project-symfony-kx-forms", "lib-react-kx-forms-app"]


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        (
            "Project Symphony KX Forms, sag Claude, er soll die Read Me zusammenfassen.",
            "Project Symfony KX Forms, sag Claude, er soll die README zusammenfassen.",
        ),
        ("Öffne das Projekt Möwe.", "Öffne das Projekt moewe."),
        ("Schließe Stadt Wolkental.", "Schließe Stadt Wolkenthal."),
        ("Öffne das Projekt Wolkental", "Öffne das Projekt Wolkenthal"),
        # Ordinary words that only look like project words stay as they are.
        ("Ich möchte heute Formen bauen.", "Ich möchte heute Formen bauen."),
        ("Die Forum Seite ist kaputt.", "Die Forum Seite ist kaputt."),
        ("Wie wird das Wetter morgen?", "Wie wird das Wetter morgen?"),
    ],
)
def test_correct(heard: str, expected: str) -> None:
    assert Vocabulary(lambda: PROJECTS).correct(heard) == expected


@pytest.mark.parametrize(
    ("text", "noise"),
    [
        ("Matchup.", True),
        ("Hm.", True),
        ("", True),
        ("Ja.", False),
        ("Danke!", False),
        ("moewe", False),
        ("Öffne moewe", False),
    ],
)
def test_is_noise(text: str, noise: bool) -> None:
    assert is_noise(text, PROJECTS) is noise


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("Sagt Lord, er soll Tests schreiben.", "Sagt Claude, er soll Tests schreiben."),
        ("Frag Lot, warum der Build fehlschlägt.", "Frag Claude, warum der Build fehlschlägt."),
        ("Tell Claw to update the change block.", "Tell Claude to update the change block."),
        # Only right after a verb that addresses someone.
        ("Der Herr der Ringe ist ein Lord.", "Der Herr der Ringe ist ein Lord."),
    ],
)
def test_misheard_claude_after_a_request_verb(heard: str, expected: str) -> None:
    assert Vocabulary(lambda: []).correct(heard) == expected
