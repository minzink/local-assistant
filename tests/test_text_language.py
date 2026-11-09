import pytest

from flay.core.language import Language
from flay.core.text_language import detect_text_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wie wird das Wetter morgen?", Language.DE),
        ("Schöne Grüße", Language.DE),
        ("What is the capital of France?", Language.EN),
        ("Can you remind me tomorrow?", Language.EN),
        ("Paris", Language.EN),  # undecidable → fallback
    ],
)
def test_detect_text_language(text: str, expected: Language) -> None:
    assert detect_text_language(text, fallback=Language.EN) is expected
