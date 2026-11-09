import pytest

from flay.core.sentences import SentenceSplitter, clean_for_speech


def split(*deltas: str) -> list[str]:
    splitter = SentenceSplitter()
    sentences = [s for delta in deltas for s in splitter.feed(delta)]
    return [*sentences, *splitter.flush()]


def test_sentences_are_emitted_as_soon_as_complete() -> None:
    splitter = SentenceSplitter()

    assert list(splitter.feed("Hallo! Wie ")) == ["Hallo!"]
    assert list(splitter.feed("geht's? Gut")) == ["Wie geht's?"]
    assert list(splitter.flush()) == ["Gut"]


def test_token_by_token_stream() -> None:
    text = "Das ist der erste Satz. Und hier der zweite."

    assert split(*text) == ["Das ist der erste Satz.", "Und hier der zweite."]


@pytest.mark.parametrize(
    "text",
    [
        "Das ist z.B. ein Beispiel.",
        "Wir treffen uns am 3. Oktober.",
        "Frag Dr. Müller danach.",
        "Use a list, e.g. this one.",
        "Der Plan von A. Schmidt steht.",
    ],
)
def test_abbreviations_do_not_split(text: str) -> None:
    assert split(text, " ") == [text]


def test_newlines_split_and_markdown_is_removed() -> None:
    assert split("**Tipp:** Viel trinken\n- Früh schlafen 😴") == [
        "Tipp: Viel trinken",
        "Früh schlafen",
    ]


def test_quotes_after_punctuation_stay_with_sentence() -> None:
    assert split('Er sagte: "Ja!" Dann ging er.') == ['Er sagte: "Ja!"', "Dann ging er."]


def test_clean_for_speech() -> None:
    assert clean_for_speech("  ### Titel  `code` 🚀 ") == "Titel code"
    assert clean_for_speech("2. Wasser trinken") == "Wasser trinken"


def test_long_first_sentence_is_cut_at_a_clause() -> None:
    text = (
        "Sonnenberg ist eine Stadt in Nordrhein-Westfalen, die für ihre Universität bekannt ist "
        "und einen berühmten Witz hat."
    )

    assert split(*(word + " " for word in text.split(" "))) == [
        "Sonnenberg ist eine Stadt in Nordrhein-Westfalen,",
        "die für ihre Universität bekannt ist und einen berühmten Witz hat.",
    ]


def test_short_sentences_are_not_cut() -> None:
    assert split("Ja, gerne. Nein, danke.") == ["Ja, gerne.", "Nein, danke."]


def test_first_chunk_ends_at_its_first_clause() -> None:
    text = "Ein Container ist eine isolierte Umgebung, die ein Programm samt Bibliotheken hat."

    assert split(*(word + " " for word in text.split(" "))) == [
        "Ein Container ist eine isolierte Umgebung,",
        "die ein Programm samt Bibliotheken hat.",
    ]
    assert split("Klar, mache ich. Gut, fertig.") == ["Klar, mache ich.", "Gut, fertig."]
