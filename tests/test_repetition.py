from cassi.repetition import RepeatGuard, key, repeated_sentences

# How Flay answered after a few turns (the chat that showed the problem).
BEFORE = [
    "Alles klar – ich bin hier. Es ist spät, aber ich bleibe wach, solange du brauchst. "
    "Was ist dran?",
]


def test_a_sentence_already_said_is_noticed() -> None:
    reply = (
        "Entschuldigung – ich habe das jetzt nicht mehr nötig. Es ist spät, aber ich bleibe "
        "wach, solange du brauchst. Was ist dran?"
    )

    assert repeated_sentences(reply, BEFORE) == [
        "Es ist spät, aber ich bleibe wach, solange du brauchst."
    ]  # the short question may repeat


def test_case_punctuation_and_spacing_do_not_matter() -> None:
    assert key("Es ist  spät,   aber ICH bleibe wach!") == "es ist spät aber ich bleibe wach"
    guard = RepeatGuard(["Das ist ein ganz normaler Satz."])
    assert not guard.repeated("Erster Satz der Antwort ist neu.")
    assert guard.repeated("das ist ein ganz normaler Satz!")


def test_the_first_sentence_never_counts() -> None:
    assert repeated_sentences(BEFORE[0], BEFORE) == [
        "Es ist spät, aber ich bleibe wach, solange du brauchst."
    ]


def test_only_the_last_replies_count() -> None:
    old = ["Das habe ich vor langer Zeit gesagt."] + ["Etwas anderes ganz Neues hier."] * 3

    assert repeated_sentences("Neu. Das habe ich vor langer Zeit gesagt.", old) == []
