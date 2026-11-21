from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cassi import screen_watch
from cassi.screen_watch import (
    LOOKS,
    WATCH,
    Reading,
    ScreenWatcher,
    about_screen,
    describe,
    read_switch,
    reading_order,
    write_switch,
)
from cassi.tools import ocr, screen
from cassi.tools.ocr import TextBox
from cassi.tools.screen import PRIVACY, Window


def box(text: str, x: float, y: float, w: float = 100, h: float = 10) -> TextBox:
    return TextBox(text, (x, y, w, h))


def test_reading_order_drops_single_words_of_a_line() -> None:
    boxes = [
        box("Zweite Zeile", 0, 40),
        box("Fehler: Datei fehlt", 0, 10, 200),
        box("Fehler:", 5, 10, 50),  # a word of the line above
        box("Fehler: Datei fehlt", 0, 10, 200),  # repeated
    ]
    assert reading_order(boxes, 1000) == "Fehler: Datei fehlt\nZweite Zeile"
    assert len(reading_order(boxes, 10)) == 10


def test_describe_only_fresh_text() -> None:
    reading = Reading("Mail", "Posteingang", "Hallo Anna", at=100.0)
    assert "Mail – Posteingang" in str(describe(reading, now=105.0))
    assert "Hallo Anna" in str(describe(reading, now=105.0))
    assert describe(reading, now=200.0) is None
    assert describe(None) is None


def test_which_requests_get_the_text() -> None:
    assert about_screen("Was heißt diese Fehlermeldung?")
    assert about_screen("Antworte auf die Mail")
    assert not about_screen("Wie wird das Wetter morgen?")
    assert LOOKS.search("Wie sieht das Diagramm aus?")
    assert not LOOKS.search("Was steht da?")


class FakeScreen:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.window = Window(1, "Mail", "Posteingang", (0, 0, 400, 300))
        self.windows: list[Window] = [self.window]
        self.grey = 100
        self.ocr_calls = 0
        monkeypatch.setattr(screen, "list_windows", lambda: self.windows)
        monkeypatch.setattr(screen_watch, "_window_image", lambda w: self.image())
        monkeypatch.setattr(ocr, "read_image", self.read)
        monkeypatch.setattr(PRIVACY, "apps", ("keepass",))
        monkeypatch.setattr(PRIVACY, "titles", ())
        monkeypatch.setattr(PRIVACY, "paused", False)

    def image(self) -> Image.Image:
        return Image.new("RGB", (400, 300), (self.grey,) * 3)

    def read(self, image: Any) -> list[TextBox]:
        self.ocr_calls += 1
        return [box(f"Text {self.ocr_calls}", 0, 0)]


def test_reads_only_when_the_window_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeScreen(monkeypatch)
    watcher = ScreenWatcher(None, default=True, ocr_seconds=2.0)

    watcher.step(now=10.0)
    assert fake.ocr_calls == 1 and watcher.reading is not None
    assert watcher.reading.text == "Text 1"
    watcher.step(now=11.0)  # unchanged: no OCR, but still current
    assert fake.ocr_calls == 1 and watcher.reading.at == 11.0

    fake.grey = 200  # changed, but only 1 s after the last read: wait
    watcher.step(now=11.5)
    assert fake.ocr_calls == 1
    watcher.step(now=12.5)
    assert fake.ocr_calls == 2 and watcher.reading.text == "Text 2"

    fake.windows = [Window(2, "Editor", "notes.txt", (0, 0, 400, 300))]  # another window
    watcher.step(now=12.6)
    assert fake.ocr_calls == 3 and watcher.reading.app == "Editor"


def test_private_or_paused_is_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeScreen(monkeypatch)
    watcher = ScreenWatcher(None, default=True)
    watcher.step(now=1.0)
    assert watcher.reading is not None

    fake.windows = [Window(3, "KeePassXC", "Tresor", (0, 0, 400, 300))]
    watcher.step(now=5.0)
    assert watcher.reading is None and fake.ocr_calls == 1

    fake.windows = [fake.window]
    monkeypatch.setattr(PRIVACY, "paused", True)
    watcher.step(now=9.0)
    assert watcher.reading is None and fake.ocr_calls == 1


def test_switch_file_wins_over_the_setting(tmp_path: Path) -> None:
    switch = tmp_path / "screen.json"
    watcher = ScreenWatcher(switch, default=False)
    assert not WATCH.enabled and read_switch(switch) is None

    write_switch(switch, True)  # tray or management UI
    watcher._sync()
    assert WATCH.enabled and read_switch(switch) is True
    watcher.reading = Reading("Mail", "", "Hallo", at=float("inf"))
    assert watcher.current() is not None

    write_switch(switch, False)
    switch.touch()
    watcher._stamp = None
    watcher._sync()
    assert not WATCH.enabled and watcher.current() is None
