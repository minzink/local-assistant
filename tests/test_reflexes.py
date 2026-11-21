import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from cassi.app import App
from cassi.config import Settings as CassiSettings
from cassi.direct import direct_arguments
from cassi.intents import computer_intent
from cassi.reflex_tab import reflex_tab
from cassi.reflexes.capture import Target, crop
from cassi.reflexes.engine import Decision, Observer, Settings, _text_matches
from cassi.reflexes.store import Action, Reflex, ReflexStore, effective_strength
from cassi.reflexes.tool import (
    ObserveTool,
    area_words,
    flatten,
    parse_color,
    parse_region,
    pointer_region,
)


class FakeEncoder:
    """Impressions from the mean colour: alike pictures, alike vectors."""

    available = True

    def image(self, picture: Image.Image) -> np.ndarray:
        r, g, b = np.asarray(picture.convert("RGB"), np.float32).reshape(-1, 3).mean(axis=0)
        vector = np.array([r, g, b, 30.0], np.float32)
        return vector / np.linalg.norm(vector)

    def chance(self, impression: np.ndarray, label: str, others: list[str]) -> float:
        return 0.9 if "red" in label and impression[0] > 0.9 else 0.1


class FakeSource:
    def __init__(self) -> None:
        self.picture = Image.new("RGB", (200, 100), (20, 20, 20))

    def private(self) -> str | None:
        return None

    def grab(self) -> Image.Image:
        return self.picture

    def close(self) -> None:
        pass


def make_observer(tmp_path: Path, **kwargs: Any) -> tuple[Observer, FakeSource, list[str]]:
    said: list[str] = []
    store = ReflexStore(tmp_path / "reflexes.json")
    observer = Observer(store, FakeEncoder(), said.append, **kwargs)  # type: ignore[arg-type]
    source = FakeSource()
    observer._source = source  # type: ignore[assignment]
    observer.status.started = 1.0
    return observer, source, said


def red_top_half(source: FakeSource) -> None:
    picture = Image.new("RGB", (200, 100), (20, 20, 20))
    picture.paste((230, 30, 30), (0, 0, 200, 50))
    source.picture = picture


def test_store_judges_and_forgets(tmp_path: Path) -> None:
    store = ReflexStore(tmp_path / "reflexes.json", clock=lambda: 1000.0)
    store.put(
        Reflex("Leben", "color", Action(say="Achtung"), (0, 0, 1000, 100), {"rgb": [255, 0, 0]})
    )
    assert store.get("leben") is not None and not store.may_press
    assert store.judge("Leben", good=True).strength == pytest.approx(1.25)  # type: ignore[union-attr]
    assert store.judge("Leben", good=False).strength == pytest.approx(0.75)  # type: ignore[union-attr]
    store.judge("Leben", good=False)
    gone = store.judge("Leben", good=False)
    assert gone is not None and gone.strength == 0 and store.get("Leben") is None

    store.set_may_press(True)
    store.request_stop()
    other = ReflexStore(store.path)  # the running Cassi reads what the UI wrote
    assert other.may_press and other.stop_at == 1000.0


def test_unused_learned_reflexes_fade(tmp_path: Path) -> None:
    day = 86400.0
    now = [100 * day]
    store = ReflexStore(tmp_path / "reflexes.json", clock=lambda: now[0])
    old = Reflex("Alt", "learned", Action(say="x"), vector=(1.0,), source="practice",
                 last_used=now[0] - 2 * day, strength=1.0)  # fmt: skip
    store.put(old)
    assert effective_strength(old, now[0]) == pytest.approx(0.97)
    assert store.fade() == []
    now[0] += 200 * day
    assert store.fade() == ["Alt"] and store.reflexes == []


def test_rules_fire_once_when_they_become_true(tmp_path: Path) -> None:
    observer, source, _ = make_observer(tmp_path)
    observer.store.put(
        Reflex("Rot oben", "color", Action(say="Rot!"), (0, 0, 1000, 500),
               {"rgb": [230, 30, 30], "tolerance": 40, "share": 0.5}, cooldown=1.0)
    )  # fmt: skip
    assert observer.look(10.0) == ([], False)
    red_top_half(source)
    fired, _ = observer.look(10.2)
    assert [r.name for r in fired] == ["Rot oben"]
    assert observer.look(10.4)[0] == []  # still red: not again
    source.picture = Image.new("RGB", (200, 100), (20, 20, 20))
    observer.look(10.6)
    red_top_half(source)
    assert observer.look(10.8)[0] == []  # resting (cooldown 1 s)
    source.picture = Image.new("RGB", (200, 100), (20, 20, 20))
    observer.look(11.0)
    red_top_half(source)
    assert [r.name for r in observer.look(11.5)[0]] == ["Rot oben"]


def test_text_rule() -> None:
    assert _text_matches("HP 15 / 100", {"below": 20})
    assert not _text_matches("HP 45 / 100", {"below": 20})
    assert _text_matches("Download abgeschlossen", {"contains": "abgeschlossen"})
    assert not _text_matches("anything", {})


def test_scene_needs_two_pictures_and_learned_needs_a_close_match(tmp_path: Path) -> None:
    observer, source, _ = make_observer(tmp_path, settings=Settings(encoder_hz=100))
    observer.store.put(Reflex("Rot", "scene", Action(say="Rot"), params={"label": "a red screen"}))
    learned = FakeEncoder().image(Image.new("RGB", (10, 10), (20, 200, 20)))
    observer.store.put(Reflex("Grün", "learned", Action(say="Grün"), vector=tuple(learned)))

    source.picture = Image.new("RGB", (200, 100), (240, 10, 10))
    assert observer.look(1.0)[0] == []  # one odd frame is no scene
    source.picture = Image.new("RGB", (200, 100), (241, 10, 10))
    assert [r.name for r in observer.look(2.0)[0]] == ["Rot"]
    source.picture = Image.new("RGB", (200, 100), (20, 198, 22))
    assert [r.name for r in observer.look(3.0)[0]] == ["Grün"]
    source.picture = Image.new("RGB", (200, 100), (20, 20, 200))
    assert observer.look(4.0)[0] == []


def test_practice_becomes_a_reflex(tmp_path: Path) -> None:
    observer, _, _ = make_observer(tmp_path, settings=Settings(promote_after=2))
    impression = FakeEncoder().image(Image.new("RGB", (10, 10), (200, 200, 20)))
    decision = Decision(Action(say="Ladebildschirm"), "loading screen")

    first = observer.practise(impression, decision)
    assert first.name == "loading screen" and not first.active
    second = observer.practise(impression, decision)
    assert second.seen == 2 and second.active
    other = observer.practise(impression, Decision(Action(say="Anders"), "loading screen"))
    assert other.name == "loading screen 2"  # another reaction: another reflex


async def test_keys_only_with_leave(tmp_path: Path) -> None:
    pressed: list[str] = []
    observer, _, said = make_observer(tmp_path, press=pressed.append)
    reflex = Reflex("Springen", "learned", Action(say="Hopp", keys="space"))
    observer.store.put(reflex)
    await observer._fire(reflex)
    assert pressed == [] and said == ["Hopp"]
    observer.store.set_may_press(True)
    await observer._fire(reflex)
    assert pressed == ["space"]


async def test_thinking_decides_and_learns(tmp_path: Path) -> None:
    async def think(picture: Image.Image, goal: str, known: list[str]) -> Decision:
        return Decision(Action(say="Neues Level"), "new level")

    observer, source, said = make_observer(
        tmp_path, thinker=think, settings=Settings(encoder_hz=100, think_seconds=0.0)
    )
    observer._goal = "Sag, wenn ein neues Level beginnt"
    _, novel = observer.look(1.0)
    assert novel
    observer._think_soon()
    assert observer._thinking is not None
    await observer._thinking
    assert said == ["Neues Level"]
    assert observer.store.get("new level") is not None


def test_crop_and_parsing() -> None:
    picture = Image.new("RGB", (200, 100))
    assert crop(picture, (0, 0, 500, 500)).size == (100, 50)
    assert crop(picture, None).size == (200, 100)
    assert parse_region([0, 0, 1000, 100]) == (0, 0, 1000, 100)
    assert parse_region([1, 2]) is None and parse_region([0, 0, 2, 2]) is None
    assert parse_color("rot") == (220, 40, 40) and parse_color("#00ff00") == (0, 255, 0)
    assert parse_color("lila-ish") is None


async def test_tool_sets_up_reflexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observer, _, _ = make_observer(tmp_path)
    tool = ObserveTool(observer)
    rule = {"action": "rule", "name": "Leben", "kind": "color", "color": "red",
            "region": [0, 0, 300, 50], "say": "Pass auf!"}  # fmt: skip
    assert (await tool.run(rule)).ok
    assert not (await tool.run({**rule, "region": None})).ok
    silent = await tool.run({**rule, "name": "", "say": ""})  # name and saying: from the place
    assert silent.ok and observer.store.get("Leben") is not None
    assert (await tool.run({"action": "scene", "name": "Ende", "label": "a game over screen",
                            "keys": "enter"})).ok  # fmt: skip
    assert "not allowed" in (await tool.run({"action": "scene", "name": "Ende2",
                                              "label": "x", "keys": "enter"})).text  # fmt: skip
    monkeypatch.setattr(observer, "impression_now", lambda region=None: np.ones(4, np.float32) / 2)
    taught = await tool.run({"action": "teach", "name": "Boss", "say": "Der Boss!"})
    assert taught.ok and observer.store.get("Boss").kind == "learned"  # type: ignore[union-attr]
    observer.last_fired = "Boss"
    assert (await tool.run({"action": "feedback", "good": True})).ok
    assert "Boss" in (await tool.run({"action": "list"})).text
    assert (await tool.run({"action": "remove", "name": "Boss"})).ok
    assert (await tool.run({"action": "stop"})).text == "Nothing was being watched."


async def test_start_and_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observer, source, _ = make_observer(tmp_path, settings=Settings(fps=30))
    monkeypatch.setattr("cassi.reflexes.engine.FrameSource", lambda target: source)
    observer.start(Target(), "")
    await asyncio.sleep(0.15)
    assert observer.watching and observer.status.frames > 1
    observer.store.request_stop()  # tray or management UI
    await asyncio.sleep(0.1)
    assert not observer.watching


def test_recognized_by_voice() -> None:
    assert computer_intent("Schau mir beim Spielen zu") == "observe"
    assert computer_intent("Gut so", watching=True) == "observe"
    assert computer_intent("Gut so") != "observe"
    assert direct_arguments("observe", "Hör auf zu beobachten") == {"action": "stop"}
    assert direct_arguments("observe", "Falsch") == {"action": "feedback", "good": False}


def test_reflex_tab(tmp_path: Path) -> None:
    settings = CassiSettings(data_dir=tmp_path)
    store = ReflexStore(tmp_path / "reflexes.json")
    store.put(
        Reflex("Leben", "color", Action(say="Achtung"), (0, 0, 1000, 100), {"rgb": [255, 0, 0]})
    )
    tab = reflex_tab(App("flay", None, None), lambda: settings)
    actions = {a.id: a for a in tab.actions}
    lines: list[str] = []
    watching, reflexes, keys = tab.sections()
    assert watching.rows[0].value == "nichts"
    assert reflexes.rows[0].label == "Leben"
    assert keys.rows[0].value == "nein"
    assert actions["reflex-press"].run(lines.append, {"allowed": "yes"})
    assert actions["reflex-stop"].run(lines.append, {})
    saved = json.loads((tmp_path / "reflexes.json").read_text())
    assert saved["may_press"] and saved["stop_at"] > 0
    assert actions["reflex-remove"].run(lines.append, {"name": "leben"})


def test_arguments_as_models_send_them() -> None:
    nested = {"action": "rule", "where": "das hier", "kind": "color", "color": "red",
              "reaction": {"say": "Achtung"}}  # fmt: skip
    flat = flatten(nested)
    assert flat["say"] == "Achtung" and flat["at_pointer"] and "where" not in flat
    assert flatten({"action": "rule", "where": "die Lebensleiste"})["where"] == "die Lebensleiste"


def test_region_around_the_pointer_and_said_back() -> None:
    area = (0.0, 0.0, 1000.0, 500.0)
    assert pointer_region(area, (500.0, 250.0), (100.0, 50.0)) == (450, 450, 550, 550)
    assert pointer_region(area, (2000.0, 10.0)) is None
    assert area_words((0, 0, 200, 100)) == "oben links"
    assert area_words((400, 400, 600, 600)) == "in der Mitte"
    assert area_words((700, 800, 990, 990)) == "unten rechts"


async def test_place_found_at_the_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observer, source, _ = make_observer(tmp_path)
    monkeypatch.setattr(observer, "view", lambda: (source.picture, (0.0, 0.0, 1000.0, 500.0)))
    monkeypatch.setattr("cassi.tools.screen.pointer_position", lambda: (100.0, 50.0))
    tool = ObserveTool(observer)
    result = await tool.run(
        {"action": "rule", "kind": "color", "color": "rot", "where": "das hier", "say": "Achtung"}
    )
    assert result.ok and "oben links" not in result.text
    reflex = observer.store.reflexes[0]
    assert reflex.region == (0, 10, 220, 190)
    assert "Mauszeiger" in str(result.say)


@pytest.mark.parametrize(
    ("said", "keys"),
    [
        ("Wenn dieser Bildschirm kommt, drück Enter.", "enter"),
        ("drück Strg S", "ctrl+s"),
        ("drück dreimal Pfeil runter", "down"),
        ("drück die Leertaste", "space"),
        ("drück auf den grünen Knopf", ""),
        ("Sag Achtung", ""),
    ],
)
def test_keys_from_spoken_words(said: str, keys: str) -> None:
    from cassi.reflexes.tool import keys_from_words

    assert keys_from_words(said) == keys


async def test_reaction_put_in_the_wrong_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observer, _, _ = make_observer(tmp_path)
    monkeypatch.setattr(observer, "impression_now", lambda region=None: np.ones(4, np.float32) / 2)
    tool = ObserveTool(observer)
    # As the model sent it: the whole request as the saying.
    await tool.run({"action": "teach", "say": "Wenn dieser Bildschirm kommt, drück Enter."})
    reflex = observer.store.reflexes[0]
    assert (reflex.name, reflex.action.keys, reflex.action.say) == ("enter drücken", "enter", "")
    # The searched word in the goal, no saying: found and said anyway.
    monkeypatch.setattr(
        observer, "view", lambda: (Image.new("RGB", (100, 100)), (0.0, 0.0, 100.0, 100.0))
    )
    await tool.run({"action": "rule", "kind": "text", "goal": "Fehler", "region": [0, 0, 500, 500]})
    text_rule = observer.store.get("Fehler")
    assert text_rule is not None and text_rule.params == {"contains": "Fehler"}
    assert text_rule.action.say == "Fehler ist da."
