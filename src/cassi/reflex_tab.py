"""The management UI's "Reflexe" tab: what Cassi watches, its reflexes, and whether they
may press keys."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from cassi.app import App
from cassi.maintenance import Output
from cassi.reflexes.engine import read_status
from cassi.reflexes.store import ReflexStore, effective_strength

if TYPE_CHECKING:
    from cassi.config import Settings
    from cassi.manage import Tab

# A status older than this: the watching ended without saying so (Cassi was closed).
STALE_SECONDS = 10.0
SOURCES = {
    "rule": "Regel",
    "described": "beschrieben",
    "shown": "gezeigt",
    "practice": "aus Übung",
}


def reflex_tab(app: App, now: Callable[[], Settings]) -> Tab:
    from cassi.manage import Action, Option, Row, Section, Tab

    title = app.title

    def store() -> ReflexStore:
        return ReflexStore(now().data_dir / "reflexes.json")

    def sections() -> list[Section]:
        settings = now()
        status = read_status(settings.data_dir / "reflexes-status.json") or {}
        fresh = time.time() - float(status.get("written", 0)) < STALE_SECONDS
        watching = bool(status.get("watching")) and fresh
        rows = [Row("Beobachtet", status.get("target", "?") if watching else "nichts",
                    "ok" if watching else "")]  # fmt: skip
        if watching:
            rows += [
                Row("Ziel", status.get("goal") or "–"),
                Row("Bilder/s", str(status.get("fps", "?"))),
                Row("Reflexe ausgelöst", str(status.get("fired", 0))),
                Row("Nachgedacht", f"{status.get('thoughts', 0)}×"),
            ]
            if status.get("note"):
                rows.append(Row("Hinweis", str(status["note"]), "warn"))
            for line in status.get("recent", [])[:5]:
                rows.append(Row("", str(line)))
        reflexes = store().reflexes
        moment = time.time()
        listed = [
            Row(
                r.name,
                f"{r.describe()} · {SOURCES.get(r.source, r.source)}"
                + (f" · Stärke {effective_strength(r, moment):.1f}" if r.source != "rule" else "")
                + (f" · {r.uses}× ausgelöst" if r.uses else ""),
                "" if r.active else "warn",
            )
            for r in reflexes
        ]
        pressing = store().may_press
        return [
            Section(
                "Beobachten",
                rows,
                f"„Schau mir beim Spielen zu“, „Beobachte das Video“ – {title} reagiert dann "
                "in Ebenen wie ein Mensch: feste Regeln (Farbe, Änderung, Text) in Millisekunden, "
                "gelernte Reflexe über ein kleines Bildmodell (~40 ms), und nur bei Neuem denkt "
                "das Sprachmodell nach (Sekunden) – was es entscheidet, wird nach ein paar Malen "
                "selbst zum Reflex. „Gut so“ / „Falsch“ stärkt oder schwächt den letzten. "
                "Beenden: „Hör auf zu beobachten“, das Menüleisten-Symbol oder hier.",
            ),
            Section(
                "Reflexe",
                listed or [Row("Keine", "noch keine Reflexe")],
                "Gelb: noch in Übung (wird erst nach ein paar gleichen Entscheidungen aktiv). "
                "Unbenutzte gelernte Reflexe verblassen mit der Zeit.",
            ),
            Section(
                "Tasten",
                [
                    Row(
                        "Reflexe dürfen Tasten drücken",
                        "ja" if pressing else "nein",
                        "warn" if pressing else "",
                    )
                ],  # fmt: skip
                "Ohne diese Erlaubnis sprechen Reflexe nur. In Online-Spielen können "
                "automatische Tastendrücke gegen die Regeln verstoßen.",
            ),
        ]

    def measure(out: Output, chosen: Mapping[str, Any]) -> bool:
        from cassi.reflexes.bench import measure as run

        return run(now(), out, think=bool(chosen.get("think")))

    def stop(out: Output, _: Mapping[str, Any]) -> bool:
        store().request_stop()
        out(f"✓ {title} hört auf zu beobachten (innerhalb einer Sekunde).")
        return True

    def remove(out: Output, chosen: Mapping[str, Any]) -> bool:
        name = str(chosen.get("name", ""))
        if not store().remove(name):
            out(f"✗ Kein Reflex „{name}“.")
            return False
        out(f"✓ Entfernt: {name}")
        return True

    def pressing(out: Output, chosen: Mapping[str, Any]) -> bool:
        allowed = chosen.get("allowed") == "yes"
        store().set_may_press(allowed)
        out(f"✓ Reflexe dürfen Tasten drücken: {'ja' if allowed else 'nein'}")
        return True

    return Tab(
        "reflexes",
        "Reflexe",
        sections,
        [
            Action("reflex-stop", "Beobachten beenden", "Not-Aus: beendet das Beobachten.", stop),
            Action(
                "reflex-measure",
                "Geschwindigkeit messen",
                "Wie schnell Aufnahme, kleines Modell und Texterkennung hier sind (~10 s).",
                measure,
                options=(Option("think", "auch einmal nachdenken lassen (Bildmodell, langsamer)"),),
            ),
            Action(
                "reflex-remove",
                "Reflex entfernen",
                "Vergisst den Reflex mit diesem Namen.",
                remove,
                danger=True,
                options=(Option("name", "Name", "", kind="text", placeholder="Lebensleiste rot"),),
            ),
            Action(
                "reflex-press",
                "Tasten für Reflexe",
                "Ob Reflexe Tasten drücken dürfen.",
                pressing,
                options=(
                    Option(
                        "allowed",
                        "Tasten drücken",
                        "no",
                        kind="choice",
                        choices=[("no", "nein, nur sprechen"), ("yes", "ja")],
                    ),
                ),
            ),
        ],
    )
