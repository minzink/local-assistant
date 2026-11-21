"""The management UI's "Tastatur" tab: the virtual keys the assistant made, and whether it
may press keys at all."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from cassi.app import App
from cassi.maintenance import Output
from cassi.tools.virtual_keys import DIRECT_LABELS, DIRECT_MODES, DirectMode, KeyStore

if TYPE_CHECKING:
    from cassi.config import Settings
    from cassi.manage import Tab


def store_for(settings: Settings) -> KeyStore:
    return KeyStore(settings.data_dir / "keyboard.json")


def keyboard_tab(app: App, now: Callable[[], Settings]) -> Tab:
    from cassi.manage import Action, Option, Row, Section, Tab

    title = app.title

    def sections() -> list[Section]:
        store = store_for(now())
        direct = store.direct
        return [
            Section(
                "Tasten drücken",
                [
                    Row(
                        f"{title} darf Tasten drücken",
                        DIRECT_LABELS[direct],
                        "warn" if direct == "always" else "",
                    )
                ],
                f"Ob {title} von sich aus Tasten drücken darf – virtuelle Tasten und alle "
                "anderen, wenn es damit erledigt, was du willst (z.B. Enter, Strg+Shift+T). "
                "Enter kann Nachrichten abschicken und Dialoge bestätigen: „nach Rückfrage“ "
                "fragt jedes Mal, „immer“ drückt sofort. Diktieren und die festen Kürzel "
                "(Speichern, Rückgängig …) gehen wie bisher.",
            ),
            Section(
                "Virtuelle Tasten",
                [
                    Row(k.name, f"{k.keys} – {k.purpose}" if k.purpose else k.keys)
                    for k in store.keys
                ]
                or [Row("Keine", "noch keine virtuelle Taste")],
                f"{title} legt virtuelle Tasten selbst an („Leg eine Taste an, mit der du in "
                "OBS die Aufnahme startest“): eine freie Tastenkombination wie "
                "Strg+Alt+Shift+F13, die keine Tastatur hat und kein Programm belegt. Trag sie "
                "im Programm als Tastenkürzel ein – oder lass sie drücken, während das Feld für "
                f"das Kürzel wartet. Danach steuert {title} das Programm darüber („Starte die "
                "Aufnahme“). Hier lassen sie sich auch von Hand anlegen, ändern und entfernen; "
                "Änderungen wirken sofort.",
            ),
        ]

    def save(out: Output, chosen: Mapping[str, Any]) -> bool:
        name = str(chosen.get("name", ""))
        store = store_for(now())
        existed = store.get(name) is not None
        made = store.put(name, str(chosen.get("purpose", "")), str(chosen.get("keys", "")))
        if isinstance(made, str):
            out(f"✗ {made}")
            return False
        out(f"✓ {'Geändert' if existed else 'Angelegt'}: {made.name} = {made.keys}")
        if not existed:
            out("Diese Tastenkombination im Programm als Tastenkürzel eintragen.")
        return True

    def remove(out: Output, chosen: Mapping[str, Any]) -> bool:
        name = str(chosen.get("name", ""))
        if not store_for(now()).remove(name):
            out(f"✗ Keine Taste „{name}“.")
            return False
        out(f"✓ Entfernt: {name}")
        return True

    def direct(out: Output, chosen: Mapping[str, Any]) -> bool:
        mode: DirectMode = next((m for m in DIRECT_MODES if m == chosen.get("mode")), "never")
        store_for(now()).set_direct(mode)
        out(f"✓ {title} darf Tasten drücken: {DIRECT_LABELS[mode]}")
        return True

    return Tab(
        "keyboard",
        "Tastatur",
        sections,
        [
            Action(
                "key-save",
                "Virtuelle Taste speichern",
                "Legt eine Taste an oder ändert die mit demselben Namen.",
                save,
                options=(
                    Option("name", "Name", "", kind="text", placeholder="OBS Aufnahme starten"),
                    Option(
                        "purpose", "Wofür", "", kind="text", placeholder="OBS: Aufnahme starten"
                    ),
                    Option(
                        "keys",
                        "Tasten",
                        "",
                        kind="keys",
                        placeholder="leer lassen = nächste freie Taste",
                        help="Leer lassen: es gibt die nächste freie Kombination (Strg+Alt+"
                        "Shift+F13 …). Oder „Aufnehmen“ und die Tasten drücken. Mehrere "
                        "nacheinander mit Komma: „ctrl+k, ctrl+s“. Vom Browser belegte Kürzel "
                        "(z.B. Strg+W) lassen sich nur eintippen.",
                    ),
                ),
            ),
            Action(
                "key-remove",
                "Virtuelle Taste entfernen",
                "Entfernt die Taste mit diesem Namen (im Programm danach das Kürzel löschen).",
                remove,
                danger=True,
                options=(
                    Option("name", "Name", "", kind="text", placeholder="OBS Aufnahme starten"),
                ),
            ),
            Action(
                "key-direct",
                "Tasten drücken erlauben",
                f"Ob {title} von sich aus Tasten drücken darf.",
                direct,
                options=(
                    Option(
                        "mode",
                        "Tasten drücken",
                        "never",
                        kind="choice",
                        choices=[(m, DIRECT_LABELS[m]) for m in DIRECT_MODES],
                    ),
                ),
            ),
        ],
    )
