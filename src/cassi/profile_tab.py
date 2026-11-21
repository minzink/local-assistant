"""The management UI's "Profile" tab: separate AIs – each with its own definition, memory and
what it learned (cassi.profiles)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cassi import profiles
from cassi.app import App
from cassi.maintenance import Output

if TYPE_CHECKING:
    from cassi.config import Settings
    from cassi.manage import Option, Tab

COPY_CHOICES = [
    ("definition", "nur die Definition (Einstellungen, Persönlichkeit – ohne Erinnerungen)"),
    ("nothing", "nichts – nur, worauf sie aufbaut"),
    ("everything", "alles – eine Kopie mit Gedächtnis und Gelerntem"),
]


def _size(folder: Path) -> str:
    total = profiles.size(folder)
    return f"{total / 1e6:.1f} MB" if total >= 1e5 else f"{total / 1e3:.0f} kB"


KEEP = ""  # in a choice: as it is (as the template has it)
VOICE_CHOICES = [
    (KEEP, "wie bisher / wie die Vorlage"),
    ("auto", "automatisch (eigene Aufnahme, sonst Ramona)"),
    ("ramona", "Ramona (weiblich)"),
    ("hokuspokus", "Hokuspokus (weiblich)"),
    ("thorsten", "Thorsten (männlich)"),
    ("own", "die eigene Aufnahme"),
]
PITCH_CHOICES = [(KEEP, "wie bisher")] + [
    (str(n), f"{n:+d} Halbtöne" if n else "normal") for n in range(-4, 5)
]
SPEED_CHOICES = [(KEEP, "wie bisher")] + [
    (value, label)
    for value, label in (
        ("0.9", "langsamer (0,9)"),
        ("1.0", "normal"),
        ("1.05", "etwas schneller (1,05)"),
        ("1.1", "schneller (1,1)"),
        ("1.2", "deutlich schneller (1,2)"),
    )
]


def _models(settings: Settings) -> list[tuple[str, str]]:
    """The language models Ollama has (for an AI's own model) – not embedding models, not
    variants Cassi made with another context, not models built from a Modelfile (an adapter
    makes those itself, "flay:77a848a1")."""
    from cassi.llm import ollama

    names = ollama.models(settings.llm.base_url)  # empty without Ollama
    made = re.compile(r"(-ctx\d+k|:[0-9a-f]{8})$")
    shown = sorted(
        n for n in names if not ("embed" in n or n.endswith(":latest") or made.search(n))
    )
    return [(KEEP, "wie bisher / wie die Vorlage")] + [(n, n) for n in shown]


def _definition(chosen: Mapping[str, Any]) -> profiles.Definition:
    def text(key: str) -> str:
        return str(chosen.get(key) or "").strip()

    def number(key: str) -> float | None:
        return float(text(key)) if text(key) else None

    return profiles.Definition(
        name=text("persona_name"),
        identity=text("identity"),
        speaking_style=text("speaking_style"),
        voice=text("voice"),
        pitch=number("pitch"),
        speed=number("speed"),
        model=text("model"),
    )


def _definition_options(models: list[tuple[str, str]]) -> tuple[Option, ...]:
    from cassi.manage import Option

    return (
        Option("persona_name", "Name der KI", "", kind="text", placeholder="z. B. Kasimir"),
        Option(
            "identity",
            "Wer sie ist",
            "",
            kind="longtext",
            placeholder="z. B. A calm chess coach who loves endgames and explains moves simply.",
            help="Charakter, Wissen, Haltung – auf Englisch folgen kleine Modelle am besten "
            "(geantwortet wird trotzdem auf Deutsch). Leer: bleibt.",
        ),
        Option(
            "speaking_style",
            "Wie sie spricht",
            "",
            kind="text",
            placeholder="z. B. Short, patient, informal (du).",
        ),
        Option(
            "voice",
            "Stimme",
            KEEP,
            kind="choice",
            choices=VOICE_CHOICES,
            help="Gilt für Qwen3-TTS (macOS); Tonhöhe und Tempo auch für Piper.",
        ),  # fmt: skip
        Option("pitch", "Tonhöhe", KEEP, kind="choice", choices=PITCH_CHOICES),
        Option("speed", "Tempo", KEEP, kind="choice", choices=SPEED_CHOICES),
        Option(
            "model",
            "Sprachmodell",
            KEEP,
            kind="choice",
            choices=models,
            help="Die Modelle, die Ollama schon hat (neue lädt der Tab „Modelle“).",
        ),
    )


def profile_tab(
    app: App,
    settings: Settings,
    config: Path | None,
    now: Callable[[], Settings],
    quit_soon: Callable[[], None],
) -> Tab:
    from cassi.manage import Action, Launcher, Option, Row, Section, Tab

    base = settings.shared_dir
    active = settings.profile or profiles.BASE

    def in_use() -> str:
        return now().profile or active

    def sections() -> list[Section]:
        current = now()
        rows = []
        for profile in profiles.available(base):
            built = ", ".join(profile.adapters) or "Cassi"
            detail = f"auf {built} · {_size(profile.folder)}"
            if profile.description:
                detail = f"{profile.description} · {detail}"
            used = profile.name == in_use()
            label = profile.label + (" (in Gebrauch)" if used else "")
            rows.append(Row(label, detail, "ok" if used else ""))
        voice = current.tts.voice
        ai = [
            Row("Name", current.persona.name),
            Row("Sprachmodell", current.llm.model),
            Row("Stimme", f"{voice.name}, Tonhöhe {voice.pitch:+g}, Tempo {voice.speed:g}"),
            Row("Ordner", str(current.data_dir)),
        ]
        return [
            Section(
                "KI-Profile",
                rows,
                "Jedes Profil ist eine eigene KI: eigene Definition (Name, Persönlichkeit, "
                "Stimme, Modell), eigene Erinnerungen (Gedächtnis, Verlauf) und eigenes "
                "Gelerntes (Notizen, Reflexe, Tasten). Gemeinsam bleiben cassi.yaml (Geräte, "
                "Server), heruntergeladene Modelle und der Export-Ordner. Alles lässt sich hier "
                "anlegen und ändern – ohne Dateien zu bearbeiten.",
            ),
            Section("Die KI in Gebrauch", ai),
        ]

    def restart(out: Output, name: str) -> None:
        """Use ``name`` from now on: the assistant (if it runs) and this page start again."""
        os.environ[profiles.ENVIRONMENT] = name  # for what this process starts
        restart_assistant(out)
        out("Die Verwaltung startet neu (mit den Tabs des Profils) – neues Fenster öffnet sich.")
        # A little later, when this one is gone (it gives up its address when it ends).
        command = [sys.executable, "-c", _LATER, *app.command("manage")]
        kwargs: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(command, stderr=subprocess.DEVNULL, **kwargs)
        quit_soon()

    def restart_assistant(out: Output) -> None:
        launcher = Launcher(app, settings, config)
        if launcher.running():
            problem = launcher.stop() or launcher.start()
            out(f"✗ {app.title} nicht neu gestartet: {problem}" if problem else
                f"✓ {app.title} startet neu")  # fmt: skip

    def switch(out: Output, chosen: Mapping[str, Any]) -> bool:
        try:
            profile = profiles.find(base, str(chosen.get("name") or ""))
        except profiles.ProfileError as exc:
            out(f"✗ {exc}")
            return False
        profiles.choose(base, profile.name)
        out(f"✓ Profil „{profile.label}“ gewählt ({profile.folder})")
        restart(out, profile.name)
        return True

    def create(out: Output, chosen: Mapping[str, Any]) -> bool:
        title = str(chosen.get("title") or "").strip()
        try:
            template = profiles.find(base, str(chosen.get("template") or active))
            profile = profiles.create(
                base,
                title,
                str(chosen.get("description") or ""),
                template,
                str(chosen.get("copy") or "definition"),  # type: ignore[arg-type]
            )
        except (profiles.ProfileError, OSError) as exc:
            out(f"✗ {exc}")
            return False
        out(f"✓ Profil „{profile.label}“ angelegt, ausgehend von „{template.label}“")
        out(f"  Ordner: {profile.folder}")
        if definition := _definition(chosen):
            try:
                profiles.define(base, profile.name, definition)
            except (profiles.ProfileError, OSError, ValueError) as exc:
                out(f"✗ Definition nicht gespeichert: {exc} – „KI bearbeiten“ versucht es neu")
                return False
            out("✓ Name, Persönlichkeit, Stimme und Modell wie angegeben gespeichert")
        if chosen.get("switch"):
            profiles.choose(base, profile.name)
            restart(out, profile.name)
        else:
            out("Wechseln: „Profil wechseln“ unten.")
        return True

    def edit(out: Output, chosen: Mapping[str, Any]) -> bool:
        name = str(chosen.get("name") or in_use())
        definition = _definition(chosen)
        title = str(chosen.get("title") or "")
        description = str(chosen.get("description") or "")
        if not (definition or title.strip() or description.strip()):
            out("Nichts geändert – leere Felder bleiben, wie sie sind.")
            return True
        try:
            profile = profiles.define(base, name, definition, title, description)
        except (profiles.ProfileError, OSError, ValueError) as exc:
            out(f"✗ {exc}")
            return False
        out(f"✓ „{profile.label}“ geändert")
        if profile.name == in_use():
            restart_assistant(out)
            out("Die Anzeige oben zeigt die Änderung nach dem Neuladen der Verwaltung.")
        return True

    def remove(out: Output, chosen: Mapping[str, Any]) -> bool:
        try:
            archive = profiles.remove(base, str(chosen.get("name") or ""), in_use())
        except (profiles.ProfileError, OSError) as exc:
            out(f"✗ {exc}")
            return False
        out(f"✓ Gelöscht – zur Sicherheit aufbewahrt in {archive}")
        return True

    def open_folder(out: Output, _: Mapping[str, Any]) -> bool:
        from cassi.maintenance import open_folder as reveal

        folder = now().data_dir
        out(f"Öffne {folder}")
        return reveal(folder)

    def actions() -> list[Action]:
        # Made anew each time: a new profile is in the lists right after it was made.
        known = profiles.available(base)
        choices = [(p.name, p.label) for p in known]
        own = [(p.name, p.label) for p in known if not p.builtin]
        editable = [(p.name, p.label) for p in known if p.name != profiles.BASE]
        current = in_use()
        definition = _definition_options(_models(now()))
        return [
            Action(
                "create",
                "Neue KI anlegen",
                "Eine weitere KI, getrennt von den anderen: eigener Name, eigene Persönlichkeit, "
                "Stimme und Modell, eigene Erinnerungen. Sie geht von einem Profil aus – worauf "
                "es aufbaut (z. B. Flay) und, wenn gewünscht, seine Definition oder alles samt "
                "Erinnerungen. Leere Felder: wie die Vorlage.",
                create,
                options=(
                    Option("title", "Name des Profils", "", kind="text", placeholder="Schach-Flay"),
                    Option("description", "Wofür", "", kind="text", placeholder="optional"),
                    Option("template", "Ausgehend von", current, kind="choice", choices=choices),
                    Option("copy", "Übernehmen", "definition", kind="choice", choices=COPY_CHOICES),
                    *definition,
                    Option("switch", "danach zu dieser KI wechseln", True),
                ),
            ),
            Action(
                "switch",
                "Profil wechseln",
                f"{app.title} arbeitet ab jetzt mit dieser KI – ihrer Definition, ihren "
                "Erinnerungen und dem, was sie gelernt hat. Läuft der Assistent, startet er neu.",
                switch,
                options=(Option("name", "Profil", current, kind="choice", choices=choices),),
            ),
            Action(
                "edit",
                "KI bearbeiten",
                "Name, Persönlichkeit, Stimme oder Modell einer KI ändern; leere Felder bleiben, "
                "wie sie sind. Ihre Erinnerungen bleiben. Ist sie in Gebrauch, startet der "
                "Assistent neu. (Cassi selbst: Tab „Einstellungen“.)",
                edit,
                options=(
                    Option(
                        "name",
                        "KI",
                        current if current != profiles.BASE else "",
                        kind="choice",
                        choices=editable,
                    ),
                    Option(
                        "title",
                        "Neuer Name des Profils",
                        "",
                        kind="text",
                        placeholder="leer: bleibt",
                    ),  # fmt: skip
                    Option("description", "Wofür", "", kind="text", placeholder="leer: bleibt"),
                    *definition,
                ),
            ),
            Action("folder", "Ordner öffnen", "Der Ordner des Profils in Gebrauch.", open_folder),
            Action(
                "remove",
                "Profil löschen",
                "Entfernt ein eigenes Profil samt Erinnerungen (eingebaute und das in Gebrauch "
                "nicht). Vorher wird es als Zip unter backups aufbewahrt.",
                remove,
                confirm="Profil mit allen Erinnerungen und allem Gelernten löschen?",
                danger=True,
                options=(Option("name", "Profil", "", kind="choice", choices=own),),
            ),
        ]

    return Tab("profiles", "Profile", sections, make_actions=actions)


# Started detached: waits until the old management UI is gone, then starts the new one.
_LATER = "import subprocess, sys, time; time.sleep(4); subprocess.Popen(sys.argv[1:])"
