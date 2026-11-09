"""Generated training rows for the LoRA adapter: commands as speech recognition hears them.

Real examples come slowly (a few per day), and the failures that matter are garbled ones:
"Slie K X E V O" for "Schließe kx-ewo", "She K Command". These rows teach the pattern with
invented projects – none of the test set's (``eval/cases.yaml``), so ``flay eval`` still
measures what the adapter learned rather than what it memorized.

Each row has its own situation (projects open in PhpStorm, Claude sessions), so the model
learns to map a garbled name onto the listed project, not onto a fixed list.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from flay.config import Settings
from flay.llm.base import ToolSpec
from flay.training import row, tool_call_text

# Invented, in the style of real ones; none of them is in eval/cases.yaml.
PROJECTS = [
    "project-symfony-kx-nvs", "project-symfony-kx-kpfb", "component-symfony-kx-mailer",
    "lib-react-kx-shop-app", "rk-portal", "billing-api", "shop-frontend", "nordwind",
    "atlas-cms", "media-hub", "stadt-sonnenberg", "orion", "lohnrechner", "event-app",
    "project-laravel-ln-portal", "wfm-cms",
]  # fmt: skip

OPEN = [
    "Öffne {n}", "Öffne das Projekt {n}", "Mach {n} auf", "Mach mal {n} auf", "Open {n}",
    "Open the project {n}", "Öffne in PhpStorm das Projekt {n}", "Öffen {n}", "Öff ne {n}",
    "Offne Projekt {n}", "Projekt {n} öffnen", "Kannst du {n} öffnen?", "Projekt {n}.",
]  # fmt: skip
CLOSE = [
    "Schließe {n}", "Schließ das Projekt {n}", "Mach {n} zu", "Close {n}", "Slie {n}",
    "Sließ {n}", "She {n}", "Schlies {n}", "Schließe bitte {n}", "Projekt {n} schließen",
    "Close the {n} project", "Sließ the {n}", "Schließe das Projekt {n} wieder",
]  # fmt: skip
TASKS = [
    ("er soll die Tests reparieren", "Repariere die fehlschlagenden Tests."),
    ("er soll die Readme zusammenfassen", "Fasse die README zusammen."),
    ("er soll den Changelog aktualisieren", "Aktualisiere den Changelog."),
    ("er soll den Login-Fehler suchen", "Finde die Ursache des Login-Fehlers."),
    ("er soll die Abhängigkeiten aktualisieren", "Aktualisiere die Abhängigkeiten."),
    ("warum der Build fehlschlägt", "Finde heraus, warum der Build fehlschlägt."),
    ("to add type hints", "Add type hints."),
    ("to write tests for the export", "Write tests for the export."),
]
CLAUDE = [
    "Sag Claude im Projekt {n}, {t}", "Sagt Claude im Projekt {n}, {t}",
    "Schreib Claude in {n}, {t}", "Frag Claude im {n} Projekt, {t}", "Tell Claude in {n} {t}",
    "Sag Cloud im Projekt {n}, {t}",
]  # fmt: skip
# Talk that names projects and verbs but asks for nothing.
TALK = [
    ("Ich hab {n} gestern noch geöffnet.", "Und, bist du weitergekommen?"),
    ("Das Projekt {n} ist geöffnet.", "Gut, dann kann es losgehen."),
    ("Wie öffne ich ein Projekt in PhpStorm?", "Über File, Open – oder sag mir einfach welches."),
    ("{n} habe ich heute schon geschlossen.", "Alles klar."),
    ("Was ist eigentlich {n} für ein Projekt?", "Das weiß ich nicht genau – frag gern Claude."),
]


def _without_prefix(name: str) -> str:
    return re.sub(r"^(?:project|component|lib)-(?:symfony|laravel|react)-", "", name)


def spoken(name: str, rng: random.Random) -> str:
    """The project name as speech recognition writes it, with typical errors."""
    short = _without_prefix(name) if rng.random() < 0.8 else name
    parts = short.split("-")
    words = []
    for part in parts:
        if len(part) <= 4 and part.isalpha() and rng.random() < 0.6:
            letters = list(part.upper())
            if rng.random() < 0.3 and len(letters) > 2:
                letters[rng.randrange(len(letters))] = rng.choice("DTEVBP")  # misheard
            words.append(" ".join(letters) if rng.random() < 0.7 else "".join(letters))
        else:
            words.append(part.capitalize() if rng.random() < 0.5 else part)
    text = " ".join(words)
    if rng.random() < 0.2:
        text = text.replace("K X", "C D").replace("KX", "CD")
    return text


def _state(rng: random.Random, must: str) -> tuple[str, list[str]]:
    open_projects = rng.sample(PROJECTS, rng.randint(2, 5))
    if must not in open_projects:
        open_projects[0] = must
    sessions = rng.sample(open_projects, rng.randint(1, len(open_projects)))
    line = f"Open in PhpStorm: {', '.join(open_projects)}. Claude Code running for: " + ", ".join(
        f"{s} (shown in PhpStorm)" for s in sessions
    )
    return line + ".", open_projects


def synthetic_rows(
    settings: Settings, specs: Sequence[ToolSpec], count: int, seed: int = 11
) -> list[dict[str, Any]]:
    """``count`` rows: about 70 % actions (open, close, Claude), 30 % talk."""
    rng = random.Random(seed)
    now = datetime.now().astimezone()
    rows = []
    for _ in range(count):
        project = rng.choice(PROJECTS)
        state, _ = _state(rng, project)
        name = spoken(project, rng)
        kind = rng.random()
        if kind < 0.25:
            request = rng.choice(OPEN).format(n=name)
            answer = tool_call_text("open_project", {"name": project})
        elif kind < 0.5:
            request = rng.choice(CLOSE).format(n=name)
            answer = tool_call_text("close_project", {"name": project})
        elif kind < 0.7:
            task, instruction = rng.choice(TASKS)
            request = rng.choice(CLAUDE).format(n=name, t=task)
            answer = tool_call_text(
                "send_to_claude", {"instruction": instruction, "project": project}
            )
        else:
            said, reply = rng.choice(TALK)
            request, answer = said.format(n=name), reply
        if rng.random() < 0.2:
            request = request.rstrip(".?!") + "."
        rows.append(row(settings, specs, request, answer, state, now))
    return rows
