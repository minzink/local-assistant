"""The management UI's "Stimme" tab: train Cassi's Piper voice (Windows, NVIDIA).

Training takes about a day, longer than the management UI lives, so it runs in its own window
(``tools/voice-training/run.ps1``). That script writes its state to ``work/status.json``; the
tab reads it, the epoch reached from the checkpoints, and which trained voices are installed.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cassi import maintenance
from cassi.app import App
from cassi.maintenance import Output

if TYPE_CHECKING:
    from cassi.config import Settings
    from cassi.manage import Level, Tab

log = logging.getLogger(__name__)

NAME = "cassi-ramona"
QUALITIES = ("medium", "high")
_EPOCH = re.compile(r"epoch=(\d+)")


def training_dir(app: App) -> Path | None:
    """``tools/voice-training`` of the source checkout, if there is one."""
    if app.source is None:
        return None
    folder = app.source / "tools" / "voice-training"
    return folder if (folder / "run.ps1").is_file() else None


def read_status(work: Path) -> dict[str, Any] | None:
    try:
        # PowerShell 5 writes UTF-8 with a byte order mark.
        data = json.loads((work / "status.json").read_text("utf-8-sig"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def epoch_reached(runs: Path) -> int | None:
    """The highest epoch a checkpoint was saved for (Lightning names them ``epoch=N-…``)."""
    epochs = [int(m.group(1)) for p in runs.rglob("*.ckpt") if (m := _EPOCH.search(p.name))]
    return max(epochs) + 1 if epochs else None


def running() -> bool:
    return any(
        "voice-training" in command and "run.ps1" in command
        for _, _, command in maintenance._processes()
    )


def _when(stamp: object) -> str:
    try:
        return datetime.fromisoformat(str(stamp)).strftime("%d.%m. %H:%M")
    except ValueError:
        return "?"


def _state(status: Mapping[str, Any], active: bool) -> tuple[str, str, Level]:
    if active:
        return "Zustand", "läuft", ""
    if status.get("ok"):
        return "Zustand", "fertig", "ok"
    # Not done and not running: its window was closed (or the computer restarted).
    return "Zustand", "fehlgeschlagen" if status.get("done") else "abgebrochen", "warn"


def _installed(voices: Path) -> list[tuple[str, float]]:
    found = []
    for quality in QUALITIES:
        model = voices / f"{NAME}-{quality}.onnx"
        if model.is_file():
            found.append((model.stem, model.stat().st_mtime))
    return found


def voice_tab(app: App, now: Callable[[], Settings]) -> Tab:
    from cassi.manage import Action, Option, Row, Section, Tab
    from cassi.tools.preferences import Preferences

    folder = training_dir(app)
    work = folder / "work" if folder else None
    title = app.title

    def problems() -> list[str]:
        found = []
        if sys.platform != "win32":
            found.append("Das Training läuft nur unter Windows mit NVIDIA-Grafikkarte.")
        elif shutil.which("nvidia-smi") is None:
            found.append("nvidia-smi fehlt: zuerst den NVIDIA-Treiber installieren.")
        if folder is None:
            found.append("Kein Cassi-Quellordner mit tools/voice-training gefunden.")
        return found

    def sections() -> list[Section]:
        settings = now()
        voices = settings.downloads / "voices"
        blockers = problems()
        result = [
            Section(
                "Stimme trainieren",
                [
                    Row("Voraussetzungen", " ".join(blockers), "fail")
                    if blockers
                    else Row("Voraussetzungen", "erfüllt", "ok")
                ],
                "Trainiert die Piper-Stimme Ramona, mit der "
                f"{title} unter Windows spricht, vollständig (2000 Epochen) – sie klingt danach "
                "deutlich sauberer. Dauert grob einen Tag und belegt so lange die Grafikkarte; "
                f"{title} dafür beenden. Läuft in einem eigenen Fenster weiter, auch wenn diese "
                "Seite zu ist. Abgebrochen? Erneut starten setzt fort. „Hohe Qualität“ klingt "
                "noch natürlicher, braucht aber je Satz länger bis zum ersten Ton.",
            )
        ]
        status = read_status(work) if work else None
        if status is not None and work is not None:
            rows = [
                Row(*_state(status, running())),
                Row("Schritt", str(status.get("text", "?"))),
                Row("Qualität", str(status.get("quality", "?"))),
                Row("Gestartet", _when(status.get("started"))),
                Row("Zuletzt", _when(status.get("updated"))),
            ]
            runs = work / ("runs-high" if status.get("quality") == "high" else "runs")
            if status.get("step") == "train" and (epoch := epoch_reached(runs)) is not None:
                rows.insert(2, Row("Epoche", f"{epoch} von {status.get('epochs', '?')}"))
            result.append(Section("Letztes Training", rows))
        installed = _installed(voices)
        chosen = Preferences.load(settings.data_dir).voice
        result.append(
            Section(
                "Trainierte Stimmen",
                [
                    Row(
                        name,
                        f"vom {datetime.fromtimestamp(t):%d.%m.%Y %H:%M} – in Verwendung",
                        "ok",
                    )
                    if name == chosen
                    else Row(name, f"vom {datetime.fromtimestamp(t):%d.%m.%Y %H:%M}")
                    for name, t in installed
                ]
                or [Row("Keine", f"noch keine trainierte Stimme in {voices}")],
            )
        )
        return result

    def start(out: Output, chosen: Mapping[str, Any]) -> bool:
        if blockers := problems():
            for blocker in blockers:
                out(f"✗ {blocker}")
            return False
        if running():
            out("Das Training läuft schon in seinem eigenen Fenster.")
            return True
        assert folder is not None
        quality = "high" if chosen.get("high") else "medium"
        voices = now().downloads / "voices"
        command = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(folder / "run.ps1"), "-Quality", quality, "-VoicesDir", str(voices),
        ]  # fmt: skip
        if not maintenance.in_own_window(command):
            out("✗ Das Training ließ sich nicht starten")
            return False
        out(f"✓ Training ({quality}) gestartet – es läuft in einem eigenen Fenster.")
        out("Den Fortschritt zeigt dieser Tab (neu laden); das Fenster nicht schließen.")
        return True

    def use(out: Output, chosen: Mapping[str, Any]) -> bool:
        settings = now()
        name = f"{NAME}-{'high' if chosen.get('high') else 'medium'}"
        if not (settings.downloads / "voices" / f"{name}.onnx").is_file():
            out(f"✗ {name} ist noch nicht trainiert.")
            return False
        preferences = Preferences.load(settings.data_dir)
        preferences.voice = name
        preferences.save()
        out(f"✓ {title} spricht mit {name}, sobald es neu gestartet ist (Piper, also Windows).")
        return True

    def open_work(out: Output, _: Mapping[str, Any]) -> bool:
        if work is None or not maintenance.open_folder(work):
            out("Noch kein Trainingsordner – das Training wurde nicht gestartet.")
            return False
        out(f"Geöffnet: {work} (run.log, status.json)")
        return True

    high = Option("high", "Hohe Qualität (natürlicher, erster Ton später)")
    return Tab(
        "voice",
        "Stimme",
        sections,
        [
            Action(
                "voice-train",
                "Training starten",
                "Einrichten, Datensatz, Training, Export – alles in einem eigenen Fenster.",
                start,
                confirm=f"Training starten? Es dauert grob einen Tag; {title} vorher beenden.",
                options=(high,),
            ),
            Action(
                "voice-use",
                "Trainierte Stimme verwenden",
                f"{title} spricht nach dem nächsten Start mit der trainierten Stimme.",
                use,
                options=(high,),
            ),
            Action("voice-folder", "Trainingsordner öffnen", "Log und Zwischenstände.", open_work),
        ],
    )
