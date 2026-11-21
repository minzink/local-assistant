"""``cassi manage``: a small local web page to look after the installation.

The header starts the assistant (or opens its chat, if it runs). Built-in tabs: an overview
(version, folders, command, autostart, models), maintenance (diagnose, repair, look for and
install updates, autostart, open folders) and uninstall. Adapters add their own tabs with
``Adapter.manage_tabs`` – described as data (``Tab``, ``Section``, ``Row``, ``Action``), which
the page renders, so an adapter writes no HTML and cannot break the page.

Safety: the server listens on 127.0.0.1 only, on a random port. Every API call needs the
token from the URL the command opens (in the fragment, so it is never sent in a request
line or logged), and actions must come from the page's own origin – other websites in the
browser cannot start anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aiohttp import web

from cassi import maintenance
from cassi.app import App
from cassi.maintenance import Output

if TYPE_CHECKING:
    from cassi.adapters import Adapter
    from cassi.config import Settings

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
MAX_TEXT = 500  # characters of a text option
Level = Literal["", "ok", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class Row:
    label: str
    value: str
    # Shown as ✓ / ! / ✗ in front of the value.
    level: Level = ""


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    rows: Sequence[Row] = ()
    # A paragraph below the rows (plain text).
    text: str = ""


@dataclass(frozen=True, slots=True)
class Option:
    """An input shown with an action; ``run`` gets its value by ``id``: a checkbox (bool),
    a text field, a key combination (a text field that can also record keys pressed, as
    "ctrl+alt+r") or a choice among ``choices`` (value, label) – all but the checkbox str."""

    id: str
    label: str
    default: bool | str = False
    kind: Literal["check", "text", "keys", "choice"] = "check"
    placeholder: str = ""
    choices: Sequence[tuple[str, str]] = ()
    # A hint below the field.
    help: str = ""

    def value(self, raw: object) -> bool | str:
        if self.kind == "check":
            return bool(self.default if raw is None else raw)
        text = str(self.default if raw is None else raw).strip()[:MAX_TEXT]
        if self.kind == "choice" and text not in {value for value, _ in self.choices}:
            return str(self.default)
        return text


@dataclass(frozen=True, slots=True)
class Action:
    id: str
    label: str
    description: str
    # Runs in a worker thread; each call of the output function adds a line to the page.
    # True: it worked.
    run: Callable[[Output, Mapping[str, Any]], bool]
    # Asked before running (e.g. "Wirklich deinstallieren?").
    confirm: str = ""
    # Shown in red: removes something.
    danger: bool = False
    options: Sequence[Option] = ()
    # Asks for a file first (its label, e.g. "Export-Datei (.zip)"); ``run`` gets its local
    # path as ``options["file"]``. ``accept`` limits the choice, e.g. ".zip".
    file: str = ""
    accept: str = ""


@dataclass(frozen=True, slots=True)
class Tab:
    id: str
    title: str
    # Called whenever the tab is shown, in a worker thread: may ask git, Ollama, files.
    sections: Callable[[], Sequence[Section]]
    actions: Sequence[Action] = ()


def _command_args(config: Path | None) -> list[str]:
    return ["-c", str(config)] if config is not None else []


def builtin_tabs(
    app: App,
    settings: Settings,
    adapters: Sequence[Adapter],
    config: Path | None,
    quit_soon: Callable[[], None],
    current: Callable[[], Settings] | None = None,
) -> list[Tab]:
    """``current``: the configuration as it is now – the management UI runs for hours, and
    cassi.yaml or an adapter's model may change meanwhile (else ``settings`` as at start)."""

    def now() -> Settings:
        if current is None:
            return settings
        try:
            return current()
        except Exception:  # an invalid cassi.yaml must not break the page
            log.warning("Konfiguration nicht neu ladbar – nutze die vom Start", exc_info=True)
            return settings

    from cassi import autostart
    from cassi.config import resolve_config_path
    from cassi.keyboard_tab import keyboard_tab
    from cassi.voice_training import voice_tab

    options = _command_args(config)

    def overview() -> list[Section]:
        settings = now()
        from cassi.llm import ollama

        shim = maintenance.shim_path(app)
        config_path = resolve_config_path(config)
        data = settings.data_dir
        size = maintenance.human_size(maintenance.folder_size(data)) if data.exists() else ""
        try:
            autostart_on = autostart.is_enabled()
        except RuntimeError:
            autostart_on = False
        installation = [
            Row("Programm", f"{app.title} {app.version}"),
            Row("Programmordner", str(app.project or "–")),
        ]
        if app.source is not None and app.source != app.project:
            installation.append(Row("Cassi (Quelle)", str(app.source)))
        installation += [
            Row(
                "Befehl",
                str(shim) if shim.exists() else "nicht eingerichtet – „Reparieren“ richtet ihn ein",
                "ok" if shim.exists() else "warn",
            ),
            Row("Konfiguration", str(config_path) if config_path else "keine (Standardwerte)"),
            Row("Datenordner", f"{data} ({size})" if size else f"{data} (noch leer)"),
            Row("Autostart", "an" if autostart_on else "aus"),
            Row("Adapter", ", ".join(a.name for a in adapters) or "keine"),
        ]
        versions = []
        for root in app.repositories():
            state = maintenance.repo_state(root)
            if state is None:
                versions.append(Row(root.name, "kein Git-Stand lesbar", "warn"))
                continue
            value = f"{state.commit} auf {state.branch}"
            level: Level = "ok"
            if state.changes:
                value += f", {state.changes} geänderte Datei(en)"
                level = "warn"
            if state.behind:
                value += f", {state.behind} Update(s) verfügbar"
                level = "warn"
            versions.append(Row(root.name, value, level))
        url = settings.llm.base_url
        model_rows = []
        if ollama.is_ollama(url):
            have = ollama.models(url)
            model_rows.append(Row("LLM-Server", f"Ollama auf {url}", "ok"))
            present = settings.llm.model in have
            model_rows.append(
                Row(
                    "Sprachmodell",
                    settings.llm.model if present else f"{settings.llm.model} fehlt – „Reparieren“",
                    "ok" if present else "fail",
                )
            )
        else:
            model_rows.append(Row("LLM-Server", f"{url} antwortet nicht – Ollama starten", "fail"))
        return [
            Section("Installation", installation),
            Section(
                "Versionen",
                versions,
                "" if versions else "Kein Git-Checkout – Updates über eine neue Installation.",
            ),
            Section("Sprachmodell", model_rows),
        ]

    def repair(out: Output, _: Mapping[str, Any]) -> bool:
        command = maintenance.setup_command(app)
        if command is None:
            out("✗ Kein Quellordner mit scripts/setup gefunden – Cassi neu installieren")
            return False
        if sys.platform == "win32":
            # Packages in use cannot be replaced on Windows: repair in its own window.
            if not maintenance.in_own_window(command):
                out("✗ Reparatur ließ sich nicht starten")
                return False
            out("Die Reparatur läuft in einem eigenen Fenster. Die Verwaltung schließt sich;")
            out(f"danach neu öffnen mit: {app.name} manage")
            quit_soon()
            return True
        ok = maintenance.run_process(command, out)
        out("✓ Repariert" if ok else "✗ Reparatur fehlgeschlagen – Details oben und im Log")
        return ok

    def update(out: Output, options_: Mapping[str, Any]) -> bool:
        if not maintenance.pull(app, out):
            return False
        out("Neuer Stand geladen – jetzt Pakete und Modelle prüfen:")
        ok = repair(out, options_)
        if ok and sys.platform != "win32":
            out(f"Fertig. {app.title} neu starten, damit die Änderungen wirken; die Verwaltung")
            out("zeigt den neuen Stand nach dem Neuladen der Seite.")
        return ok

    def doctor(out: Output, _: Mapping[str, Any]) -> bool:
        return maintenance.run_process(app.command(*options, "doctor"), out)

    def check(out: Output, _: Mapping[str, Any]) -> bool:
        return maintenance.check_updates(app, out)

    def autostart_on(out: Output, _: Mapping[str, Any]) -> bool:
        return maintenance.run_process(app.command(*options, "autostart", "on"), out)

    def autostart_off(out: Output, _: Mapping[str, Any]) -> bool:
        return maintenance.run_process(app.command(*options, "autostart", "off"), out)

    def open_data(out: Output, _: Mapping[str, Any]) -> bool:
        settings = now()
        ok = maintenance.open_folder(settings.data_dir)
        out(f"Geöffnet: {settings.data_dir}" if ok else "Der Datenordner existiert noch nicht")
        return ok

    def open_logs(out: Output, _: Mapping[str, Any]) -> bool:
        folder = (app.source or Path.cwd()) / "logs"
        ok = maintenance.open_folder(folder)
        out(f"Geöffnet: {folder}" if ok else f"Noch keine Logs in {folder}")
        return ok

    def remove(out: Output, chosen: Mapping[str, Any]) -> bool:
        settings = now()
        return maintenance.uninstall(
            app,
            settings,
            adapters,
            out,
            data=chosen.get("data", False),
            models=chosen.get("models", False),
        )

    def uninstall_info() -> list[Section]:
        settings = now()
        data = settings.data_dir
        size = maintenance.human_size(maintenance.folder_size(data)) if data.exists() else "leer"
        return [
            Section(
                "Was entfernt wird",
                [
                    Row("Immer", "Autostart, der Befehl " + app.name + ", Spuren der Adapter"),
                    Row("Auf Wunsch", f"Datenordner ({size}): Gedächtnis, Aufnahmen, Stimmen"),
                    Row(
                        "Auf Wunsch",
                        "Modelle in Ollama: " + ", ".join(maintenance.ollama_models(settings)),
                    ),  # fmt: skip
                    Row("Selbst löschen", str(app.project or "den Programmordner")),
                ],
                "uv, Ollama und Git bleiben installiert, ebenso Modelle im Hugging-Face-Cache – "
                "andere Programme können sie nutzen. Läuft der Assistent, vorher beenden.",
            )
        ]

    def data_info() -> list[Section]:
        settings = now()
        from cassi import transfer

        folder = settings.data_dir
        personal = transfer.contents(folder)
        everything = transfer.contents(folder, recordings=True)
        size = maintenance.human_size
        rows = [
            Row("Datenordner", str(folder)),
            Row(
                "Im Export",
                f"{personal.files} Dateien, {size(personal.size)} – Gedächtnis, Verlauf, "
                "Abläufe, Einstellungen, eigene Modelle und Stimme",
            ),
            Row(
                "Aufnahmen (auf Wunsch)",
                f"{everything.files - personal.files} Dateien, "
                f"{size(everything.size - personal.size)}",
            ),
        ]
        if settings.downloads != folder:
            rows.append(Row("Downloads (geteilt, nicht im Export)", str(settings.downloads)))
        backups = sorted((folder / "backups").glob("*.zip"))
        if backups:
            rows.append(Row("Letzte Sicherung", backups[-1].name))
        target = transfer.export_folder(app, settings)
        problem = transfer.check_folder(target)
        source = (
            "gewählt"
            if transfer.chosen_folder(app)
            else "aus cassi.yaml"
            if settings.export_dir
            else "Standard"
        )
        exports = transfer.exports_in(app, target)
        places = [
            Row(
                "Export-Ordner",
                f"{target} ({source})" + (f" – {problem}" if problem else ""),
                "fail" if problem else "ok",
            ),
            Row(
                "Exporte dort",
                f"{len(exports)}, neuester: {exports[0].name}" if exports else "keine",
            ),
        ]
        return [
            Section(
                "Persönliche Daten",
                rows,
                "Gedächtnis und eigene Modelle liegen nur hier, nie im Repository. Mit dem "
                "Export nimmst du sie auf einen anderen Rechner mit oder sicherst sie; "
                "heruntergeladene Modelle lädt dort das Setup.",
            ),
            Section(
                "Übertragen",
                places,
                "Für einen anderen Rechner: hier den Export-Ordner auf die externe Festplatte "
                "stellen und exportieren; dort denselben Ordner wählen und den neuesten Export "
                "importieren. Die Wahl gilt nur für diesen Rechner.",
            ),
        ]

    def export_data(out: Output, chosen: Mapping[str, Any]) -> bool:
        settings = now()
        from cassi import transfer

        folder = transfer.export_folder(app, settings)
        if problem := transfer.check_folder(folder):
            out(f"✗ {problem}")
            return False
        target = transfer.default_target(app, settings)
        done = transfer.export(app, settings, target, recordings=bool(chosen.get("recordings")))
        out(f"✓ {done.files} Dateien ({maintenance.human_size(done.size)}) exportiert:")
        out(f"  {target}")
        maintenance.open_folder(target.parent)
        return True

    def import_data(out: Output, chosen: Mapping[str, Any]) -> bool:
        settings = now()
        from cassi import transfer

        if Launcher(app, settings, config).running():
            out(f"✗ {app.title} läuft – erst oben mit „Beenden“ schließen, dann importieren.")
            return False
        source = chosen.get("file")
        if source is None:  # the newest export in the export folder
            folder = transfer.export_folder(app, settings)
            if problem := transfer.check_folder(folder):
                out(f"✗ {problem}")
                return False
            exports = transfer.exports_in(app, folder)
            if not exports:
                out(f"✗ Kein {app.title}-Export in {folder}")
                return False
            source = exports[0]
            out(f"Importiere {Path(source).name} …")
        try:
            transfer.import_(app, settings, Path(str(source)), out)
        except transfer.TransferError as exc:
            out(f"✗ {exc}")
            return False
        out(f"Fertig. Beim nächsten Start nutzt {app.title} die importierten Daten.")
        return True

    def pick_folder(out: Output, _: Mapping[str, Any]) -> bool:
        from cassi import transfer

        settings = now()
        out("Der Ordner-Dialog ist auf diesem Rechner geöffnet …")
        folder = maintenance.ask_for_folder(
            f"Ordner für {app.title}-Exporte (z.B. auf einer externen Festplatte)",
            transfer.export_folder(app, settings),
        )
        if folder is None:
            out("Kein Ordner gewählt – alles bleibt, wie es war.")
            return True
        if problem := transfer.check_folder(folder):
            out(f"✗ {problem}")
            return False
        transfer.choose_folder(app, folder)
        out(f"✓ Exporte gehen jetzt nach {folder}")
        return True

    def reset_folder(out: Output, _: Mapping[str, Any]) -> bool:
        from cassi import transfer

        transfer.choose_folder(app, None)
        out(f"✓ Exporte gehen wieder nach {transfer.export_folder(app, now())}")
        return True

    def open_backups(out: Output, _: Mapping[str, Any]) -> bool:
        settings = now()
        folder = settings.data_dir / "backups"
        ok = maintenance.open_folder(folder)
        out(f"Geöffnet: {folder}" if ok else "Noch keine Sicherungen")
        return ok

    def stop_others(out: Output, _: Mapping[str, Any]) -> bool:
        if not end_others(app, out):
            out("Keine weiteren Verwaltungen.")
        return True

    title = app.title
    return [
        Tab("overview", "Übersicht", overview),
        Tab(
            "maintenance",
            "Wartung",
            lambda: [
                Section(
                    "Wartung",
                    text="Reparieren ist immer sicher: Es prüft alles und ergänzt nur, was fehlt.",
                ),
                Section(
                    "Verwaltung",
                    [
                        Row("Diese", f"Prozess {os.getpid()}"),
                        Row(
                            "Weitere",
                            f"{len(others)} – „Andere Verwaltungen beenden“"
                            if (others := maintenance.other_manage_processes(app.name))
                            else "keine",
                            "warn" if others else "ok",
                        ),
                    ],
                    "Es läuft immer nur eine: Ein neuer Aufruf öffnet sie oder ersetzt eine "
                    "ältere Version. Ist kein Fenster mehr offen, beendet sie sich nach etwa "
                    f"{IDLE_SECONDS // 60} Minuten; sofort: „Verwaltung beenden“ oben.",
                ),
            ],
            [
                Action("doctor", "Diagnose", "Prüft GPU, Ollama, Modelle und Kontext.", doctor),
                Action(
                    "repair",
                    "Reparieren",
                    "Installiert fehlende Programme und Pakete, startet Ollama, lädt fehlende "
                    f"Modelle und richtet den Befehl {app.name} ein.",
                    repair,
                ),
                Action("check", "Nach Updates suchen", "Fragt GitHub nach neuen Ständen.", check),
                Action(
                    "update",
                    "Updaten",
                    "Lädt den neuesten Stand (nur ohne eigene Änderungen) und repariert danach.",
                    update,
                    confirm=f"{title} jetzt updaten?",
                ),
                Action(
                    "autostart-on",
                    "Autostart an",
                    f"{title} startet bei der Anmeldung.",
                    autostart_on,
                ),
                Action("autostart-off", "Autostart aus", "Entfernt den Autostart.", autostart_off),
                Action("data", "Datenordner öffnen", "Gedächtnis, Aufnahmen, Stimmen.", open_data),
                Action(
                    "others",
                    "Andere Verwaltungen beenden",
                    f"Beendet weitere laufende {title}-Verwaltungen (z.B. ältere Versionen).",
                    stop_others,
                ),
                Action("logs", "Setup-Logs öffnen", "Protokolle der Installation.", open_logs),
            ],
        ),
        Tab(
            "data",
            "Daten",
            data_info,
            [
                Action(
                    "export",
                    "Exportieren",
                    "Speichert die persönlichen Daten als Zip-Datei im Export-Ordner.",
                    export_data,
                    options=(Option("recordings", "Aufnahmen einschließen (größer)"),),
                ),
                Action(
                    "folder",
                    "Export-Ordner wählen …",
                    "Öffnet den Ordner-Dialog – z.B. für eine externe Festplatte, um die Daten "
                    "auf einen anderen Rechner mitzunehmen.",
                    pick_folder,
                ),
                Action(
                    "folder-reset",
                    "Export-Ordner zurücksetzen",
                    "Exporte gehen wieder in den Download-Ordner (oder export_dir aus cassi.yaml).",
                    reset_folder,
                ),
                Action(
                    "import-latest",
                    "Neuesten Export importieren",
                    "Übernimmt den neuesten Export aus dem Export-Ordner (z.B. von der "
                    "Festplatte). Die bisherigen Daten werden vorher gesichert.",
                    import_data,
                    confirm="Die persönlichen Daten durch den neuesten Export im Export-Ordner "
                    "ersetzen? Die bisherigen werden vorher gesichert.",
                    danger=True,
                ),
                Action(
                    "import",
                    "Importieren",
                    "Übernimmt eine beliebige Export-Datei. Die bisherigen Daten werden vorher "
                    "gesichert.",
                    import_data,
                    confirm="Die persönlichen Daten durch den Export ersetzen? "
                    "Die bisherigen werden vorher gesichert.",
                    danger=True,
                    file="Export-Datei (.zip)",
                    accept=".zip",
                ),
                Action("backups", "Sicherungen öffnen", "Stände vor einem Import.", open_backups),
            ],
        ),
        keyboard_tab(app, now),
        voice_tab(app, now),
        Tab(
            "uninstall",
            "Deinstallieren",
            uninstall_info,
            [
                Action(
                    "uninstall",
                    "Deinstallieren",
                    "Entfernt Autostart und Befehl; Daten und Modelle nur, wenn angehakt.",
                    remove,
                    confirm=f"{title} wirklich deinstallieren?",
                    danger=True,
                    options=(
                        Option("data", "Datenordner löschen (Gedächtnis, Aufnahmen, Stimmen)"),
                        Option("models", "Modelle in Ollama löschen"),
                    ),
                )
            ],
        ),
    ]


class Launcher:
    """Starts the assistant and tells whether it runs: its chat UI answers on its port.

    Shown in the page header, so starting is the first thing one sees.
    """

    def __init__(self, app: App, settings: Settings, config: Path | None) -> None:
        self.app = app
        self.settings = settings
        self.config = config

    @property
    def chat_url(self) -> str | None:
        ui = self.settings.ui
        return f"http://{HOST}:{ui.port}/" if ui.enabled else None

    def running(self) -> bool | None:
        """None: unknown (the chat UI is switched off, nothing to ask)."""
        return self.status()[0]

    def status(self) -> tuple[bool | None, str]:
        """Whether something runs on the chat port, and what is wrong with it ("" if fine).

        An error answer still counts as running – the port is taken, a second start would
        fail – but the chat cannot be opened: e.g. an old install whose files are gone.
        """
        url = self.chat_url
        if url is None:
            return None, ""
        import urllib.error

        from cassi.net import DIRECT

        try:
            with DIRECT.open(url, timeout=0.5):
                return True, ""
        except urllib.error.HTTPError as exc:
            port = self.settings.ui.port
            return True, (
                f"Auf Port {port} antwortet ein Programm mit Fehler {exc.code} – vermutlich eine "
                "alte oder kaputte Installation. Oben mit „Beenden“ schließen, dann neu starten."
            )
        except OSError:
            return False, ""

    def stop(self) -> str | None:
        """End whatever Cassi runs on the chat port – also an old or broken install that
        only answers with errors. Only Cassi itself or an adapter's command is ended,
        never another program that happens to use the port. The reason if not."""
        port = self.settings.ui.port
        pid = maintenance.port_owner(port)
        if pid is None:
            return None if not self.running() else f"Kein Prozess auf Port {port} gefunden"
        command = maintenance.process_command(pid)
        names = {"cassi", self.app.name}
        if not any(re.search(rf"\b{re.escape(name)}\b", command) for name in names):
            return (
                f"Port {port} gehört einem anderen Programm ({command or f'Prozess {pid}'}) – "
                "nicht beendet"
            )
        log.info("Beende Prozess %s auf Port %s: %s", pid, port, command)
        if not maintenance.stop_process(pid):
            return f"Prozess {pid} ließ sich nicht beenden"
        return None

    def start(self) -> str | None:
        """Start ``<program> run`` on its own; the reason if that failed."""
        command = self.app.command(*_command_args(self.config), "run")
        try:
            if sys.platform == "win32":
                flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                subprocess.Popen(command, creationflags=flags)
            else:
                subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
        except OSError as exc:
            return str(exc)
        return None


@dataclass
class Job:
    lines: list[str] = field(default_factory=list)
    done: bool = False
    ok: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)


def _row(row: Row) -> dict[str, str]:
    return {"label": row.label, "value": row.value, "level": row.level}


class ManageServer:
    def __init__(
        self,
        app: App,
        tabs: Sequence[Tab],
        port: int = 0,
        token: str | None = None,
        launcher: Launcher | None = None,
    ) -> None:
        self.app_info = app
        self.launcher = launcher
        self.tabs = {tab.id: tab for tab in tabs}
        self.order = [tab.id for tab in tabs]
        self.token = token or secrets.token_urlsafe(24)
        self.port = port
        self.jobs: dict[str, Job] = {}
        self.last_seen = time.monotonic()
        self.visited = False
        # Files chosen on the page (an export to import), by id until an action takes them.
        self.uploads: dict[str, Path] = {}
        self._upload_dir = Path(tempfile.mkdtemp(prefix="cassi-manage-"))
        self.stopped = asyncio.Event()
        self._runner: web.AppRunner | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.web = web.Application(middlewares=[self._guard])
        self.web.router.add_get("/", self._index)
        self.web.router.add_get("/api/tabs", self._tab_list)
        self.web.router.add_get("/api/tabs/{tab}", self._tab)
        self.web.router.add_post("/api/run", self._run)
        self.web.router.add_post("/api/upload", self._upload)
        self.web.router.add_get("/api/jobs/{job}", self._job)
        self.web.router.add_post("/api/quit", self._quit)
        self.web.router.add_get("/api/status", self._status)
        self.web.router.add_post("/api/start", self._start)
        self.web.router.add_post("/api/stop", self._stop)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}/#token={self.token}"

    def origins(self) -> set[str]:
        return {f"http://{HOST}:{self.port}", f"http://localhost:{self.port}"}

    @web.middleware
    async def _guard(self, request: web.Request, handler: Any) -> web.StreamResponse:
        if request.path.startswith("/api/"):
            if not secrets.compare_digest(request.headers.get("X-Cassi-Token", ""), self.token):
                raise web.HTTPForbidden(text="token")
            # An open page asks every few seconds (status in the header): it is still open.
            self.last_seen = time.monotonic()
            self.visited = True
            if request.method != "GET" and request.headers.get("Origin") not in self.origins():
                raise web.HTTPForbidden(text="origin")
        response: web.StreamResponse = await handler(request)
        return response

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._runner = web.AppRunner(self.web)
        await self._runner.setup()
        site = web.TCPSite(self._runner, HOST, self.port)
        await site.start()
        if self.port == 0:  # the port the system chose
            server = getattr(site, "_server", None)
            sockets = getattr(server, "sockets", None) or []
            self.port = int(sockets[0].getsockname()[1])

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        shutil.rmtree(self._upload_dir, ignore_errors=True)

    def quit_soon(self) -> None:
        """Stop after the page fetched the last lines (from any thread)."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(lambda: loop.call_later(2.0, self.stopped.set))

    async def _index(self, request: web.Request) -> web.Response:
        html = resources.files("cassi.ui").joinpath("static/manage.html").read_text("utf-8")
        return web.Response(text=html, content_type="text/html")

    async def _tab_list(self, request: web.Request) -> web.Response:
        app = self.app_info
        tabs = [{"id": tab_id, "title": self.tabs[tab_id].title} for tab_id in self.order]
        return web.json_response({"title": app.title, "version": app.version, "tabs": tabs})

    async def _tab(self, request: web.Request) -> web.Response:
        tab = self.tabs.get(request.match_info["tab"])
        if tab is None:
            raise web.HTTPNotFound()
        try:
            sections = await asyncio.to_thread(tab.sections)
        except Exception as exc:  # an adapter's tab must not break the page
            log.exception("Tab %s", tab.id)
            sections = [Section("Fehler", text=f"Dieser Tab ließ sich nicht laden: {exc}")]
        return web.json_response(
            {
                "sections": [
                    {"title": s.title, "rows": [_row(r) for r in s.rows], "text": s.text}
                    for s in sections
                ],
                "actions": [
                    {
                        "id": a.id,
                        "label": a.label,
                        "description": a.description,
                        "confirm": a.confirm,
                        "danger": a.danger,
                        "options": [
                            {
                                "id": o.id,
                                "label": o.label,
                                "default": o.default,
                                "kind": o.kind,
                                "placeholder": o.placeholder,
                                "help": o.help,
                                "choices": [list(c) for c in o.choices],
                            }
                            for o in a.options
                        ],
                        "file": a.file,
                        "accept": a.accept,
                    }
                    for a in tab.actions
                ],
            }
        )

    async def _run(self, request: web.Request) -> web.Response:
        body = await request.json()
        tab = self.tabs.get(str(body.get("tab")))
        action = next((a for a in tab.actions if a.id == body.get("action")), None) if tab else None
        if action is None:
            raise web.HTTPNotFound()
        if any(not job.done for job in self.jobs.values()):
            raise web.HTTPConflict(text="Es läuft schon eine Aktion.")
        raw = body.get("options") if isinstance(body.get("options"), dict) else {}
        chosen: dict[str, Any] = {o.id: o.value(raw.get(o.id)) for o in action.options}
        uploaded = self.uploads.pop(str(body.get("upload")), None) if action.file else None
        if action.file:
            if uploaded is None:
                raise web.HTTPBadRequest(text="Erst eine Datei auswählen.")
            chosen["file"] = str(uploaded)
        job_id = secrets.token_hex(6)
        job = self.jobs[job_id] = Job()

        def work() -> None:
            try:
                job.ok = action.run(job.add, chosen)
            except Exception as exc:
                log.exception("Aktion %s", action.id)
                job.add(f"✗ Unerwarteter Fehler: {exc}")
                job.ok = False
            finally:
                job.done = True
                if uploaded is not None:
                    uploaded.unlink(missing_ok=True)

        threading.Thread(target=work, name=f"manage-{action.id}", daemon=True).start()
        return web.json_response({"job": job_id})

    async def _upload(self, request: web.Request) -> web.Response:
        """A file chosen on the page, streamed to disk (exports with recordings are large)."""
        upload_id = secrets.token_hex(8)
        path = self._upload_dir / upload_id
        with path.open("wb") as file:
            async for chunk in request.content.iter_chunked(1 << 20):
                file.write(chunk)
        self.uploads[upload_id] = path
        return web.json_response({"upload": upload_id})

    async def _job(self, request: web.Request) -> web.Response:
        job = self.jobs.get(request.match_info["job"])
        if job is None:
            raise web.HTTPNotFound()
        try:
            since = max(0, int(request.query.get("since", "0")))
        except ValueError:
            since = 0
        with job.lock:
            lines = job.lines[since:]
        return web.json_response({"lines": lines, "done": job.done, "ok": job.ok})

    async def _status(self, request: web.Request) -> web.Response:
        if self.launcher is None:
            return web.json_response({"startable": False})
        running, problem = await asyncio.to_thread(self.launcher.status)
        return web.json_response(
            {
                "startable": True,
                "running": running,
                "chat": self.launcher.chat_url,
                "problem": problem,
            }
        )

    async def _start(self, request: web.Request) -> web.Response:
        if self.launcher is None:
            raise web.HTTPNotFound()
        if await asyncio.to_thread(self.launcher.running):
            return web.json_response({"ok": True, "message": "läuft schon"})
        problem = await asyncio.to_thread(self.launcher.start)
        return web.json_response({"ok": problem is None, "message": problem or ""})

    async def _stop(self, request: web.Request) -> web.Response:
        if self.launcher is None:
            raise web.HTTPNotFound()
        problem = await asyncio.to_thread(self.launcher.stop)
        return web.json_response({"ok": problem is None, "message": problem or ""})

    async def _quit(self, request: web.Request) -> web.Response:
        self.stopped.set()
        return web.json_response({"ok": True})


def tabs_for(
    app: App,
    settings: Settings,
    adapters: Sequence[Adapter],
    config: Path | None,
    quit_soon: Callable[[], None],
    current: Callable[[], Settings] | None = None,
) -> list[Tab]:
    tabs = builtin_tabs(app, settings, adapters, config, quit_soon, current)
    for adapter in adapters:
        try:
            extra = adapter.manage_tabs(settings)
        except Exception:  # a broken adapter must not take the management UI with it
            log.exception("Adapter %s: Tabs", adapter.name)
            continue
        # Adapter tabs go before "Deinstallieren", which stays last.
        tabs[-1:-1] = [t for t in extra if t.id not in {tab.id for tab in tabs}]
    return tabs


# One management UI per program: a second `manage` (tray, install script, terminal) opens the
# running one instead of starting another. Its address lives here, readable only by the user.


# One management UI per program, wherever its data folder is: a second `manage` (tray, install
# script, terminal) opens the running one – unless that one runs older code, which it ends.
# Its address lives in Cassi's folder, readable only by the user (the token allows actions).
IDLE_SECONDS = 180  # no page polled for so long: no window open any more – the server ends
FIRST_VISIT_SECONDS = 600  # until the page is opened the first time


def instance_file(app: App) -> Path:
    from cassi.config import default_data_dir

    return default_data_dir() / "run" / f"manage-{app.name}.json"


def read_instance(path: Path) -> dict[str, Any] | None:
    """The record of a management UI that still answers; a stale one (crashed, killed) is
    removed."""
    import json
    import urllib.request

    from cassi.net import DIRECT

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        port, token = int(data["port"]), str(data["token"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    request = urllib.request.Request(
        f"http://{HOST}:{port}/api/tabs", headers={"X-Cassi-Token": token}
    )
    try:
        with DIRECT.open(request, timeout=1.5) as response:
            if response.status == 200:
                return {**data, "url": f"http://{HOST}:{port}/#token={token}"}
    except OSError:
        pass
    with contextlib.suppress(OSError):
        path.unlink()
    return None


def running_instance(path: Path) -> str | None:
    """The address of the recorded management UI, if it still answers."""
    record = read_instance(path)
    return str(record["url"]) if record else None


def ask_to_quit(record: dict[str, Any], wait: float = 5.0) -> bool:
    """Ask a running management UI to end (its own quit, as its page would)."""
    import time
    import urllib.request

    from cassi.net import DIRECT

    port, token = int(record["port"]), str(record["token"])
    request = urllib.request.Request(
        f"http://{HOST}:{port}/api/quit",
        data=b"{}",
        method="POST",
        headers={"X-Cassi-Token": token, "Origin": f"http://{HOST}:{port}"},
    )
    try:
        with DIRECT.open(request, timeout=2):
            pass
    except OSError:
        return False
    end = time.monotonic() + wait
    while time.monotonic() < end:
        try:
            with DIRECT.open(f"http://{HOST}:{port}/", timeout=0.5):
                pass
        except OSError:
            return True
        time.sleep(0.2)
    return False


def remember_instance(path: Path, server: ManageServer, fingerprint: str = "") -> None:
    import json
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "port": server.port,
        "token": server.token,
        "pid": os.getpid(),
        "fingerprint": fingerprint,
    }
    # Created readable for the user only: the token allows to repair or uninstall.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump(record, file)


def forget_instance(path: Path, server: ManageServer) -> None:
    """Remove the record – only if it is still this server's (not a newer one's)."""
    import json

    try:
        if json.loads(path.read_text(encoding="utf-8")).get("token") == server.token:
            path.unlink()
    except (OSError, ValueError, AttributeError):
        pass


def end_others(app: App, out: Output) -> int:
    """End this program's other management UIs (leftovers of older versions); the count."""
    ended = 0
    for pid, _ in maintenance.other_manage_processes(app.name):
        if maintenance.stop_process(pid, timeout=5):
            out(f"✓ Ältere Verwaltung beendet (Prozess {pid})")
            ended += 1
        else:
            out(f"✗ Verwaltung (Prozess {pid}) ließ sich nicht beenden")
    return ended


async def _close_when_unused(server: ManageServer) -> None:
    """End the server once no page has asked anything for a while: its window was closed."""
    import time

    while not server.stopped.is_set():
        await asyncio.sleep(15)
        idle = time.monotonic() - server.last_seen
        limit = IDLE_SECONDS if server.visited else FIRST_VISIT_SECONDS
        busy = any(not job.done for job in server.jobs.values())
        if idle > limit and not busy:
            log.info("Kein Fenster der Verwaltung mehr offen – beendet")
            server.stopped.set()


async def serve(
    app: App,
    settings: Settings,
    adapters: Sequence[Adapter],
    config: Path | None,
    open_browser: bool = True,
    port: int = 0,
) -> int:
    from cassi import updates

    record = instance_file(app)
    fingerprint = (await asyncio.to_thread(updates.installed, app)).fingerprint
    existing = await asyncio.to_thread(read_instance, record)
    if existing is not None and existing.get("fingerprint") == fingerprint:
        print(f"{app.title}-Verwaltung läuft schon: {existing['url']}", flush=True)
        if open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(str(existing["url"]))
        return 0
    if existing is not None:
        print("Eine ältere Version der Verwaltung läuft – sie wird beendet.", flush=True)
        await asyncio.to_thread(ask_to_quit, existing)
    # Older ones without a record (from before it was kept here) – never two at once.
    await asyncio.to_thread(end_others, app, lambda line: print(line, flush=True))
    holder: list[ManageServer] = []

    def current() -> Settings:
        from cassi.adapters import configure
        from cassi.config import load_settings

        return configure(adapters, load_settings(config))

    tabs = tabs_for(app, settings, adapters, config, lambda: holder[0].quit_soon(), current)
    server = ManageServer(app, tabs, port, launcher=Launcher(app, settings, config))
    holder.append(server)
    await server.start()
    remember_instance(record, server, fingerprint)
    closer = asyncio.create_task(_close_when_unused(server))
    print(f"{app.title}-Verwaltung: {server.url}", flush=True)
    print(
        "Beenden mit Strg+C oder „Verwaltung beenden“ auf der Seite; ohne offenes Fenster "
        f"beendet sie sich nach etwa {IDLE_SECONDS // 60} Minuten.",
        flush=True,
    )
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(server.url)
    try:
        await server.stopped.wait()
    finally:
        closer.cancel()
        forget_instance(record, server)
        await server.stop()
    return 0
