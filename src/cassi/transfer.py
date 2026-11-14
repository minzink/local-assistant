"""Export and import the personal data: memory, history, learned workflows and preferences,
own models and the own voice – everything in the data folder that is the user's, never the
repository's (``cassi export``, ``cassi import``, and the "Daten" tab of ``cassi manage``).

Downloaded models (Piper voices, speech detection, wake word features) are not part of it:
``cassi setup`` downloads them again. Recordings (own wake word takes, test recordings,
voice training data) only on request – they can be large.

An export is a zip file with a manifest (``cassi-export.json``) naming the program it came
from; an import accepts only exports of the same program (Flay's memory is not Cassi's).
Before importing, the current data is saved to ``<data_dir>/backups``.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from cassi.app import App

if TYPE_CHECKING:
    from cassi.config import Settings

MANIFEST = "cassi-export.json"
FORMAT = 1
# Large, made by recording: only with ``recordings=True``.
RECORDINGS = ("eval", "wake", "voice_dataset", "training")
# Never exported: downloads (voices, feature models), earlier exports and backups.
SKIPPED_FOLDERS = ("voices", "backups", "exports")
# Downloaded model files in ``models/``; own trained models there (.npz) are personal.
DOWNLOADED_SUFFIXES = (".onnx",)
MEMORY = "memory.sqlite3"


class TransferError(RuntimeError):
    """The export cannot be imported; the message says why (German, for the user)."""


@dataclass(frozen=True, slots=True)
class Contents:
    files: int
    size: int


def _skipped(relative: PurePosixPath, recordings: bool) -> bool:
    top = relative.parts[0]
    name = relative.name
    if top in SKIPPED_FOLDERS or (not recordings and top in RECORDINGS):
        return True
    if top == "models" and relative.suffix in DOWNLOADED_SUFFIXES:
        return True
    # Logs, the management UI's address, SQLite's side files (the copy is made consistent).
    return (
        name.endswith(".log")
        or (name.startswith("manage-") and name.endswith(".json"))
        or name.endswith(("-wal", "-shm", "-journal"))
        or name == ".DS_Store"
    )


def personal_files(data_dir: Path, recordings: bool = False) -> Iterator[PurePosixPath]:
    """The files an export contains, relative to the data folder."""
    if not data_dir.is_dir():
        return
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            relative = PurePosixPath(path.relative_to(data_dir).as_posix())
            if not _skipped(relative, recordings):
                yield relative


def contents(data_dir: Path, recordings: bool = False) -> Contents:
    files = list(personal_files(data_dir, recordings))
    return Contents(len(files), sum((data_dir / f).stat().st_size for f in files))


def default_target(app: App) -> Path:
    """Where an export goes without a path: the downloads folder, else the home folder."""
    downloads = Path.home() / "Downloads"
    folder = downloads if downloads.is_dir() else Path.home()
    return folder / f"{app.name}-export-{time.strftime('%Y-%m-%d-%H%M')}.zip"


def export(app: App, settings: Settings, target: Path, recordings: bool = False) -> Contents:
    """Write the personal data to the zip file ``target``."""
    data_dir = settings.data_dir
    files = list(personal_files(data_dir, recordings))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    size = 0
    with (
        tempfile.TemporaryDirectory() as scratch,
        zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        manifest = {
            "format": FORMAT,
            "app": app.name,
            "version": app.version,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "recordings": recordings,
            "files": len(files),
        }
        archive.writestr(MANIFEST, json.dumps(manifest, indent=1))
        for relative in files:
            source = data_dir / relative
            if str(relative) == MEMORY:
                # A consistent copy, even while Cassi runs and writes to it.
                copy = Path(scratch) / MEMORY
                _backup_sqlite(source, copy)
                source = copy
            archive.write(source, str(relative))
            size += source.stat().st_size
    partial.replace(target)
    return Contents(len(files), size)


def _backup_sqlite(source: Path, target: Path) -> None:
    with sqlite3.connect(source) as original, sqlite3.connect(target) as copy:
        original.backup(copy)
    copy.close()
    original.close()


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read(MANIFEST))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise TransferError(f"Keine Export-Datei von Cassi: {path.name} ({exc})") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise TransferError(f"Unbekanntes Export-Format in {path.name}")
    return data


def import_(
    app: App,
    settings: Settings,
    source: Path,
    out: Callable[[str], None],
    allow_other: bool = False,
) -> Path:
    """Replace the personal data with the export ``source``; returns the backup made of the
    data before. The assistant must not run (it holds the memory open)."""
    manifest = read_manifest(source)
    origin = str(manifest.get("app", "?"))
    if origin != app.name and not allow_other:
        raise TransferError(
            f"Das ist ein Export von {origin.capitalize()}, nicht von {app.title} – "
            "Gedächtnis und Modelle gehören zu ihm."
        )
    data_dir = settings.data_dir
    with zipfile.ZipFile(source) as archive:
        members = [m for m in archive.namelist() if m != MANIFEST and not m.endswith("/")]
        for member in members:
            relative = PurePosixPath(member)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member:
                raise TransferError(f"Ungültiger Pfad in der Export-Datei: {member}")
            if _skipped(relative, recordings=True):
                raise TransferError(f"Unerwartete Datei in der Export-Datei: {member}")
        backup = data_dir / "backups" / f"vor-import-{time.strftime('%Y-%m-%d-%H%M%S')}.zip"
        saved = export(app, settings, backup, recordings=True)
        out(f"✓ Bisherige Daten gesichert ({saved.files} Dateien): {backup}")
        for member in members:
            target = data_dir / PurePosixPath(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as packed, target.open("wb") as unpacked:
                while chunk := packed.read(1 << 20):
                    unpacked.write(chunk)
        if MEMORY in members:
            for side in ("-wal", "-shm"):  # belonged to the replaced database
                (data_dir / f"{MEMORY}{side}").unlink(missing_ok=True)
    created = manifest.get("created", "?")
    out(f"✓ {len(members)} Dateien übernommen (Export von {origin.capitalize()}, {created})")
    return backup
