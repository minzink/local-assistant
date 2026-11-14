import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from cassi import transfer
from cassi.app import App
from cassi.config import Settings
from cassi.transfer import MANIFEST, TransferError


def fill(data: Path) -> None:
    """A data folder as Cassi leaves it: personal data, recordings, downloads, runtime files."""
    (data / "models").mkdir(parents=True)
    (data / "voices").mkdir()
    (data / "wake").mkdir()
    with sqlite3.connect(data / "memory.sqlite3") as db:
        db.execute("CREATE TABLE facts (text TEXT)")
        db.execute("INSERT INTO facts VALUES ('Ich mag Tee')")
    db.close()
    (data / "workflows.json").write_text("{}", encoding="utf-8")
    (data / "models" / "hey_cassi.npz").write_bytes(b"own")  # trained here: personal
    (data / "models" / "silero_vad_v5.onnx").write_bytes(b"download")
    (data / "voices" / "de_DE-thorsten-high.onnx").write_bytes(b"download")
    (data / "wake" / "hey_cassi_001.wav").write_bytes(b"recording")
    (data / "cassi.log").write_text("log", encoding="utf-8")
    (data / "manage-cassi.json").write_text("{}", encoding="utf-8")


def test_the_export_holds_personal_data_only(tmp_path: Path) -> None:
    data = tmp_path / "data"
    fill(data)

    files = {str(f) for f in transfer.personal_files(data)}
    assert files == {"memory.sqlite3", "workflows.json", "models/hey_cassi.npz"}
    with_recordings = {str(f) for f in transfer.personal_files(data, recordings=True)}
    assert with_recordings == files | {"wake/hey_cassi_001.wav"}


def test_export_and_import_carry_memory_to_another_computer(tmp_path: Path) -> None:
    app = App("flay", None, None)
    here = tmp_path / "here"
    fill(here)
    target = tmp_path / "flay-export.zip"

    done = transfer.export(app, Settings(data_dir=here), target)

    assert done.files == 3 and target.exists()
    manifest = json.loads(zipfile.ZipFile(target).read(MANIFEST))
    assert manifest["app"] == "flay" and manifest["recordings"] is False

    there = tmp_path / "there"
    (there).mkdir()
    (there / "workflows.json").write_text('{"alt": 1}', encoding="utf-8")
    lines: list[str] = []
    backup = transfer.import_(app, Settings(data_dir=there), target, lines.append)

    with sqlite3.connect(there / "memory.sqlite3") as db:
        assert db.execute("SELECT text FROM facts").fetchall() == [("Ich mag Tee",)]
    db.close()
    assert (there / "workflows.json").read_text(encoding="utf-8") == "{}"
    # What was there before is kept in the backup.
    assert zipfile.ZipFile(backup).read("workflows.json") == b'{"alt": 1}'
    assert "gesichert" in lines[0]


def test_an_export_of_another_assistant_is_refused(tmp_path: Path) -> None:
    data = tmp_path / "data"
    fill(data)
    target = tmp_path / "cassi.zip"
    transfer.export(App("cassi", None, None), Settings(data_dir=data), target)

    with pytest.raises(TransferError, match="Export von Cassi, nicht von Flay"):
        transfer.import_(App("flay", None, None), Settings(data_dir=tmp_path / "f"), target, print)


@pytest.mark.parametrize("member", ["../evil.txt", "/etc/evil", "voices/x.onnx"])
def test_files_outside_the_personal_data_are_refused(tmp_path: Path, member: str) -> None:
    target = tmp_path / "bad.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(MANIFEST, json.dumps({"format": 1, "app": "cassi"}))
        archive.writestr(member, "x")
    data = tmp_path / "data"

    with pytest.raises(TransferError):
        transfer.import_(App("cassi", None, None), Settings(data_dir=data), target, print)
    assert not (tmp_path / "evil.txt").exists() and not data.exists()


def test_a_foreign_file_is_no_export(tmp_path: Path) -> None:
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as archive:
        archive.writestr("readme.txt", "x")

    with pytest.raises(TransferError, match="Keine Export-Datei"):
        transfer.read_manifest(other)
