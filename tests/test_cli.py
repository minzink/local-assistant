from pathlib import Path

import pytest

from cassi.cli import main

EXAMPLE_CONFIG = Path(__file__).parent.parent / "cassi.example.yaml"


def test_info_prints_resolved_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(EXAMPLE_CONFIG), "info"]) == 0

    out = capsys.readouterr().out
    assert "Beschleuniger" in out
    assert "cassi-ramona-medium" in out


def test_info_reports_invalid_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "cassi.yaml"
    config.write_text("language:\n  default: fr\n", encoding="utf-8")

    assert main(["--config", str(config), "info"]) == 1
    assert "ungültig" in capsys.readouterr().out


def test_memory_command_lists_forgets_and_clears(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cassi.memory.store import MemoryStore

    config = tmp_path / "cassi.yaml"
    config.write_text(f"data_dir: {tmp_path}\n", encoding="utf-8")
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add_fact("Heißt Mia.", now=1)
    store.add_fact("Mag Tee.", now=1)
    store.close()

    assert main(["-c", str(config), "memory"]) == 0
    assert "Heißt Mia." in capsys.readouterr().out

    assert main(["-c", str(config), "memory", "forget", "1"]) == 0
    assert main(["-c", str(config), "memory"]) == 0
    out = capsys.readouterr().out
    assert "Heißt Mia." not in out.split("gelöscht.")[-1]
    assert "Mag Tee." in out

    assert main(["-c", str(config), "memory", "clear-chat"]) == 0
    assert main(["-c", str(config), "memory"]) == 0
    assert "Mag Tee." in capsys.readouterr().out

    assert main(["-c", str(config), "memory", "clear"]) == 0
    assert main(["-c", str(config), "memory"]) == 0
    assert "Erinnerungen: keine" in capsys.readouterr().out


def test_export_and_import_latest_go_through_the_export_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cassi import app, manage, transfer

    monkeypatch.setattr(manage.Launcher, "running", lambda self: False)
    drive = tmp_path / "Stick"
    drive.mkdir()
    here, there = tmp_path / "here.yaml", tmp_path / "there.yaml"
    here.write_text(f"data_dir: {tmp_path / 'a'}\nexport_dir: {drive}\nadapters: []\n")
    there.write_text(f"data_dir: {tmp_path / 'b'}\nexport_dir: {drive}\nadapters: []\n")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "workflows.json").write_text('{"x": 1}')

    assert main(["-c", str(here), "export"]) == 0
    assert len(transfer.exports_in(app.current(), drive)) == 1
    assert main(["-c", str(there), "import"]) == 2  # neither a file nor --latest
    assert main(["-c", str(there), "import", "--latest", "--yes"]) == 0
    assert (tmp_path / "b" / "workflows.json").read_text() == '{"x": 1}'
