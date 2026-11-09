from pathlib import Path

import pytest

from flay.cli import main

EXAMPLE_CONFIG = Path(__file__).parent.parent / "flay.example.yaml"


def test_info_prints_resolved_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(EXAMPLE_CONFIG), "info"]) == 0

    out = capsys.readouterr().out
    assert "Beschleuniger" in out
    assert "de_DE-kerstin-low" in out


def test_info_reports_invalid_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "flay.yaml"
    config.write_text("language:\n  default: fr\n", encoding="utf-8")

    assert main(["--config", str(config), "info"]) == 1
    assert "ungültig" in capsys.readouterr().out


def test_memory_command_lists_forgets_and_clears(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from flay.memory.store import MemoryStore

    config = tmp_path / "flay.yaml"
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
