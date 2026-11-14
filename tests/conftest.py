from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_real_data_folders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests never touch the user's real data folders: `cli.main` with default settings
    would otherwise move the folder from before the rename (Flay → Cassi)."""
    from cassi import config

    monkeypatch.setattr(
        config, "user_data_path", lambda name, appauthor=False: tmp_path / "user-data" / name
    )
