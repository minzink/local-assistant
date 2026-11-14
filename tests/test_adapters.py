from pathlib import Path
from typing import Any

import pytest

from cassi import adapters, cli
from cassi.adapters import Adapter, AdapterError
from cassi.config import Settings


class Renaming(Adapter):
    name = "renaming"

    def __init__(self) -> None:
        self.prepared: list[bool] = []

    def configure(self, settings: Settings) -> Settings:
        llm = settings.llm.model_copy(update={"model": "personal"})
        return settings.model_copy(update={"llm": llm})

    def prepare(self, settings: Settings, download: bool) -> list[str]:
        self.prepared.append(download)
        return [f"  ✓ {settings.llm.model}"]


class Broken(Adapter):
    def __init__(self) -> None:
        raise ValueError("kaputt")


def registered(monkeypatch: pytest.MonkeyPatch, **factories: Any) -> None:
    monkeypatch.setattr(
        adapters, "installed", lambda: {name: (lambda f=f: f) for name, f in factories.items()}
    )


def test_without_adapters_cassi_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    registered(monkeypatch)
    loaded = adapters.load()
    settings = Settings()
    assert loaded == []
    assert adapters.configure(loaded, settings) == settings


def test_an_adapter_changes_the_settings_and_prepares(monkeypatch: pytest.MonkeyPatch) -> None:
    registered(monkeypatch, renaming=Renaming)
    loaded = adapters.load()
    settings = adapters.configure(loaded, Settings())
    assert settings.llm.model == "personal"
    assert adapters.prepare(loaded, settings, download=True) == ["  ✓ personal"]


def test_the_config_chooses_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    registered(monkeypatch, renaming=Renaming)
    assert adapters.load([]) == []
    with pytest.raises(AdapterError, match="nicht installiert: other"):
        adapters.load(["other"])


def test_a_broken_adapter_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    registered(monkeypatch, broken=Broken)
    with pytest.raises(AdapterError, match="broken lässt sich nicht laden: kaputt"):
        adapters.load()


def test_an_adapter_must_be_one(monkeypatch: pytest.MonkeyPatch) -> None:
    registered(monkeypatch, odd=lambda: object())
    with pytest.raises(AdapterError, match="keine Unterklasse"):
        adapters.load()


def test_the_cli_prepares_adapters_before_llm_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter = Renaming()
    registered(monkeypatch, renaming=lambda: adapter)
    seen: list[str] = []
    monkeypatch.setattr(cli, "_info", lambda settings, config: seen.append("info") or 0)
    config = tmp_path / "cassi.yaml"
    config.write_text(f"data_dir: {tmp_path}\n")
    assert cli.main(["--config", str(config), "info"]) == 0
    assert seen == ["info"] and adapter.prepared == []  # info does not use the LLM


def test_the_cli_stops_on_an_adapter_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registered(monkeypatch, broken=Broken)
    config = tmp_path / "cassi.yaml"
    config.write_text(f"data_dir: {tmp_path}\n")
    assert cli.main(["--config", str(config), "info"]) == 1
    assert "Adapter broken lässt sich nicht laden" in capsys.readouterr().out


def test_llm_commands_prepare_without_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = Renaming()
    registered(monkeypatch, renaming=lambda: adapter)
    used: list[str] = []

    async def bench(settings: Settings, *args: Any) -> int:
        used.append(settings.llm.model)
        return 0

    monkeypatch.setattr(cli, "_bench", bench)
    config = tmp_path / "cassi.yaml"
    config.write_text(f"data_dir: {tmp_path}\n")
    assert cli.main(["--config", str(config), "bench"]) == 0
    assert adapter.prepared == [False] and used == ["personal"]
