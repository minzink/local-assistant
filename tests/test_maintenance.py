import subprocess
import sys
from pathlib import Path

import pytest

from cassi import autostart, maintenance
from cassi.adapters import Adapter
from cassi.app import App
from cassi.config import Settings
from cassi.llm import ollama


def app(tmp_path: Path, name: str = "cassi") -> App:
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    return App(name, project, None)


def test_command_launcher_runs_the_program_from_its_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(maintenance.shutil, "which", lambda name: "/opt/uv")
    program = app(tmp_path, "flay")
    lines: list[str] = []

    assert maintenance.install_command(program, lines.append, tmp_path / "bin")

    path = maintenance.shim_path(program, tmp_path / "bin")
    text = path.read_text(encoding="utf-8")
    assert maintenance.SHIM_MARK in text
    assert f'run --quiet --project "{program.project}" flay' in text
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o111
    maintenance.remove_command(program, lines.append, tmp_path / "bin")
    assert not path.exists()


def test_a_foreign_command_is_never_overwritten_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(maintenance.shutil, "which", lambda name: "/opt/uv")
    program = app(tmp_path)
    path = maintenance.shim_path(program, tmp_path)
    path.write_text("someone else's cassi", encoding="utf-8")
    lines: list[str] = []

    assert not maintenance.install_command(program, lines.append, tmp_path)
    maintenance.remove_command(program, lines.append, tmp_path)

    assert path.read_text(encoding="utf-8") == "someone else's cassi"
    assert "anderen Programm" in lines[0]


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, Path]:
    """An upstream repository and a clone of it, one commit each."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-q", "-b", "main")
    git(upstream, "config", "user.email", "test@example.org")
    git(upstream, "config", "user.name", "Test")
    (upstream / "a.txt").write_text("1", encoding="utf-8")
    git(upstream, "add", ".")
    git(upstream, "commit", "-q", "-m", "one")
    clone = tmp_path / "project"
    git(tmp_path, "clone", "-q", str(upstream), str(clone))
    return upstream, clone


def test_updates_are_found_and_pulled(checkout: tuple[Path, Path]) -> None:
    upstream, clone = checkout
    (upstream / "a.txt").write_text("2", encoding="utf-8")
    git(upstream, "commit", "-q", "-am", "two")
    program = App("cassi", clone, None)
    lines: list[str] = []

    assert maintenance.check_updates(program, lines.append)
    assert any("1 neue Änderung" in line for line in lines)
    assert maintenance.pull(program, lines.append)
    assert (clone / "a.txt").read_text(encoding="utf-8") == "2"
    state = maintenance.repo_state(clone)
    assert state is not None and state.behind == 0 and state.changes == 0


def test_local_changes_stop_the_update(checkout: tuple[Path, Path]) -> None:
    _, clone = checkout
    (clone / "a.txt").write_text("mine", encoding="utf-8")
    lines: list[str] = []

    assert not maintenance.pull(App("cassi", clone, None), lines.append)

    assert (clone / "a.txt").read_text(encoding="utf-8") == "mine"
    assert "geänderte Datei" in lines[-1]


def test_repair_runs_cassis_setup_for_the_project(tmp_path: Path) -> None:
    program = App("flay", tmp_path / "flay", tmp_path / "local-assistant")

    command = maintenance.setup_command(program)

    assert command is not None
    assert str(tmp_path / "flay") in command and "flay" in command
    assert any("setup." in part for part in command)
    assert maintenance.setup_command(App("cassi", tmp_path, None)) is None


class Cleaning(Adapter):
    name = "cleaning"

    def __init__(self) -> None:
        self.asked: list[bool] = []

    def uninstall(self, settings: Settings, models: bool) -> list[str]:
        self.asked.append(models)
        return ["✓ adapter clean"]


def test_uninstall_keeps_data_and_models_unless_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autostart, "disable", lambda: True)
    monkeypatch.setattr(maintenance, "bin_dir", lambda: tmp_path / "bin")
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    settings = Settings(data_dir=data)
    adapter = Cleaning()
    lines: list[str] = []

    assert maintenance.uninstall(app(tmp_path), settings, [adapter], lines.append)

    assert data.exists()
    assert adapter.asked == [False]
    assert "✓ Autostart entfernt" in lines and "✓ adapter clean" in lines


def test_uninstall_removes_data_and_cassis_models_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autostart, "disable", lambda: False)
    monkeypatch.setattr(maintenance, "bin_dir", lambda: tmp_path / "bin")
    have = {"qwen3-vl:8b-instruct", "qwen3-embedding:0.6b-ctx2k", "someone-elses:7b"}
    deleted: list[str] = []
    monkeypatch.setattr(ollama, "is_ollama", lambda url: True)
    monkeypatch.setattr(ollama, "models", lambda url: set(have))
    monkeypatch.setattr(ollama, "delete", lambda url, name: deleted.append(name) or True)
    data = tmp_path / "data"
    data.mkdir()
    lines: list[str] = []

    ok = maintenance.uninstall(
        app(tmp_path), Settings(data_dir=data), [], lines.append, data=True, models=True
    )

    assert ok and not data.exists()
    assert sorted(deleted) == ["qwen3-embedding:0.6b-ctx2k", "qwen3-vl:8b-instruct"]


def test_sizes_read_well() -> None:
    assert maintenance.human_size(512) == "512 B"
    assert maintenance.human_size(3 * 1024**3) == "3.0 GB"


def test_the_process_on_a_port_is_found_and_ended() -> None:
    import socket
    import time

    # A separate process that listens, like a running (old) Cassi.
    code = (
        "import socket, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
        "print(s.getsockname()[1], flush=True); time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None
        port = int(child.stdout.readline())
        owner = None
        for _ in range(20):  # the OS lists the socket a moment later on some systems
            owner = maintenance.port_owner(port)
            if owner:
                break
            time.sleep(0.1)
        # On Windows a venv's python.exe is a launcher that starts the real Python as its
        # child – the port belongs to that one, so compare command lines, not ids.
        assert owner is not None
        assert "time.sleep(60)" in maintenance.process_command(owner)

        assert maintenance.stop_process(owner, timeout=5)
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    finally:
        child.kill()
        child.wait()
