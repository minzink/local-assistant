import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from cassi import cli, updates
from cassi.app import App
from cassi.config import Settings
from cassi.core.events import EventBus, Notice
from cassi.maintenance import SHIM_MARK


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def commit(root: Path, text: str) -> None:
    (root / "a.txt").write_text(text, encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", text)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "flay"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.org")
    git(root, "config", "user.name", "Test")
    commit(root, "eins")
    return root


@pytest.fixture(autouse=True)
def no_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real ~/.local/bin/cassi or flay may count in these tests."""
    monkeypatch.setattr(updates.shutil, "which", lambda name: None)


def launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Path) -> Path:
    path = tmp_path / "bin" / "flay"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f'#!/bin/sh\n# {SHIM_MARK}.\nexec uv run --project "{target}" flay\n')
    monkeypatch.setattr(updates.shutil, "which", lambda name: str(path))
    return path


def test_the_running_version_is_current_until_something_new_is_installed(project: Path) -> None:
    app = App("flay", project, None)
    started = updates.installed(app)
    assert updates.newer(app, started) is None

    commit(project, "zwei")  # what `git pull` or `flay update` does

    reason = updates.newer(app, started)
    assert reason is not None and "→" in reason
    assert started.label.split("(")[1].rstrip(")") in reason


def test_a_command_that_now_starts_another_installation_is_newer(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = App("flay", project, None)
    started = updates.installed(app)
    elsewhere = tmp_path / "neu" / "flay"
    elsewhere.mkdir(parents=True)

    launcher(tmp_path, monkeypatch, elsewhere)

    assert updates.newer(app, started) == f"flay startet jetzt die Installation in {elsewhere}"
    command = updates.restart_command(app, ["run"])
    assert command is not None and command[0].endswith("flay")
    assert command[1:] == ["run", "--background"]


def test_a_deleted_installation_ends_when_nothing_can_take_over(project: Path) -> None:
    app = App("flay", project, None)
    started = updates.installed(app)
    gone = App("flay", project.parent / "weg", None)

    assert updates.newer(gone, started) == "diese Installation wurde entfernt"
    assert updates.restart_command(gone, ["run"]) is None
    assert updates.restart_command(app, ["run"]) == [
        sys.executable, "-m", "flay", "run", "--background",
    ]  # fmt: skip


def test_the_new_version_starts_without_another_browser_tab() -> None:
    assert updates.restart_arguments([]) == ["run", "--background"]
    assert updates.restart_arguments(["-c", "x.yaml"]) == ["-c", "x.yaml", "run", "--background"]
    assert updates.restart_arguments(["run", "--background"]) == ["run", "--background"]


async def test_the_watcher_reports_once() -> None:
    reasons: list[str] = []
    checks = iter([None, "neu"])
    watcher = updates.UpdateWatcher(
        App("cassi", None, None), reasons.append, interval=0, check=lambda app, s: next(checks)
    )

    await asyncio.wait_for(watcher.run(), 2)

    assert reasons == ["neu"]


class IdleAssistant:
    def __init__(self) -> None:
        self.events = EventBus()
        self.idle = asyncio.Event()

    async def wait_until_idle(self) -> None:
        await self.idle.wait()


async def test_cassi_says_so_in_the_chat_and_switches_once_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Immediate:
        def __init__(self, app: App, on_newer: object, interval: float) -> None:
            self.on_newer = on_newer

        async def run(self) -> None:
            self.on_newer("Flay 0.2.0 (aaa) → Flay 0.2.1 (bbb)")  # type: ignore[operator]

    monkeypatch.setattr(updates, "UpdateWatcher", Immediate)
    monkeypatch.setattr(cli, "_SWITCH", [])
    monkeypatch.setattr(updates, "restart_command", lambda app, argv: ["flay", "run"])
    assistant = IdleAssistant()
    background: list[asyncio.Task[object]] = []

    async def run() -> None:
        async with assistant.events.subscribe() as events:
            cli._watch_for_updates(Settings(), assistant, background)  # type: ignore[arg-type]
            first = await asyncio.wait_for(events.get(), 1)
            assert isinstance(first, Notice) and "starte neu, sobald" in first.text
            assert cli._SWITCH == []  # busy: not yet
            assistant.idle.set()
            second = await asyncio.wait_for(events.get(), 1)
            assert isinstance(second, Notice) and "jetzt neu" in second.text
            await asyncio.sleep(10)  # cancelled by the switch

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.create_task(run())
    assert cli._SWITCH == ["Flay 0.2.0 (aaa) → Flay 0.2.1 (bbb)"]
    for task in background:
        task.cancel()


_real_sleep = asyncio.sleep


async def _fast_sleep(delay: float, result: object = None) -> object:
    return await _real_sleep(min(delay, 0.01), result)


def test_after_a_switch_the_chat_says_what_replaced_what(
    monkeypatch: pytest.MonkeyPatch, project: Path
) -> None:
    monkeypatch.setenv(updates.SWITCHED_ENV, "Flay 0.2.0 (aaa) → Flay 0.2.1 (bbb)")
    monkeypatch.setattr(cli.os, "environ", dict(cli.os.environ))
    settings = Settings.model_validate({"updates": {"switch": False}})

    notices = cli._watch_for_updates(settings, IdleAssistant(), [])  # type: ignore[arg-type]

    assert len(notices) == 1 and notices[0].startswith("Neu gestartet mit Cassi")
    assert "Flay 0.2.0 (aaa) → Flay 0.2.1 (bbb)" in notices[0]
    assert updates.SWITCHED_ENV not in cli.os.environ  # only once
