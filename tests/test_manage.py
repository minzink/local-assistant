import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cassi import app as cassi_app
from cassi.adapters import Adapter
from cassi.app import App
from cassi.config import Settings
from cassi.maintenance import Output
from cassi.manage import Action, Launcher, ManageServer, Option, Row, Section, Tab, tabs_for

TOKEN = "secret"


def greet(out: Output, options: Mapping[str, bool]) -> bool:
    out("hallo")
    out("laut" if options.get("loud") else "leise")
    return True


class Extending(Adapter):
    name = "extending"

    def manage_tabs(self, settings: Settings) -> list[Tab]:
        return [
            Tab(
                "extra",
                "Extra",
                lambda: [Section("Info", [Row("Modell", "m1", "ok")], "<b>kein HTML</b>")],
                [Action("greet", "Grüßen", "Sagt hallo.", greet, options=[Option("loud", "Laut")])],
            )
        ]


class Broken(Adapter):
    name = "broken"

    def manage_tabs(self, settings: Settings) -> list[Tab]:
        return [Tab("bad", "Kaputt", lambda: 1 / 0)]  # type: ignore[arg-type,return-value]


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[TestClient]:
    program = App("cassi", tmp_path, None)
    settings = Settings(data_dir=tmp_path / "data")
    tabs = tabs_for(program, settings, [Extending(), Broken()], None, lambda: None)
    server = ManageServer(program, tabs, token=TOKEN)
    async with TestClient(TestServer(server.web)) as test_client:
        server.port = test_client.port
        yield test_client


def auth(client: TestClient, post: bool = False) -> dict[str, str]:
    headers = {"X-Cassi-Token": TOKEN}
    if post:
        headers["Origin"] = f"http://127.0.0.1:{client.port}"
    return headers


async def test_the_api_needs_the_token(client: TestClient) -> None:
    assert (await client.get("/api/tabs")).status == 403
    assert (await client.get("/api/tabs", headers={"X-Cassi-Token": "wrong"})).status == 403
    assert (await client.get("/")).status == 200  # the page itself holds nothing


async def test_actions_only_from_the_pages_own_origin(client: TestClient) -> None:
    body = {"tab": "extra", "action": "greet"}
    headers = {"X-Cassi-Token": TOKEN, "Origin": "https://evil.example.org"}

    assert (await client.post("/api/run", json=body, headers=headers)).status == 403


async def test_adapter_tabs_come_before_uninstall(client: TestClient) -> None:
    data = await (await client.get("/api/tabs", headers=auth(client))).json()

    ids = [tab["id"] for tab in data["tabs"]]
    assert ids[:2] == ["overview", "maintenance"]
    assert ids[-1] == "uninstall"
    assert "extra" in ids


async def test_an_adapter_tab_is_data_and_its_action_runs(client: TestClient) -> None:
    tab = await (await client.get("/api/tabs/extra", headers=auth(client))).json()
    assert tab["sections"][0]["rows"] == [{"label": "Modell", "value": "m1", "level": "ok"}]
    assert tab["actions"][0]["options"] == [
        {
            "id": "loud",
            "label": "Laut",
            "default": False,
            "kind": "check",
            "placeholder": "",
            "help": "",
            "choices": [],
        }
    ]

    body = {"tab": "extra", "action": "greet", "options": {"loud": True}}
    response = await client.post("/api/run", json=body, headers=auth(client, post=True))
    job = (await response.json())["job"]
    for _ in range(50):
        state = await (await client.get(f"/api/jobs/{job}", headers=auth(client))).json()
        if state["done"]:
            break
        await asyncio.sleep(0.02)

    assert state == {"lines": ["hallo", "laut"], "done": True, "ok": True}


async def test_a_broken_tab_shows_an_error_instead_of_breaking_the_page(
    client: TestClient,
) -> None:
    tab = await (await client.get("/api/tabs/bad", headers=auth(client))).json()

    assert tab["sections"][0]["title"] == "Fehler"


async def test_builtin_uninstall_tab_offers_data_and_models(client: TestClient) -> None:
    tab = await (await client.get("/api/tabs/uninstall", headers=auth(client))).json()

    action = tab["actions"][0]
    assert action["danger"] and action["confirm"]
    assert [o["id"] for o in action["options"]] == ["data", "models"]
    assert not any(o["default"] for o in action["options"])


def test_flay_runs_under_its_own_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassi import autostart

    monkeypatch.setattr(cassi_app, "_name", "flay")
    monkeypatch.setattr(autostart.shutil, "which", lambda name: None)

    assert cassi_app.current().name == "flay"
    assert autostart.program_command()[-2:] == ["-m", "flay"]


class FakeLauncher(Launcher):
    def __init__(self) -> None:
        super().__init__(App("flay", None, None), Settings(), None)
        self.is_running = False
        self.started = 0

    def status(self) -> tuple[bool | None, str]:
        return self.is_running, ""

    def start(self) -> str | None:
        self.started += 1
        self.is_running = True
        return None


async def test_the_header_starts_the_assistant_once(tmp_path: Path) -> None:
    launcher = FakeLauncher()
    server = ManageServer(App("flay", tmp_path, None), [], token=TOKEN, launcher=launcher)
    async with TestClient(TestServer(server.web)) as client:
        server.port = client.port
        status = await (await client.get("/api/status", headers=auth(client))).json()
        assert status == {
            "startable": True,
            "running": False,
            "chat": "http://127.0.0.1:8765/",
            "problem": "",
        }

        assert (await client.post("/api/start", headers={"X-Cassi-Token": TOKEN})).status == 403
        first = await client.post("/api/start", headers=auth(client, post=True))
        again = await client.post("/api/start", headers=auth(client, post=True))

        assert (await first.json())["ok"] and (await again.json())["message"] == "läuft schon"
        assert launcher.started == 1


def test_running_means_the_chat_ui_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = Settings.model_validate({"ui": {"port": port}})
    launcher = Launcher(App("cassi", None, None), settings, None)
    assert launcher.running() is False

    off = Launcher(
        App("cassi", None, None), Settings.model_validate({"ui": {"enabled": False}}), None
    )
    assert off.running() is None and off.chat_url is None


def test_a_broken_program_on_the_chat_port_is_a_problem_not_a_chat() -> None:
    import http.server
    import threading

    class Failing(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_error(500)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Failing)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        settings = Settings.model_validate({"ui": {"port": server.server_address[1]}})
        running, problem = Launcher(App("flay", None, None), settings, None).status()
    finally:
        server.shutdown()

    assert running is True  # the port is taken: starting again would fail
    assert "Fehler 500" in problem


def test_stop_ends_only_cassi_on_the_chat_port(monkeypatch: pytest.MonkeyPatch) -> None:
    from cassi import maintenance

    stopped: list[int] = []
    monkeypatch.setattr(maintenance, "port_owner", lambda port: 4242)
    monkeypatch.setattr(maintenance, "stop_process", lambda pid: stopped.append(pid) or True)
    launcher = Launcher(App("flay", None, None), Settings(), None)

    old_flay = "/home/x/.local/share/uv/tools/flay/bin/python /home/x/.local/bin/flay run"
    monkeypatch.setattr(maintenance, "process_command", lambda pid: old_flay)
    assert launcher.stop() is None and stopped == [4242]

    monkeypatch.setattr(maintenance, "process_command", lambda pid: "/usr/bin/python3 server.py")
    problem = launcher.stop()
    assert problem is not None and "anderen Programm" in problem
    assert stopped == [4242]  # the other program was left alone


async def test_the_header_ends_the_assistant(tmp_path: Path) -> None:
    class Stoppable(FakeLauncher):
        def stop(self) -> str | None:
            self.is_running = False
            return None

    launcher = Stoppable()
    launcher.is_running = True
    server = ManageServer(App("flay", tmp_path, None), [], token=TOKEN, launcher=launcher)
    async with TestClient(TestServer(server.web)) as client:
        server.port = client.port
        assert (await client.post("/api/stop", headers={"X-Cassi-Token": TOKEN})).status == 403
        response = await client.post("/api/stop", headers=auth(client, post=True))

        assert (await response.json())["ok"] and not launcher.is_running


async def test_a_second_manage_opens_the_running_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from cassi.manage import instance_file, running_instance, serve

    program = App("flay", tmp_path, None)
    settings = Settings(data_dir=tmp_path / "data")
    record = instance_file(program)
    first = asyncio.create_task(serve(program, settings, [], None, open_browser=False))
    for _ in range(100):
        if record.exists():
            break
        await asyncio.sleep(0.02)
    data = json.loads(record.read_text(encoding="utf-8"))

    # A second call does not start another server: it names (and would open) the first.
    assert await serve(program, settings, [], None, open_browser=False) == 0
    assert "läuft schon" in capsys.readouterr().out
    assert not first.done()

    url = await asyncio.to_thread(running_instance, record)
    assert url == f"http://127.0.0.1:{data['port']}/#token={data['token']}"

    from aiohttp import ClientSession

    async with ClientSession() as session:
        headers = {"X-Cassi-Token": data["token"], "Origin": f"http://127.0.0.1:{data['port']}"}
        await session.post(f"http://127.0.0.1:{data['port']}/api/quit", headers=headers)
    assert await asyncio.wait_for(first, 5) == 0
    assert not record.exists()  # closed: the next manage starts a new one


def test_a_stale_record_is_removed(tmp_path: Path) -> None:
    import socket

    from cassi.manage import running_instance

    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    record = tmp_path / "manage-cassi.json"
    record.write_text(f'{{"port": {port}, "token": "x", "pid": 1}}', encoding="utf-8")

    assert running_instance(record) is None
    assert not record.exists()
    assert running_instance(tmp_path / "missing.json") is None


async def test_an_export_is_imported_through_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from cassi import manage, transfer

    monkeypatch.setattr(manage.Launcher, "running", lambda self: False)
    source = tmp_path / "old"
    source.mkdir()
    with sqlite3.connect(source / "memory.sqlite3") as db:
        db.execute("CREATE TABLE facts (text TEXT)")
    db.close()
    program = App("cassi", tmp_path, None)
    export = tmp_path / "cassi-export.zip"
    transfer.export(program, Settings(data_dir=source), export)

    settings = Settings(data_dir=tmp_path / "new")
    server = ManageServer(program, tabs_for(program, settings, [], None, lambda: None), token=TOKEN)
    async with TestClient(TestServer(server.web)) as client:
        server.port = client.port
        tab = await (await client.get("/api/tabs/data", headers=auth(client))).json()
        importing = next(a for a in tab["actions"] if a["id"] == "import")
        assert importing["file"] and importing["accept"] == ".zip" and importing["danger"]

        body = {"tab": "data", "action": "import"}
        refused = await client.post("/api/run", json=body, headers=auth(client, post=True))
        assert refused.status == 400  # no file chosen

        sent = await client.post(
            "/api/upload", data=export.read_bytes(), headers=auth(client, post=True)
        )
        body["upload"] = (await sent.json())["upload"]
        job = (
            await (await client.post("/api/run", json=body, headers=auth(client, post=True))).json()
        )["job"]
        for _ in range(100):
            state = await (await client.get(f"/api/jobs/{job}", headers=auth(client))).json()
            if state["done"]:
                break
            await asyncio.sleep(0.02)

    assert state["ok"], state["lines"]
    assert (tmp_path / "new" / "memory.sqlite3").exists()
    assert not server.uploads  # the uploaded file was used up


def test_the_page_shows_the_configuration_as_it_is_now(tmp_path: Path) -> None:
    program = App("cassi", tmp_path, None)
    at_start = Settings(data_dir=tmp_path / "old")
    changed = {"settings": Settings(data_dir=tmp_path / "new")}

    tabs = {
        tab.id: tab
        for tab in tabs_for(program, at_start, [], None, lambda: None, lambda: changed["settings"])
    }
    section = tabs["data"].sections()[0]

    assert {row.label: row.value for row in section.rows}["Datenordner"] == str(tmp_path / "new")

    def broken() -> Settings:
        raise ValueError("cassi.yaml kaputt")

    tabs = {t.id: t for t in tabs_for(program, at_start, [], None, lambda: None, broken)}
    section = tabs["data"].sections()[0]  # falls back to the configuration from the start
    assert {row.label: row.value for row in section.rows}["Datenordner"] == str(tmp_path / "old")


async def test_an_older_version_is_replaced_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from cassi import updates
    from cassi.manage import instance_file, serve
    from cassi.updates import Installed

    program = App("flay", tmp_path, None)
    settings = Settings(data_dir=tmp_path / "data")
    record = instance_file(program)
    monkeypatch.setattr(updates, "installed", lambda app: Installed("v1", "Flay v1"))
    old = asyncio.create_task(serve(program, settings, [], None, open_browser=False))
    for _ in range(100):
        if record.exists():
            break
        await asyncio.sleep(0.02)
    first = json.loads(record.read_text(encoding="utf-8"))

    monkeypatch.setattr(updates, "installed", lambda app: Installed("v2", "Flay v2"))
    new = asyncio.create_task(serve(program, settings, [], None, open_browser=False))
    assert await asyncio.wait_for(old, 10) == 0  # the old one was asked to end
    for _ in range(100):
        if (
            record.exists()
            and json.loads(record.read_text(encoding="utf-8"))["port"] != first["port"]
        ):
            break
        await asyncio.sleep(0.05)
    second = json.loads(record.read_text(encoding="utf-8"))

    assert second["fingerprint"] == "v2" and second["token"] != first["token"]
    assert "ältere Version" in capsys.readouterr().out
    from cassi.manage import ask_to_quit

    assert await asyncio.to_thread(ask_to_quit, second)
    assert await asyncio.wait_for(new, 10) == 0


async def test_the_server_ends_when_no_window_is_open(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from cassi import manage

    server = ManageServer(App("cassi", None, None), [], token=TOKEN)
    server.visited = True
    server.last_seen = time.monotonic() - manage.IDLE_SECONDS - 1
    real_sleep = asyncio.sleep
    monkeypatch.setattr(manage.asyncio, "sleep", lambda seconds: real_sleep(0))

    await asyncio.wait_for(manage._close_when_unused(server), 2)

    assert server.stopped.is_set()


async def test_a_running_action_keeps_the_server_open(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from cassi import manage
    from cassi.manage import Job

    server = ManageServer(App("cassi", None, None), [], token=TOKEN)
    server.visited = True
    server.last_seen = time.monotonic() - manage.IDLE_SECONDS - 1
    server.jobs["repair"] = Job()  # not done
    real_sleep = asyncio.sleep
    calls = 0

    async def few(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls > 3:
            server.jobs["repair"].done = True
        await real_sleep(0)

    monkeypatch.setattr(manage.asyncio, "sleep", few)
    await asyncio.wait_for(manage._close_when_unused(server), 2)

    assert calls > 3  # waited for the action before ending


def test_the_export_folder_is_chosen_in_the_page_and_used_both_ways(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from cassi import maintenance, manage, transfer

    monkeypatch.setattr(manage.Launcher, "running", lambda self: False)
    drive = tmp_path / "Stick"
    drive.mkdir()
    asked: list[str] = []

    def dialog(prompt: str, start: Path | None = None) -> Path:
        asked.append(prompt)
        return drive

    monkeypatch.setattr(maintenance, "ask_for_folder", dialog)
    monkeypatch.setattr(maintenance, "open_folder", lambda path: True)
    program = App("flay", tmp_path, None)

    here = Settings(data_dir=tmp_path / "here")
    here.data_dir.mkdir()
    with sqlite3.connect(here.data_dir / "memory.sqlite3") as db:
        db.execute("CREATE TABLE facts (text TEXT)")
    db.close()
    tabs = {t.id: t for t in tabs_for(program, here, [], None, lambda: None)}
    actions = {a.id: a for a in tabs["data"].actions}
    lines: list[str] = []

    assert actions["folder"].run(lines.append, {}) and asked
    assert actions["export"].run(lines.append, {"recordings": False})
    assert len(transfer.exports_in(program, drive)) == 1
    rows = {r.label: r.value for r in tabs["data"].sections()[1].rows}
    assert rows["Export-Ordner"].startswith(f"{drive} (gewählt)")

    there = Settings(data_dir=tmp_path / "there")
    tabs = {t.id: t for t in tabs_for(program, there, [], None, lambda: None)}
    latest = next(a for a in tabs["data"].actions if a.id == "import-latest")
    assert latest.run(lines.append, {}), lines
    assert (tmp_path / "there" / "memory.sqlite3").exists()


def test_exporting_to_a_drive_that_is_not_attached_says_so(tmp_path: Path) -> None:
    from cassi import transfer

    program = App("flay", tmp_path, None)
    transfer.choose_folder(program, tmp_path / "Volumes" / "weg")
    tabs = {t.id: t for t in tabs_for(program, Settings(data_dir=tmp_path), [], None, lambda: None)}
    lines: list[str] = []

    assert not next(a for a in tabs["data"].actions if a.id == "export").run(lines.append, {})
    assert "Festplatte" in lines[0]
