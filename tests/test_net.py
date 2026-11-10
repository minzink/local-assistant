"""Local servers are reached directly, even when a (system) proxy is configured."""

from __future__ import annotations

import contextlib
import io
import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from flay import net
from flay.llm import ollama
from flay.llm.openai_compatible import create_client

# Nothing listens here: a request that goes to this "proxy" fails like behind a dead
# corporate proxy (on Windows urllib.getproxies() returns the registry proxy the same way).
DEAD_PROXY = "http://127.0.0.1:9"


class _Models(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"object": "list", "data": [{"id": "m", "object": "model"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), _Models)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, DEAD_PROXY)


@pytest.mark.usefixtures("proxy")
async def test_llm_client_bypasses_proxy_for_local_server(server: str) -> None:
    models = [model.id async for model in create_client(server).models.list()]
    assert models == ["m"]


@pytest.mark.usefixtures("proxy")
def test_unreachable_bypasses_proxy_for_local_server(server: str) -> None:
    assert net.unreachable(server + "/models") is None


def test_unreachable_names_the_reason() -> None:
    reason = net.unreachable("http://127.0.0.1:9/v1/models", timeout=1)
    assert reason


def test_remote_servers_keep_the_proxy() -> None:
    assert not net.trust_env("http://127.0.0.1:11434/v1")
    assert not net.trust_env("http://localhost:11434/v1")
    assert net.trust_env("https://api.example.com/v1")


def test_message_mentions_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "proxies", lambda: {"http": "http://proxy.corp:8080"})
    monkeypatch.setattr(net, "probe", lambda host, port: [f"{host}:{port} – abgelehnt"])
    message = net.llm_unreachable_message("http://localhost:11434/v1", "refused")
    assert "proxy.corp" in message and "localhost:11434 – abgelehnt" in message


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_probe_explains_refused_connections() -> None:
    lines = net.probe("127.0.0.1", _free_port())
    assert len(lines) == 1 and "abgelehnt" in lines[0]


def test_probe_tries_every_address(server: str) -> None:
    port = int(server.rsplit(":", 1)[1].split("/")[0])
    assert net.probe("127.0.0.1", port) == [f"127.0.0.1:{port} – verbunden"]


NETSH = """
Protokoll tcp Portausschlussbereiche

Startport    Endport
----------    --------
      5357        5357
     11400       11499
     50000       50059     *

* - Verwaltete Portausschlüsse.
"""


def test_excluded_port_ranges() -> None:
    assert net.excluded_port_ranges(NETSH) == [(5357, 5357), (11400, 11499), (50000, 50059)]


def test_diagnose_names_reserved_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "probe", lambda host, port: ["abgelehnt"])
    monkeypatch.setattr(net, "_windows_reserved", lambda port: (11400, 11499))
    monkeypatch.setattr(net, "_ollama_process_running", lambda: True)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    findings = net.diagnose("http://127.0.0.1:11434/v1")
    assert any("reservierten Bereich (11400–11499" in f for f in findings)
    assert any("Ollama läuft, antwortet aber nicht" in f for f in findings)


def test_diagnose_names_missing_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "probe", lambda host, port: ["abgelehnt"])
    monkeypatch.setattr(net, "_windows_reserved", lambda port: None)
    monkeypatch.setattr(net, "_ollama_process_running", lambda: False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    findings = net.diagnose("http://127.0.0.1:11434/v1")
    assert any("Ollama läuft nicht" in f for f in findings)


def test_wait_until_reachable_gives_a_starting_server_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(["refused", "refused", None])
    monkeypatch.setattr(net, "unreachable", lambda url: next(answers))
    waited: list[str] = []
    assert net.wait_until_reachable("http://x", 5, waited.append, interval=0) is None
    assert waited == ["refused"]


def test_wait_until_reachable_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(net, "unreachable", lambda url: "refused")
    assert net.wait_until_reachable("http://x", 0.05, interval=0.01) == "refused"


def _response(lines: list[dict[str, Any]]) -> contextlib.AbstractContextManager[io.BytesIO]:
    data = b"".join(json.dumps(line).encode() + b"\n" for line in lines)
    return contextlib.nullcontext(io.BytesIO(data))


def test_pull_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling a", "total": 1000, "completed": 0},
        {"status": "pulling a", "total": 1000, "completed": 550},
        {"status": "pulling a", "total": 1000, "completed": 1000},
        {"status": "success"},
    ]
    monkeypatch.setattr(ollama, "_open", lambda request, timeout: _response(lines))
    printed: list[str] = []
    assert ollama.pull("http://127.0.0.1:11434/v1", "m", printed.append)
    assert [p.split(":")[1].split("%")[0].strip() for p in printed] == ["0", "50", "100"]


def test_pull_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [{"error": "pull model manifest: file does not exist"}]
    monkeypatch.setattr(ollama, "_open", lambda request, timeout: _response(lines))
    printed: list[str] = []
    assert not ollama.pull("http://127.0.0.1:11434/v1", "nope", printed.append)
    assert "does not exist" in printed[0]
