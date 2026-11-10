"""Connections to servers on this computer: the LLM server, Ollama, PhpStorm's MCP server.

They never go through a proxy. httpx (used by the OpenAI and MCP clients) takes proxies from
``urllib.request.getproxies()``, which on Windows falls back to the system proxy in the
registry – but ignores its exceptions (``<local>``).

"All connection attempts failed" from the OpenAI client means that every address the host
name resolves to (``::1`` and ``127.0.0.1`` for localhost) refused or dropped the
connection: nothing was listening there at that moment. :func:`diagnose` finds out why, so
the log says more than that.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlparse
from urllib.request import getproxies

from flay.offline import is_local_url

# Plain urllib requests without any proxy, for local servers only.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# Windows error codes (WSAE*) next to their POSIX counterparts.
_REFUSED = {errno.ECONNREFUSED, 10061}
_FORBIDDEN = {errno.EACCES, 10013}
_TIMED_OUT = {errno.ETIMEDOUT, 10060}


def trust_env(url: str) -> bool:
    """Whether an HTTP client for ``url`` may use proxies from the environment/system."""
    return not is_local_url(url)


def unreachable(url: str, timeout: float = 2.0) -> str | None:
    """None if a local HTTP server answers at ``url`` (any status), else the reason."""
    opener = DIRECT if is_local_url(url) else urllib.request.build_opener()
    try:
        with opener.open(url, timeout=timeout):
            return None
    except urllib.error.HTTPError:
        return None  # it answered, just not with 200 – that's reachable
    except (OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        return str(reason) or type(exc).__name__


def wait_until_reachable(
    url: str,
    seconds: float,
    waiting: Callable[[str], None] = lambda reason: None,
    interval: float = 1.0,
) -> str | None:
    """Like :func:`unreachable`, but gives a starting server up to ``seconds``.

    At login Flay and Ollama start at the same time (both from the autostart), and the
    Ollama app needs a few seconds until its server listens.
    """
    deadline = time.monotonic() + seconds
    problem = unreachable(url)
    if problem is not None:
        waiting(problem)
    while problem is not None and time.monotonic() < deadline:
        time.sleep(interval)
        problem = unreachable(url)
    return problem


def proxies() -> dict[str, str]:
    """Proxies Python sees (environment, or on Windows/macOS the system settings)."""
    return {scheme: str(value) for scheme, value in getproxies().items()}


def _explain(exc: OSError) -> str:
    code = getattr(exc, "winerror", None) or exc.errno
    if isinstance(exc, TimeoutError) or code in _TIMED_OUT:
        return "keine Antwort (eine Firewall verwirft die Verbindung?)"
    if code in _REFUSED:
        return "abgelehnt – dort lauscht kein Server"
    if code in _FORBIDDEN:
        return "Zugriff verweigert (Firewall oder Sicherheitssoftware blockiert Python?)"
    return f"{type(exc).__name__}: {exc}"


def probe(host: str, port: int, timeout: float = 2.0) -> list[str]:
    """Try each address ``host`` resolves to, like the OpenAI client does; one line each."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        return [f"{host} lässt sich nicht auflösen: {exc}"]
    lines: list[str] = []
    seen: set[str] = set()
    for family, kind, protocol, _, address in infos:
        ip = str(address[0])
        if ip in seen:
            continue
        seen.add(ip)
        shown = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
        try:
            with socket.socket(family, kind, protocol) as connection:
                connection.settimeout(timeout)
                connection.connect(address)
            lines.append(f"{shown} – verbunden")
        except OSError as exc:
            lines.append(f"{shown} – {_explain(exc)}")
    return lines


def excluded_port_ranges(netsh_output: str) -> list[tuple[int, int]]:
    """Port ranges from ``netsh interface ipv4 show excludedportrange protocol=tcp``."""
    return [
        (int(start), int(end))
        for start, end in re.findall(r"^\s*(\d+)\s+(\d+)\b", netsh_output, re.MULTILINE)
    ]


def _windows_reserved(port: int) -> tuple[int, int] | None:
    """The range Windows (Hyper-V, WSL 2, Docker) reserved that contains ``port``."""
    if sys.platform != "win32" or not shutil.which("netsh"):
        return None
    try:
        output = subprocess.run(
            ["netsh", "interface", "ipv4", "show", "excludedportrange", "protocol=tcp"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return next((r for r in excluded_port_ranges(output) if r[0] <= port <= r[1]), None)


def _ollama_process_running() -> bool | None:
    """Whether an Ollama server process exists; None if that can't be checked."""
    try:
        if sys.platform == "win32":
            output = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ollama.exe", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            return "ollama.exe" in output.lower()
        if shutil.which("pgrep"):
            result = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, timeout=5)
            return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def diagnose(base_url: str) -> list[str]:
    """Why a local server does not answer: one finding per line, most useful first."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    findings = [f"Verbindungsversuche: {'; '.join(probe(host, port))}"]
    if not is_local_url(base_url):
        return findings
    reserved = _windows_reserved(port)
    if reserved is not None:
        findings.append(
            f"Port {port} liegt in einem von Windows reservierten Bereich "
            f"({reserved[0]}–{reserved[1]}, Hyper-V/WSL 2/Docker) – Ollama kann ihn nicht "
            "öffnen. Abhilfe: Benutzer-Umgebungsvariable OLLAMA_HOST=127.0.0.1:11500 setzen, "
            "Ollama neu starten und llm.base_url auf http://127.0.0.1:11500/v1 stellen."
        )
    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        findings.append(f"OLLAMA_HOST={ollama_host} – Ollama lauscht dann dort, nicht auf {port}.")
    running = _ollama_process_running() if port == 11434 or ollama_host else None
    if running is False:
        findings.append(
            "Kein Ollama-Prozess gefunden – Ollama läuft nicht. Windows: Ollama aus dem "
            "Startmenü starten (Symbol im Infobereich); macOS: Ollama-App öffnen."
        )
    elif running:
        findings.append(
            "Ollama läuft, antwortet aber nicht auf diesem Port – Ollama-Log prüfen "
            "(Windows: %LOCALAPPDATA%\\Ollama\\server.log, macOS: ~/.ollama/logs/server.log)."
        )
    found = proxies()
    if found:
        listed = ", ".join(f"{k}={v}" for k, v in sorted(found.items()))
        findings.append(f"Proxy erkannt ({listed}); Flay umgeht ihn für lokale Server.")
    return findings


def llm_unreachable_message(base_url: str, reason: str) -> str:
    """What to tell the user when the LLM server does not answer."""
    lines = [f"LLM-Server {base_url} nicht erreichbar ({reason})."]
    lines += [f"  · {finding}" for finding in diagnose(base_url)]
    lines.append(
        "  · Prüfen: im Browser http://127.0.0.1:11434 öffnen – dort sollte "
        "'Ollama is running' stehen. Einrichtung prüfen: scripts/setup.ps1 bzw. setup.sh."
    )
    return "\n".join(lines)
