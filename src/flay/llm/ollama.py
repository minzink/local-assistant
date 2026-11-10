"""Small-context variants of Flay's helper models, so they fit next to the main model.

Ollama loads each model with its context length, and the KV cache grows with it: the
embedding model took 4.8 GB at 32k context, a second vision model for pointing 8 GB – with
the 8B main model (10 GB) more than an M2 Pro with 32 GB keeps loaded, so models were
swapped on every click (seconds each time). With 2k context for embeddings and 4k for
pointing (one screenshot is ~1100 tokens) all three take ~15 GB.

``flay setup`` creates the variants (``/api/create``, no download beyond the base models);
"auto" in the config uses a variant when the server has it, else the usual model.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flay.net import DIRECT
from flay.offline import is_local_url

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    name: str
    base: str
    context: int


# The names keep the base's family: memory search recognizes Qwen3 embeddings by name.
EMBEDDING = Variant("qwen3-embedding:0.6b-ctx2k", "qwen3-embedding:0.6b", 2048)
POINTER = Variant("qwen3-vl:4b-instruct-ctx4k", "qwen3-vl:4b-instruct", 4096)
# A second model next to the main one needs this much (unified) memory or VRAM: the 4B
# with 4k context takes ~3.7 GB on top of ~10 GB for the 8B with 32k.
POINTER_MIN_GB = 20.0


def server_root(base_url: str) -> str:
    """http://host:11434 for an OpenAI-compatible base URL like http://host:11434/v1."""
    root = base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


def _open(request: str | urllib.request.Request, timeout: float) -> Any:
    """urlopen, without proxy for a server on this computer (see flay.net)."""
    url = request if isinstance(request, str) else request.full_url
    if is_local_url(url):
        return DIRECT.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def _get(root: str, path: str, timeout: float) -> dict[str, object] | None:
    try:
        with _open(root + path, timeout) as response:
            data = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return data if isinstance(data, dict) else None


def is_ollama(base_url: str, timeout: float = 1.0) -> bool:
    return _get(server_root(base_url), "/api/version", timeout) is not None


def models(base_url: str, timeout: float = 1.0) -> set[str]:
    """Model names the Ollama server has ("name:latest" also as "name")."""
    data = _get(server_root(base_url), "/api/tags", timeout) or {}
    listed = data.get("models")
    names: set[str] = set()
    for model in listed if isinstance(listed, list) else []:
        name = str(model.get("name", "")) if isinstance(model, dict) else ""
        names |= {name, name.removesuffix(":latest")} if name else set()
    return names


def resolve(setting: str | None, variant: Variant, fallback: str, base_url: str) -> str:
    """ "auto": the variant if the server has it, else ``fallback``; else the setting."""
    if setting not in (None, "auto"):
        return str(setting)
    if setting is None:
        return fallback
    return variant.name if variant.name in models(base_url) else fallback


def create(base_url: str, variant: Variant, timeout: float = 120.0) -> bool:
    body = json.dumps(
        {
            "model": variant.name,
            "from": variant.base,
            "parameters": {"num_ctx": variant.context},
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        server_root(base_url) + "/api/create",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with _open(request, timeout) as response:
            result = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        log.info("Variante %s nicht angelegt: %s", variant.name, exc)
        return False
    return bool(isinstance(result, dict) and result.get("status") == "success")


def pull(
    base_url: str,
    model: str,
    progress: Callable[[str], None] = print,
    timeout: float = 120.0,
) -> bool:
    """Download ``model`` into Ollama (``/api/pull``); reports every 10 %, True on success."""
    request = urllib.request.Request(
        server_root(base_url) + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    reported = -1
    try:
        # The timeout applies per read: Ollama reports progress several times a second.
        with _open(request, timeout) as response:
            for raw in response:
                line = json.loads(raw) if raw.strip() else {}
                if not isinstance(line, dict):
                    continue
                if line.get("error"):
                    progress(f"    Fehler: {line['error']}")
                    return False
                total, completed = line.get("total"), line.get("completed")
                if isinstance(total, int) and isinstance(completed, int) and total > 0:
                    percent = completed * 100 // total // 10 * 10
                    if percent != reported:
                        reported = percent
                        progress(f"    {model}: {percent} % von {total / 1e9:.1f} GB")
                if line.get("status") == "success":
                    return True
    except (OSError, ValueError, urllib.error.URLError) as exc:
        log.info("%s nicht geladen: %s", model, exc)
        progress(f"    Fehler: {exc}")
    return False


def memory_gb() -> float | None:
    """Memory a second model can use: unified memory on a Mac, VRAM of an NVIDIA GPU."""
    try:
        if sys.platform == "darwin":
            output = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            ).stdout
            # macOS lets the GPU use about two thirds of it.
            return int(output.strip()) / 1e9 * 2 / 3
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return max(float(line) for line in output.split()) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def setup(base_url: str) -> list[str]:
    """Create the variants (pulling missing base models); lines to print for `flay setup`."""
    if not is_ollama(base_url):
        return []
    lines: list[str] = []
    have = models(base_url)
    wanted = [EMBEDDING]
    memory = memory_gb()
    if memory is not None and memory >= POINTER_MIN_GB:
        wanted.append(POINTER)
    else:
        lines.append(
            "  · Ortungsmodell: zu wenig Speicher für ein zweites Bildmodell – "
            "click_on_screen nutzt das Hauptmodell"
        )
    for variant in wanted:
        if variant.name in have:
            lines.append(f"  ✓ {variant.name}")
            continue
        if variant.base not in have:
            print(f"  ↓ {variant.base} wird geladen ...", flush=True)
            if not pull(base_url, variant.base):
                lines.append(f"  ✗ {variant.base} konnte nicht geladen werden")
                continue
        if create(base_url, variant):
            lines.append(f"  ✓ {variant.name} angelegt (Kontext {variant.context})")
        else:
            lines.append(f"  ✗ {variant.name} konnte nicht angelegt werden")
    return lines
