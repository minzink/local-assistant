"""``flay doctor``: checks what Flay needs on this computer and says what to do about it.

Each check prints one line – ✓ fine, ! works but slower or limited, ✗ broken – with the
fix. Nothing is downloaded or changed, except that the LLM is loaded once to see how it
fits the GPU.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from flay.config import Settings
from flay.core.platform import Accelerator, detect_hardware


class Level(StrEnum):
    OK = "✓"
    WARN = "!"
    FAIL = "✗"


@dataclass(frozen=True, slots=True)
class Check:
    level: Level
    name: str
    detail: str
    fix: str = ""

    def line(self) -> str:
        text = f"  {self.level} {self.name}: {self.detail}"
        return text + (f"\n      → {self.fix}" if self.fix else "")


def _gpu() -> Check:
    hardware = detect_hardware()
    if hardware.accelerator is Accelerator.APPLE_SILICON:
        return Check(Level.OK, "Beschleuniger", f"Apple Silicon ({platform.machine()})")
    if hardware.accelerator is Accelerator.CUDA:
        from flay.llm.ollama import memory_gb

        memory = memory_gb()
        detail = f"NVIDIA, {memory:.0f} GB Grafikspeicher" if memory else "NVIDIA"
        if memory is not None and memory < 10:
            return Check(
                Level.WARN,
                "Beschleuniger",
                detail,
                "Unter 10 GB passt das 8B-Modell kaum: ein kleineres Modell wählen (llm.model)",
            )
        return Check(Level.OK, "Beschleuniger", detail)
    return Check(
        Level.WARN,
        "Beschleuniger",
        "keine GPU erkannt – Antworten dauern Sekunden",
        "NVIDIA-Treiber prüfen (nvidia-smi muss laufen)",
    )


def _ollama(settings: Settings) -> list[Check]:
    from flay import net
    from flay.llm import ollama

    url = settings.llm.base_url
    checks: list[Check] = []
    problem = net.wait_until_reachable(url.rstrip("/") + "/models", 3, lambda _: None)
    if problem is not None:
        return [
            Check(
                Level.FAIL,
                "LLM-Server",
                f"{url} antwortet nicht",
                net.llm_unreachable_message(url, problem),
            )
        ]
    if not ollama.is_ollama(url):
        return [Check(Level.OK, "LLM-Server", f"{url} (kein Ollama, Prüfungen übersprungen)")]
    version = (ollama._get(ollama.server_root(url), "/api/version", 2) or {}).get("version", "?")
    old = _older(str(version), "0.30")
    checks.append(
        Check(
            Level.WARN if old else Level.OK,
            "Ollama",
            f"Version {version}",
            "Ollama aktualisieren (mind. 0.30)" if old else "",
        )
    )
    have = ollama.models(url)
    model = ollama.main_model(url, settings.llm.model)
    if settings.llm.model not in have:
        checks.append(Check(Level.FAIL, "LLM", f"{settings.llm.model} fehlt", "flay setup"))
        return checks
    checks.append(Check(Level.OK, "LLM", model))
    # Load it once to see how it fits.
    from flay.bench import run_prompt
    from flay.llm.factory import create_chat_model

    llm = create_chat_model(settings.llm.model_copy(update={"model": model}))
    asyncio.run(run_prompt(llm, settings.persona.system_prompt, "Hallo"))
    warnings, context = ollama.check_loaded(url, model)
    for warning in warnings:
        checks.append(
            Check(
                Level.WARN,
                "Grafikspeicher",
                warning.split(" – ")[0],
                warning.split("Abhilfe: ")[-1],
            )
        )
    if not warnings:
        checks.append(Check(Level.OK, "Grafikspeicher", "alle geladenen Modelle ganz auf der GPU"))
    if context is not None:
        small = context < settings.llm.context_tokens
        checks.append(
            Check(
                Level.WARN if small else Level.OK,
                "Kontext",
                f"{context} Token",
                f"llm.context_tokens ({settings.llm.context_tokens}) ist größer – Flay kürzt "
                "auf den echten Wert"
                if small
                else "",
            )
        )
    return checks


def _older(version: str, minimum: str) -> bool:
    def parts(text: str) -> tuple[int, ...]:
        return tuple(int(p) for p in text.split(".")[:3] if p.isdigit())

    found = parts(version)
    return bool(found) and found < parts(minimum)


def _voice(settings: Settings) -> Check:
    from flay.tts.factory import resolve_backend
    from flay.tts.piper import MIN_MODEL_BYTES, REPO_VOICES

    backend = resolve_backend(settings.tts.backend, detect_hardware())
    if backend == "qwen3":
        return Check(Level.OK, "Stimme", "Qwen3-TTS")
    voice_dir = settings.data_dir / "voices"
    missing = [
        n for n in set(settings.tts.voices.values()) if not (voice_dir / f"{n}.onnx").exists()
    ]
    if not missing:
        return Check(
            Level.OK, "Stimme", "Piper: " + ", ".join(sorted(set(settings.tts.voices.values())))
        )
    in_repo = [n for n in missing if (REPO_VOICES / f"{n}.onnx").is_file()]
    pointers = [n for n in in_repo if (REPO_VOICES / f"{n}.onnx").stat().st_size < MIN_MODEL_BYTES]
    if pointers:
        return Check(
            Level.WARN,
            "Stimme",
            f"{', '.join(pointers)} ist nur ein Git-LFS-Zeiger",
            "git lfs install && git lfs pull",
        )
    if len(in_repo) == len(missing):
        return Check(Level.OK, "Stimme", "Piper – wird beim Start aus dem Repository übernommen")
    return Check(Level.FAIL, "Stimme", f"Piper-Stimme fehlt: {', '.join(missing)}", "flay setup")


def _vad(settings: Settings) -> Check:
    from flay.audio.vad import MODEL_FILE

    if (settings.data_dir / "models" / MODEL_FILE).exists():
        return Check(Level.OK, "Stille-Erkennung", "Silero VAD")
    return Check(Level.FAIL, "Stille-Erkennung", "Modell fehlt", "flay setup")


def _microphone(settings: Settings) -> Check:
    try:
        import sounddevice as sd

        device = sd.query_devices(settings.audio.input_device, "input")
    except Exception as exc:
        return Check(Level.FAIL, "Mikrofon", str(exc), "flay devices; audio.input_device setzen")
    return Check(Level.OK, "Mikrofon", str(device["name"]))


def _permissions() -> list[Check]:
    if sys.platform != "darwin":
        return []
    from flay.tools.pointer import accessibility_allowed

    checks = []
    if accessibility_allowed():
        checks.append(Check(Level.OK, "Bedienungshilfen", "erlaubt"))
    else:
        checks.append(
            Check(
                Level.FAIL,
                "Bedienungshilfen",
                "nicht erlaubt (Hotkey, Klicken)",
                "Systemeinstellungen → Datenschutz & Sicherheit → Bedienungshilfen",
            )
        )
    import Quartz

    if Quartz.CGPreflightScreenCaptureAccess():
        checks.append(Check(Level.OK, "Bildschirmaufnahme", "erlaubt"))
    else:
        checks.append(
            Check(
                Level.WARN,
                "Bildschirmaufnahme",
                "nicht erlaubt (Bildschirm ansehen)",
                "Systemeinstellungen → Datenschutz & Sicherheit → Bildschirmaufnahme",
            )
        )
    return checks


def _phpstorm(settings: Settings) -> Check:
    if not settings.tools.phpstorm.use_mcp:
        return Check(Level.OK, "PhpStorm", "MCP abgeschaltet (tools.phpstorm.use_mcp)")
    from flay.tools.phpstorm_mcp import PhpStormMcp

    url = asyncio.run(PhpStormMcp(settings.tools.phpstorm.mcp_url).url())
    if url is None:
        return Check(
            Level.WARN,
            "PhpStorm",
            "MCP-Server nicht gefunden (läuft PhpStorm?)",
            "PhpStorm: Settings → Tools → MCP Server aktivieren",
        )
    return Check(Level.OK, "PhpStorm", f"MCP-Server {url}")


def _claude() -> Check:
    if shutil.which("tmux"):
        return Check(Level.OK, "Claude Code", "tmux vorhanden (flay claude, Statusmeldungen)")
    return Check(
        Level.WARN,
        "Claude Code",
        "kein tmux – Aufträge gehen über die Zwischenablage",
        "brew install tmux" if sys.platform == "darwin" else "",
    )


def run(settings: Settings, print_line: Callable[[str], None] = print) -> int:
    """Run all checks; 1 if any is broken."""
    steps: list[Callable[[], Check | list[Check]]] = [
        _gpu,
        lambda: _ollama(settings),
        lambda: _voice(settings),
        lambda: _vad(settings),
        lambda: _microphone(settings),
        _permissions,
        lambda: _phpstorm(settings),
        _claude,
    ]
    print_line(f"Flay prüft diesen Rechner ({platform.system()} {platform.machine()}):")
    failed = False
    for step in steps:
        try:
            found = step()
        except Exception as exc:  # a check must never stop the others
            found = Check(Level.WARN, getattr(step, "__name__", "Prüfung"), f"Fehler: {exc}")
        for check in found if isinstance(found, list) else [found]:
            print_line(check.line())
            failed |= check.level is Level.FAIL
    print_line("Alles bereit." if not failed else "Mit ✗ markierte Punkte zuerst beheben.")
    return 1 if failed else 0
