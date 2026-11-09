from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from openai import OpenAIError
from pydantic import ValidationError

from flay import __version__
from flay.config import Settings, load_settings, resolve_config_path
from flay.core.language import Language
from flay.core.platform import detect_hardware
from flay.input.base import InputMode

if TYPE_CHECKING:
    from flay.training import Counts
    from flay.ui.tray import Tray
    from flay.wake import WakeWord
from flay.llm.base import ChatModel
from flay.memory.base import Fact, Memory, ShortTermMemory
from flay.offline import MissingModelError, enforce_offline, is_local_url

log = logging.getLogger("flay")

# Learned examples from which a LoRA adapter is worth another try (docs/roadmap.md).
TRAINING_WORTHWHILE = 300


def _info(settings: Settings, config: Path | None) -> int:
    hardware = detect_hardware()
    print(f"flay {__version__}")
    print(f"System:        {hardware.system} ({hardware.machine})")
    print(f"Beschleuniger: {hardware.accelerator}")
    print(f"Config:        {resolve_config_path(config) or 'Standardwerte'}")
    print()
    print(yaml.safe_dump(settings.model_dump(mode="json"), allow_unicode=True, sort_keys=False))
    return 0


def _devices() -> int:
    import sounddevice as sd

    print(sd.query_devices())
    return 0


async def _eval(
    settings: Settings,
    cases: Path | None,
    repeat: int,
    label: str | None,
    only: str,
    examples: Path | None = None,
) -> int:
    from flay.evaluation import DEFAULT_CASES, evaluate, save
    from flay.llm.factory import create_chat_model

    print(f"Auswertung: {settings.llm.model} @ {settings.llm.base_url}")
    report = await evaluate(
        settings,
        create_chat_model(settings.llm),
        cases or DEFAULT_CASES,
        repeat,
        label or settings.llm.model.replace("/", "_").replace(":", "_"),
        only=[int(n) for n in only.split(",") if n.strip()],
        examples=examples,
    )
    print()
    for key, value in report.summary().items():
        print(f"{key:22} {value}")
    print(f"Ergebnisse: {save(report, settings.data_dir / 'eval')}")
    return 0


async def _eval_stt(settings: Settings, recordings: str, backends: str) -> int:
    from flay.stt.factory import create_stt
    from flay.stt.vocabulary import Vocabulary
    from flay.stt_evaluation import (
        evaluate_stt,
        load_sentences,
        recordings_dir,
        save_report,
        synthesize_set,
    )
    from flay.tools.phpstorm import recent_projects
    from flay.tts.factory import create_tts

    sentences = load_sentences()
    directory = recordings_dir(settings.data_dir, recordings)
    if recordings == "synthetic" and not (directory / "01.wav").exists():
        print("Erzeuge synthetische Aufnahmen mit Piper ...")
        tts = create_tts(settings, detect_hardware(), backend="piper")
        await asyncio.to_thread(tts.load)
        await synthesize_set(tts, sentences, directory)
    if not (directory / "01.wav").exists():
        print(f"Keine Aufnahmen in {directory} – erst `flay record --set {recordings}`.")
        return 1
    vocabulary = Vocabulary(
        lambda: [p.name for p in recent_projects(settings.tools.phpstorm.config_dir)]
    )
    hardware = detect_hardware()
    summaries = []
    for backend in [b.strip() for b in backends.split(",") if b.strip()]:
        name = backend.removesuffix("+prompt")
        stt_settings = settings.stt.model_copy(
            update={"backend": "auto" if name == "whisper" else name}
        )
        stt = create_stt(settings.model_copy(update={"stt": stt_settings}), hardware)
        if backend.endswith("+prompt"):
            stt.prompt = vocabulary.prompt  # type: ignore[union-attr]
        print(f"\n== {backend} ({recordings})")
        try:
            report = await evaluate_stt(
                backend, stt, sentences, directory, correct=vocabulary.correct
            )
        except MissingModelError as exc:
            print(f"Modell fehlt: {exc} – mit --download laden.")
            continue
        summaries.append(report.summary())
        print(f"Ergebnisse: {save_report(report, settings.data_dir)}")
    print()
    for summary in summaries:
        print("  ".join(f"{k}={v}" for k, v in summary.items()))
    return 0


def _train_wake(settings: Settings, repeats: int) -> int:
    from flay.tts.voice import BUNDLED_DIR
    from flay.wake import FEATURE_MODELS
    from flay.wake_training import build, free_voices

    models = settings.data_dir / "models"
    if not all((models / name).exists() for name in FEATURE_MODELS):
        print("Merkmalsmodelle fehlen – erst 'flay setup' ausführen.")
        return 1
    repo_voices = Path(__file__).resolve().parents[2] / "voices"
    voices = free_voices(settings.data_dir / "voices", repo_voices, BUNDLED_DIR)
    # Kristin is checked last (held out): Ramona was trained from her, so holding out
    # Ramona would look better than it is.
    voices.sort(key=lambda path: "kristin" in path.name)
    if len(voices) < 2:
        print("Zu wenige freie Piper-Stimmen gefunden – erst 'flay setup' ausführen.")
        return 1
    own = settings.data_dir / "wake"
    print(f"Trainiere „Hey Flay“ mit {', '.join(v.stem for v in voices)}", end="")
    print(f" und {len(list(own.glob('*.wav')))} eigenen Aufnahmen …" if own.is_dir() else " …")
    classifier, report = build(voices, models, own, repeats=repeats)
    target = models / "hey_flay.npz"
    classifier.save(target)
    print(
        f"Geprüft mit der zurückgehaltenen Stimme {report.voice}: „Hey Flay“ erkannt "
        f"{report.recall:.0%}, Fehlauslösungen {report.false_accepts} von {report.negatives} "
        f"anderen Sätzen ({report.negative_minutes:.1f} min, Schwelle {report.threshold})."
    )
    print(f"Gespeichert: {target}")
    return 0


def _train_export(
    settings: Settings, examples_file: Path | None, out: Path | None, synthetic: int = 0
) -> int:
    from flay.evaluation import DEFAULT_CASES, _state_line, _tool_specs, load_cases
    from flay.memory.base import MISTAKE
    from flay.memory.store import MemoryStore
    from flay.training import export, from_file, from_memory

    mistakes = []
    if examples_file is not None:
        examples = from_file(examples_file)
    else:
        store = MemoryStore(_memory_path(settings))
        examples = from_memory(store)
        mistakes = from_memory(store, MISTAKE)
        store.close()
    if not examples and not synthetic:
        print(
            "Keine Beispiele – Flay lernt sie im Betrieb; sonst --examples eval/examples.yaml "
            "oder --synthetic 200."
        )
        return 1
    # The usual situation (open projects, Claude sessions) as in the test set.
    state, _ = load_cases(DEFAULT_CASES)
    target = out or settings.data_dir / "training"
    extra = []
    if synthetic:
        from flay.training_synthetic import synthetic_rows

        extra = synthetic_rows(settings, _tool_specs(), synthetic)
    train, valid = export(
        settings, _tool_specs(), examples, target, _state_line(state) or None, mistakes, extra
    )
    print(f"{len(examples)} Beispiele, {len(mistakes)} Korrekturen")
    print(f"{train} Trainings- und {valid} Prüfzeilen in {target}")
    return 0


def _record(settings: Settings, name: str, overwrite: bool) -> int:
    from flay.audio.microphone import Microphone
    from flay.stt_evaluation import SAMPLE_RATE, load_sentences, recordings_dir, save_wav

    sentences = load_sentences()
    directory = recordings_dir(settings.data_dir, name)
    microphone = Microphone(SAMPLE_RATE, settings.audio.input_device)
    print(f"Mikrofon: {microphone.device_name}")
    print(f"Aufnahmen landen in {directory} (bleiben lokal).")
    print("Sprich jeden Satz so, wie du ihn Flay sagen würdest.\n")
    microphone.open()
    try:
        for index, sentence in enumerate(sentences, start=1):
            path = directory / f"{index:02}.wav"
            if path.exists() and not overwrite:
                continue
            while True:
                input(f"[{index}/{len(sentences)}] {sentence.text}\n   Enter = Aufnahme starten ")
                microphone.start_recording()
                input("   ... sprechen, Enter = stoppen ")
                save_wav(path, microphone.stop_recording())
                if input("   Enter = weiter, r = nochmal ").strip().lower() != "r":
                    break
    finally:
        microphone.close()
    print(f"\nFertig. Vergleich: uv run flay eval-stt --set {name}")
    return 0


VOICE_SCRIPT = {
    Language.DE: (
        "Guten Morgen! Ich bin Flay, dein Sprachassistent. Ich öffne Projekte, starte Befehle "
        "im Terminal und beantworte deine Fragen. Sag einfach, was du brauchst."
    ),
    Language.EN: (
        "Good morning! I'm Flay, your voice assistant. I open projects, run commands in the "
        "terminal and answer your questions. Just tell me what you need."
    ),
}
VOICE_TEST = {
    Language.DE: "Hallo! So klinge ich auf Deutsch. Ich öffne das Projekt KX Forms.",
    Language.EN: "Hi! This is how I sound in English. I'm opening the project KX Forms.",
}
# A clone needs a few seconds of speech; much more only slows down every sentence.
VOICE_SECONDS = (5.0, 20.0)


def _voice_record(settings: Settings, languages: list[Language]) -> int:
    from flay.assistant import SILENCE_DBFS
    from flay.audio.microphone import Microphone
    from flay.stt_evaluation import save_wav
    from flay.tts.qwen3 import SAMPLE_RATE
    from flay.tts.voice import own_recording

    microphone = Microphone(SAMPLE_RATE, settings.audio.input_device)
    print(f"Mikrofon: {microphone.device_name}")
    print("Lies den Text ruhig und deutlich vor, so wie Flay klingen soll. Die Aufnahme")
    print("bleibt auf diesem Rechner; Flay spricht danach in beiden Sprachen mit deiner Stimme.\n")
    microphone.open()
    try:
        for language in languages:
            path = own_recording(settings.data_dir, language)
            while True:
                input(f"[{language}] {VOICE_SCRIPT[language]}\n   Enter = Aufnahme starten ")
                microphone.start_recording()
                input("   ... vorlesen, Enter = stoppen ")
                audio = microphone.stop_recording()
                low, high = VOICE_SECONDS
                if audio.level_dbfs < SILENCE_DBFS:
                    print("   Nur Stille aufgenommen – Mikrofon prüfen (flay mic-test).")
                elif not low <= audio.duration <= high:
                    print(f"   {audio.duration:.0f} s – bitte zwischen {low:.0f} und {high:.0f} s.")
                else:
                    save_wav(path, audio)
                    path.with_suffix(".txt").write_text(VOICE_SCRIPT[language] + "\n", "utf-8")
                    print(f"   Gespeichert: {path}")
                    break
                if input("   Enter = nochmal, q = abbrechen ").strip().lower() == "q":
                    return 1
    finally:
        microphone.close()
    print("\nFertig. Probehören: flay voice test (tts.voice.name: auto oder own)")
    return 0


def _voice_dataset(settings: Settings, languages: list[Language]) -> int:
    """Read the training sentences aloud for an own Piper voice; resumable."""
    from flay.assistant import SILENCE_DBFS
    from flay.audio.microphone import Microphone
    from flay.stt_evaluation import save_wav
    from flay.voice_recording import DATASET_RATE, Progress, check_take, sentences

    folder = settings.data_dir / "voice_dataset"
    progress = Progress(folder)
    done = progress.done()
    microphone = Microphone(DATASET_RATE, settings.audio.input_device)
    print(f"Mikrofon: {microphone.device_name}")
    print("Lies jeden Satz ruhig und natürlich vor, immer gleich weit vom Mikrofon, in einem")
    print("leisen Raum. 30–60 Minuten ergeben eine brauchbare Stimme; aufhören (q) und später")
    print(f"weitermachen geht jederzeit. Aufnahmen: {folder}\n")
    microphone.open()
    try:
        for language in languages:
            todo = [s for s in sentences(language) if (language.value, s) not in done]
            total = len(sentences(language))
            for text in todo:
                number = total - len(todo) + todo.index(text) + 1
                minutes = progress.seconds() / 60
                while True:
                    print(f"[{language} {number}/{total}, {minutes:.0f} min] {text}")
                    if input("   Enter = Aufnahme, q = Ende ").strip().lower() == "q":
                        return 0
                    microphone.start_recording()
                    input("   ... vorlesen, Enter = stoppen ")
                    audio = microphone.stop_recording()
                    check = check_take(audio, text, SILENCE_DBFS)
                    if not check.ok:
                        print(f"   Bitte nochmal: {check.problem}.")
                        continue
                    answer = input("   Enter = übernehmen, r = nochmal, s = überspringen ")
                    if answer.strip().lower() == "r":
                        continue
                    if answer.strip().lower() != "s":
                        name = f"{language.value}_{number:04d}.wav"
                        save_wav(folder / name, audio)
                        progress.add(language, text, name, round(audio.duration, 2))
                    break
    finally:
        microphone.close()
    print(f"\nFertig: {progress.seconds() / 60:.0f} min. Training: tools/voice-training/README.md")
    return 0


def _voice_wake(settings: Settings) -> int:
    """Record "Hey Flay" twenty times for flay train-wake."""
    from flay.assistant import SILENCE_DBFS
    from flay.audio.microphone import Microphone
    from flay.stt_evaluation import save_wav
    from flay.voice_recording import WAKE_RATE, WAKE_TAKES

    folder = settings.data_dir / "wake"
    microphone = Microphone(WAKE_RATE, settings.audio.input_device)
    print(f"Mikrofon: {microphone.device_name}")
    print(
        f"Sag {len(WAKE_TAKES)}-mal „Hey Flay“, jedes Mal etwas anders. Danach: flay train-wake\n"
    )
    microphone.open()
    try:
        number = len(list(folder.glob("hey_flay_*.wav"))) if folder.is_dir() else 0
        for take in WAKE_TAKES:
            while True:
                if input(f"„Hey Flay“ – {take}. Enter = Aufnahme, q = Ende ").strip() == "q":
                    return 0
                microphone.start_recording()
                input("   ... sprechen, Enter = stoppen ")
                audio = microphone.stop_recording()
                if audio.level_dbfs < SILENCE_DBFS or audio.duration < 0.4:
                    print("   Nichts gehört – nochmal.")
                    continue
                number += 1
                save_wav(folder / f"hey_flay_{number:03d}.wav", audio)
                break
    finally:
        microphone.close()
    print(f"\nGespeichert in {folder}. Jetzt: flay train-wake")
    return 0


async def _voice_test(settings: Settings, name: str | None) -> int:
    from flay.audio.speaker import Speaker
    from flay.tts.factory import create_tts

    if name:
        voice = settings.tts.voice.model_copy(update={"name": name})
        settings = settings.model_copy(
            update={"tts": settings.tts.model_copy(update={"voice": voice})}
        )
    tts = create_tts(settings, detect_hardware())
    await asyncio.to_thread(tts.load)
    speaker = Speaker(settings.audio.output_device)
    speaker.prepare(tts.sample_rate)
    try:
        for language, text in VOICE_TEST.items():
            print(f"[{language}] {text}")
            await speaker.play(tts.synthesize(text, language))
    finally:
        speaker.close()
    return 0


async def _bench(settings: Settings, rounds: int) -> int:
    from flay.bench import benchmark, format_report
    from flay.llm.factory import create_chat_model

    print(f"Benchmark: {settings.llm.model} @ {settings.llm.base_url}")
    results = await benchmark(
        create_chat_model(settings.llm), settings.persona.system_prompt, rounds
    )
    print(format_report(results))
    return 0


async def _mic_test(settings: Settings, seconds: float) -> int:
    from flay.assistant import SILENCE_DBFS
    from flay.audio.microphone import Microphone
    from flay.stt.factory import create_stt
    from flay.stt.whisper import WHISPER_SAMPLE_RATE

    microphone = Microphone(WHISPER_SAMPLE_RATE, settings.audio.input_device)
    stt = create_stt(settings, detect_hardware())
    await stt.warm_up()
    print(f"Mikrofon: {microphone.device_name}")
    print(f"Bitte {seconds:.0f} Sekunden sprechen ...")
    microphone.open()
    try:
        microphone.start_recording()
        await asyncio.sleep(seconds)
        audio = microphone.stop_recording()
    finally:
        microphone.close()

    print(f"Aufgenommen: {audio.duration:.1f}s, Pegel {audio.level_dbfs:.0f} dBFS")
    if audio.duration == 0:
        print("Keine Audiodaten erhalten – Eingabegerät prüfen ('flay devices').")
        return 1
    if audio.level_dbfs < SILENCE_DBFS:
        print(
            "Das Mikrofon liefert (fast) Stille. Unter macOS: Systemeinstellungen → Datenschutz "
            "& Sicherheit → Mikrofon → Terminal-App erlauben; sonst Eingabegerät prüfen."
        )
        return 1
    transcript = await stt.transcribe(audio)
    print(f"Erkannt ({transcript.language}): {transcript.text or '–'}")
    return 0


def _memory_path(settings: Settings) -> Path:
    return settings.data_dir / "memory.sqlite3"


def _create_memory(settings: Settings, llm: ChatModel) -> Memory:
    from flay.memory.persistent import PersistentMemory
    from flay.memory.store import MemoryStore

    if not settings.memory.enabled:
        return ShortTermMemory(settings.memory.recent_messages)
    from flay.memory.embeddings import create_embedder

    log.info("Gedächtnis: %s", _memory_path(settings))
    return PersistentMemory(
        MemoryStore(_memory_path(settings)),
        llm,
        settings.memory,
        embedder=create_embedder(settings),
    )


def _memory_line(fact: Fact) -> str:
    from flay.memory.base import IMPORTANCE_NAMES

    written = time.strftime("%Y-%m-%d %H:%M", time.localtime(fact.created_at))
    kind = {
        "fact": "Fakt", "episode": "Gespräch", "example": "Beispiel", "mistake": "Fehler",
        "verb": "Verb",
    }.get(fact.kind, fact.kind)  # fmt: skip
    line = f"  {fact.id:>4}  {written}  {kind:8} {IMPORTANCE_NAMES[fact.importance]:13} {fact.text}"
    if fact.kind == "example" and fact.data:
        meant = json.loads(fact.data)
        sure = "bestätigt" if meant.get("confirmed") else f"{meant.get('times', 1)}×"
        line += f"  → {meant['tool']}({meant.get('arguments')}) [{sure}]"
    if fact.kind == "verb" and fact.data:
        line += f"  → {json.loads(fact.data)['tool']}"
    if fact.kind == "mistake" and fact.data:
        meant = json.loads(fact.data)
        line += f"  ≠ {meant['tool']}({meant.get('arguments')}) [{meant.get('times', 1)}×]"
    return line


def _training_line(counts: Counts) -> str:
    """How much has been collected for a LoRA adapter (``flay train-export``)."""
    line = (
        f"{counts.examples} Beispiele gesammelt ({counts.confirmed} bestätigt), "
        f"{counts.mistakes} Korrekturen"
    )
    if counts.examples < TRAINING_WORTHWHILE:
        line += f" – ein LoRA-Versuch lohnt sich ab etwa {TRAINING_WORTHWHILE}"
    return line


def _memory_command(settings: Settings, action: str, words: list[str]) -> int:
    from flay.memory.embeddings import create_embedder
    from flay.memory.recall import Recall
    from flay.memory.store import MemoryStore

    store = MemoryStore(_memory_path(settings))
    try:
        if action == "forget":
            removed = store.remove_facts([int(word) for word in words])
            print(f"{removed} Erinnerung(en) gelöscht.")
        elif action == "search":
            recall = Recall(store, create_embedder(settings), settings.memory.recall)

            async def search() -> list[Fact]:
                await recall.embed_pending()
                return await recall.search(" ".join(words))

            for fact in asyncio.run(search()):
                print(_memory_line(fact))
        elif action == "clear":
            store.clear()
            print("Gedächtnis gelöscht (Gespräche, Zusammenfassungen, Fakten).")
        elif action == "clear-chat":
            store.clear(facts=False)
            print("Gesprächsverlauf und Zusammenfassungen gelöscht, Fakten behalten.")
        else:
            from flay.training import counts

            facts = store.facts()
            print(f"Gedächtnis: {_memory_path(settings)}")
            print(f"Training: {_training_line(counts(store))}")
            print("Erinnerungen:" if facts else "Erinnerungen: keine")
            for fact in facts:
                print(_memory_line(fact))
    finally:
        store.close()
    return 0


def _claude(claude_args: list[str]) -> int:
    """Start Claude Code in a tmux session named after the current project."""
    from flay.tools.claude_code import session_name

    claude = shutil.which("claude")
    if claude is None:
        print("Claude Code ('claude') wurde nicht gefunden.")
        return 1
    tmux = shutil.which("tmux")
    if tmux is None or sys.platform == "win32":
        print("tmux ist nicht verfügbar – starte Claude direkt. Flay nutzt die Zwischenablage.")
        return subprocess.call([claude, *claude_args])
    project = Path.cwd()
    name = session_name(project)
    # -A attaches if the session already exists. No status bar and mouse scrolling, so it
    # looks and feels like a plain terminal inside PhpStorm.
    command = [tmux, "new-session", "-A", "-s", name, "-c", str(project), claude, *claude_args]
    options = [";", "set-option", "-t", name, "status", "off"]
    options += [";", "set-option", "-t", name, "mouse", "on"]
    os.execvp(tmux, command + options)


def _wake_word(settings: Settings) -> WakeWord:
    from flay.wake import WakeWord, classifier_path

    models = settings.data_dir / "models"
    try:
        return WakeWord(
            models, classifier_path(settings.data_dir), settings.input.continuous.wake_threshold
        )
    except Exception as exc:  # feature models missing: `flay setup`
        raise MissingModelError("Aktivierungswort (flay setup)") from exc


async def _setup(settings: Settings) -> int:
    """The only command that downloads anything: all models Flay needs, once."""
    from openai import AsyncOpenAI

    from flay.audio.vad import SileroVAD
    from flay.stt.factory import create_stt
    from flay.tts.factory import create_tts, resolve_backend

    print("Lade fehlende Modelle (danach arbeitet Flay komplett offline):")
    hardware = detect_hardware()
    stt = create_stt(settings, hardware)
    stt.download()
    print(f"  ✓ Spracherkennung ({type(stt).__name__})")
    # Piper also speaks the synthetic test recordings of eval-stt.
    create_tts(settings, hardware, backend="piper").download()
    print("  ✓ Stimmen (Piper)")
    if resolve_backend(settings.tts.backend, hardware) == "qwen3":
        create_tts(settings, hardware).download()
        print("  ✓ Stimme (Qwen3-TTS)")
    SileroVAD(settings.data_dir / "models").download()
    print("  ✓ Stille-Erkennung (Silero VAD)")
    from flay.wake import download_features

    download_features(settings.data_dir / "models")
    print("  ✓ Aktivierungswort-Merkmale (openWakeWord, Apache 2.0)")

    # The LLM is served by its own program; Flay only checks that the model is there.
    client = AsyncOpenAI(base_url=settings.llm.base_url, api_key="not-needed")
    try:
        models = {model.id async for model in client.models.list()}
    except OpenAIError as exc:
        print(f"  ? LLM-Server {settings.llm.base_url} nicht erreichbar: {exc}")
        return 1
    if settings.llm.model in models:
        print(f"  ✓ LLM {settings.llm.model}")
    else:
        print(f"  ✗ LLM {settings.llm.model} fehlt – z.B. 'ollama pull {settings.llm.model}'")
        return 1
    from flay.llm import ollama
    from flay.memory.embeddings import embedding_model

    # Small-context variants so the helper models fit next to the main model.
    for line in await asyncio.to_thread(ollama.setup, settings.llm.base_url):
        print(line)
    recall = settings.memory.recall
    if settings.memory.enabled and recall.enabled:
        base_url = recall.base_url or settings.llm.base_url
        if recall.base_url and recall.base_url != settings.llm.base_url:
            client = AsyncOpenAI(base_url=recall.base_url, api_key="not-needed")
        models = {model.id async for model in client.models.list()}
        model = await asyncio.to_thread(embedding_model, recall.model, base_url)
        if model in models:
            print(f"  ✓ Gedächtnis-Suche {model}")
        else:
            print(
                f"  ✗ Gedächtnis-Suche {model} fehlt – z.B. 'ollama pull {model}' "
                "(ohne sucht Flay nur nach Stichwörtern)"
            )
            return 1
    return 0


def _autostart(settings: Settings, config: Path | None, action: str) -> int:
    from flay import autostart

    if action == "on":
        path = autostart.enable(resolve_config_path(config), settings.data_dir)
        print(f"Autostart eingerichtet: {path}")
        print("Flay startet ab der nächsten Anmeldung im Hintergrund (Symbol in der Menüleiste).")
        if sys.platform == "darwin":
            print(
                "Hinweis: macOS fragt dann für Flay erneut nach Mikrofon, Bedienungshilfen und "
                "Bildschirmaufnahme. Log: " + str(settings.data_dir / "flay.log")
            )
    elif action == "off":
        print("Autostart entfernt." if autostart.disable() else "Autostart war nicht eingerichtet.")
    else:
        print(f"Autostart: {'an' if autostart.is_enabled() else 'aus'} ({autostart.entry_path()})")
    return 0


def _start(settings: Settings) -> int:
    """Run Flay, with a menu bar / tray icon if enabled."""
    if not settings.ui.tray:
        return asyncio.run(_run(settings))
    try:
        from flay.ui.tray import Tray

        tray = Tray(f"http://127.0.0.1:{settings.ui.port}/" if settings.ui.enabled else None)
    except Exception as exc:  # e.g. no graphical session
        log.warning("Menüleisten-Symbol nicht verfügbar (%s) – Flay läuft ohne.", exc)
        return asyncio.run(_run(settings))
    return tray.run(lambda tray: _run(settings, tray))


async def _run(settings: Settings, tray: Tray | None = None) -> int:
    # Heavy imports (audio, ML) only when the assistant actually starts.
    from flay.assistant import Assistant
    from flay.audio.microphone import Microphone
    from flay.audio.speaker import Speaker
    from flay.audio.vad import SileroVAD
    from flay.bench import run_prompt
    from flay.core.events import AssistantState
    from flay.input.base import InputMode, UtteranceSource
    from flay.input.continuous import ContinuousListener, SegmenterConfig
    from flay.input.push_to_talk import HotkeyPushToTalk, TerminalPushToTalk
    from flay.input.switchable import SwitchableSource
    from flay.llm.factory import create_chat_model
    from flay.memory.persistent import PersistentMemory
    from flay.stt.factory import create_stt
    from flay.stt.vocabulary import Vocabulary
    from flay.stt.whisper import WHISPER_SAMPLE_RATE
    from flay.tools.factory import create_state, create_tools
    from flay.tools.phpstorm import recent_projects
    from flay.tts.factory import create_tts
    from flay.ui.server import WebUI

    if not is_local_url(settings.llm.base_url):
        log.warning(
            "ACHTUNG: llm.base_url (%s) ist nicht dieser Rechner – Gespräche, Gedächtnis und "
            "Screenshots verlassen damit den Rechner.",
            settings.llm.base_url,
        )
    hardware = detect_hardware()
    stt = create_stt(settings, hardware)

    def project_names() -> list[str]:
        return [p.name for p in recent_projects(settings.tools.phpstorm.config_dir)]

    vocabulary = Vocabulary(project_names)
    if settings.stt.vocabulary_prompt and hasattr(stt, "prompt"):
        stt.prompt = vocabulary.prompt
    tts = create_tts(settings, hardware)
    llm = create_chat_model(settings.llm)
    continuous = settings.input.mode is InputMode.CONTINUOUS
    # Always loaded: continuous listening (switchable in the UI) finds utterances with it.
    vad = SileroVAD(settings.data_dir / "models")

    log.info("Lade Modelle ...")
    await asyncio.gather(
        stt.warm_up(),
        asyncio.to_thread(tts.load),
        asyncio.to_thread(vad.load),
        run_prompt(llm, settings.persona.system_prompt, "Hallo"),
    )

    speaker = Speaker(settings.audio.output_device)
    speaker.prepare(tts.sample_rate)
    microphone = Microphone(WHISPER_SAMPLE_RATE, settings.audio.input_device)
    log.info("Mikrofon: %s", microphone.device_name)

    def interrupt() -> None:
        assistant.interrupt()

    push_to_talk: UtteranceSource
    if settings.input.trigger == "terminal":
        push_to_talk = TerminalPushToTalk(microphone, on_press=interrupt)
        ready = "Enter startet die Aufnahme, Enter beendet sie."
    else:
        push_to_talk = HotkeyPushToTalk(
            microphone, settings.input.push_to_talk_key, on_press=interrupt
        )
        ready = f"Taste '{settings.input.push_to_talk_key}' gedrückt halten und sprechen."
    options = settings.input.continuous

    # Loaded when it can be: "Hey Flay" can be switched on in UI and tray at any time.
    try:
        wake: WakeWord | None = _wake_word(settings)
    except MissingModelError:
        if options.wake_word:
            raise
        log.debug("Aktivierungswort nicht verfügbar (flay setup)")
        wake = None

    def continuous_listener() -> UtteranceSource:
        # Built anew whenever listening (or the wake word) is switched on.
        wake_on = wake is not None and bool(assistant.wake_word)
        listener = ContinuousListener(
            vad,
            SegmenterConfig(
                threshold=options.threshold,
                end_silence_ms=options.end_silence_ms,
                min_speech_ms=options.min_speech_ms,
                max_utterance_ms=int(options.max_utterance_seconds * 1000),
                barge_in=options.barge_in or wake_on,
            ),
            # With the wake word only a confirmed "Hey Flay …" interrupts; speech around the
            # user neither stops Flay nor costs recognition time.
            on_speech_start=(lambda: None) if wake_on else interrupt,
            is_speaking=lambda: assistant.state is AssistantState.SPEAKING,
            device=settings.audio.input_device,
            on_tentative=None if wake_on else (lambda audio: assistant.prefetch(audio)),
        )
        if not wake_on or wake is None:
            return listener
        from flay.wake import WakeGate

        # No interrupting yet: the assistant does so once the transcript says "Hey Flay".
        return WakeGate(listener, wake)

    # Push-to-talk always works; continuous listening can be switched on in UI and tray.
    source = SwitchableSource(push_to_talk, continuous_listener, start_continuous=continuous)
    if continuous:
        ready += " Dauerhaftes Zuhören ist an – einfach sprechen."
        if options.barge_in:
            ready += " Unterbrechen per Sprache ist an (Headset empfohlen)."

    memory = _create_memory(settings, llm)
    if isinstance(memory, PersistentMemory):
        log.info("Training: %s", _training_line(memory.training_counts()))
    # Loads the embedding model in the background; until then recall uses keywords only.
    warm_memory = getattr(memory, "warm_up", None)
    memory_ready = asyncio.create_task(warm_memory()) if callable(warm_memory) else None
    tools = create_tools(settings, memory if isinstance(memory, PersistentMemory) else None)
    if tools:
        log.info("Werkzeuge: %s", ", ".join(spec.name for spec in tools.specs))
    if "open_project" in tools:
        from flay.tools.phpstorm import open_project_setting

        if open_project_setting(settings.tools.phpstorm.config_dir) not in (0, None):
            log.info(
                "Hinweis: PhpStorm fragt beim Öffnen eines Projekts nach dem Fenster. Für "
                "'immer neues Fenster': Settings → Appearance & Behavior → System Settings → "
                "Project → Open project in: New window."
            )
    if "send_to_claude" in tools:
        from flay.tools.claude_code import Tmux

        sessions = [s.name for s in await Tmux().sessions()]
        log.info(
            "Claude-Sitzungen: %s",
            ", ".join(sessions)
            or "keine – Claude im PhpStorm-Terminal mit 'flay claude' starten, sonst nutzt "
            "Flay die Zwischenablage",
        )
    assistant = Assistant(
        settings,
        source,
        stt,
        llm,
        tts,
        speaker,
        vad=vad if settings.audio.vad else None,
        memory=memory,
        tools=tools,
        vocabulary=vocabulary,
        state=state.describe if (state := create_state(settings)) else None,
        projects=project_names,
        wake_word_available=wake is not None,
    )
    await assistant.prepare_fillers()
    if tray is not None:
        tray.attach(assistant)
    ui = WebUI(assistant, settings.ui.port) if settings.ui.enabled else None
    try:
        if ui is not None:
            try:
                await ui.start()
            except OSError as exc:
                log.error("UI konnte nicht starten (Port %s belegt?): %s", settings.ui.port, exc)
                ui = None
            else:
                log.info("UI: %s", ui.url)
                if settings.ui.open_browser:
                    webbrowser.open(ui.url)
        log.info("Bereit. %s Ctrl+C beendet flay.", ready)
        await assistant.run()
    finally:
        if ui is not None:
            await ui.stop()
        speaker.close()
        if memory_ready is not None:
            memory_ready.cancel()
        await memory.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flay", description="Lokaler Sprachassistent")
    parser.add_argument("--version", action="version", version=f"flay {__version__}")
    parser.add_argument("-c", "--config", type=Path, help="Pfad zur YAML-Konfiguration")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-Ausgaben")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("setup", help="Alle Modelle einmalig herunterladen (sonst offline)")
    autostart = commands.add_parser("autostart", help="Flay bei der Anmeldung starten")
    autostart.add_argument("action", nargs="?", choices=["on", "off", "status"], default="status")
    commands.add_parser("info", help="Hardware und aufgelöste Konfiguration anzeigen")
    commands.add_parser("devices", help="Audio-Geräte auflisten")
    memory = commands.add_parser(
        "memory", help="Erinnerungen anzeigen, durchsuchen, vergessen, löschen"
    )
    memory.add_argument(
        "action",
        nargs="?",
        choices=["list", "search", "forget", "clear", "clear-chat"],
        default="list",
    )
    memory.add_argument("words", nargs="*", help="IDs für 'forget', Suchtext für 'search'")
    eval_stt = commands.add_parser(
        "eval-stt", help="Spracherkenner mit denselben Aufnahmen vergleichen"
    )
    eval_stt.add_argument(
        "--set", default="synthetic", help="Aufnahmen: synthetic (Piper) oder aus flay record"
    )
    eval_stt.add_argument(
        "--backends",
        default="parakeet,canary,whisper,whisper+prompt",
        help="Komma-Liste; +prompt = Whisper mit Projektnamen als Vorgabe",
    )
    eval_stt.add_argument("--download", action="store_true", help="fehlende Modelle herunterladen")
    train_export = commands.add_parser(
        "train-export", help="Trainingsdaten für einen LoRA-Adapter aus gelernten Beispielen"
    )
    train_export.add_argument(
        "--examples", type=Path, help="Beispiele aus dieser Datei statt aus dem Gedächtnis"
    )
    train_export.add_argument("--out", type=Path, help="Zielordner (Standard: Datenordner)")
    train_export.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="zusätzlich so viele erzeugte Zeilen (genuschelte Befehle, erfundene Projekte)",
    )
    train_wake = commands.add_parser(
        "train-wake", help="Aktivierungswort „Hey Flay“ trainieren (eigene Erkennung)"
    )
    train_wake.add_argument(
        "--repeats", type=int, default=120, help="„Hey Flay“-Beispiele je Stimme (Standard 120)"
    )
    record = commands.add_parser("record", help="Testsätze für eval-stt einsprechen")
    record.add_argument("--set", default="mine", help="Name der Aufnahmen")
    record.add_argument("--overwrite", action="store_true", help="vorhandene neu aufnehmen")
    voice = commands.add_parser("voice", help="Eigene Stimme aufnehmen oder Stimmen probehören")
    voice.add_argument("action", choices=["record", "test", "dataset", "wake"])
    voice.add_argument(
        "--language", choices=[lang.value for lang in Language], help="nur diese Sprache aufnehmen"
    )
    voice.add_argument("--name", help="test: diese Stimme statt tts.voice.name (z.B. ramona)")
    mic_test = commands.add_parser("mic-test", help="Mikrofon aufnehmen, Pegel und Text zeigen")
    mic_test.add_argument("--seconds", type=float, default=4.0)
    run = commands.add_parser("run", help="Assistent starten")
    run.add_argument("--terminal", action="store_true", help="Enter statt Hotkey verwenden")
    run.add_argument("--continuous", action="store_true", help="Dauerhaft zuhören (ohne Taste)")
    run.add_argument("--no-ui", action="store_true", help="Web-UI nicht starten")
    run.add_argument("--no-tray", action="store_true", help="Kein Menüleisten-/Tray-Symbol")
    run.add_argument(
        "--background", action="store_true", help="Browser nicht öffnen (z.B. beim Autostart)"
    )
    claude = commands.add_parser(
        "claude", help="Claude Code in einer tmux-Sitzung starten, in die Flay schreiben kann"
    )
    claude.add_argument("claude_args", nargs=argparse.REMAINDER, help="Argumente für claude")
    evaluation = commands.add_parser(
        "eval", help="Trefferquote und Antwortzeit des LLM mit dem Testsatz messen"
    )
    evaluation.add_argument("--cases", type=Path, help="Testsatz (Standard: eval/cases.yaml)")
    evaluation.add_argument("--repeat", type=int, default=2, help="Durchläufe pro Fall")
    evaluation.add_argument("--label", help="Name für die Ergebnisdatei")
    evaluation.add_argument("--only", default="", help="nur diese Fälle, z.B. 3,8,11")
    evaluation.add_argument("--model", help="Modell aus der Config überschreiben")
    evaluation.add_argument("--base-url", help="Server-URL aus der Config überschreiben")
    evaluation.add_argument(
        "--structured-routing", action="store_true", help="Weiterleitung per Formular testen"
    )
    evaluation.add_argument(
        "--thinking-model", help="Denken auf Wunsch mit diesem Modell testen (llm.thinking)"
    )
    evaluation.add_argument(
        "--examples", type=Path, help="Beispiele vorab ins Gedächtnis laden (eval/examples.yaml)"
    )
    evaluation.add_argument(
        "--extra-body",
        help="zusätzliche Request-Felder als JSON, z.B. für llama.cpp + Qwen3.5: "
        '\'{"chat_template_kwargs": {"enable_thinking": false}}\'',
    )

    bench = commands.add_parser("bench", help="LLM-Latenz messen")
    bench.add_argument("--rounds", type=int, default=2)
    bench.add_argument("--model", help="Modell aus der Config überschreiben")
    bench.add_argument("--base-url", help="Server-URL aus der Config überschreiben")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    enforce_offline(
        allow_downloads=args.command == "setup" or (args.command == "eval-stt" and args.download)
    )

    # Third-party libraries only report warnings; flay's own loggers are verbose.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # flay logs everything (the UI's debug window shows DEBUG too); the console shows
    # INFO, or DEBUG with -v.
    log.setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    try:
        settings = load_settings(args.config)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        print(f"Konfiguration ungültig: {exc}")
        return 1

    try:
        if args.command == "autostart":
            return _autostart(settings, args.config, args.action)
        if args.command == "setup":
            return asyncio.run(_setup(settings))
        if args.command == "info":
            return _info(settings, args.config)
        if args.command == "devices":
            return _devices()
        if args.command == "claude":
            return _claude(args.claude_args)
        if args.command == "memory":
            return _memory_command(settings, args.action, args.words)
        if args.command == "voice" and args.action == "dataset":
            languages = [Language(args.language)] if args.language else list(Language)
            return _voice_dataset(settings, languages)
        if args.command == "voice" and args.action == "wake":
            return _voice_wake(settings)
        if args.command == "voice" and args.action == "record":
            languages = [Language(args.language)] if args.language else list(Language)
            return _voice_record(settings, languages)
        if args.command == "voice":
            return asyncio.run(_voice_test(settings, args.name))
        if args.command == "mic-test":
            return asyncio.run(_mic_test(settings, args.seconds))
        if args.command == "eval-stt":
            return asyncio.run(_eval_stt(settings, args.set, args.backends))
        if args.command == "train-export":
            return _train_export(settings, args.examples, args.out, args.synthetic)
        if args.command == "train-wake":
            return _train_wake(settings, args.repeats)
        if args.command == "record":
            return _record(settings, args.set, args.overwrite)
        if args.command == "eval":
            overrides: dict[str, Any] = {"model": args.model, "base_url": args.base_url}
            if args.extra_body:
                overrides["extra_body"] = json.loads(args.extra_body)
            if args.structured_routing:
                overrides["structured_routing"] = True
            if args.thinking_model:
                overrides["thinking"] = settings.llm.thinking.model_copy(
                    update={"enabled": True, "model": args.thinking_model}
                )
            llm = settings.llm.model_copy(update={k: v for k, v in overrides.items() if v})
            return asyncio.run(
                _eval(
                    settings.model_copy(update={"llm": llm}),
                    args.cases,
                    args.repeat,
                    args.label,
                    args.only,
                    args.examples,
                )
            )
        if args.command == "bench":
            overrides = {"model": args.model, "base_url": args.base_url}
            llm = settings.llm.model_copy(update={k: v for k, v in overrides.items() if v})
            return asyncio.run(_bench(settings.model_copy(update={"llm": llm}), args.rounds))
        if args.terminal:
            input_settings = settings.input.model_copy(update={"trigger": "terminal"})
            settings = settings.model_copy(update={"input": input_settings})
        if args.continuous:
            input_settings = settings.input.model_copy(update={"mode": InputMode.CONTINUOUS})
            settings = settings.model_copy(update={"input": input_settings})
        ui_updates: dict[str, bool] = {}
        if args.no_ui:
            ui_updates["enabled"] = False
        if args.background:
            ui_updates["open_browser"] = False
        if args.no_tray:
            ui_updates["tray"] = False
        if ui_updates:
            settings = settings.model_copy(update={"ui": settings.ui.model_copy(update=ui_updates)})
        return _start(settings)
    except KeyboardInterrupt:
        return 0
    except OpenAIError as exc:
        print(f"LLM-Server nicht erreichbar ({settings.llm.base_url}): {exc}")
        return 1
    except MissingModelError as exc:
        print(exc)
        return 1
