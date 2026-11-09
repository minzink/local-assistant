"""Qwen3-TTS 0.6B (Apache 2.0): one cloned voice for German and English.

Apple Silicon runs the 4-bit MLX version (mlx-audio, MIT) and streams: on an M2 Pro the
first sound comes after about 0.2 s and a sentence takes about 0.4 s per second of speech.
Elsewhere the official qwen-tts package runs it with PyTorch on the NVIDIA GPU, a sentence
at a time. The voice is a short recording (see ``flay.tts.voice``) the model imitates;
pitch and speed are applied to that recording once, so they cost nothing per sentence.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from flay.audio.base import AudioBuffer
from flay.config import VoiceSettings
from flay.core.language import Language
from flay.offline import MissingModelError
from flay.tts.shaping import shape
from flay.tts.voice import Reference, read_reference, resample, resolve

log = logging.getLogger(__name__)

Engine = Literal["mlx", "torch"]
MODELS: dict[Engine, str] = {
    "mlx": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
    "torch": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
SAMPLE_RATE = 24_000
# Audio per streamed chunk: smaller starts sooner but costs more decoding steps.
STREAMING_INTERVAL = 0.32
LANGUAGE_NAMES = {Language.DE: "German", Language.EN: "English"}

_DONE = object()


class _Model(Protocol):
    def chunks(self, text: str, language: Language) -> Generator[np.ndarray]: ...


class Qwen3TTS:
    def __init__(self, voice: VoiceSettings, data_dir: Path, engine: Engine) -> None:
        self._voice = voice
        self._data_dir = data_dir
        self._engine: Engine = engine
        self._model: _Model | None = None
        # MLX keeps its GPU streams per thread: load and speak in the same one.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flay-tts")
        # One sentence at a time: the streaming decoder keeps its state in the model.
        self._speaking = asyncio.Lock()

    def download(self) -> None:
        """Fetch the model (only used by `flay setup`)."""
        from huggingface_hub import snapshot_download

        log.info("Lade Qwen3-TTS (%s) von huggingface.co herunter ...", MODELS[self._engine])
        snapshot_download(MODELS[self._engine])

    def load(self) -> None:
        """Load the model and the voice; speaks one word so the first answer is quick."""
        self._executor.submit(self._load).result()

    def _load(self) -> None:
        if self._model is not None:
            return
        references = self._references()
        path = _cached(MODELS[self._engine])
        model: _Model = (
            _MlxModel(path, references) if self._engine == "mlx" else _TorchModel(path, references)
        )
        for _ in model.chunks("Hallo.", Language.DE):
            pass
        self._model = model

    def _references(self) -> dict[Language, Reference]:
        paths = resolve(self._voice.name, self._data_dir)
        log.info("Stimme: %s", ", ".join(f"{lang} {path.name}" for lang, path in paths.items()))
        shaped: dict[Path, Reference] = {}
        for path in set(paths.values()):
            reference = read_reference(path, SAMPLE_RATE)
            samples = shape(reference.samples, SAMPLE_RATE, self._voice.pitch, self._voice.speed)
            shaped[path] = Reference(samples, SAMPLE_RATE, reference.text)
        return {language: shaped[path] for language, path in paths.items()}

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    async def synthesize(self, text: str, language: Language) -> AsyncIterator[AudioBuffer]:
        loop = asyncio.get_running_loop()
        if self._model is None:
            await loop.run_in_executor(self._executor, self._load)
        assert self._model is not None
        async with self._speaking:
            chunks = self._model.chunks(text, language)
            try:
                while (
                    chunk := await loop.run_in_executor(self._executor, next, chunks, _DONE)
                ) is not _DONE:
                    assert isinstance(chunk, np.ndarray)
                    pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    yield AudioBuffer(pcm=pcm, sample_rate=SAMPLE_RATE)
            finally:
                # An interrupted answer ends its generation before the next one starts.
                await asyncio.shield(loop.run_in_executor(self._executor, chunks.close))


def _cached(repo: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return str(snapshot_download(repo, local_files_only=True))
    except FileNotFoundError:
        raise MissingModelError(f"Stimme Qwen3-TTS ({repo})") from None


class _MlxModel:
    def __init__(self, path: str, references: dict[Language, Reference]) -> None:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model

        self._model = load_model(path)
        # The same array each time lets mlx-audio reuse the encoded reference.
        self._references = {
            language: (mx.array(reference.samples), reference.text)
            for language, reference in references.items()
        }

    def chunks(self, text: str, language: Language) -> Generator[np.ndarray]:
        audio, transcript = self._references[language]
        results = self._model.generate(
            text=text,
            lang_code=LANGUAGE_NAMES[language].lower(),
            ref_audio=audio,
            ref_text=transcript,
            stream=True,
            streaming_interval=STREAMING_INTERVAL,
        )
        for result in results:
            yield np.asarray(result.audio, dtype=np.float32)


class _TorchModel:
    def __init__(self, path: str, references: dict[Language, Reference]) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        cuda = torch.cuda.is_available()
        if not cuda:
            log.warning("Qwen3-TTS ohne CUDA-GPU: deutlich langsamer (tts.backend: piper?)")
        self._model: Any = Qwen3TTSModel.from_pretrained(
            path,
            device_map="cuda:0" if cuda else "cpu",
            dtype=torch.bfloat16 if cuda else torch.float32,
        )
        self._prompts = {
            language: self._model.create_voice_clone_prompt(
                ref_audio=(reference.samples, reference.sample_rate), ref_text=reference.text
            )
            for language, reference in references.items()
        }

    def chunks(self, text: str, language: Language) -> Generator[np.ndarray]:
        wavs, rate = self._model.generate_voice_clone(
            text=text, language=LANGUAGE_NAMES[language], voice_clone_prompt=self._prompts[language]
        )
        yield resample(np.asarray(wavs[0], dtype=np.float32), int(rate), SAMPLE_RATE)
