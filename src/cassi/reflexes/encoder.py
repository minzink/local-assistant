"""The small model of the reflexes: SigLIP's image and text encoders (Google, Apache 2.0),
run with ONNX.

It writes no text: each picture becomes a vector – an impression of the situation – in
~40 ms on a CPU (quantized base model, 224 px; M2 Pro), so the graphics card stays free for
a game. Pictures are compared with learned situations (similar impression → the same
reaction) and with descriptions ("a loading screen"), whose vectors come from the text
encoder. Downloaded once by `cassi setup` (~210 MB), never while running.

CLIP (OpenAI) was faster (~10 ms) and MIT-licensed, but its model card rules out any
deployed use; SigLIP has no such reservation and recognizes scenes more surely.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cassi.offline import MissingModelError

log = logging.getLogger(__name__)

REPO = "Xenova/siglip-base-patch16-224"
FILES = (
    "onnx/vision_model_quantized.onnx",
    "onnx/text_model_quantized.onnx",
    "tokenizer.json",
    "preprocessor_config.json",
)
SIZE = 224
# SigLIP was trained on texts padded to 64 tokens with </s> (id 1), lower case.
CONTEXT = 64
PAD = 1
# Similarities ×100 before a softmax: which of several descriptions fits best.
LOGIT_SCALE = 100.0


def model_dir(downloads: Path) -> Path:
    return downloads / "models" / "siglip-base-patch16-224"


def download(downloads: Path) -> None:
    """Fetch the encoder (only `cassi setup` does this)."""
    from huggingface_hub import hf_hub_download

    target = model_dir(downloads)
    for name in FILES:
        if not (target / name).is_file():
            hf_hub_download(REPO, name, local_dir=target)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norm == 0, 1, norm)


class Encoder:
    """Pictures and descriptions as unit vectors (cosine similarity = dot product)."""

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._lock = threading.Lock()
        self._vision: Any = None
        self._text: Any = None
        self._tokenizer: Any = None
        self._mean = np.zeros(3, np.float32)
        self._std = np.ones(3, np.float32)
        self._texts: dict[str, np.ndarray] = {}

    @property
    def available(self) -> bool:
        return all((self._folder / name).is_file() for name in FILES)

    def _load(self) -> None:
        if self._vision is not None:
            return
        if not self.available:
            raise MissingModelError("Das Reflex-Modell (SigLIP)")
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        # A few threads: fast enough, and the rest of the CPU stays with the game.
        options.intra_op_num_threads = 4
        cpu = ["CPUExecutionProvider"]
        self._vision = ort.InferenceSession(str(self._folder / FILES[0]), options, providers=cpu)
        self._text = ort.InferenceSession(str(self._folder / FILES[1]), options, providers=cpu)
        tokenizer = Tokenizer.from_file(str(self._folder / FILES[2]))
        tokenizer.enable_padding(length=CONTEXT, pad_id=PAD)
        tokenizer.enable_truncation(CONTEXT)
        self._tokenizer = tokenizer
        config = json.loads((self._folder / FILES[3]).read_text("utf-8"))
        self._mean = np.array(config["image_mean"], np.float32)
        self._std = np.array(config["image_std"], np.float32)
        log.info("Reflex-Modell geladen (SigLIP, ONNX)")

    def image(self, picture: Image.Image) -> np.ndarray:
        """The impression of a picture (unit vector)."""
        with self._lock:
            self._load()
            pixels = picture.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            array = (np.asarray(pixels, np.float32) / 255 - self._mean) / self._std
            name = self._vision.get_inputs()[0].name
            vector = self._vision.run(["pooler_output"], {name: array.transpose(2, 0, 1)[None]})[0][
                0
            ]
        return normalize(np.asarray(vector, np.float32))

    def text(self, description: str) -> np.ndarray:
        """A description's vector, comparable with pictures (kept once computed)."""
        if (known := self._texts.get(description)) is not None:
            return known
        with self._lock:
            self._load()
            ids = np.array([self._tokenizer.encode(description.lower()).ids], np.int64)
            vector = self._text.run(["pooler_output"], {"input_ids": ids})[0][0]
        found = normalize(np.asarray(vector, np.float32))
        self._texts[description] = found
        return found

    def chance(self, impression: np.ndarray, label: str, others: list[str]) -> float:
        """How likely the picture shows ``label`` rather than one of ``others`` (0–1)."""
        texts = np.stack([self.text(t) for t in [label, *others]])
        logits = LOGIT_SCALE * (texts @ impression)
        weights = np.exp(logits - logits.max())
        return float(weights[0] / weights.sum())
