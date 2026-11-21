"""How fast the reflexes are on this computer: `cassi reflex-test`, or "Geschwindigkeit
messen" in the management UI (tab Reflexe).

Measures what each layer costs: taking pictures (dxcam or mss), the small model, OCR of a
small region – and, if asked, one decision of the thinking model.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from cassi.reflexes.capture import FrameSource, Target, crop

if TYPE_CHECKING:
    from cassi.config import Settings

SECONDS = 3.0


def _median_ms(work: Callable[[], object], times: int) -> float:
    taken = []
    for _ in range(times):
        start = time.perf_counter()
        work()
        taken.append(time.perf_counter() - start)
    return statistics.median(taken) * 1000


def measure(settings: Settings, out: Callable[[str], None], think: bool = False) -> bool:
    from cassi.reflexes import encoder
    from cassi.tools import ocr

    source = FrameSource(Target())
    if (private := source.private()) is not None:
        out(f"✗ Der Bildschirm ist gerade nicht erlaubt ({private}) – nichts gemessen.")
        return False
    try:
        picture = source.grab()
        frames, start = 0, time.perf_counter()
        while time.perf_counter() - start < SECONDS:
            picture = source.grab()
            frames += 1
        way = "mss" if source._dxcam is None else "dxcam"
        out(f"✓ Bilder aufnehmen ({way}, {picture.width}×{picture.height}): "
            f"{frames / SECONDS:.0f} Bilder/s")  # fmt: skip
    finally:
        source.close()

    model = encoder.Encoder(encoder.model_dir(settings.downloads))
    ok = True
    if model.available:
        model.image(picture)  # loading
        ms = _median_ms(lambda: model.image(picture), 10)
        out(f"✓ Kleines Modell (SigLIP, Reflexe): {ms:.0f} ms je Bild → bis {1000 / ms:.0f}/s")
    else:
        out("✗ Kleines Modell fehlt – einmal 'setup' ausführen (Reparieren in der Verwaltung)")
        ok = False

    if ocr.available():
        region = crop(picture, (0, 0, 250, 80))
        ocr.read_image(region)
        ms = _median_ms(lambda: ocr.read_image(region), 5)
        out(f"✓ Text lesen (OCR, kleiner Ausschnitt): {ms:.0f} ms")
    else:
        out("! Keine Texterkennung: Text-Regeln fallen aus")

    if think:
        out("Nachdenken (Bildmodell mit Bild) …")
        try:
            seconds = asyncio.run(_think_once(settings, picture))
            out(f"✓ Nachdenken: {seconds:.1f} s")
        except Exception as exc:
            out(f"✗ Nachdenken fehlgeschlagen: {exc}")
            ok = False
    return ok


async def _think_once(settings: Settings, picture: object) -> float:
    from cassi.llm.base import StructuredChat
    from cassi.llm.factory import create_chat_model
    from cassi.reflexes.tool import make_thinker

    llm = settings.llm
    if settings.tools.reflexes.model:
        from cassi.llm.ollama import POINTER, resolve

        model = resolve(settings.tools.reflexes.model, POINTER, llm.model, llm.base_url)
        llm = llm.model_copy(update={"model": model})
    chat = create_chat_model(llm)
    if not isinstance(chat, StructuredChat):
        raise RuntimeError("das Modell kann keine strukturierten Antworten")
    think = make_thinker(chat, lambda: False)
    start = time.perf_counter()
    await think(picture, "Sag kurz, was zu sehen ist.", [])  # type: ignore[arg-type]
    return time.perf_counter() - start
