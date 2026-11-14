"""Text on the screen through the system's own text recognition (OCR).

macOS: the Vision framework; Windows: Windows.Media.Ocr. Both run locally in a fraction of
a second and find text the accessibility interfaces do not report – PhpStorm's toolbars,
web content in some browsers, remote desktops – faster and more exactly than the vision
model, which guesses a point.

The picture is taken at full resolution (OCR needs the pixels the vision model is spared);
boxes come back in screen points, like windows and the mouse.
"""

from __future__ import annotations

import asyncio
import difflib
import io
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from cassi.tools.pointer import Frame, normalize
from cassi.tools.screen import _grab, ensure_allowed

log = logging.getLogger(__name__)

Region = tuple[float, float, float, float]
# A label must fit this well to count ("Speichern" read as "Speichem" still counts).
MIN_SIMILARITY = 0.8


@dataclass(frozen=True, slots=True)
class TextBox:
    text: str
    # x, y, width, height in screen points
    frame: Frame

    @property
    def centre(self) -> tuple[float, float]:
        x, y, width, height = self.frame
        return x + width / 2, y + height / 2


def available() -> bool:
    """Whether this system's OCR can be used."""
    try:
        if sys.platform == "darwin":
            import Vision  # noqa: F401

            return True
        if sys.platform == "win32":
            from winrt.windows.media.ocr import OcrEngine

            return OcrEngine.try_create_from_user_profile_languages() is not None
    except Exception as exc:
        log.debug("Texterkennung nicht verfügbar: %s", exc)
    return False


def read_screen(region: Region) -> list[TextBox]:
    """The text lines (and words, on Windows) in a part of the screen."""
    ensure_allowed(region)
    left, top, width, height = (round(v) for v in region)
    image, _ = _grab({"left": left, "top": top, "width": width, "height": height})
    scale = image.width / width if width else 1.0  # 2 on Retina displays
    boxes = read_image(image)
    return [
        TextBox(
            box.text,
            (
                left + box.frame[0] / scale,
                top + box.frame[1] / scale,
                box.frame[2] / scale,
                box.frame[3] / scale,
            ),
        )
        for box in boxes
    ]


def read_image(image: Image.Image) -> list[TextBox]:
    """Text in an image, boxes in its pixels."""
    if sys.platform == "darwin":
        return _vision(image)
    if sys.platform == "win32":
        return asyncio.run(_windows(image))
    return []


def _vision(image: Image.Image) -> list[TextBox]:
    import Vision
    from Foundation import NSData, NSMakeRange

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = NSData.dataWithBytes_length_(buffer.getvalue(), len(buffer.getvalue()))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(["de-DE", "en-US"])
    # Labels and names are no sentences: correcting them makes "KX" into "Kx" or worse.
    request.setUsesLanguageCorrection_(False)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision: {error}")
    width, height = image.size
    boxes = []
    for observation in request.results() or []:
        candidate = observation.topCandidates_(1)
        if not candidate:
            continue
        best = candidate[0]
        text = str(best.string())
        boxes.append(TextBox(text, _flip(observation.boundingBox(), width, height)))
        # Each word on its own too: a line "Datei Bearbeiten Ansicht" holds three targets.
        words = text.split()
        if len(words) > 1:
            start = 0
            for word in words:
                index = text.index(word, start)
                start = index + len(word)
                box, _ = best.boundingBoxForRange_error_(NSMakeRange(index, len(word)), None)
                if box is not None:
                    boxes.append(TextBox(word, _flip(box.boundingBox(), width, height)))
    return boxes


def _flip(box: object, width: int, height: int) -> Frame:
    """Vision's normalized box (origin bottom left) → pixels (origin top left)."""
    origin, size = box.origin, box.size  # type: ignore[attr-defined]
    return (
        origin.x * width,
        (1 - origin.y - size.height) * height,
        size.width * width,
        size.height * height,
    )


async def _windows(image: Image.Image) -> list[TextBox]:
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        return []
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(buffer.getvalue())
    await writer.store_async()
    stream.seek(0)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    result = await engine.recognize_async(bitmap)
    boxes = []
    for line in result.lines:
        words = list(line.words)
        rects = [w.bounding_rect for w in words]
        if rects:
            left = min(r.x for r in rects)
            top = min(r.y for r in rects)
            right = max(r.x + r.width for r in rects)
            bottom = max(r.y + r.height for r in rects)
            boxes.append(TextBox(line.text, (left, top, right - left, bottom - top)))
        if len(words) > 1:
            for word, rect in zip(words, rects, strict=True):
                boxes.append(TextBox(word.text, (rect.x, rect.y, rect.width, rect.height)))
    return boxes


def find_text(wanted: str, boxes: Sequence[TextBox]) -> list[TextBox]:
    """The boxes showing ``wanted``: exact first, then close misreadings.

    Several boxes may show the same text; the caller decides between them.
    """
    target = normalize(wanted)
    if not target:
        return []
    exact = [b for b in boxes if normalize(b.text) == target]
    if exact:
        return exact
    scored = [(difflib.SequenceMatcher(None, target, normalize(b.text)).ratio(), b) for b in boxes]
    close = [b for score, b in sorted(scored, key=lambda s: -s[0]) if score >= MIN_SIMILARITY]
    return close[:5]


def describe(boxes: Sequence[TextBox], limit: int = 80) -> str:
    """The text lines of a screen as a compact list for the model (words left out)."""
    lines = []
    seen: set[str] = set()
    for box in boxes:
        x, y, _, _ = box.frame
        if box.text in seen:
            continue
        seen.add(box.text)
        lines.append(f"'{box.text}' at {x:.0f},{y:.0f}")
        if len(lines) >= limit:
            break
    return "; ".join(lines)
