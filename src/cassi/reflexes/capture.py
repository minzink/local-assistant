"""Perception: pictures of what is watched – a monitor or one window – many times a second.

macOS: mss (~25 ms per full display; per-window capture is gone there). Windows: dxcam
(DXGI Desktop Duplication), which also sees full-screen games that GDI capture shows
black; mss if dxcam is missing or fails.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from cassi.reflexes.store import Box
from cassi.tools.screen import Window

log = logging.getLogger(__name__)

# The watched window may move: its bounds are looked up again this often.
BOUNDS_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class Target:
    """What is watched: a window (by app or title) or a monitor (1 = primary)."""

    window: str | None = None
    monitor: int = 1

    def describe(self) -> str:
        return self.window or ("Bildschirm" if self.monitor == 1 else f"Bildschirm {self.monitor}")


def crop(picture: Image.Image, box: Box | None) -> Image.Image:
    """The part of ``picture`` in ``box`` (0–1000 of its width and height)."""
    if box is None:
        return picture
    x1, y1, x2, y2 = box
    width, height = picture.size
    left, right = sorted((max(0, x1) * width // 1000, min(1000, x2) * width // 1000))
    top, bottom = sorted((max(0, y1) * height // 1000, min(1000, y2) * height // 1000))
    if right - left < 2 or bottom - top < 2:
        return picture
    return picture.crop((left, top, right, bottom))


class FrameSource:
    def __init__(self, target: Target) -> None:
        self.target = target
        self._window: Window | None = None
        self._window_at = 0.0
        self._dxcam: Any = None
        self._dxcam_failed = sys.platform != "win32"
        self._last: Image.Image | None = None
        self._monitors: list[dict[str, int]] = []
        self._monitors_at = 0.0
        self._dxcam_monitor: dict[str, int] | None = None

    def close(self) -> None:
        if self._dxcam is not None:
            with contextlib.suppress(Exception):  # already gone
                self._dxcam.release()
            self._dxcam = None

    def _bounds(self) -> tuple[int, int, int, int] | None:
        """The watched window's bounds (screen points), or None for the whole monitor."""
        if self.target.window is None:
            return None
        now = time.monotonic()
        if self._window is None or now - self._window_at > BOUNDS_SECONDS:
            from cassi.tools.screen import find_window, list_windows

            self._window = find_window(self.target.window, list_windows())
            self._window_at = now
        if self._window is None:
            raise LookupError(f"Kein Fenster „{self.target.window}“")
        return self._window.bounds

    def private(self) -> str | None:
        """A private window that would be seen, or a paused screen: then nothing is taken."""
        from cassi.tools.screen import PRIVACY, monitor_bounds, private_window

        if PRIVACY.paused:
            return "pausiert"
        region = self._bounds() or monitor_bounds(self.target.monitor)
        window = private_window(region)
        return (window.app or window.title) if window else None

    def grab(self) -> Image.Image:
        bounds = self._bounds()
        picture = self._grab_dxcam(bounds) if not self._dxcam_failed else None
        if picture is None:
            picture = self._grab_mss(bounds)
        self._last = picture
        return picture

    def _monitor(self, bounds: tuple[int, int, int, int] | None) -> dict[str, int]:
        """The monitor shown (screen points): the target's, or the one under the window."""
        now = time.monotonic()
        if not self._monitors or now - self._monitors_at > BOUNDS_SECONDS:
            from cassi.tools.screen import _mss

            with _mss() as screen:
                self._monitors = list(screen.monitors[1:] or screen.monitors)
            self._monitors_at = now
        monitors = self._monitors
        if bounds is None:
            return monitors[min(self.target.monitor, len(monitors)) - 1]
        x, y, width, height = bounds
        centre = (x + width / 2, y + height / 2)
        return next(
            (
                m
                for m in monitors
                if m["left"] <= centre[0] < m["left"] + m["width"]
                and m["top"] <= centre[1] < m["top"] + m["height"]
            ),
            monitors[0],
        )

    @staticmethod
    def _cut(
        picture: Image.Image, monitor: dict[str, int], bounds: tuple[int, int, int, int] | None
    ) -> Image.Image:
        """The window's part of a whole-monitor picture. The picture may have more pixels
        than the monitor has points (Retina, Windows scaling): scaled the same way."""
        if bounds is None:
            return picture
        scale = picture.width / monitor["width"]
        x, y, width, height = bounds
        left = max(0.0, (x - monitor["left"]) * scale)
        top = max(0.0, (y - monitor["top"]) * scale)
        right = min(float(picture.width), (x - monitor["left"] + width) * scale)
        bottom = min(float(picture.height), (y - monitor["top"] + height) * scale)
        if right - left < 2 or bottom - top < 2:
            return picture
        return picture.crop((round(left), round(top), round(right), round(bottom)))

    def _grab_mss(self, bounds: tuple[int, int, int, int] | None) -> Image.Image:
        from cassi.tools.screen import _mss

        monitor = self._monitor(bounds)
        with _mss() as screen:
            shot = screen.grab(monitor)
        picture = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        return self._cut(picture, monitor, bounds)

    def _grab_dxcam(self, bounds: tuple[int, int, int, int] | None) -> Image.Image | None:
        monitor = self._monitor(bounds)
        try:
            if self._dxcam is None:
                import dxcam

                index = self._monitors.index(monitor) if monitor in self._monitors else 0
                self._dxcam = dxcam.create(output_idx=index)
                self._dxcam_monitor = monitor
            elif monitor != self._dxcam_monitor:  # the window moved to another monitor
                self.close()
                return self._grab_dxcam(bounds)
            # The whole monitor in its own pixels, cut like mss: no coordinate systems to mix.
            frame = self._dxcam.grab()
        except Exception as exc:  # no dxcam, another GPU output …
            log.info("dxcam nicht nutzbar (%s) – Aufnahme mit mss", exc)
            self._dxcam_failed = True
            self.close()
            return None
        if frame is None:  # nothing new on screen since the last picture
            return self._last
        return self._cut(Image.fromarray(frame), monitor, bounds)
