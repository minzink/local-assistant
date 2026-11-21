"""Following an object with two keys – an inborn reflex, like an eye following a ball.

Each picture: find the bright object that moves (the ball) and the bright object one
controls (a paddle in its column), then hold "up" or "down" until both are level. Pure
pixel work, no model: ~5 ms per picture, so it keeps up at 30 pictures a second.

Pictures of a window include the browser's own bars; the game is the large dark area in it,
found again every few seconds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Grey value from which a pixel belongs to a bright object, and below which to the dark field.
BRIGHT = 180
# Pure black: a dark browser theme's bars (grey ~30–45) are not the game.
DARK = 12
WIDTH = 320  # pictures are looked at this wide
FIELD_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class Layout:
    """Where things are, as shares (0–1) of the dark field."""

    own_column: tuple[float, float] = (0.93, 0.99)  # x range of the controlled paddle
    ball_area: tuple[float, float, float, float] = (0.07, 0.12, 0.93, 0.94)  # x1, y1, x2, y2
    # Level enough: no key held within this share of the height.
    deadzone: float = 0.035
    # More bright pixels than this many balls in the area: text, not a ball (game over).
    max_ball_pixels: int = 160
    # Aim where the ball will arrive (bounces off top and bottom included) instead of where
    # it is – a trained player rather than a pure reflex.
    predict: bool = False


@dataclass
class Sight:
    """What one picture showed (shares of the field), for tests and the status."""

    ball: tuple[float, float] | None = None
    own: float | None = None
    move: str | None = None  # "up", "down" or None


def dark_field(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """The largest dark rectangle (the game), as top, bottom, left, right rows/columns."""
    dark = gray < DARK

    def longest(shares: np.ndarray) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        start = None
        # Half dark is enough: big text ("GAME OVER") must not split the field.
        for index, value in enumerate([*list(shares > 0.5), False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                if best is None or index - start > best[1] - best[0]:
                    best = (start, index)
                start = None
        return best

    rows = longest(dark.mean(axis=1))
    if rows is None or rows[1] - rows[0] < gray.shape[0] * 0.3:
        return None
    columns = longest(dark[rows[0] : rows[1]].mean(axis=0))
    if columns is None or columns[1] - columns[0] < gray.shape[1] * 0.3:
        return None
    return rows[0], rows[1], columns[0], columns[1]


class Tracker:
    """Holds ``up``/``down`` so the own object follows the moving one."""

    def __init__(
        self,
        press: Callable[[str], None],
        release: Callable[[str], None],
        layout: Layout | None = None,
        up: str = "up",
        down: str = "down",
    ) -> None:
        self._press = press
        self._release = release
        self.layout = layout or Layout()
        self.up, self.down = up, down
        self.held: str | None = None
        self._field: tuple[int, int, int, int] | None = None
        self._field_at = 0.0
        self._previous_ball: tuple[float, float] | None = None
        # Where the paddle is headed, kept while the ball is briefly out of sight.
        self._target = 0.5

    def _hold(self, key: str | None) -> None:
        if key == self.held:
            return
        if self.held is not None:
            self._release(self.held)
        if key is not None:
            self._press(key)
        self.held = key

    def let_go(self) -> None:
        self._hold(None)

    def see(self, picture: Image.Image, now: float | None = None) -> Sight:
        """Look at one picture and move (hold or release keys)."""
        now = time.monotonic() if now is None else now
        scale = WIDTH / picture.width
        small = picture.convert("L").resize((WIDTH, max(1, round(picture.height * scale))))
        gray = np.asarray(small, dtype=np.int16)
        if self._field is None or now - self._field_at > FIELD_SECONDS:
            self._field = dark_field(gray)
            self._field_at = now
        if self._field is None:
            self.let_go()
            return Sight()
        top, bottom, left, right = self._field
        field = gray[top:bottom, left:right]
        height, width = field.shape
        bright = field >= BRIGHT
        layout = self.layout

        x1, x2 = (round(v * width) for v in layout.own_column)
        own = _longest_run(bright[:, x1:x2].any(axis=1), height)

        bx1, by1, bx2, by2 = layout.ball_area
        area = bright[
            round(by1 * height) : round(by2 * height), round(bx1 * width) : round(bx2 * width)
        ]
        ys, xs = np.nonzero(area)
        ball = None
        if 0 < ys.size <= layout.max_ball_pixels:
            ball = (
                bx1 + float(xs.mean()) / width,
                by1 + float(ys.mean()) / height,
            )
        sight = Sight(ball=ball, own=own)
        if own is None:
            self.let_go()
            return sight
        # Like a player: follow the ball while it comes, go back to the middle once it leaves.
        # Without a clear sight (ball not seen, e.g. right in front of the paddle where it
        # leaves the ball area, or seen once only) keep the aim instead of giving it up.
        towards = 1 if sum(layout.own_column) / 2 > 0.5 else -1  # the own side: right or left
        previous = self._previous_ball
        if ball is not None:
            self._previous_ball = ball
            if previous is not None and ball != previous:
                if (ball[0] - previous[0]) * towards >= 0:
                    self._target = ball[1]
                    if layout.predict:
                        column = sum(layout.own_column) / 2
                        self._target = _arrival(previous, ball, column, layout.ball_area)
                else:
                    self._target = 0.5
        target = self._target
        if target < own - layout.deadzone:
            sight.move = self.up
        elif target > own + layout.deadzone:
            sight.move = self.down
        self._hold(sight.move)
        return sight


def _longest_run(rows: np.ndarray, height: int) -> float | None:
    """The middle of the longest run of bright rows (the paddle, not a ball beside it)."""
    best, begin, length = (0, 0), None, 0
    for index, value in enumerate([*rows.tolist(), False]):
        if value:
            begin = index if begin is None else begin
        elif begin is not None:
            if index - begin > length:
                best, length = (begin, index), index - begin
            begin = None
    if length == 0:
        return None
    return (best[0] + best[1]) / 2 / height


def _arrival(
    before: tuple[float, float],
    now: tuple[float, float],
    column: float,
    area: tuple[float, float, float, float],
) -> float:
    """Where a ball moving from ``before`` to ``now`` reaches ``column``, bouncing off the
    top and bottom of ``area``."""
    dx, dy = now[0] - before[0], now[1] - before[1]
    if abs(dx) < 1e-4:
        return now[1]
    y = now[1] + dy * (column - now[0]) / dx
    top, bottom = area[1], area[3]
    span = bottom - top
    if span <= 0:
        return min(1.0, max(0.0, y))
    # Unfold the bounces: reflect into the field.
    folded = (y - top) % (2 * span)
    return top + (folded if folded <= span else 2 * span - folded)
