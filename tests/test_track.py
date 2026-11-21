from PIL import Image, ImageDraw

from cassi.reflexes.track import Layout, Tracker, dark_field

W, H = 800, 500


def game(ball: tuple[float, float] | None, own: float, text: bool = False) -> Image.Image:
    """A browser window: light bars on top, the dark game below (like Flay Pong)."""
    picture = Image.new("RGB", (W, H + 80), (235, 235, 235))  # tabs and address bar
    draw = ImageDraw.Draw(picture)
    draw.rectangle((0, 80, W, H + 80), fill=(0, 0, 0))
    paddle_h = 0.18 * H
    x = 0.96 * W
    draw.rectangle(
        (x - 5, 80 + own * H - paddle_h / 2, x + 5, 80 + own * H + paddle_h / 2), fill="white"
    )
    draw.rectangle((0.04 * W - 5, 80 + 200, 0.04 * W + 5, 80 + 290), fill="white")  # the other
    if ball is not None:
        bx, by = ball[0] * W, 80 + ball[1] * H
        draw.rectangle((bx - 6, by - 6, bx + 6, by + 6), fill="white")
    if text:
        draw.rectangle((0.3 * W, 80 + 0.5 * H, 0.7 * W, 80 + 0.6 * H), fill="white")
    return picture


class Keys:
    def __init__(self) -> None:
        self.down: set[str] = set()
        self.log: list[str] = []

    def press(self, key: str) -> None:
        self.down.add(key)
        self.log.append(f"+{key}")

    def release(self, key: str) -> None:
        self.down.discard(key)
        self.log.append(f"-{key}")


def test_finds_the_game_in_the_browser_window() -> None:
    import numpy as np

    gray = np.asarray(game((0.5, 0.5), 0.5).convert("L"), dtype=np.int16)
    top, bottom, left, right = dark_field(gray)  # type: ignore[misc]
    assert abs(top - 80) <= 2 and bottom >= H + 70 and left <= 2 and right >= W - 2


def test_follows_the_ball_while_it_comes() -> None:
    keys = Keys()
    tracker = Tracker(keys.press, keys.release)
    tracker.see(game((0.40, 0.20), 0.5), now=0.0)  # first sight: direction unknown
    sight = tracker.see(game((0.45, 0.22), 0.5), now=0.03)  # coming, above the paddle
    assert sight.ball is not None and abs(sight.ball[1] - 0.22) < 0.03
    assert sight.own is not None and abs(sight.own - 0.5) < 0.03
    assert keys.down == {"up"}
    tracker.see(
        game(
            (0.50, 0.80),
            0.5,
        ),
        now=0.06,
    )  # now below: switch keys
    assert keys.down == {"down"} and keys.log == ["+up", "-up", "+down"]
    tracker.see(game((0.55, 0.51), 0.5), now=0.09)  # level: let go
    assert keys.down == set()


def test_goes_back_to_the_middle_when_the_ball_leaves() -> None:
    keys = Keys()
    tracker = Tracker(keys.press, keys.release)
    tracker.see(game((0.60, 0.2), 0.2), now=0.0)
    tracker.see(game((0.55, 0.2), 0.2), now=0.03)  # going away, paddle high
    assert keys.down == {"down"}


def test_text_is_no_ball_and_no_field_means_no_keys() -> None:
    keys = Keys()
    tracker = Tracker(keys.press, keys.release)
    tracker.see(game(None, 0.8, text=True), now=0.0)
    sight = tracker.see(game(None, 0.8, text=True), now=0.03)
    assert sight.ball is None and keys.down == {"up"}  # back to the middle
    blank = Image.new("RGB", (W, H), (240, 240, 240))
    assert tracker.see(blank, now=5.0).own is None and keys.down == set()


def test_own_paddle_on_the_left() -> None:
    keys = Keys()
    tracker = Tracker(keys.press, keys.release, Layout(own_column=(0.01, 0.07)))
    tracker.see(game((0.60, 0.9), 0.5), now=0.0)
    tracker.see(game((0.55, 0.9), 0.5), now=0.03)  # moving left: towards the own side
    assert keys.down == {"down"}


def test_aims_where_the_ball_will_arrive() -> None:
    from cassi.reflexes.track import _arrival

    area = (0.07, 0.12, 0.93, 0.94)
    # Straight on: where the line meets the column.
    assert abs(_arrival((0.5, 0.5), (0.6, 0.55), 0.96, area) - 0.73) < 0.01
    # Steeply down: bounces off the bottom (0.94) and is far up again at the column:
    # 0.9 + 2 × 0.36 = 1.62, folded back into 0.12–0.94 → 0.26.
    assert abs(_arrival((0.5, 0.7), (0.6, 0.9), 0.96, area) - 0.26) < 0.01


def test_keeps_its_aim_when_the_ball_is_briefly_not_seen() -> None:
    # Just before the paddle the ball leaves the area where balls are looked for, and single
    # pictures can miss it: that must not send the paddle back to the middle.
    keys = Keys()
    tracker = Tracker(keys.press, keys.release)
    tracker.see(game((0.80, 0.2), 0.2), now=0.0)
    tracker.see(game((0.85, 0.2), 0.2), now=0.03)  # coming, level with the paddle
    assert keys.down == set()
    tracker.see(game(None, 0.2), now=0.06)  # one picture without the ball
    assert keys.down == set()
    sight = tracker.see(game((0.945, 0.2), 0.2), now=0.09)  # right in front of the paddle
    assert keys.down == set() and sight.own is not None and abs(sight.own - 0.2) < 0.02
    tracker.see(game((0.90, 0.2), 0.2), now=0.12)  # back in sight, still coming
    assert keys.down == set()


def test_the_ball_beside_the_paddle_is_not_taken_for_it() -> None:
    keys = Keys()
    tracker = Tracker(keys.press, keys.release)
    # The ball just below the paddle's lower end, inside the paddle's column.
    sight = tracker.see(game((0.945, 0.33), 0.2), now=0.0)
    assert sight.own is not None and abs(sight.own - 0.2) < 0.008
