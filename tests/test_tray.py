from flay.core.events import AssistantState
from flay.ui.tray import STATE_COLORS, icon_image


def test_icon_colors_follow_state() -> None:
    for state, color in STATE_COLORS.items():
        image = icon_image(state, microphone_muted=False)
        assert image.getpixel((32, 32))[:3] == color


def test_muted_microphone_is_marked() -> None:
    plain = icon_image(AssistantState.IDLE, microphone_muted=False)
    muted = icon_image(AssistantState.IDLE, microphone_muted=True)

    assert plain.tobytes() != muted.tobytes()
    assert muted.getpixel((32, 32))[:3] == (255, 59, 48)
