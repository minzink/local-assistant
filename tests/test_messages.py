import pytest

from flay.core.messages import ImagePart, Message, Modality, Role, TextPart
from flay.llm.base import ModelCapabilities, UnsupportedModalityError


def test_text_message() -> None:
    message = Message.from_text(Role.USER, "Hallo")

    assert message.text == "Hallo"
    assert message.modalities == {Modality.TEXT}


def test_mixed_message_text_ignores_other_parts() -> None:
    message = Message(
        role=Role.USER,
        parts=(TextPart("Was ist "), ImagePart(b"\x89PNG", "image/png"), TextPart("das?")),
    )

    assert message.text == "Was ist das?"
    assert message.modalities == {Modality.TEXT, Modality.IMAGE}


def test_text_only_model_rejects_image() -> None:
    message = Message(role=Role.USER, parts=(ImagePart(b"\x89PNG", "image/png"),))

    with pytest.raises(UnsupportedModalityError, match="image"):
        ModelCapabilities().ensure_supported([message])


def test_multimodal_model_accepts_image() -> None:
    message = Message(role=Role.USER, parts=(ImagePart(b"\x89PNG", "image/png"),))
    capabilities = ModelCapabilities(input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}))

    capabilities.ensure_supported([message])
