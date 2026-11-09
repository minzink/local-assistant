import asyncio
from collections.abc import AsyncIterator

import pytest
from aiohttp import WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

from flay.assistant import Assistant
from flay.config import Settings
from flay.core.messages import Message, Role
from flay.memory.base import ShortTermMemory
from flay.ui.server import WebUI
from tests.test_assistant import FakeLLM, FakeOutput, FakeSource, FakeSTT, FakeTTS

PORT = 8765


@pytest.fixture
async def client() -> AsyncIterator[TestClient]:
    memory = ShortTermMemory()
    memory.record(Message.from_text(Role.USER, "Hallo"), Message.from_text(Role.ASSISTANT, "Hi!"))
    assistant = Assistant(
        Settings(),
        FakeSource(),
        FakeSTT("x"),
        FakeLLM("Antwort."),
        FakeTTS(),
        FakeOutput(),
        memory=memory,
    )
    ui = WebUI(assistant, PORT)
    async with TestClient(TestServer(ui.app)) as test_client:
        yield test_client


ORIGIN = {"Origin": f"http://127.0.0.1:{PORT}"}


async def test_index_page(client: TestClient) -> None:
    response = await client.get("/")

    assert response.status == 200
    assert "<title>Flay</title>" in await response.text()


@pytest.mark.parametrize("headers", [{}, {"Origin": "https://evil.example"}])
async def test_websocket_rejects_foreign_origins(
    client: TestClient, headers: dict[str, str]
) -> None:
    with pytest.raises(WSServerHandshakeError) as error:
        await client.ws_connect("/ws", headers=headers)

    assert error.value.status == 403


async def test_websocket_sends_history_and_handles_commands(client: TestClient) -> None:
    async with client.ws_connect("/ws", headers=ORIGIN) as ws:
        init = await ws.receive_json()
        assert init["type"] == "Init"
        assert init["history"] == [
            {"role": "user", "text": "Hallo", "interrupted": False},
            {"role": "assistant", "text": "Hi!", "interrupted": False},
        ]
        assert init["state"] == "idle"

        await ws.send_json({"type": "mute", "speaker": True})
        assert await ws.receive_json() == {
            "type": "MuteChanged",
            "microphone": False,
            "speaker": True,
        }

        await ws.send_json({"type": "send", "text": "Wie geht es dir?"})
        received = []
        async with asyncio.timeout(2):
            while not received or received[-1]["type"] != "AssistantMessage":
                received.append(await ws.receive_json())

    types = [e["type"] for e in received]
    assert types[0] == "UserMessage"
    assert received[0]["text"] == "Wie geht es dir?"
    assert received[0]["source"] == "text"
    assert "AssistantDelta" in types
    assert received[-1]["text"] == "Antwort."


async def test_invalid_commands_are_ignored(client: TestClient) -> None:
    async with client.ws_connect("/ws", headers=ORIGIN) as ws:
        await ws.receive_json()
        await ws.send_str("kein json")
        await ws.send_json(["liste"])
        await ws.send_json({"type": "mute", "speaker": "ja"})
        await ws.send_json({"type": "mute", "microphone": True})

        assert await ws.receive_json() == {
            "type": "MuteChanged",
            "microphone": True,
            "speaker": False,
        }


async def test_logs_are_streamed_and_replayed(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    # Without `flay run` (or an earlier test) setting it, INFO/DEBUG would be dropped.
    caplog.set_level(logging.DEBUG, logger="flay")
    async with client.ws_connect("/ws", headers=ORIGIN) as ws:
        await ws.receive_json()
        logging.getLogger("flay.tools.phpstorm").info("PhpStorm: Projekt „flay“ geöffnet")
        logging.getLogger("flay.assistant").debug("Aufnahme 1.2s")
        live = [await ws.receive_json(), await ws.receive_json()]

    assert live[0] == {
        "type": "LogEntry",
        "time": live[0]["time"],
        "level": "INFO",
        "source": "tools.phpstorm",
        "message": "PhpStorm: Projekt „flay“ geöffnet",
        "action": True,
    }
    assert live[1]["action"] is False and live[1]["level"] == "DEBUG"

    # A page opened later gets the recent entries with its initial state.
    async with client.ws_connect("/ws", headers=ORIGIN) as ws:
        init = await ws.receive_json()
    assert [entry["message"] for entry in init["logs"]][-2:] == [
        "PhpStorm: Projekt „flay“ geöffnet",
        "Aufnahme 1.2s",
    ]


async def test_listening_mode_can_be_switched_from_the_ui() -> None:
    from flay.input.switchable import SwitchableSource
    from tests.test_switchable import QueueSource

    assistant = Assistant(
        Settings(),
        SwitchableSource(QueueSource(), QueueSource),
        FakeSTT("x"),
        FakeLLM("x"),
        FakeTTS(),
        FakeOutput(),
    )
    async with (
        TestClient(TestServer(WebUI(assistant, PORT).app)) as client,
        client.ws_connect("/ws", headers=ORIGIN) as ws,
    ):
        assert (await ws.receive_json())["continuous"] is False
        await ws.send_json({"type": "listen", "continuous": True})
        event = await ws.receive_json()

    assert event == {"type": "ListeningChanged", "continuous": True}
    assert assistant.continuous is True


async def test_init_reports_no_listening_switch_for_plain_sources(client: TestClient) -> None:
    async with client.ws_connect("/ws", headers=ORIGIN) as ws:
        assert (await ws.receive_json())["continuous"] is None


async def test_wake_word_can_be_switched_from_the_ui() -> None:
    from flay.input.switchable import SwitchableSource
    from tests.test_switchable import QueueSource

    built: list[QueueSource] = []

    def make() -> QueueSource:
        built.append(QueueSource())
        return built[-1]

    assistant = Assistant(
        Settings(),
        SwitchableSource(None, make, start_continuous=True),
        FakeSTT("x"),
        FakeLLM("x"),
        FakeTTS(),
        FakeOutput(),
    )
    run = asyncio.create_task(assistant.run())
    await asyncio.sleep(0.01)
    async with (
        TestClient(TestServer(WebUI(assistant, PORT).app)) as client,
        client.ws_connect("/ws", headers=ORIGIN) as ws,
    ):
        init = await ws.receive_json()
        assert init["wake_word"] is False and init["recording"] is None
        await ws.send_json({"type": "wake", "enabled": True})
        event = await ws.receive_json()
        while event["type"] != "WakeWordChanged":
            event = await ws.receive_json()

    assert event == {"type": "WakeWordChanged", "enabled": True}
    assert assistant.wake_word is True
    assert len(built) == 2  # the running listener was rebuilt for the wake word
    run.cancel()


def test_without_the_wake_word_model_the_switch_is_hidden() -> None:
    assistant = Assistant(
        Settings.model_validate({"input": {"continuous": {"wake_word": True}}}),
        FakeSource(),
        FakeSTT("x"),
        FakeLLM("x"),
        FakeTTS(),
        FakeOutput(),
        wake_word_available=False,
    )

    assert assistant.wake_word is None
    assistant.set_wake_word(True)
    assert assistant.wake_word is None
