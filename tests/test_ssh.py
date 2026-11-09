"""ssh_connect: servers from the ssh config, opened in a new terminal window."""

from pathlib import Path
from typing import Any

import pytest

from flay.assistant import Assistant, tool_intent
from flay.config import Settings
from flay.tools.base import ToolRegistry
from flay.tools.ssh import SshConnectTool, SshHost, read_ssh_hosts, terminal_command
from tests.test_assistant import FakeOutput, FakeSource, FakeSTT, FakeTTS
from tests.test_examples import EagerLLM

CONFIG = """\
# Comments and options before the first host are ignored.
Include conf.d/*.conf

Host prod-web-01 web
    HostName 10.0.0.1
    User deploy

Host=staging
Host *.internal !secret -oProxyCommand=evil
    User x

Host db
  HostName=db.example.com
"""


@pytest.fixture
def config(tmp_path: Path) -> Path:
    (tmp_path / "conf.d").mkdir()
    (tmp_path / "conf.d" / "team.conf").write_text("Host jump\n  HostName jump.example.com\n")
    path = tmp_path / "config"
    path.write_text(CONFIG)
    return path


def test_hosts_are_read_with_includes_but_without_patterns(config: Path) -> None:
    hosts = read_ssh_hosts(config)

    assert [h.alias for h in hosts] == ["jump", "prod-web-01", "web", "staging", "db"]
    assert hosts[1] == SshHost("prod-web-01", "10.0.0.1")
    assert hosts[4].hostname == "db.example.com"
    # No wildcard, negation or anything ssh would read as an option.
    assert not any(h.alias.startswith(("-", "!")) or "*" in h.alias for h in hosts)
    assert read_ssh_hosts(config.parent / "missing") == []


def test_terminal_commands_run_ssh_with_the_alias_only() -> None:
    mac = terminal_command("prod-web-01", "terminal")
    assert mac[0] == "osascript" and 'do script "ssh prod-web-01"' in mac
    assert terminal_command("db", "windows-terminal") == [
        "wt", "new-tab", "--title", "db", "ssh", "db"
    ]  # fmt: skip
    with pytest.raises(ValueError):
        terminal_command("-oProxyCommand=evil", "terminal")
    with pytest.raises(ValueError):
        terminal_command('x"; rm -rf ~; "', "terminal")


class Started:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    async def __call__(self, command: list[str]) -> None:
        self.commands.append(command)


def tool(config: Path, started: Started) -> SshConnectTool:
    return SshConnectTool(lambda: read_ssh_hosts(config), "terminal", started)


async def test_spoken_name_finds_the_host(config: Path) -> None:
    started = Started()

    result = await tool(config, started).run({"host": "prod web 01"})

    assert result.ok and result.say is not None
    assert 'do script "ssh prod-web-01"' in started.commands[0]
    # By the HostName behind an alias too.
    await tool(config, started).run({"host": "db.example.com"})
    assert 'do script "ssh db"' in started.commands[1]


async def test_unknown_or_ambiguous_names_connect_to_nothing(config: Path) -> None:
    started = Started()
    ssh = tool(config, started)

    unknown = await ssh.run({"host": "moewe"})
    hosts = [SshHost("web-1"), SshHost("web-2")]
    ambiguous = await SshConnectTool(lambda: hosts, "terminal", started).run({"host": "web"})

    assert not unknown.ok and not ambiguous.ok
    assert ambiguous.say is not None and "web 1" in next(iter(ambiguous.say.values()))
    assert started.commands == []


def test_ssh_requests_are_recognized() -> None:
    assert tool_intent("SSH auf prod web 01") == "ssh_connect"
    assert tool_intent("Verbinde dich mit dem Server staging.") == "ssh_connect"
    assert tool_intent("Connect to the db server.") == "ssh_connect"
    assert tool_intent("Wie verbinde ich mich per SSH?") is None
    assert tool_intent("Ich habe mich gestern per SSH verbunden.") is None


def assistant(llm: Any, ssh: SshConnectTool) -> Assistant:
    return Assistant(
        Settings(), FakeSource(), FakeSTT("x"), llm, FakeTTS(), FakeOutput(),
        tools=ToolRegistry([ssh]),
    )  # fmt: skip


async def test_a_server_the_user_never_named_is_not_connected(config: Path) -> None:
    started = Started()
    llm = EagerLLM("ssh_connect", {"host": "db"})
    flay = assistant(llm, tool(config, started))

    await flay.handle_text("Verbinde dich mit dem Server.")

    assert started.commands == []
    assert flay.history[-1].text == "Mit welchem Server?"


async def test_ssh_request_opens_the_terminal(config: Path) -> None:
    started = Started()
    llm = EagerLLM("ssh_connect", {"host": "staging"})
    flay = assistant(llm, tool(config, started))

    await flay.handle_text("SSH auf staging.")

    assert 'do script "ssh staging"' in started.commands[0]
    assert flay.history[-1].text == "Ich verbinde dich mit staging."
