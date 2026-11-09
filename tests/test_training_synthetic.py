import json
import random

import pytest

from flay.config import Settings
from flay.evaluation import DEFAULT_CASES, _tool_specs, load_cases
from flay.llm import ollama
from flay.llm.ollama import EMBEDDING, POINTER, resolve
from flay.training_synthetic import PROJECTS, spoken, synthetic_rows


def test_generated_projects_are_not_the_test_sets() -> None:
    state, cases = load_cases(DEFAULT_CASES)
    tested = " ".join([*state["recent_projects"], *(str(case["say"]) for case in cases)]).lower()

    assert not [p for p in PROJECTS if p in tested]


def test_names_are_garbled_like_speech_recognition() -> None:
    rng = random.Random(1)
    heard = {spoken("project-symfony-kx-nvs", rng) for _ in range(40)}

    assert any(" " in name and len(name.split()) >= 3 for name in heard)  # spelled out
    assert any(name.startswith(("CD", "C D")) for name in heard)  # misheard


def test_rows_teach_actions_on_the_listed_project_and_talk() -> None:
    rows = synthetic_rows(Settings(), _tool_specs(), 200, seed=5)

    answers = [row["messages"][2]["content"] for row in rows]
    calls = [json.loads(a.split("\n")[1]) for a in answers if a.startswith("<tool_call>")]
    assert 0.55 < len(calls) / len(rows) < 0.85
    for row in rows:
        user = row["messages"][1]["content"]
        answer = row["messages"][2]["content"]
        if answer.startswith("<tool_call>"):
            call = json.loads(answer.split("\n")[1])
            project = call["arguments"].get("name") or call["arguments"].get("project")
            assert project in user  # the project is in the situation line
    assert {c["name"] for c in calls} == {"open_project", "close_project", "send_to_claude"}


def test_variants_are_used_when_the_server_has_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "models", lambda base_url, timeout=1.0: {POINTER.name})

    assert resolve("auto", POINTER, "qwen3-vl:8b-instruct", "http://x/v1") == POINTER.name
    assert resolve("auto", EMBEDDING, EMBEDDING.base, "http://x/v1") == EMBEDDING.base
    assert resolve(None, POINTER, "main", "http://x/v1") == "main"
    assert resolve("own-model", POINTER, "main", "http://x/v1") == "own-model"
    assert ollama.server_root("http://localhost:11434/v1/") == "http://localhost:11434"


def test_setup_creates_what_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []
    monkeypatch.setattr(ollama, "is_ollama", lambda base_url, timeout=1.0: True)
    monkeypatch.setattr(
        ollama, "models", lambda base_url, timeout=1.0: {EMBEDDING.base, POINTER.base}
    )
    monkeypatch.setattr(
        ollama, "create", lambda url, v, timeout=120.0: created.append(v.name) or True
    )

    monkeypatch.setattr(ollama, "memory_gb", lambda: 8.0)  # Windows, 12 GB GPU minus the 8B
    lines = ollama.setup("http://localhost:11434/v1")
    assert created == [EMBEDDING.name] and any("zu wenig Speicher" in line for line in lines)

    created.clear()
    monkeypatch.setattr(ollama, "memory_gb", lambda: 22.9)  # M2 Pro, 32 GB
    ollama.setup("http://localhost:11434/v1")
    assert created == [EMBEDDING.name, POINTER.name]
