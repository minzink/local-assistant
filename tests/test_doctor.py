from __future__ import annotations

from flay import doctor
from flay.config import Settings
from flay.doctor import Check, Level


def test_older() -> None:
    assert doctor._older("0.29.1", "0.30")
    assert not doctor._older("0.40.11", "0.30")
    assert not doctor._older("unbekannt", "0.30")


def test_check_line_with_fix() -> None:
    line = Check(Level.FAIL, "LLM", "fehlt", "flay setup").line()
    assert line == "  ✗ LLM: fehlt\n      → flay setup"


def test_run_reports_failures_and_survives_broken_checks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def broken() -> Check:
        raise RuntimeError("kaputt")

    monkeypatch.setattr(doctor, "_gpu", broken)
    monkeypatch.setattr(doctor, "_ollama", lambda s: [Check(Level.FAIL, "LLM-Server", "aus")])
    for name in ("_voice", "_vad", "_microphone", "_phpstorm"):
        monkeypatch.setattr(doctor, name, lambda s: Check(Level.OK, "x", "ok"))
    monkeypatch.setattr(doctor, "_permissions", lambda: [])
    monkeypatch.setattr(doctor, "_claude", lambda: Check(Level.OK, "x", "ok"))
    lines: list[str] = []

    assert doctor.run(Settings(), lines.append) == 1
    assert any("Fehler: kaputt" in line for line in lines)
    assert lines[-1].startswith("Mit ✗")
