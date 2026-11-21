from pathlib import Path
from typing import Any

import pytest
import yaml

from cassi import cli, profiles
from cassi.config import Settings, default_data_dir
from cassi.profiles import AdapterInfo, ProfileError

FLAY = AdapterInfo("flay", True, "Flay", ("Modelfile.local", "persona.local.yaml"))
TOOL = AdapterInfo("tool")  # an adapter for every AI
INFOS = {"flay": FLAY, "tool": TOOL}


def base_settings(tmp_path: Path, **values: object) -> Settings:
    return Settings(data_dir=tmp_path / "cassi", **values)  # type: ignore[arg-type]


def test_built_in_profiles_cassi_and_each_ai_adapter(tmp_path: Path) -> None:
    found = profiles.available(tmp_path / "cassi", INFOS)
    assert [p.name for p in found] == ["cassi", "flay"]
    assert found[1].folder == default_data_dir("flay") and found[1].adapters == ("flay",)
    assert all(p.builtin for p in found)


def test_the_profile_decides_folder_adapters_and_settings(tmp_path: Path) -> None:
    settings = base_settings(tmp_path)
    flay = profiles.resolve(settings, "flay", infos=INFOS)
    assert flay.data_dir == default_data_dir("flay")
    assert flay.downloads == tmp_path / "cassi"  # downloads shared
    assert flay.shared_dir == tmp_path / "cassi" and flay.profile == "flay"
    assert flay.adapters == ["tool", "flay"]  # tools for all, Flay only in its profile

    cassi = profiles.resolve(settings, "cassi", infos=INFOS)
    assert cassi.data_dir == tmp_path / "cassi" and cassi.adapters == ["tool"]


def test_which_profile_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = base_settings(tmp_path)
    assert profiles.active_name(settings, infos=INFOS) == "flay"  # an AI adapter's own
    assert profiles.active_name(settings, infos={"tool": TOOL}) == "cassi"
    assert profiles.active_name(base_settings(tmp_path, adapters=[]), infos=INFOS) == "cassi"
    assert profiles.active_name(base_settings(tmp_path, profile="cassi"), infos=INFOS) == "cassi"
    profiles.choose(tmp_path / "cassi", "cassi")  # chosen in the UI wins over cassi.yaml
    assert profiles.active_name(base_settings(tmp_path, profile="flay"), infos=INFOS) == "cassi"
    monkeypatch.setenv(profiles.ENVIRONMENT, "flay")
    assert profiles.active_name(settings, infos=INFOS) == "flay"
    assert profiles.active_name(settings, "cassi", infos=INFOS) == "cassi"  # --profil first


def test_a_new_profile_is_a_separate_ai(tmp_path: Path) -> None:
    base = tmp_path / "cassi"
    flay = profiles.find(base, "flay", INFOS)
    flay.folder.mkdir(parents=True)
    (flay.folder / "Modelfile.local").write_text("PARAMETER temperature 0.2")
    (flay.folder / "memory.sqlite3").write_text("Erinnerungen")
    (flay.folder / "persona.learned.yaml").write_text("Gelerntes")

    chess = profiles.create(
        base, "Schach-Flay", "nur Schach", flay, "definition",
        {"persona": {"name": "Kasimir"}}, infos=INFOS,
    )  # fmt: skip
    assert chess.name == "schach-flay" and chess.folder == base / "profiles" / "schach-flay"
    assert chess.adapters == ("flay",) and chess.settings == {"persona": {"name": "Kasimir"}}
    assert (chess.folder / "Modelfile.local").is_file()  # how it is defined: copied
    assert not (chess.folder / "memory.sqlite3").exists()  # what it remembers: not
    assert not (chess.folder / "persona.learned.yaml").exists()  # what it learned: not

    copy = profiles.create(base, "Flay Kopie", template=flay, copy="everything", infos=INFOS)
    assert (copy.folder / "memory.sqlite3").read_text() == "Erinnerungen"

    settings = profiles.resolve(base_settings(tmp_path), "Schach-Flay", infos=INFOS)
    assert settings.data_dir == chess.folder and settings.persona.name == "Kasimir"
    assert settings.adapters == ["tool", "flay"]
    with pytest.raises(ProfileError, match="gibt es schon"):
        profiles.create(base, "schach flay", infos=INFOS)


def test_a_copy_of_cassi_does_not_take_the_other_profiles(tmp_path: Path) -> None:
    base = tmp_path / "cassi"
    base.mkdir()
    (base / "memory.sqlite3").write_text("m")
    plain = profiles.create(base, "Leer", infos=INFOS)
    cassi = profiles.find(base, "cassi", INFOS)
    copy = profiles.create(base, "Kopie", template=cassi, copy="everything", infos=INFOS)
    assert (copy.folder / "memory.sqlite3").is_file()
    assert not (copy.folder / "profiles").exists() and plain.adapters == ()


def test_removing_keeps_a_backup_and_spares_built_in_and_active(tmp_path: Path) -> None:
    base = tmp_path / "cassi"
    test = profiles.create(base, "Test", infos=INFOS)
    with pytest.raises(ProfileError, match="eingebaut"):
        profiles.remove(base, "flay", "test", INFOS)
    with pytest.raises(ProfileError, match="in Gebrauch"):
        profiles.remove(base, "test", "test", INFOS)
    archive = profiles.remove(base, "Test", "cassi", INFOS)
    assert archive.is_file() and archive.parent == base / "backups"
    assert not test.folder.exists()


def test_profile_definition_errors_are_reported(tmp_path: Path) -> None:
    folder = tmp_path / "cassi" / "profiles" / "kaputt"
    folder.mkdir(parents=True)
    (folder / "profile.yaml").write_text(yaml.safe_dump({"adapters": "flay"}))
    assert [p.name for p in profiles.available(tmp_path / "cassi", INFOS)] == ["cassi", "flay"]
    assert profiles.slug("Ünd Schöne Grüße!") == "uend-schoene-gruesse"


def test_profil_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "cassi.yaml"
    config.write_text(yaml.safe_dump({"data_dir": str(tmp_path / "cassi")}))
    assert cli.main(["-c", str(config), "profil", "neu", "Kasimir", "--vorlage", "cassi"]) == 0
    assert cli.main(["-c", str(config), "profil", "wechseln", "kasimir"]) == 0
    capsys.readouterr()
    assert cli.main(["-c", str(config), "profil"]) == 0
    listed = capsys.readouterr().out
    assert "* kasimir" in listed and "  cassi" in listed
    assert cli.main(["-c", str(config), "info"]) == 0
    assert "Profil:        kasimir" in capsys.readouterr().out
    assert cli.main(["-c", str(config), "--profil", "gibtsnicht", "info"]) == 1


def test_the_profiles_tab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cassi import profile_tab
    from cassi.app import current
    from cassi.llm import ollama
    from cassi.manage import Launcher

    settings = profiles.resolve(base_settings(tmp_path), infos={})
    started: list[list[str]] = []
    monkeypatch.setattr(profile_tab.subprocess, "Popen", lambda c, **k: started.append(c))
    monkeypatch.setattr(Launcher, "running", lambda self: False)
    monkeypatch.setattr(
        ollama, "models", lambda url: {"qwen3:4b", "qwen3:4b:latest", "nomic-embed"}
    )
    quits: list[bool] = []
    tab = profile_tab.profile_tab(
        current(), settings, None, lambda: settings, lambda: quits.append(True)
    )

    def actions() -> dict[str, Any]:
        return {a.id: a for a in tab.current_actions()}

    create = actions()["create"]
    model = next(o for o in create.options if o.id == "model")
    assert [value for value, _ in model.choices] == ["", "qwen3:4b"]
    lines: list[str] = []
    # Everything that defines the AI, in the page's fields – no file to edit.
    made = {o.id: o.value(None) for o in create.options}
    made.update(title="Kasimir", template="cassi", switch=False, persona_name="Kasimir",
                identity="A calm chess coach.", voice="thorsten", pitch="-2", speed="1.1",
                model="qwen3:4b")  # fmt: skip
    assert create.run(lines.append, made), lines
    kasimir = profiles.find(tmp_path / "cassi", "kasimir", {})
    assert kasimir.settings["persona"]["name"] == "Kasimir"
    assert "A calm chess coach." in kasimir.settings["persona"]["system_prompt"]
    assert kasimir.settings["tts"] == {"voice": {"name": "thorsten", "pitch": -2.0, "speed": 1.1}}
    assert kasimir.settings["llm"] == {"model": "qwen3:4b"}
    assert any(r.label == "Kasimir" for r in tab.sections()[0].rows) and not started
    # The new AI is in the lists right away (no restart of the page).
    assert ("kasimir", "Kasimir") in next(o for o in actions()["switch"].options).choices

    edit = actions()["edit"]
    changed = {o.id: o.value(None) for o in edit.options}
    changed.update(name="kasimir", speaking_style="Short and patient.")
    assert edit.run(lines.append, changed)
    prompt = profiles.find(tmp_path / "cassi", "kasimir", {}).settings["persona"]["system_prompt"]
    assert "A calm chess coach." in prompt and "Short and patient." in prompt  # both kept
    assert not edit.run(lines.append, {**changed, "name": "cassi"})  # built in: cassi.yaml

    assert actions()["switch"].run(lines.append, {"name": "kasimir"})
    assert profiles.chosen(tmp_path / "cassi") == "kasimir" and quits == [True]
    assert started[0][-1] == "manage"  # the management UI starts again, with its tabs

    assert not actions()["remove"].run(lines.append, {"name": "cassi"})  # built in
    assert actions()["remove"].run(lines.append, {"name": "kasimir"})
    assert not kasimir.folder.exists()


def test_an_adapter_defines_its_ais_in_its_own_files(tmp_path: Path) -> None:
    base = tmp_path / "cassi"
    written: list[profiles.Definition] = []

    def define(folder: Path, definition: profiles.Definition) -> dict[str, Any]:
        written.append(definition)
        (folder / "character.txt").write_text(definition.name)
        return {}

    infos = {"flay": AdapterInfo("flay", True, "Flay", define=define)}
    flay = profiles.find(base, "flay", infos)
    made = profiles.create(base, "Schach", template=flay, infos=infos)
    profiles.define(base, "schach", profiles.Definition(name="Kasimir"), infos=infos)
    assert (made.folder / "character.txt").read_text() == "Kasimir"
    assert profiles.find(base, "schach", infos).settings == {}  # nothing in Cassi's persona
    # Flay's own profile too – its files, not cassi.yaml.
    profiles.define(base, "flay", profiles.Definition(speed=1.1), infos=infos)
    assert written[-1].speed == 1.1
    with pytest.raises(ProfileError, match="eingebaut"):
        profiles.define(base, "flay", profiles.Definition(), "Neu", infos=infos)
    assert not profiles.Definition() and profiles.Definition(pitch=0.0)


def test_a_long_text_field_takes_a_whole_description() -> None:
    from cassi.manage import MAX_LONG_TEXT, MAX_TEXT, Option

    text = "x" * 5000
    assert len(str(Option("a", "A", "", kind="longtext").value(text))) == MAX_LONG_TEXT
    assert len(str(Option("a", "A", "", kind="text").value(text))) == MAX_TEXT
