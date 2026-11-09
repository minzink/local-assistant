from pathlib import Path

import pytest
from pydantic import ValidationError

from flay.config import Settings, load_settings
from flay.core.language import Language
from flay.core.messages import Modality

EXAMPLE_CONFIG = Path(__file__).parent.parent / "flay.example.yaml"


def test_defaults_without_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FLAY_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings == Settings()
    assert settings.llm.input_modalities == {Modality.TEXT, Modality.IMAGE}


def test_example_config_is_valid() -> None:
    settings = load_settings(EXAMPLE_CONFIG)

    assert settings.language.default is Language.DE
    assert set(settings.tts.voices) == set(Language)


def test_partial_override(tmp_path: Path) -> None:
    config = tmp_path / "flay.yaml"
    config.write_text("llm:\n  model: phi4\n", encoding="utf-8")

    settings = load_settings(config)

    assert settings.llm.model == "phi4"
    assert settings.llm.base_url == Settings().llm.base_url


def test_env_var_selects_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = tmp_path / "custom.yaml"
    config.write_text("language:\n  default: en\n", encoding="utf-8")
    monkeypatch.setenv("FLAY_CONFIG", str(config))

    assert load_settings().language.default is Language.EN


@pytest.mark.parametrize(
    "yaml_text",
    [
        "language:\n  default: fr\n",
        "llm:\n  input_modalities: [image]\n",
        "tts:\n  voices:\n    de: de_DE-thorsten-medium\n",
        "input:\n  continuous:\n    end_silence_ms: 50\n",
        "llm:\n  modell: typo\n",
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, yaml_text: str) -> None:
    config = tmp_path / "flay.yaml"
    config.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config)


def test_multimodal_capabilities_from_config(tmp_path: Path) -> None:
    config = tmp_path / "flay.yaml"
    config.write_text("llm:\n  input_modalities: [text, image]\n", encoding="utf-8")

    capabilities = load_settings(config).llm.capabilities

    assert capabilities.input_modalities == {Modality.TEXT, Modality.IMAGE}


def test_api_key_is_masked_in_dump(tmp_path: Path) -> None:
    config = tmp_path / "flay.yaml"
    config.write_text("llm:\n  api_key: geheim\n", encoding="utf-8")

    settings = load_settings(config)

    assert settings.llm.api_key is not None
    assert settings.llm.api_key.get_secret_value() == "geheim"
    assert "geheim" not in str(settings.model_dump(mode="json"))


def test_continuous_mode_is_allowed(tmp_path: Path) -> None:
    config = tmp_path / "flay.yaml"
    config.write_text("input:\n  mode: continuous\n", encoding="utf-8")

    assert load_settings(config).input.mode == "continuous"
