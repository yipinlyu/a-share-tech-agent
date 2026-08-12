from __future__ import annotations

import logging

import pytest
from pydantic import SecretStr, ValidationError

from stock_agent.config import Settings


def test_settings_hide_api_credentials_from_repr_and_json() -> None:
    settings = Settings(tushare_token="t-secret", deepseek_api_key="sk-secret")

    rendered = repr(settings)
    dumped_json = settings.model_dump_json()

    assert "t-secret" not in rendered
    assert "sk-secret" not in rendered
    assert "tushare_token" not in rendered
    assert "deepseek_api_key" not in rendered
    assert "t-secret" not in dumped_json
    assert "sk-secret" not in dumped_json


def test_settings_use_frozen_deepseek_defaults() -> None:
    settings = Settings()

    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-v4-flash"
    with pytest.raises(ValidationError):
        settings.deepseek_model = "another-model"  # type: ignore[misc]


def test_settings_report_capabilities_from_present_secrets() -> None:
    missing = Settings(tushare_token=None, deepseek_api_key=None)
    configured = Settings(
        tushare_token=SecretStr("t-token"),
        deepseek_api_key=SecretStr("sk-key"),
    )

    assert missing.data_enabled is False
    assert missing.ai_enabled is False
    assert configured.data_enabled is True
    assert configured.ai_enabled is True


def test_settings_normalize_blank_credentials_to_missing() -> None:
    settings = Settings(tushare_token="   ", deepseek_api_key="")

    assert settings.tushare_token is None
    assert settings.deepseek_api_key is None
    assert settings.data_enabled is False
    assert settings.ai_enabled is False


def test_streamlit_secrets_override_environment_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    secrets = {
        "TUSHARE_TOKEN": "streamlit-tushare",
        "DEEPSEEK_API_KEY": "streamlit-deepseek",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com/",
        "DEEPSEEK_MODEL": "streamlit-model",
    }
    environ = {
        "TUSHARE_TOKEN": "environment-tushare",
        "DEEPSEEK_API_KEY": "environment-deepseek",
        "DEEPSEEK_BASE_URL": "https://environment.example/v1",
        "DEEPSEEK_MODEL": "environment-model",
    }

    settings = Settings.from_sources(secrets=secrets, environ=environ)

    assert settings.tushare_token == SecretStr("streamlit-tushare")
    assert settings.deepseek_api_key == SecretStr("streamlit-deepseek")
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "streamlit-model"
    for sensitive_value in (*secrets.values(), *environ.values()):
        assert sensitive_value not in caplog.text


def test_environment_is_used_when_streamlit_secret_is_absent() -> None:
    settings = Settings.from_sources(
        secrets={},
        environ={
            "TUSHARE_TOKEN": "environment-tushare",
            "DEEPSEEK_API_KEY": "environment-deepseek",
        },
    )

    assert settings.tushare_token == SecretStr("environment-tushare")
    assert settings.deepseek_api_key == SecretStr("environment-deepseek")
    assert settings.deepseek_base_url == "https://api.deepseek.com"
    assert settings.deepseek_model == "deepseek-v4-flash"


def test_settings_reject_unknown_configuration() -> None:
    with pytest.raises(ValidationError):
        Settings(unknown="value")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.deepseek.com",
        "https://user:pass@api.deepseek.com",
        "https://example.com",
        "https://api.deepseek.com.evil.example",
        "https://evil-api.deepseek.com",
    ],
)
def test_settings_reject_non_official_or_insecure_deepseek_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        Settings(deepseek_base_url=base_url)


def test_settings_normalize_official_deepseek_url_trailing_slash() -> None:
    settings = Settings(deepseek_base_url="https://api.deepseek.com/")

    assert settings.deepseek_base_url == "https://api.deepseek.com"
