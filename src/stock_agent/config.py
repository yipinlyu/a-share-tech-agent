"""Safe application configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class Settings(BaseModel):
    """Deployment settings loaded without exposing credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    tushare_token: SecretStr | None = Field(default=None, repr=False, exclude=True)
    deepseek_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    @field_validator("tushare_token", "deepseek_api_key", mode="before")
    @classmethod
    def blank_secret_is_missing(cls, value: Any) -> Any:
        if isinstance(value, SecretStr):
            return value if value.get_secret_value().strip() else None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("deepseek_base_url")
    @classmethod
    def official_deepseek_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.deepseek.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("deepseek_base_url must be the official HTTPS endpoint")
        return "https://api.deepseek.com"

    @field_validator("deepseek_model")
    @classmethod
    def public_setting_cannot_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("configuration value cannot be blank")
        return value

    @property
    def data_enabled(self) -> bool:
        return self.tushare_token is not None

    @property
    def ai_enabled(self) -> bool:
        return self.deepseek_api_key is not None

    @classmethod
    def from_sources(
        cls,
        secrets: Mapping[str, object] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Self:
        """Load settings, giving explicitly supplied Streamlit Secrets precedence."""

        secret_values = secrets if secrets is not None else {}
        environment_values = environ if environ is not None else os.environ

        def choose(name: str, default: object = None) -> object:
            if name in secret_values:
                return secret_values[name]
            return environment_values.get(name, default)

        return cls(
            tushare_token=choose("TUSHARE_TOKEN"),
            deepseek_api_key=choose("DEEPSEEK_API_KEY"),
            deepseek_base_url=choose("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=choose("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        )
