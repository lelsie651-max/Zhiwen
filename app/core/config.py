from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "zhiwen"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-this-in-local-env"
    database_url: str = "postgresql+asyncpg://zhiwen:zhiwen@localhost:5432/zhiwen"
    port: int = Field(default=8000, ge=1, le=65535)
    upload_temp_dir: str = "./data/tmp"
    file_storage_root: str = "./data/files"
    max_upload_bytes: int = Field(default=26214400, gt=0)
    upload_chunk_bytes: int = Field(default=1048576, gt=0)
    processing_stale_minutes: int = Field(default=30, gt=0)

    # LLM (OpenAI-compatible) settings. Works with DeepSeek official API,
    # Alibaba Cloud Bailian (DashScope compatible-mode), or any compatible host.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        return self.database_url.replace("+asyncpg", "+psycopg", 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    @computed_field  # type: ignore[prop-decorator]
    @property
    def runtime_debug(self) -> bool:
        # Production always suppresses interactive debug traces.
        return self.debug and not self.is_production

    @computed_field  # type: ignore[prop-decorator]
    @property
    def upload_temp_dir_path(self) -> Path:
        return Path(self.upload_temp_dir).expanduser().resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def file_storage_root_path(self) -> Path:
        return Path(self.file_storage_root).expanduser().resolve()

    @property
    def masked_database_url(self) -> str:
        if "@" not in self.database_url or "://" not in self.database_url:
            return self.database_url

        scheme, remainder = self.database_url.split("://", 1)
        credentials, host_part = remainder.split("@", 1)

        if ":" not in credentials:
            return f"{scheme}://***@{host_part}"

        username, _password = credentials.split(":", 1)
        return f"{scheme}://{username}:***@{host_part}"

    @model_validator(mode="after")
    def validate_production_secret_key(self) -> "Settings":
        weak_secret_values = {
            "",
            "change-this-in-local-env",
            "changeme",
            "secret",
            "default-secret-key",
        }
        if self.is_production and self.secret_key in weak_secret_values:
            raise ValueError("SECRET_KEY must be replaced in production environments.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
