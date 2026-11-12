from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    bot_token: SecretStr | None = None
    bot_enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    group_delay_seconds: float = Field(default=1.0, ge=0.1, le=30)
    ollama_enabled: bool = False
    ollama_base_url: HttpUrl = HttpUrl("https://ollama.com")
    ollama_api_key: SecretStr | None = None
    ollama_model: str = Field(default="gemma4:31b", min_length=1)
    ollama_timeout_seconds: float = Field(default=30.0, ge=1, le=120)
    classification_timeout_seconds: float = Field(default=2.0, ge=0.05, le=5)
    history_path: Path = PROJECT_ROOT / ".run" / "conversations.sqlite3"
    history_context_chars: int = Field(default=12_000, ge=4_000, le=16_000)
    ollama_max_concurrency: int = Field(default=4, ge=1, le=16)

    @field_validator("history_path")
    @classmethod
    def resolve_history_path(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_origin(cls, value):
        if value.username or value.password or value.query or value.fragment or value.path not in (None, "/"):
            raise ValueError("OLLAMA_BASE_URL must be an origin without credentials, path or query")
        if value.scheme != "https" and value.host not in ("localhost", "127.0.0.1", "::1", "[::1]"):
            raise ValueError("Remote Ollama requires HTTPS; HTTP is allowed only on localhost")
        return value
