from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from alfa_ai.settings import DEFAULT_DIGEST, DEFAULT_MODEL, Settings as AISettings

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
    demo_users_enabled: bool = True
    accounts_path: Path = PROJECT_ROOT / ".run" / "accounts.sqlite3"
    ai_enabled: bool = False
    ai_corpus_path: Path = PROJECT_ROOT / "data" / "local" / "pilot_chunks.jsonl"
    ai_ollama_url: str = "http://127.0.0.1:11434"
    ai_model: str = DEFAULT_MODEL
    ai_model_digest: str = DEFAULT_DIGEST
    ai_request_timeout: float = 60
    ai_http_timeout: float = 45
    ai_max_workers: int = 1
    ai_max_pending: int = 4
    ai_podcast_corpus_path: Path | None = None
    ai_podcast_corpus_sha256: str | None = None
    ai_e5_model_dir: Path | None = None

    def ai_settings(self) -> AISettings:
        return AISettings(
            corpus_path=self.ai_corpus_path,
            ollama_url=self.ai_ollama_url,
            model=self.ai_model,
            model_digest=self.ai_model_digest,
            request_timeout=self.ai_request_timeout,
            http_timeout=self.ai_http_timeout,
            max_workers=self.ai_max_workers,
            max_pending=self.ai_max_pending,
            podcast_corpus_path=self.ai_podcast_corpus_path,
            podcast_corpus_sha256=self.ai_podcast_corpus_sha256,
            e5_model_dir=self.ai_e5_model_dir,
        )

    @field_validator(
        "history_path", "accounts_path", "ai_corpus_path", "ai_podcast_corpus_path", "ai_e5_model_dir",
        mode="before",
    )
    @classmethod
    def resolve_project_path(cls, value):
        if value is None or value == "":
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @field_validator("ai_podcast_corpus_sha256", mode="before")
    @classmethod
    def empty_digest(cls, value):
        return value or None

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_origin(cls, value):
        if value.username or value.password or value.query or value.fragment or value.path not in (None, "/"):
            raise ValueError("OLLAMA_BASE_URL must be an origin without credentials, path or query")
        if value.scheme != "https" and value.host not in ("localhost", "127.0.0.1", "::1", "[::1]"):
            raise ValueError("Remote Ollama requires HTTPS; HTTP is allowed only on localhost")
        return value
