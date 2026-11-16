"""Configuration for the local model and reviewed corpora."""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_MODEL = "qwen35-unc-local:latest"
DEFAULT_DIGEST = "4cde2d71a709d75cf51d3d0e45432995ac55457175f73e96ca5d47376c721173"
DEFAULT_CORPUS_PATH = Path(__file__).resolve().parents[2] / "assets" / "reviewed_articles.jsonl"
PACKAGED_CORPUS_SHA256 = "6dad98fed9b35db108b45911d1a6787cf081e678b1b2ed33d79e38849dec4ef7"


@dataclass(frozen=True, slots=True)
class Settings:
    corpus_path: Path = DEFAULT_CORPUS_PATH
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_internal_host: str | None = None
    model: str = DEFAULT_MODEL
    model_digest: str = DEFAULT_DIGEST
    request_timeout: float = 60.0
    http_timeout: float = 45.0
    max_workers: int = 1
    max_pending: int = 4
    podcast_corpus_path: Path | None = None
    podcast_corpus_sha256: str | None = None
    e5_model_dir: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "corpus_path", Path(self.corpus_path).expanduser().resolve())
        if self.e5_model_dir is not None:
            object.__setattr__(self, "e5_model_dir", Path(self.e5_model_dir).expanduser().resolve())
        if (self.podcast_corpus_path is None) != (self.podcast_corpus_sha256 is None):
            raise ValueError(
                "Задайте одновременно AI_PODCAST_CORPUS_PATH и AI_PODCAST_CORPUS_SHA256."
            )
        if self.podcast_corpus_path is not None:
            object.__setattr__(
                self, "podcast_corpus_path", Path(self.podcast_corpus_path).expanduser().resolve()
            )
            if not re.fullmatch(r"[a-f0-9]{64}", self.podcast_corpus_sha256):
                raise ValueError("AI_PODCAST_CORPUS_SHA256 должен содержать SHA-256 файла.")
        url = urlsplit(self.ollama_url)
        if self.ollama_internal_host is not None and (
            self.ollama_internal_host != "host.docker.internal"
            and not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", self.ollama_internal_host)
        ):
            raise ValueError("AI_OLLAMA_INTERNAL_HOST должен быть именем внутреннего сервиса.")
        allowed_hosts = {"127.0.0.1", "::1"}
        if self.ollama_internal_host:
            allowed_hosts.add(self.ollama_internal_host)
        if (
            url.scheme != "http"
            or url.hostname not in allowed_hosts
            or url.username
            or url.password
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("AI_OLLAMA_URL должен указывать на loopback или явно заданный внутренний сервис без пути.")
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError("Некорректный порт Ollama.")
        object.__setattr__(self, "ollama_url", self.ollama_url.rstrip("/"))
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or self.model.endswith(":cloud")
        ):
            raise ValueError("Требуется имя локальной модели Ollama.")
        if not re.fullmatch(r"[a-f0-9]{64}", self.model_digest):
            raise ValueError("AI_MODEL_DIGEST должен содержать SHA-256 проверенной модели.")
        for value in (self.request_timeout, self.http_timeout):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 300:
                raise ValueError("Таймаут должен быть конечным числом в диапазоне (0, 300].")
        for value in (self.max_workers, self.max_pending):
            if type(value) is not int or not 1 <= value <= 16:
                raise ValueError("Число запросов должно быть целым числом от 1 до 16.")
        if self.max_pending < self.max_workers:
            raise ValueError("AI_MAX_PENDING должен быть не меньше AI_MAX_WORKERS.")

    @classmethod
    def from_env(cls):
        return cls(
            corpus_path=Path(os.getenv("AI_CORPUS_PATH", str(DEFAULT_CORPUS_PATH))),
            ollama_url=os.getenv("AI_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_internal_host=os.getenv("AI_OLLAMA_INTERNAL_HOST") or None,
            model=os.getenv("AI_MODEL", DEFAULT_MODEL),
            model_digest=os.getenv("AI_MODEL_DIGEST", DEFAULT_DIGEST),
            request_timeout=float(os.getenv("AI_REQUEST_TIMEOUT", "60")),
            http_timeout=float(os.getenv("AI_HTTP_TIMEOUT", "45")),
            max_workers=int(os.getenv("AI_MAX_WORKERS", "1")),
            max_pending=int(os.getenv("AI_MAX_PENDING", "4")),
            podcast_corpus_path=os.getenv("AI_PODCAST_CORPUS_PATH") or None,
            podcast_corpus_sha256=os.getenv("AI_PODCAST_CORPUS_SHA256") or None,
            e5_model_dir=os.getenv("AI_E5_MODEL_DIR") or None,
        )
