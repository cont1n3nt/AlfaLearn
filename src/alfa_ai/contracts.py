"""Immutable request and response contracts for the local AI module."""

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from .podcasts import validate_interval

Status = Literal["answer", "clarify", "no_answer", "action_proposal", "error"]


@dataclass(frozen=True, slots=True)
class LearningProfile:
    stage: Literal["starting", "running"] | None = None
    preferred_format: Literal["article", "course", "podcast"] | None = None
    free_only: bool | None = None
    industry: Literal["salon", "cafe", "retail"] | None = None

    def __post_init__(self):
        if self.stage not in (None, "starting", "running"):
            raise ValueError("Неизвестная стадия бизнеса.")
        if self.preferred_format not in (None, "article", "course", "podcast"):
            raise ValueError("Неизвестный формат материала.")
        if self.free_only is not None and type(self.free_only) is not bool:
            raise ValueError("free_only должен быть логическим значением.")
        if self.industry not in (None, "salon", "cafe", "retail"):
            raise ValueError("Неизвестная сфера бизнеса.")


@dataclass(frozen=True, slots=True)
class AIRequest:
    query: str
    profile: LearningProfile | None = None
    previous_query: str | None = None
    now: datetime | None = None
    timezone: str | None = None

    def __post_init__(self):
        if not isinstance(self.query, str) or not 1 <= len(self.query.strip()) <= 1000:
            raise ValueError("query должен содержать от 1 до 1000 символов.")
        object.__setattr__(self, "query", self.query.strip())
        if self.profile is not None and not isinstance(self.profile, LearningProfile):
            raise ValueError("Ожидается LearningProfile.")
        if self.previous_query is not None:
            if (
                not isinstance(self.previous_query, str)
                or not 1 <= len(self.previous_query.strip()) <= 1000
            ):
                raise ValueError("previous_query должен содержать от 1 до 1000 символов.")
            object.__setattr__(self, "previous_query", self.previous_query.strip())
        if self.now is not None and (
            not isinstance(self.now, datetime) or self.now.utcoffset() is None
        ):
            raise ValueError("now должен содержать часовой пояс или смещение UTC.")
        if self.timezone is not None and (
            not isinstance(self.timezone, str) or not 1 <= len(self.timezone) <= 80
        ):
            raise ValueError("Ожидается имя часового пояса IANA.")


@dataclass(frozen=True, slots=True)
class ReminderProposal:
    action: Literal["create_reminder", "update_reminder", "cancel_reminder"]
    subject: str | None = None
    reminder_id: str | None = None
    local_datetime: str | None = None
    timezone: str | None = None
    repeat: Literal["once", "daily", "weekly"] | None = None
    weekday: int | None = None
    missing_fields: tuple[str, ...] = ()
    needs_confirmation: bool = True


@dataclass(frozen=True, slots=True)
class Recommendation:
    material_id: str
    title: str
    kind: Literal["article", "course", "podcast"]
    access: Literal["free", "subscription", "paid"]
    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    document_id: str
    chunk_id: str
    title: str
    source_block_orders: tuple[int, ...]
    url: str
    start_sec: float | None = None
    end_sec: float | None = None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not re.fullmatch(r"S[1-9]\d*", self.source_id):
            raise ValueError("Некорректная метка источника.")
        for value in (self.document_id, self.chunk_id, self.title):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Некорректные метаданные источника.")
        object.__setattr__(self, "source_block_orders", tuple(self.source_block_orders))
        if any(type(n) is not int or n < 0 for n in self.source_block_orders):
            raise ValueError("Некорректные номера блоков.")
        if self.document_id.startswith("podcast:"):
            validate_interval(self.start_sec, self.end_sec)
            if self.source_block_orders:
                raise ValueError("Источник подкаста использует время вместо номеров блоков.")
        elif not self.source_block_orders or self.start_sec is not None or self.end_sec is not None:
            raise ValueError("Источник статьи использует номера блоков.")
        url = urlsplit(self.url)
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("Источник должен иметь проверенный HTTPS URL.")


@dataclass(frozen=True, slots=True)
class AIResponse:
    status: Status
    text: str
    trace_id: str
    sources: tuple[Source, ...] = ()
    error_code: str | None = None
    recommendations: tuple[Recommendation, ...] = ()
    profile_patch: LearningProfile | None = None
    proposed_action: ReminderProposal | None = None

    def __post_init__(self):
        if self.status not in {"answer", "clarify", "no_answer", "action_proposal", "error"}:
            raise ValueError("Неизвестный статус ответа.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Пустой текст ответа.")
        object.__setattr__(self, "sources", tuple(self.sources))
        if (self.status == "answer") != bool(self.sources):
            raise ValueError("Источники обязательны только для answer.")
        if any(not isinstance(source, Source) for source in self.sources):
            raise ValueError("Некорректный источник ответа.")
        labels = [source.source_id for source in self.sources]
        if len(labels) != len(set(labels)) or set(re.findall(r"\[(S\d+)\]", self.text)) != set(
            labels
        ):
            raise ValueError("Метки ответа не совпадают с источниками.")
        if (self.status == "error") != bool(self.error_code):
            raise ValueError("error_code обязателен только для error.")
        object.__setattr__(self, "recommendations", tuple(self.recommendations))
        if any(not isinstance(item, Recommendation) for item in self.recommendations):
            raise ValueError("Некорректная карточка рекомендации.")
        if self.recommendations and self.status not in {"answer", "no_answer"}:
            raise ValueError("Рекомендации доступны для определённого запроса.")
        if self.profile_patch is not None and not isinstance(self.profile_patch, LearningProfile):
            raise ValueError("Некорректное изменение профиля.")
        if self.proposed_action is not None and not isinstance(
            self.proposed_action, ReminderProposal
        ):
            raise ValueError("Некорректное действие с напоминанием.")
        if self.status == "action_proposal" and (
            self.proposed_action is None or self.proposed_action.missing_fields
        ):
            raise ValueError("Предложение действия требует всех обязательных параметров.")
        if self.proposed_action is not None and self.status not in {"action_proposal", "clarify"}:
            raise ValueError("Действие несовместимо со статусом ответа.")

    def to_dict(self):
        result = asdict(self)
        result["sources"] = [
            {
                **{key: value for key, value in asdict(source).items() if value is not None},
                "source_block_orders": list(source.source_block_orders),
            }
            for source in self.sources
        ]
        result["recommendations"] = [asdict(item) for item in self.recommendations]
        result["profile_patch"] = {
            key: value
            for key, value in (asdict(self.profile_patch) if self.profile_patch else {}).items()
            if value is not None
        }
        if self.proposed_action is not None:
            result["proposed_action"]["missing_fields"] = list(self.proposed_action.missing_fields)
        return result
