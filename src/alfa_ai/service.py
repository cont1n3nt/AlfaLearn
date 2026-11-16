"""Asynchronous interface for answers, recommendations, and reminder proposals."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError
from uuid import uuid4

from .contracts import AIRequest, AIResponse, Source
from .dialogue import resolve_dialogue
from .generation import ModelUnavailable
from .recommendations import lookup_requested_materials, recommend_materials
from .reminders import plan_reminder
from .settings import Settings

logger = logging.getLogger(__name__)
ERROR_TEXT = {
    "busy": "Сейчас обрабатывается слишком много запросов. Попробуйте чуть позже.",
    "timeout": "Ответ не успел подготовиться. Попробуйте ещё раз позже.",
    "unavailable": "Сервис ответов временно недоступен. Попробуйте позже.",
    "model_unavailable": "Нужная модель сейчас недоступна.",
    "invalid_response": "Не удалось получить корректный ответ. Попробуйте переформулировать вопрос.",
    "internal_error": "Не удалось обработать запрос. Попробуйте позже.",
    "closed": "Сервис ответов остановлен.",
}


def error_response(code, trace_id):
    return AIResponse(status="error", text=ERROR_TEXT[code], trace_id=trace_id, error_code=code)


class AIService:
    """One instance per application/event loop; no user history or global state.

    max_pending includes running and queued jobs. Caller cancellation/timeouts
    do not free a running worker: urllib cannot be forcibly cancelled. The slot
    is held until the actual call ends; queued jobs can be cancelled immediately.
    """

    def __init__(self, engine, settings: Settings):
        self._loop = asyncio.get_running_loop()
        self._engine = engine
        self.settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_workers, thread_name_prefix="alfa-ai"
        )
        self._jobs = set()
        self._closed = False

    @classmethod
    async def create(cls, settings: Settings | None = None) -> "AIService":
        settings = settings or Settings.from_env()

        def load_engine():
            # First import of NumPy/sklearn can also take time; keep it off the loop.
            from .engine import RagEngine

            return RagEngine(settings=settings)

        engine = await asyncio.to_thread(load_engine)
        return cls(engine, settings)

    def _check_loop(self):
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("AIService должен использоваться в том же event loop, где создан.")

    def _completed(self, job):
        try:
            self._loop.call_soon_threadsafe(self._jobs.discard, job)
        except RuntimeError:
            pass

    def _run(self, request: AIRequest, trace_id: str) -> AIResponse:
        try:
            reminder = plan_reminder(request)
            if reminder:
                proposal, text = reminder
                status = "clarify" if proposal.missing_fields else "action_proposal"
                return AIResponse(status, text, trace_id, proposed_action=proposal)
            effective, patch, clarification = resolve_dialogue(request)
            if clarification:
                return AIResponse("clarify", clarification, trace_id, profile_patch=patch)
            cards = lookup_requested_materials(effective)
            if cards is not None:
                text = (
                    "В каталоге есть подходящий материал."
                    if len(cards) == 1
                    else "В каталоге есть подходящие материалы."
                    if cards
                    else "В проверенном каталоге нет подходящих материалов по вашему запросу."
                )
                return AIResponse(
                    "no_answer", text, trace_id, recommendations=cards, profile_patch=patch
                )
            history = (
                {"previous_query": effective.previous_query} if effective.previous_query else {}
            )
            result = self._engine.answer(effective.query, **history)
            if not isinstance(result, dict) or result.keys() != {"status", "text", "sources"}:
                raise ValueError("Unexpected engine output")
            status = result["status"]
            sources = tuple(Source(**source) for source in result["sources"])
            cards = ()
            if status in {"answer", "no_answer"}:
                cards = recommend_materials(
                    effective, excluded_ids={source.document_id for source in sources}
                )
            return AIResponse(
                status=status,
                text=result["text"],
                trace_id=trace_id,
                sources=sources,
                recommendations=cards,
                profile_patch=patch,
            )
        except ModelUnavailable:
            code = "model_unavailable"
        except TimeoutError:
            code = "timeout"
        except URLError as exc:
            code = "timeout" if isinstance(exc.reason, TimeoutError) else "unavailable"
        except OSError:
            code = "unavailable"
        except (ValueError, TypeError, KeyError, AttributeError):
            code = "invalid_response"
        except Exception:
            code = "internal_error"
        # Do not log question, full exception, model response or private context.
        logger.warning("AI request failed: trace_id=%s error_code=%s", trace_id, code)
        return error_response(code, trace_id)

    async def answer(self, request: AIRequest) -> AIResponse:
        self._check_loop()
        if not isinstance(request, AIRequest):
            raise TypeError("answer принимает AIRequest.")
        trace_id = str(uuid4())
        if self._closed:
            return error_response("closed", trace_id)
        self._jobs = {job for job in self._jobs if not job.done()}
        if len(self._jobs) >= self.settings.max_pending:
            return error_response("busy", trace_id)
        job = self._executor.submit(self._run, request, trace_id)
        self._jobs.add(job)
        job.add_done_callback(self._completed)
        wrapped = asyncio.wrap_future(job)
        try:
            return await asyncio.wait_for(
                asyncio.shield(wrapped), timeout=self.settings.request_timeout
            )
        except TimeoutError:
            job.cancel()  # Only queued work is cancellable; running work retains capacity.
            return error_response("timeout", trace_id)
        except asyncio.CancelledError:
            job.cancel()
            raise

    async def readiness(self) -> bool:
        self._check_loop()
        if self._closed:
            return False
        try:
            return bool(await asyncio.to_thread(self._engine.readiness))
        except Exception:
            return False

    async def aclose(self) -> None:
        self._check_loop()
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        jobs = tuple(self._jobs)
        if jobs:
            await asyncio.shield(
                asyncio.gather(
                    *(asyncio.wrap_future(job) for job in jobs),
                    return_exceptions=True,
                )
            )

    async def __aenter__(self) -> "AIService":
        self._check_loop()
        if self._closed:
            raise RuntimeError("AIService уже остановлен.")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        await self.aclose()
