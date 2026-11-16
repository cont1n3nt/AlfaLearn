import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from aiogram import Bot
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from alfa_ai import AIRequest, AIService
from alfalearn.accounts.api import create_accounts_router
from alfalearn.analytics import EventStore
from alfalearn.accounts.service import AccountService
from alfalearn.accounts.store import AccountStore
from alfalearn.answers import KIND_LABELS, MessageKind, build_answer
from alfalearn.bot import HELLO_MESSAGE, create_dispatcher
from alfalearn.config import Settings
from alfalearn.grouping import OLLAMA_UNAVAILABLE
from alfalearn.knowledge import TOO_LONG, format_answer
from alfalearn.llm import OllamaClient, OllamaError
from alfalearn.memory import ConversationStore
from alfalearn.reminder_runtime import ReminderDialogue, ReminderStore, ReminderWorker

logger = logging.getLogger(__name__)


class AnswerRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_000, description="Сообщение пользователя")

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value.strip()


class AnswerResponse(BaseModel):
    answer: str


class GroupAnswerRequest(BaseModel):
    types: list[MessageKind] = Field(min_length=1, max_length=100)


def typed_answer_endpoint(kind: MessageKind):
    async def answer() -> AnswerResponse:
        return AnswerResponse(answer=build_answer([kind]))

    return answer


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def telegram_lifespan(app: FastAPI):
        app.state.polling_task = None
        app.state.reminder_task = None
        app.state.bot_username = None
        if not settings.bot_enabled:
            yield
            return

        if not settings.bot_token or not settings.bot_token.get_secret_value():
            raise RuntimeError("Set BOT_TOKEN in .env, or set BOT_ENABLED=false for API only.")

        async with Bot(token=settings.bot_token.get_secret_value()) as bot:
            # Fail startup for invalid credentials or an incompatible webhook setup.
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
            if webhook.url:
                raise RuntimeError("Bot has an active webhook. Use a bot configured for polling.")
            app.state.bot_username = me.username
            reminders = (
                ReminderDialogue(app.state.reminder_store)
                if app.state.reminder_store is not None else None
            )
            dispatcher = create_dispatcher(
                settings.group_delay_seconds,
                llm=app.state.llm,
                classification_timeout_seconds=settings.classification_timeout_seconds,
                memory=app.state.memory, history_context_chars=settings.history_context_chars,
                accounts=app.state.accounts,
                ai=app.state.ai,
                reminders=reminders,
                analytics=app.state.analytics,
            )
            if app.state.reminder_store is not None:
                async def send_reminder(row):
                    await bot.send_message(
                        row["chat_id"], f"🔔 Напоминание #{row['id']}: {row['subject']}",
                        message_thread_id=row["thread_id"],
                    )

                worker = ReminderWorker(app.state.reminder_store, send_reminder)
                app.state.reminder_task = asyncio.create_task(
                    worker.run(), name="telegram-reminders",
                )
            task = asyncio.create_task(
                dispatcher.start_polling(
                    bot,
                    handle_signals=False,
                    close_bot_session=False,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                ),
                name="telegram-polling",
            )
            app.state.polling_task = task

            def report_failure(done: asyncio.Task) -> None:
                if not done.cancelled() and done.exception() is not None:
                    # Avoid logging exception text that could contain credentials.
                    logger.error("Telegram polling stopped (%s)", type(done.exception()).__name__)

            task.add_done_callback(report_failure)
            try:
                yield
            finally:
                if app.state.reminder_task is not None:
                    app.state.reminder_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await app.state.reminder_task
                task.cancel()
                try:
                    with suppress(asyncio.CancelledError):
                        await task
                finally:
                    await dispatcher["answer_buffer"].close()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.llm = None
        app.state.memory = None
        app.state.accounts = None
        app.state.ai = None
        app.state.reminder_store = None
        app.state.analytics = None
        try:
            if settings.demo_users_enabled:
                app.state.accounts = AccountService(AccountStore(settings.accounts_path))
                app.state.accounts.seed_demo()
            if settings.ai_enabled:
                app.state.ai = await AIService.create(settings.ai_settings())
            elif settings.ollama_enabled:
                key = settings.ollama_api_key.get_secret_value() if settings.ollama_api_key else None
                if settings.ollama_base_url.host == "ollama.com" and not key:
                    raise RuntimeError("Set OLLAMA_API_KEY in .env for Ollama Cloud.")
                app.state.llm = OllamaClient(
                    base_url=str(settings.ollama_base_url), model=settings.ollama_model,
                    api_key=key, timeout_seconds=settings.ollama_timeout_seconds,
                    max_concurrency=settings.ollama_max_concurrency,
                )
            if settings.bot_enabled and (app.state.ai is not None or app.state.llm is not None):
                app.state.memory = ConversationStore(settings.history_path)
            if settings.bot_enabled and app.state.ai is not None:
                app.state.reminder_store = ReminderStore(settings.reminders_path)
                app.state.analytics = EventStore(settings.analytics_path, settings.analytics_retention_days)
            async with telegram_lifespan(app):
                yield
        finally:
            if app.state.memory is not None:
                app.state.memory.close()
            if app.state.reminder_store is not None:
                app.state.reminder_store.close()
            if app.state.analytics is not None:
                app.state.analytics.close()
            if app.state.accounts is not None:
                app.state.accounts.store.close()
            if app.state.llm is not None:
                await app.state.llm.close()
            if app.state.ai is not None:
                await app.state.ai.aclose()

    app = FastAPI(title="AlfaLearn microbot", version="0.1.0", lifespan=lifespan)

    @app.get("/")
    @app.get("/hello")
    async def hello() -> dict[str, str]:
        return {"message": HELLO_MESSAGE}

    @app.post("/answers", tags=["Ответы"], response_model=AnswerResponse)
    async def answer(request: AnswerRequest) -> AnswerResponse | JSONResponse:
        """Ответ по базе знаний, ответ Ollama или заглушка."""
        if app.state.ai is not None:
            if len(request.message) > 1000:
                return JSONResponse(status_code=422, content={"answer": TOO_LONG})
            response = await app.state.ai.answer(AIRequest(query=request.message))
            if response.status == "error":
                return JSONResponse(status_code=503, content={"answer": response.text})
            return AnswerResponse(answer=format_answer(response))
        if app.state.llm is not None:
            try:
                return AnswerResponse(answer=await app.state.llm.answer(request.message))
            except (OllamaError, TimeoutError):
                logger.warning("Ollama HTTP answer unavailable")
                return JSONResponse(status_code=503, content={"answer": OLLAMA_UNAVAILABLE})
        return AnswerResponse(answer=build_answer([MessageKind.TEXT]))

    for kind in MessageKind:
        app.add_api_route(
            f"/answers/{kind.value}", typed_answer_endpoint(kind), methods=["POST"],
            response_model=AnswerResponse, tags=["Ответы"],
            summary=KIND_LABELS[kind].capitalize(), name=f"answer_{kind.value}",
        )

    @app.post("/answers/group", tags=["Ответы"], response_model=AnswerResponse)
    async def group_answer(request: GroupAnswerRequest) -> AnswerResponse:
        """Вернуть один ответ для перечисленных типов сообщений."""
        return AnswerResponse(answer=build_answer(request.types, len(request.types)))

    if settings.demo_users_enabled:
        app.include_router(create_accounts_router())

    @app.get("/health")
    async def health() -> JSONResponse:
        task = app.state.polling_task
        running = task is not None and not task.done()
        content = {
            "status": "ok",
            "bot": "polling" if running else "stopped" if settings.bot_enabled else "disabled",
            "username": app.state.bot_username,
        }
        if settings.ai_enabled:
            content["llm"] = {"provider": "alfa_ai", "model": settings.ai_model}
        elif settings.ollama_enabled:
            # Configuration only: this endpoint does not make paid model calls.
            content["llm"] = {"provider": "ollama", "model": settings.ollama_model}
        return JSONResponse(
            status_code=200,
            content=content,
        )

    @app.get("/ready")
    async def ready() -> JSONResponse:
        task = app.state.polling_task
        bot_ready = not settings.bot_enabled or (task is not None and not task.done())
        reminder_task = app.state.reminder_task
        reminders_ready = app.state.reminder_store is None or (
            reminder_task is not None and not reminder_task.done()
        )
        model_ready = (
            await app.state.ai.readiness() if app.state.ai is not None else True
        )
        healthy = bot_ready and reminders_ready and model_ready
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ready" if healthy else "unavailable",
                "bot": bot_ready,
                "reminders": reminders_ready,
                "model": model_ready,
            },
        )

    return app
