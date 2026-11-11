import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from aiogram import Bot
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from alfalearn.answers import KIND_LABELS, MessageKind, build_answer
from alfalearn.bot import HELLO_MESSAGE, create_dispatcher
from alfalearn.config import Settings

logger = logging.getLogger(__name__)


class AnswerRequest(BaseModel):
    message: str = Field(min_length=1, description="Сообщение пользователя")


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
    async def lifespan(app: FastAPI):
        app.state.polling_task = None
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
            dispatcher = create_dispatcher(settings.group_delay_seconds)
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
                task.cancel()
                try:
                    with suppress(asyncio.CancelledError):
                        await task
                finally:
                    await dispatcher["answer_buffer"].close()

    app = FastAPI(title="AlfaLearn microbot", version="0.1.0", lifespan=lifespan)

    @app.get("/")
    @app.get("/hello")
    async def hello() -> dict[str, str]:
        return {"message": HELLO_MESSAGE}

    @app.post("/answers", tags=["Ответы"], response_model=AnswerResponse)
    async def answer(request: AnswerRequest) -> AnswerResponse:
        """Вернуть заглушку ответа; содержимое сообщения пока не обрабатывается."""
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

    @app.get("/health")
    async def health() -> JSONResponse:
        task = app.state.polling_task
        running = task is not None and not task.done()
        healthy = not settings.bot_enabled or running
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "degraded",
                "bot": "polling" if running else "stopped" if settings.bot_enabled else "disabled",
                "username": app.state.bot_username,
            },
        )

    return app
