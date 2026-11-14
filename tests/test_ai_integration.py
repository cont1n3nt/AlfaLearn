import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from aiogram import Bot
from aiogram.types import Chat, Message, Update, User
from fastapi.testclient import TestClient

from alfa_ai import AIRequest, AIResponse, LearningProfile, Recommendation, Source
from alfalearn.api import create_app
from alfalearn.bot import create_dispatcher
from alfalearn.config import Settings
from alfalearn.memory import ConversationStore


def test_http_consultation_uses_shared_service(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AI_ENABLED=true\nAI_OLLAMA_URL=http://127.0.0.1:11439\nOLLAMA_ENABLED=true\n",
        encoding="utf-8",
    )
    source = Source("S1", "podcast:demo", "chunk", "Сервис", (), "https://example.org/podcast", 61, 93)
    card = Recommendation("course:demo", "Курс", "course", "paid", "https://example.org/course", "По теме.")
    service = AsyncMock()
    service.answer.return_value = AIResponse("answer", "Ответ [S1]", "trace", (source,), recommendations=(card,))
    create = AsyncMock(return_value=service)
    monkeypatch.setattr("alfalearn.api.AIService.create", create)
    settings = Settings(_env_file=env_file, bot_enabled=False, demo_users_enabled=False)
    with TestClient(create_app(settings)) as client:
        response = client.post("/answers", json={"message": "Вопрос"})
        assert response.status_code == 200
        assert "[S1] Сервис (01:01–01:33)" in response.json()["answer"]
        assert "Курс (платно)" in response.json()["answer"]
        assert client.get("/health").json()["llm"]["provider"] == "alfa_ai"
        assert client.post("/answers", json={"message": "x" * 1001}).status_code == 422
        service.answer.assert_awaited_once_with(AIRequest("Вопрос"))
        assert create.await_args.args[0].ollama_url == "http://127.0.0.1:11439"
    service.aclose.assert_awaited_once()


def test_telegram_preserves_profile_and_clarification_until_reset(tmp_path):
    async def run():
        service = AsyncMock()
        service.answer.side_effect = [
            AIResponse("clarify", "Какую задачу решаем?", "1", profile_patch=LearningProfile(industry="cafe")),
            AIResponse("no_answer", "Пока нет сведений.", "2"),
            AIResponse("no_answer", "Пока нет сведений.", "3"),
        ]
        store = ConversationStore(tmp_path / "conversations.sqlite3")
        dispatcher = create_dispatcher(ai=service, memory=store)
        buffer = dispatcher["answer_buffer"]
        async with Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi") as bot:
            session = bot.session
            bot.session = AsyncMock()
            try:
                for number, text in enumerate(("У меня кофейня", "Как вернуть клиентов?", "/reset", "А как начать?"), 1):
                    message = Message(
                        message_id=number,
                        date=datetime.now(timezone.utc),
                        chat=Chat(id=42, type="private"),
                        from_user=User(id=42, is_bot=False, first_name="Test"),
                        text=text,
                    )
                    await dispatcher.feed_update(bot, Update(update_id=number, message=message))
                    await buffer.wait_idle()
                requests = [call.args[0] for call in service.answer.await_args_list]
                assert len(requests) == 3
                assert requests[1].profile.industry == "cafe"
                assert requests[1].previous_query == "У меня кофейня"
                assert requests[2].profile.industry is None
                assert requests[2].previous_query is None
                assert "Какую задачу решаем?" == bot.session.call_args_list[0].args[1].text
            finally:
                await buffer.close()
                store.close()
                bot.session = session

    asyncio.run(run())
