import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User
from fastapi.testclient import TestClient

from alfalearn.api import create_app
from alfalearn.answers import ANSWER_MESSAGE
from alfalearn.bot import greeting_message, help_message, create_dispatcher

GREETING = greeting_message(llm=False, accounts=False)
from alfalearn.config import Settings


def test_api_without_telegram():
    settings = Settings(_env_file=None, bot_enabled=False)
    with TestClient(create_app(settings)) as client:
        for route in ("/", "/hello"):
            response = client.get(route)
            assert response.status_code == 200
            assert response.json() == {"message": "Hello, world!"}
        assert client.get("/health").json() == {
            "status": "ok", "bot": "disabled", "username": None,
        }


def test_enabled_bot_requires_token():
    settings = Settings(_env_file=None, bot_enabled=True, bot_token=None)
    with pytest.raises(RuntimeError, match="Set BOT_TOKEN"):
        with TestClient(create_app(settings)):
            pass


def test_answers_api_returns_placeholder():
    settings = Settings(_env_file=None, bot_enabled=False)
    with TestClient(create_app(settings)) as client:
        response = client.post("/answers", json={"message": "Как начать обучение?"})
        assert response.status_code == 200
        assert response.json() == {
            "answer": "Извините я пока-что не могу читать ваши сообщения",
        }


@pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": None}, {"message": 42}])
def test_answers_api_requires_nonempty_message(payload):
    settings = Settings(_env_file=None, bot_enabled=False)
    with TestClient(create_app(settings)) as client:
        assert client.post("/answers", json=payload).status_code == 422


@pytest.mark.parametrize("text,expected", [
    ("/start", GREETING),
    ("/hello", GREETING),
    ("/help", help_message(accounts=False)),
    ("Привет", ANSWER_MESSAGE),
    ("/unknown", ANSWER_MESSAGE),
])
def test_telegram_update_routes_to_reply(text, expected):
    async def run():
        bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
        bot.session = AsyncMock()
        update = Update(
            update_id=1,
            message=Message(
                message_id=1,
                date=datetime.now(timezone.utc),
                chat=Chat(id=42, type="private"),
                from_user=User(id=42, is_bot=False, first_name="Test"),
                text=text,
            ),
        )
        dispatcher = create_dispatcher(group_delay_seconds=0.01)
        await dispatcher.feed_update(bot, update)
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        method = bot.session.call_args.args[1]
        assert isinstance(method, SendMessage)
        assert method.chat_id == 42
        assert method.text == expected

    asyncio.run(run())


@pytest.mark.parametrize("content,expected", [
    ({"photo": [{"file_id": "photo", "file_unique_id": "photo", "width": 100, "height": 100}]},
     "Извините, я пока не могу просматривать фотографии и скриншоты."),
    ({"voice": {"file_id": "voice", "file_unique_id": "voice", "duration": 1}},
     "Извините, я пока не могу прослушивать голосовые сообщения."),
    ({"sticker": {
        "file_id": "sticker", "file_unique_id": "sticker", "type": "regular",
        "width": 100, "height": 100, "is_animated": False, "is_video": False,
    }}, "Извините, я пока не могу распознавать стикеры."),
])
def test_telegram_media_routes_to_placeholder(content, expected):
    async def run():
        bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
        bot.session = AsyncMock()
        update = Update(
            update_id=1,
            message=Message(
                message_id=1,
                date=datetime.now(timezone.utc),
                chat=Chat(id=42, type="private"),
                from_user=User(id=42, is_bot=False, first_name="Test"),
                **content,
            ),
        )
        dispatcher = create_dispatcher(group_delay_seconds=0.01)
        await dispatcher.feed_update(bot, update)
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        method = bot.session.call_args.args[1]
        assert isinstance(method, SendMessage)
        assert method.chat_id == 42
        assert method.text == expected

    asyncio.run(run())
