import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User
from fastapi.testclient import TestClient

from alfalearn.api import create_app
from alfalearn.bot import HELLO_MESSAGE, START_MESSAGE, create_dispatcher
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


@pytest.mark.parametrize("text,expected", [
    ("/start", START_MESSAGE),
    ("/hello", HELLO_MESSAGE),
    ("/help", HELLO_MESSAGE),
    ("Привет", HELLO_MESSAGE),
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
        await create_dispatcher().feed_update(bot, update)
        method = bot.session.call_args.args[1]
        assert isinstance(method, SendMessage)
        assert method.chat_id == 42
        assert method.text == expected

    asyncio.run(run())
