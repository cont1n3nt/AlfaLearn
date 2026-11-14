"""Demo users: storage, HTTP endpoints and Telegram commands agree with each other."""

import asyncio
from datetime import datetime, timezone
from itertools import count
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import DeleteMessage, SendMessage
from aiogram.types import Chat, Message, Update, User
from fastapi.testclient import TestClient

from alfalearn.accounts import demo_data
from alfalearn.accounts.bot import ASK_LOGIN, ASK_PASSWORD, LOGGED_OUT, WRONG_CREDENTIALS
from alfalearn.accounts.passwords import hash_password, verify_password
from alfalearn.accounts.service import AccountService
from alfalearn.accounts.store import AccountStore
from alfalearn.answers import ANSWER_MESSAGE
from alfalearn.api import create_app
from alfalearn.bot import RESET_MESSAGE, create_dispatcher
from alfalearn.config import Settings

PASSWORDS = {user.login: user.password for user in demo_data.DEMO_USERS}
TELEGRAM_ID = 42


@pytest.fixture
def service():
    store = AccountStore()
    service = AccountService(store)
    service.seed_demo()
    yield service
    store.close()


def available(profile) -> set[str]:
    return {course.title for course in profile.courses if course.available}


def test_password_hash_roundtrip():
    stored = hash_password("secret", iterations=1_000)
    assert stored.startswith("pbkdf2_sha256$1000$") and "secret" not in stored
    assert verify_password("secret", stored)
    assert not verify_password("Secret", stored)
    assert not verify_password("secret", "broken")


def test_three_users_have_different_access(service):
    courses = {
        login: available(service.login(TELEGRAM_ID + i, login, PASSWORDS[login]))
        for i, login in enumerate(("anna", "boris", "vera"))
    }
    assert courses["anna"] == set()
    assert courses["boris"] == {demo_data.MARKETPLACE_USN.title}
    assert courses["vera"] == {course.title for course in demo_data.COURSES}

    vera = service.profile(TELEGRAM_ID + 2)
    assert [p.owned for p in vera.products] == [True, True]
    assert len(vera.courses) == 5


def test_guest_and_wrong_password(service):
    guest = service.profile(TELEGRAM_ID)
    assert guest.user is None and available(guest) == set()
    assert service.login(TELEGRAM_ID, "boris", "wrong") is None
    assert service.login(TELEGRAM_ID, "nobody", "x") is None
    assert service.profile(TELEGRAM_ID).user is None


def test_relogin_switches_user_and_logout(service):
    service.login(TELEGRAM_ID, "anna", PASSWORDS["anna"])
    assert service.login(TELEGRAM_ID, " Vera ", PASSWORDS["vera"]).user.login == "vera"
    assert service.logout(TELEGRAM_ID)
    assert not service.logout(TELEGRAM_ID)
    assert service.profile(TELEGRAM_ID).user is None


def test_seed_is_idempotent_and_reset_drops_logins(tmp_path):
    path = tmp_path / "db.sqlite3"
    store = AccountStore(path)
    service = AccountService(store)
    assert service.seed_demo() == ["anna", "boris", "vera"]
    service.login(TELEGRAM_ID, "boris", PASSWORDS["boris"])
    store.close()

    # Restart: nothing duplicated, the Telegram login survives.
    store = AccountStore(path)
    service = AccountService(store)
    assert service.seed_demo() == []
    assert service.profile(TELEGRAM_ID).user.login == "boris"

    assert service.reset_demo() == ["anna", "boris", "vera"]
    assert service.profile(TELEGRAM_ID).user is None
    assert len(service.profile(TELEGRAM_ID).products) == 2
    store.close()


def test_links_point_to_course_pages():
    assert demo_data.MARKETPLACE_USN.url == (
        "https://kurs.alfabank.ru/courses/usn-na-marketplejsah-osobennosti-nalogovogo-uchyota/"
    )
    assert demo_data.KURS_PLUS.url == "https://kurs.alfabank.ru/subscription/"


# --- HTTP -------------------------------------------------------------------


def make_client():
    return TestClient(create_app(Settings(_env_file=None, bot_enabled=False)))


def test_api_login_profile_logout():
    with make_client() as client:
        base = f"/telegram/{TELEGRAM_ID}"
        assert client.get(f"{base}/user").status_code == 404
        assert client.post(f"{base}/login", json={"login": "boris", "password": "no"}).status_code == 401

        response = client.post(f"{base}/login", json={"login": "boris", "password": PASSWORDS["boris"]})
        assert response.status_code == 200
        body = client.get(f"{base}/user").json()
        assert body == response.json()
        assert body["user"]["login"] == "boris" and "password_hash" not in body["user"]
        assert [c["title"] for c in body["courses"] if c["available"]] == [demo_data.MARKETPLACE_USN.title]
        assert {p["id"]: p["owned"] for p in body["products"]} == {"kurs_plus": False, "usn_marketplaces": True}

        assert client.delete(f"{base}/login").status_code == 204
        assert client.delete(f"{base}/login").status_code == 204
        assert client.get(f"{base}/user").status_code == 404


def test_api_rejects_bad_input():
    with make_client() as client:
        assert client.post("/telegram/abc/login", json={"login": "anna", "password": "x"}).status_code == 422
        assert client.post("/telegram/1/login", json={"login": "", "password": "x"}).status_code == 422


def test_demo_users_can_be_disabled():
    settings = Settings(_env_file=None, bot_enabled=False, demo_users_enabled=False)
    with TestClient(create_app(settings)) as client:
        assert client.get(f"/telegram/{TELEGRAM_ID}/user").status_code == 404
        assert "/telegram/{telegram_id}/user" not in client.get("/openapi.json").json()["paths"]


# --- Telegram -----------------------------------------------------------------

_ids = count(1)


def text_update(text: str) -> Update:
    return Update(update_id=next(_ids), message=Message(
        message_id=next(_ids), date=datetime.now(timezone.utc), chat=Chat(id=TELEGRAM_ID, type="private"),
        from_user=User(id=TELEGRAM_ID, is_bot=False, first_name="Test"), text=text,
    ))


def run_bot(service, scenario):
    async def run():
        bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
        bot.session = AsyncMock()
        dispatcher = create_dispatcher(group_delay_seconds=0.01, accounts=service)

        async def feed(text: str) -> tuple[list[str], int]:
            """Send a message, return bot replies and the number of deleted messages."""
            bot.session.reset_mock()
            await dispatcher.feed_update(bot, text_update(text))
            await dispatcher["answer_buffer"].wait_idle()
            methods = [call.args[1] for call in bot.session.call_args_list]
            replies = [m.text for m in methods if isinstance(m, SendMessage)]
            return replies, sum(isinstance(m, DeleteMessage) for m in methods)

        try:
            await scenario(feed)
        finally:
            await dispatcher["answer_buffer"].close()

    asyncio.run(run())


def test_bot_login_dialog_hides_password_and_shows_profile(service):
    async def scenario(feed):
        replies, _ = await feed("/profile")
        assert "Гость" in replies[0] and "Открыто 0 из 5" in replies[0]

        assert await feed("/login") == ([ASK_LOGIN], 0)
        assert await feed("vera") == ([ASK_PASSWORD], 0)
        replies, deleted = await feed(PASSWORDS["vera"])
        assert deleted == 1
        assert "Вход выполнен" in replies[0] and "Вера" in replies[0] and "Открыто 5 из 5" in replies[0]

        # The answers buffer never saw the password.
        assert PASSWORDS["vera"] not in str(service.store.user_by_login("vera"))
        assert service.profile(TELEGRAM_ID).user.login == "vera"

        replies, _ = await feed("/profile")
        assert "Вера" in replies[0]
        assert await feed("/logout") == ([LOGGED_OUT], 0)
        assert service.profile(TELEGRAM_ID).user is None

    run_bot(service, scenario)


def test_bot_one_line_login_and_wrong_password(service):
    async def scenario(feed):
        replies, deleted = await feed(f"/login boris {PASSWORDS['boris']}")
        assert deleted == 1 and "Борис" in replies[0] and "Открыто 1 из 5" in replies[0]

        replies, deleted = await feed("/login anna wrong")
        assert (replies, deleted) == ([WRONG_CREDENTIALS], 1)
        assert service.profile(TELEGRAM_ID).user.login == "boris"

        # Regular messages still reach the answers handler.
        assert await feed("Привет") == ([ANSWER_MESSAGE], 0)

    run_bot(service, scenario)


def test_api_and_bot_share_one_login():
    with make_client() as client:
        service = client.app.state.accounts
        client.post(f"/telegram/{TELEGRAM_ID}/login", json={"login": "anna", "password": PASSWORDS["anna"]})

        async def scenario(feed):
            replies, _ = await feed("/profile")
            assert "Анна" in replies[0] and "Открыто 0 из 5" in replies[0]
            await feed(f"/login vera {PASSWORDS['vera']}")

        run_bot(service, scenario)
        assert client.get(f"/telegram/{TELEGRAM_ID}/user").json()["user"]["login"] == "vera"


def test_greeting_and_help_list_account_commands(service):
    async def scenario(feed):
        greeting, _ = await feed("/start")
        assert greeting == (await feed("/hello"))[0]
        assert greeting[0].startswith("Привет! Я АльфаНавигатор")
        assert "Доступные команды" not in greeting[0]

        help_text, _ = await feed("/help")
        assert help_text[0] == (
            "Доступные команды:\n"
            "/login — войти в демо-аккаунт (anna, boris, vera)\n"
            "/profile — мой доступ к платным курсам\n"
            "/logout — выйти из аккаунта\n"
            "/reset — очистить историю диалога\n"
            "/help — показать этот список"
        )

    run_bot(service, scenario)


def test_reset_explains_what_is_cleared_and_greets(service):
    async def scenario(feed):
        await feed(f"/login boris {PASSWORDS['boris']}")
        replies, _ = await feed("/reset")
        assert replies[0].startswith(RESET_MESSAGE)
        assert "Привет! Я АльфаНавигатор" in replies[0]
        # /reset forgets the conversation, not the login.
        assert service.profile(TELEGRAM_ID).user.login == "boris"

    run_bot(service, scenario)
