import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Message, Update
from fastapi.testclient import TestClient

from alfalearn.answers import classify_message, is_emoji_only
from alfalearn.api import create_app
from alfalearn.bot import create_dispatcher
from alfalearn.config import Settings

FILE = {"file_id": "file", "file_unique_id": "unique"}
PHOTO = {"photo": [{**FILE, "width": 100, "height": 100}]}
VOICE = {"voice": {**FILE, "duration": 1}}
VIDEO = {"video": {**FILE, "width": 100, "height": 100, "duration": 1}}
ORIGINAL = {
    "message_id": 10, "date": 1700000000, "chat": {"id": 42, "type": "private"},
    "from_user": {"id": 123456789, "is_bot": True, "first_name": "Bot"},
    "text": "Сообщение бота",
}
MESSAGE_CASES = [
    ("text", {"text": "Привет"}, {"text"}, "читать ваши сообщения"),
    ("voice", VOICE, {"voice"}, "прослушивать голосовые сообщения"),
    ("photo", PHOTO, {"photo"}, "просматривать фотографии и скриншоты"),
    ("document", {"document": FILE}, {"document"}, "читать документы и файлы"),
    ("link", {"text": "https://example.com", "entities": [
        {"type": "url", "offset": 0, "length": 19},
    ]}, {"link"}, "открывать и читать ссылки"),
    ("video", VIDEO, {"video"}, "просматривать видео"),
    ("video-note", {"video_note": {**FILE, "length": 100, "duration": 1}},
     {"video-note"}, "просматривать видеокружки"),
    ("audio", {"audio": {**FILE, "duration": 1}}, {"audio"}, "прослушивать аудиофайлы"),
    ("sticker", {"sticker": {
        **FILE, "type": "regular", "width": 100, "height": 100,
        "is_animated": False, "is_video": False,
    }}, {"sticker"}, "распознавать стикеры"),
    ("gif", {"animation": {**FILE, "width": 100, "height": 100, "duration": 1},
             "document": FILE}, {"gif"}, "просматривать GIF-анимации"),
    ("emoji", {"text": "👩🏽‍💻 ❤️ 🇷🇺 1️⃣"}, {"emoji"}, "интерпретировать эмодзи"),
    ("contact", {"contact": {"phone_number": "+70000000000", "first_name": "Test"}},
     {"contact"}, "обрабатывать контакты"),
    ("location", {"location": {"latitude": 55.75, "longitude": 37.61}},
     {"location"}, "обрабатывать геолокацию"),
    ("forwarded", {"text": "Пересланный текст", "forward_origin": {
        "type": "hidden_user", "date": 1700000000, "sender_user_name": "Test",
    }}, {"text", "forwarded"}, "пересланное сообщение"),
    ("reply", {"text": "Ответ", "reply_to_message": ORIGINAL},
     {"text", "reply"}, "ответ на сообщение бота"),
    ("quote", {"text": "Комментарий", "quote": {"text": "Цитата", "position": 0}},
     {"text", "quote"}, "сообщение с цитатой"),
    ("album", {**PHOTO, "media_group_id": "album-1"},
     {"photo", "album"}, "альбом фотографий или видео"),
    ("caption", {**PHOTO, "caption": "Подпись"},
     {"photo", "caption"}, "вложение с текстовой подписью"),
]


def make_message(content, *, message_id=1, chat_id=42, user_id=42, thread_id=None):
    return Message(
        message_id=message_id, date=datetime.now(timezone.utc),
        chat={"id": chat_id, "type": "private" if chat_id > 0 else "supergroup"},
        from_user={"id": user_id, "is_bot": False, "first_name": "Test"},
        message_thread_id=thread_id, is_topic_message=thread_id is not None,
        **content,
    )


def make_bot():
    bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    bot.session = AsyncMock()
    return bot


@pytest.mark.parametrize("kind,content,expected,fragment", MESSAGE_CASES,
                         ids=[case[0] for case in MESSAGE_CASES])
def test_all_message_types_are_detected_and_answered(kind, content, expected, fragment):
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.01)
        message = make_message(content)
        assert classify_message(message, bot.id) == expected
        await dispatcher.feed_update(bot, Update(update_id=1, message=message))
        bot.session.assert_not_awaited()
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        assert fragment in bot.session.call_args.args[1].text

    asyncio.run(run())


@pytest.mark.parametrize("kind", [case[0] for case in MESSAGE_CASES])
def test_each_type_has_an_http_endpoint(kind):
    with TestClient(create_app(Settings(_env_file=None, bot_enabled=False))) as client:
        response = client.post(f"/answers/{kind}")
        assert response.status_code == 200
        assert response.json()["answer"].startswith("Извините")
        assert response.json()["answer"] != "Hello, world!"
        operation = client.get("/openapi.json").json()["paths"][f"/answers/{kind}"]["post"]
        assert operation["tags"] == ["Ответы"]
        assert "parameters" not in operation


def test_group_endpoint_uses_types_and_deduplicates_labels():
    with TestClient(create_app(Settings(_env_file=None, bot_enabled=False))) as client:
        response = client.post("/answers/group", json={"types": ["photo", "voice", "photo"]})
        assert response.status_code == 200
        assert response.json() == {"answer": (
            "Извините, я пока не могу обработать эту группу сообщений: "
            "голосовое сообщение, фото или скриншот."
        )}


@pytest.mark.parametrize("payload", [
    {}, {"types": []}, {"types": ["unknown"]}, {"types": "photo"},
    {"types": [None]}, {"types": ["photo"] * 101},
])
def test_invalid_group_is_rejected(payload):
    with TestClient(create_app(Settings(_env_file=None, bot_enabled=False))) as client:
        assert client.post("/answers/group", json=payload).status_code == 422


@pytest.mark.parametrize("text,expected", [
    ("😀", True), ("👨‍👩‍👧‍👦", True), ("👍🏽", True), ("🇷🇺", True),
    ("#️⃣", True), ("1", False), ("Привет 😀", False), (" ", False), ("你好", False),
])
def test_emoji_and_ordinary_text_are_distinguished(text, expected):
    assert is_emoji_only(text) is expected


def test_caption_link_forward_and_quote_keep_their_types():
    content = {
        **PHOTO, "caption": "https://example.com", "caption_entities": [
            {"type": "text_link", "offset": 0, "length": 19, "url": "https://example.com"},
        ], "forward_origin": {"type": "hidden_user", "date": 1700000000, "sender_user_name": "Test"},
        "quote": {"text": "Цитата", "position": 0},
    }
    assert classify_message(make_message(content)) == {"photo", "caption", "link", "forwarded", "quote"}


def test_reply_to_a_user_or_another_bot_is_not_a_reply_to_this_bot():
    for sender in (
        {"id": 77, "is_bot": False, "first_name": "User"},
        {"id": 88, "is_bot": True, "first_name": "OtherBot"},
    ):
        message = make_message({"text": "Ответ", "reply_to_message": {**ORIGINAL, "from_user": sender}})
        assert classify_message(message, bot_id=123456789) == {"text"}


def test_blockquote_entity_is_a_quote():
    message = make_message({"text": "Цитата", "entities": [
        {"type": "blockquote", "offset": 0, "length": 6},
    ]})
    assert classify_message(message) == {"text", "quote"}


def test_mixed_messages_receive_one_combined_reply():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.02)
        for index, content in enumerate(({"text": "Привет"}, PHOTO, VOICE), start=1):
            await dispatcher.feed_update(bot, Update(update_id=index, message=make_message(content, message_id=index)))
        bot.session.assert_not_awaited()
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        assert bot.session.call_args.args[1].text == (
            "Извините, я пока не могу обработать эту группу сообщений: "
            "текстовое сообщение, голосовое сообщение, фото или скриншот."
        )

    asyncio.run(run())


def test_mixed_photo_video_album_gets_one_reply():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.02)
        for index, content in enumerate((PHOTO, VIDEO, PHOTO), start=1):
            message = make_message({**content, "media_group_id": "same-album"}, message_id=index)
            await dispatcher.feed_update(bot, Update(update_id=index, message=message))
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        assert bot.session.call_args.args[1].text == (
            "Извините, я пока не могу обработать эту группу сообщений: "
            "фото или скриншот, видео, альбом фотографий или видео."
        )

    asyncio.run(run())


def test_all_18_types_in_one_burst_receive_one_reply():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.05)
        updates = [
            dispatcher.feed_update(bot, Update(
                update_id=index, message=make_message(content, message_id=index),
            ))
            for index, (_, content, _, _) in enumerate(MESSAGE_CASES, start=1)
        ]
        await asyncio.gather(*updates)
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        answer = bot.session.call_args.args[1].text
        assert "группу сообщений" in answer
        assert "видеокружок" in answer
        assert "геолокация" in answer
        assert "пересланное сообщение" in answer
        assert answer.count("фото или скриншот") == 1

    asyncio.run(run())


def test_repeated_texts_receive_one_group_reply():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.02)
        for index in range(3):
            await dispatcher.feed_update(bot, Update(
                update_id=index, message=make_message({"text": "Привет"}, message_id=index + 1),
            ))
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        assert bot.session.call_args.args[1].text == (
            "Извините, я пока не могу обработать эту группу сообщений: текстовое сообщение."
        )

    asyncio.run(run())


def test_new_group_does_not_cancel_a_reply_already_being_sent():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.01)
        send_started = asyncio.Event()
        allow_send = asyncio.Event()

        async def send(*args, **kwargs):
            send_started.set()
            await allow_send.wait()

        bot.session.side_effect = send
        await dispatcher.feed_update(bot, Update(update_id=1, message=make_message(PHOTO)))
        await asyncio.wait_for(send_started.wait(), timeout=1)
        await dispatcher.feed_update(bot, Update(update_id=2, message=make_message(VOICE)))
        allow_send.set()
        await dispatcher["answer_buffer"].wait_idle()
        assert bot.session.await_count == 2
        replies = [call.args[1].text for call in bot.session.call_args_list]
        assert any("фотографии" in reply for reply in replies)
        assert any("голосовые" in reply for reply in replies)

    asyncio.run(run())


@pytest.mark.parametrize("other", [
    {"chat_id": 43}, {"user_id": 43}, {"thread_id": 2},
])
def test_groups_do_not_mix_chats_users_or_topics(other):
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.02)
        first = make_message({"text": "Привет"}, thread_id=1)
        second = make_message(PHOTO, **{"thread_id": 1, **other})
        await dispatcher.feed_update(bot, Update(update_id=1, message=first))
        await dispatcher.feed_update(bot, Update(update_id=2, message=second))
        await dispatcher["answer_buffer"].wait_idle()
        assert bot.session.await_count == 2
        replies = [call.args[1] for call in bot.session.call_args_list]
        assert all("группу" not in reply.text for reply in replies)
        if "thread_id" in other:
            assert {reply.message_thread_id for reply in replies} == {1, 2}

    asyncio.run(run())


def test_new_message_extends_the_grouping_window():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.2)
        await dispatcher.feed_update(bot, Update(update_id=1, message=make_message(PHOTO)))
        await asyncio.sleep(0.1)
        await dispatcher.feed_update(bot, Update(update_id=2, message=make_message(VOICE)))
        await asyncio.sleep(0.13)
        bot.session.assert_not_awaited()
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()

    asyncio.run(run())


def test_messages_after_a_pause_start_a_new_group():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.01)
        await dispatcher.feed_update(bot, Update(update_id=1, message=make_message(PHOTO)))
        await dispatcher["answer_buffer"].wait_idle()
        await dispatcher.feed_update(bot, Update(update_id=2, message=make_message(VOICE)))
        await dispatcher["answer_buffer"].wait_idle()
        assert bot.session.await_count == 2

    asyncio.run(run())


def test_commands_in_a_burst_do_not_create_extra_replies():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=0.02)
        for index, content in enumerate(({"text": "/start"}, PHOTO), start=1):
            await dispatcher.feed_update(bot, Update(update_id=index, message=make_message(content)))
        await dispatcher["answer_buffer"].wait_idle()
        bot.session.assert_awaited_once()
        assert "группу сообщений" in bot.session.call_args.args[1].text

    asyncio.run(run())


def test_shutdown_cancels_pending_replies():
    async def run():
        bot = make_bot()
        dispatcher = create_dispatcher(group_delay_seconds=1)
        await dispatcher.feed_update(bot, Update(update_id=1, message=make_message(PHOTO)))
        await dispatcher.emit_shutdown()
        buffer = dispatcher["answer_buffer"]
        assert not buffer.pending
        assert not buffer.tasks
        bot.session.assert_not_awaited()

    asyncio.run(run())
