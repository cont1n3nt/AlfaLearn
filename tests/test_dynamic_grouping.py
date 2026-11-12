"""Behavioural checks for strict turn gating, isolated memory and cancellation."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Message

from alfalearn.bot import create_dispatcher
from alfalearn.llm import OllamaError

PHOTO = {"photo": [{"file_id": "photo", "file_unique_id": "photo", "width": 100, "height": 100}]}


class FakeBrain:
    def __init__(self, *, ready=False, reply="Тестовый ответ"):
        self.classify_turn = AsyncMock(return_value="QUESTION" if ready else "WAIT")
        self.answer_with_history = AsyncMock(return_value=reply)
        self.summarize = AsyncMock(return_value="Краткая память")


def make_buffer(brain, *, delay=0.05, classification_timeout=0.03, memory=None,
                history_context_chars=12000):
    bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
    bot.session = AsyncMock()
    dispatcher = create_dispatcher(
        group_delay_seconds=delay, llm=brain,
        classification_timeout_seconds=classification_timeout,
        memory=memory, history_context_chars=history_context_chars,
    )
    return dispatcher["answer_buffer"], bot


def message(bot, text=None, *, message_id=1, chat_id=42, user_id=42, thread_id=None,
            sender_chat_id=None, **content):
    values = {
        "message_id": message_id, "date": datetime.now(timezone.utc),
        "chat": {"id": chat_id, "type": "private" if chat_id > 0 else "supergroup"},
        "message_thread_id": thread_id, "is_topic_message": thread_id is not None,
        **content,
    }
    if user_id is not None:
        values["from_user"] = {"id": user_id, "is_bot": False, "first_name": "Test"}
    if sender_chat_id is not None:
        values["sender_chat"] = {"id": sender_chat_id, "type": "channel", "title": "Sender"}
    if text is not None:
        values["text"] = text
    return Message(**values).as_(bot)


def key_for(msg):
    return (msg.chat.id, msg.message_thread_id,
            msg.from_user.id if msg.from_user else None,
            msg.sender_chat.id if msg.sender_chat else None)


def sent_texts(bot):
    return [call.args[1].text for call in bot.session.call_args_list]


def contents(rows):
    return [(row["role"], row["content"]) for row in rows]


def test_obvious_question_starts_answer_without_waiting_for_group_window():
    async def run():
        brain = FakeBrain()
        started = asyncio.Event()

        async def answer(text, history):
            started.set()
            return "Прибыль = выручка минус расходы."

        brain.answer_with_history.side_effect = answer
        buffer, bot = make_buffer(brain, delay=10)
        try:
            buffer.add(message(bot, "Как рассчитать прибыль?"))
            await asyncio.wait_for(started.wait(), timeout=1)
            await buffer.wait_idle()
            brain.classify_turn.assert_not_awaited()
            brain.answer_with_history.assert_awaited_once_with("Как рассчитать прибыль?", [])
            assert sent_texts(bot) == ["Прибыль = выручка минус расходы."]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_semantic_complete_request_without_punctuation_starts_immediately():
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain, delay=10)
        try:
            text = "Объясни разницу между выручкой и прибылью"
            buffer.add(message(bot, text))
            await asyncio.wait_for(buffer.wait_idle(), timeout=1)
            brain.classify_turn.assert_awaited_once_with(text, [])
            assert sent_texts(bot) == ["Тестовый ответ"]
        finally:
            await buffer.close()
    asyncio.run(run())


@pytest.mark.parametrize("text", [
    "?", "А если?", "Как вы думаете?", "Как же так?", "Что ты думаешь?",
    "Он спросил: «Как рассчитать прибыль?»",
])
def test_question_mark_without_self_contained_intent_does_not_force_a_reply(text):
    async def run():
        brain = FakeBrain()
        buffer, bot = make_buffer(brain)
        incoming = message(bot, text)
        try:
            buffer.add(incoming)
            await buffer.wait_idle()
            brain.classify_turn.assert_awaited_once_with(text, [])
            brain.answer_with_history.assert_not_awaited()
            bot.session.assert_not_awaited()
            assert contents(buffer.store.history(key_for(incoming))) == [("user", text)]
            assert not buffer.pending
        finally:
            await buffer.close()
    asyncio.run(run())


def test_fragments_remain_silent_after_the_grouping_window():
    async def run():
        brain = FakeBrain()
        buffer, bot = make_buffer(brain, delay=0.12)
        first = message(bot, "У меня небольшая кофейня")
        try:
            buffer.add(first)
            await asyncio.sleep(0.07)
            buffer.add(message(bot, "И есть расходы на аренду", message_id=2))
            await buffer.wait_idle()
            await asyncio.sleep(0.15)
            brain.answer_with_history.assert_not_awaited()
            bot.session.assert_not_awaited()
            assert contents(buffer.store.history(key_for(first))) == [
                ("user", "У меня небольшая кофейня"), ("user", "И есть расходы на аренду"),
            ]
            assert not buffer.pending
        finally:
            await buffer.close()
    asyncio.run(run())


def test_continuation_cancels_outdated_classification():
    async def run():
        brain = FakeBrain(ready=True)
        first_started, first_cancelled = asyncio.Event(), asyncio.Event()

        async def classify(text, history):
            if text == "У меня кофейня":
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            return "QUESTION"

        brain.classify_turn.side_effect = classify
        buffer, bot = make_buffer(brain, delay=10, classification_timeout=5)
        try:
            buffer.add(message(bot, "У меня кофейня"))
            await asyncio.wait_for(first_started.wait(), timeout=1)
            buffer.add(message(bot, "Как рассчитать прибыль?", message_id=2))
            await asyncio.wait_for(buffer.wait_idle(), timeout=1)
            assert first_cancelled.is_set()
            brain.answer_with_history.assert_awaited_once_with("У меня кофейня\n\nКак рассчитать прибыль?", [])
            assert sent_texts(bot) == ["Тестовый ответ"]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_continuation_cancels_outdated_generation_before_telegram_send():
    async def run():
        brain = FakeBrain(ready=True)
        first_started, first_cancelled = asyncio.Event(), asyncio.Event()

        async def answer(text, history):
            if text == "Как рассчитать прибыль?":
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            return "Ответ с учётом доставки"

        brain.answer_with_history.side_effect = answer
        buffer, bot = make_buffer(brain, delay=10)
        try:
            buffer.add(message(bot, "Как рассчитать прибыль?"))
            await asyncio.wait_for(first_started.wait(), timeout=1)
            buffer.add(message(bot, "С учётом доставки", message_id=2))
            await asyncio.wait_for(buffer.wait_idle(), timeout=1)
            assert first_cancelled.is_set()
            assert brain.answer_with_history.call_args.args == ("Как рассчитать прибыль?\n\nС учётом доставки", [])
            assert sent_texts(bot) == ["Ответ с учётом доставки"]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_new_group_does_not_cancel_telegram_delivery_already_started():
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        send_started, allow_send = asyncio.Event(), asyncio.Event()

        async def send(*args, **kwargs):
            send_started.set()
            await allow_send.wait()

        bot.session.side_effect = send
        try:
            buffer.add(message(bot, "Как рассчитать прибыль?"))
            await asyncio.wait_for(send_started.wait(), timeout=1)
            buffer.add(message(bot, "Как рассчитать выручку?", message_id=2))
            await asyncio.sleep(0)
            assert brain.answer_with_history.await_count == 1
            allow_send.set()
            await buffer.wait_idle()
            assert brain.answer_with_history.await_count == 2
            assert bot.session.await_count == 2
            second_history = brain.answer_with_history.call_args_list[1].args[1]
            assert ("assistant", "Тестовый ответ") in contents(second_history)
        finally:
            allow_send.set()
            await buffer.close()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_classification_failure_remains_silent_and_keeps_the_message(failure):
    async def run():
        brain = FakeBrain()
        cancelled = asyncio.Event()

        async def classify(text, history):
            if failure == "error":
                raise OllamaError("unavailable")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        brain.classify_turn.side_effect = classify
        buffer, bot = make_buffer(brain, delay=0.08, classification_timeout=0.02)
        incoming = message(bot, "У меня вопрос про бизнес")
        try:
            buffer.add(incoming)
            await asyncio.wait_for(buffer.wait_idle(), timeout=1)
            await asyncio.sleep(0.1)
            brain.answer_with_history.assert_not_awaited()
            bot.session.assert_not_awaited()
            assert contents(buffer.store.history(key_for(incoming))) == [("user", incoming.text)]
            if failure == "timeout":
                assert cancelled.is_set()
        finally:
            await buffer.close()
    asyncio.run(run())


def test_generation_failure_sends_recoverable_message_and_next_group_works():
    async def run():
        brain = FakeBrain(ready=True)
        brain.answer_with_history.side_effect = [OllamaError("unavailable"), "Работа восстановлена"]
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "Как рассчитать прибыль?")
        try:
            buffer.add(incoming)
            await buffer.wait_idle()
            buffer.add(message(bot, "Как рассчитать выручку?", message_id=2))
            await buffer.wait_idle()
            assert sent_texts(bot) == [
                "Сейчас не удалось получить ответ от Ollama. Попробуйте ещё раз чуть позже.",
                "Работа восстановлена",
            ]
            assert [row["content"] for row in buffer.store.history(key_for(incoming)) if row["role"] == "assistant"] == ["Работа восстановлена"]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_photo_caption_question_and_album_are_not_sent_to_text_model():
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        try:
            buffer.add(message(bot, caption="Как рассчитать прибыль?", media_group_id="album", **PHOTO))
            buffer.add(message(bot, message_id=2, media_group_id="album", **PHOTO))
            await buffer.wait_idle()
            brain.classify_turn.assert_not_awaited()
            brain.answer_with_history.assert_not_awaited()
            assert bot.session.await_count == 1
            assert "альбом" in sent_texts(bot)[0]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_text_and_attachment_keep_one_unsupported_media_reply():
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        try:
            buffer.add(message(bot, "Посмотри мой отчёт"))
            buffer.add(message(bot, message_id=2, **PHOTO))
            await buffer.wait_idle()
            brain.answer_with_history.assert_not_awaited()
            assert bot.session.await_count == 1
            assert "фото" in sent_texts(bot)[0]
        finally:
            await buffer.close()
    asyncio.run(run())


@pytest.mark.parametrize("other", [
    {"chat_id": 43}, {"user_id": 43}, {"thread_id": 2},
    {"user_id": None, "sender_chat_id": -500},
])
def test_text_context_does_not_mix_chats_users_topics_or_sender_chats(other):
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        try:
            buffer.add(message(bot, "Первая независимая мысль", thread_id=1))
            buffer.add(message(bot, "Вторая независимая мысль", **{"thread_id": 1, **other}))
            await buffer.wait_idle()
            calls = brain.answer_with_history.call_args_list
            assert {call.args[0] for call in calls} == {"Первая независимая мысль", "Вторая независимая мысль"}
            assert all(call.args[1] == [] for call in calls)
            assert bot.session.await_count == 2
        finally:
            await buffer.close()
    asyncio.run(run())


def test_anonymous_channel_senders_have_separate_text_contexts():
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        try:
            for index, sender in enumerate((-500, -501), start=1):
                buffer.add(message(bot, f"Мысль канала {sender}", message_id=index,
                                   user_id=None, chat_id=-100, sender_chat_id=sender))
            await buffer.wait_idle()
            assert brain.answer_with_history.await_count == 2
            assert all(call.args[1] == [] for call in brain.answer_with_history.call_args_list)
            assert bot.session.await_count == 2
        finally:
            await buffer.close()
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["classification", "generation"])
def test_close_cancels_pending_model_work(phase):
    async def run():
        brain = FakeBrain(ready=True)
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def block(text, history):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        target = brain.classify_turn if phase == "classification" else brain.answer_with_history
        target.side_effect = block
        buffer, bot = make_buffer(brain, delay=10, classification_timeout=5)
        text = "У меня ещё не законченная мысль" if phase == "classification" else "Как рассчитать прибыль?"
        buffer.add(message(bot, text))
        await asyncio.wait_for(started.wait(), timeout=1)
        await buffer.close()
        assert cancelled.is_set()
        assert not buffer.pending
        assert not buffer.tasks
        bot.session.assert_not_awaited()
    asyncio.run(run())


def test_long_answer_is_split_within_telegram_utf16_limit():
    async def run():
        reply = "Прибыль 😀 " * 1000
        brain = FakeBrain(ready=True, reply=reply)
        buffer, bot = make_buffer(brain)
        try:
            incoming = message(bot, "Как рассчитать прибыль?")
            buffer.add(incoming)
            await buffer.wait_idle()
            chunks = sent_texts(bot)
            assert len(chunks) > 1
            assert "".join(chunks) == reply
            assert all(0 < len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in chunks)
            saved = [row["content"] for row in buffer.store.history(key_for(incoming)) if row["role"] == "assistant"]
            assert "".join(saved) == reply
        finally:
            await buffer.close()
    asyncio.run(run())


@pytest.mark.parametrize("overflow", ["characters", "messages"])
def test_oversized_group_is_bounded_and_does_not_call_model(overflow):
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        try:
            if overflow == "characters":
                for index in range(5):
                    buffer.add(message(bot, "x" * 4000, message_id=index + 1))
            else:
                for index in range(101):
                    buffer.add(message(bot, "часть", message_id=index + 1))
            await buffer.wait_idle()
            brain.classify_turn.assert_not_awaited()
            brain.answer_with_history.assert_not_awaited()
            assert bot.session.await_count == 1
            assert sent_texts(bot)[0]
            assert not buffer.pending
        finally:
            await buffer.close()
    asyncio.run(run())
