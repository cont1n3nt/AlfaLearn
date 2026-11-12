"""Persistent dialogue, explicit turn readiness, reset and bounded context checks."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Update

from alfalearn.bot import create_dispatcher
from alfalearn.conversation import DialogueContext
from alfalearn.memory import ConversationStore
from test_dynamic_grouping import FakeBrain, contents, key_for, make_buffer, message, sent_texts


def test_waited_fragments_are_remembered_when_a_later_question_arrives():
    async def run():
        brain = FakeBrain()
        buffer, bot = make_buffer(brain, delay=0.02)
        first = message(bot, "Моя кофейня называется Север")
        try:
            buffer.add(first)
            await buffer.wait_idle()
            await asyncio.sleep(0.05)
            assert sent_texts(bot) == []
            buffer.add(message(bot, "Аренда стоит 50000 рублей", message_id=2))
            await buffer.wait_idle()
            assert sent_texts(bot) == []
            question = "Как рассчитать прибыль?"
            buffer.add(message(bot, question, message_id=3))
            await buffer.wait_idle()
            text, history = brain.answer_with_history.call_args.args
            assert text == question
            assert contents(history) == [
                ("user", "Моя кофейня называется Север"),
                ("user", "Аренда стоит 50000 рублей"),
            ]
            assert question not in [row["content"] for row in history]
            assert contents(buffer.store.history(key_for(first))) == [
                ("user", "Моя кофейня называется Север"),
                ("user", "Аренда стоит 50000 рублей"),
                ("user", question), ("assistant", "Тестовый ответ"),
            ]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_answer_to_assistant_question_uses_previous_conversation():
    async def run():
        brain = FakeBrain()
        brain.classify_turn.return_value = "ANSWER"
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "70000 рублей", message_id=2)
        key = key_for(incoming)
        try:
            buffer.store.append_user(key, "Рассчитай мою прибыль", message_id=1)
            buffer.store.append_assistant(key, "Какие у вас расходы?")
            buffer.add(incoming)
            await buffer.wait_idle()
            assert brain.classify_turn.call_args.args[0] == "70000 рублей"
            history = brain.classify_turn.call_args.args[1]
            assert contents(history)[-1] == ("assistant", "Какие у вас расходы?")
            brain.answer_with_history.assert_awaited_once()
            assert brain.answer_with_history.call_args.args[1] == history
            assert sent_texts(bot) == ["Тестовый ответ"]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_an_answer_decision_without_a_previous_assistant_turn_is_not_enough():
    async def run():
        brain = FakeBrain()
        brain.classify_turn.return_value = "ANSWER"
        buffer, bot = make_buffer(brain)
        try:
            buffer.add(message(bot, "да"))
            await buffer.wait_idle()
            brain.answer_with_history.assert_not_awaited()
            bot.session.assert_not_awaited()
        finally:
            await buffer.close()
    asyncio.run(run())


def test_a_short_yes_is_silent_when_the_classifier_does_not_find_a_question():
    async def run():
        brain = FakeBrain()
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "да")
        try:
            buffer.store.append_assistant(key_for(incoming), "Прибыль — доходы минус расходы.")
            buffer.add(incoming)
            await buffer.wait_idle()
            assert contents(brain.classify_turn.call_args.args[1]) == [
                ("assistant", "Прибыль — доходы минус расходы."),
            ]
            brain.answer_with_history.assert_not_awaited()
            bot.session.assert_not_awaited()
        finally:
            await buffer.close()
    asyncio.run(run())


@pytest.mark.parametrize("assistant,allowed", [
    ("Прибыль составляет 30000 рублей.", False),
    ("Какая у вас выручка?", True),
    ("Укажите доходы и расходы.", True),
    ("Прибыль известна. Пожалуйста, напишите сумму расходов.", True),
    ("Например, 'Какая выручка?' — это вопрос. Прибыль считается иначе.", False),
])
def test_answer_decision_requires_an_actual_last_request_for_input(assistant, allowed):
    async def run():
        brain = FakeBrain()
        brain.classify_turn.return_value = "ANSWER"
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "100000")
        try:
            buffer.store.append_assistant(key_for(incoming), assistant)
            buffer.add(incoming)
            await buffer.wait_idle()
            assert bool(brain.answer_with_history.await_count) is allowed
            assert bool(bot.session.await_count) is allowed
        finally:
            await buffer.close()
    asyncio.run(run())


def test_history_survives_closing_and_reopening_the_application_store(tmp_path):
    async def run():
        database = tmp_path / "dialogue.sqlite3"
        store = ConversationStore(database)
        brain = FakeBrain()
        first_buffer, bot = make_buffer(brain, memory=store)
        first = message(bot, "Мой магазин называется Север")
        first_buffer.add(first)
        await first_buffer.wait_idle()
        await first_buffer.close()
        store.close()
        reopened = ConversationStore(database)
        new_brain = FakeBrain(ready=True)
        buffer, new_bot = make_buffer(new_brain, memory=reopened)
        try:
            buffer.add(message(new_bot, "Как называется мой магазин?", message_id=2))
            await buffer.wait_idle()
            assert contents(new_brain.answer_with_history.call_args.args[1]) == [
                ("user", "Мой магазин называется Север"),
            ]
        finally:
            await buffer.close()
            reopened.close()
    asyncio.run(run())


def test_current_group_fragments_appear_only_once_in_model_input():
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        first = message(bot, "Выручка 100000")
        try:
            buffer.store.append_user(key_for(first), "Раньше я писал об аренде", message_id=99)
            buffer.add(first)
            buffer.add(message(bot, "Расходы 70000. Какова прибыль?", message_id=2))
            await buffer.wait_idle()
            text, history = brain.answer_with_history.call_args.args
            assert text == "Выручка 100000\n\nРасходы 70000. Какова прибыль?"
            assert contents(history) == [("user", "Раньше я писал об аренде")]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_reset_clears_only_its_conversation_and_cancels_late_generation():
    async def run():
        brain = FakeBrain(ready=True)
        started = asyncio.Event()

        async def resistant_answer(text, history):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "Устаревший ответ"

        brain.answer_with_history.side_effect = resistant_answer
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "Как рассчитать прибыль?")
        other = message(bot, "Чужой контекст", chat_id=43)
        try:
            buffer.store.append_user(key_for(other), other.text, message_id=1)
            buffer.add(incoming)
            await asyncio.wait_for(started.wait(), timeout=1)
            buffer.reset(incoming)
            await buffer.wait_idle()
            assert buffer.store.history(key_for(incoming)) == []
            assert contents(buffer.store.history(key_for(other))) == [("user", "Чужой контекст")]
            assert sent_texts(bot) == []
            assert not buffer.pending
        finally:
            await buffer.close()
    asyncio.run(run())


def test_reset_telegram_command_clears_memory_and_sends_confirmation():
    async def run():
        store = ConversationStore()
        brain = FakeBrain()
        buffer, bot = make_buffer(brain, memory=store)
        reset_message = message(bot, "/reset", message_id=2)
        store.append_user(key_for(reset_message), "Старый контекст", message_id=1)
        dispatcher = create_dispatcher(group_delay_seconds=0.01, llm=brain, memory=store)
        try:
            await dispatcher.feed_update(bot, Update(update_id=2, message=reset_message))
            await dispatcher["answer_buffer"].wait_idle()
            assert store.history(key_for(reset_message)) == []
            assert len(sent_texts(bot)) == 1
            assert sent_texts(bot)[0]
            brain.answer_with_history.assert_not_awaited()
            brain.classify_turn.assert_not_awaited()
        finally:
            await dispatcher["answer_buffer"].close()
            await buffer.close()
            store.close()
    asyncio.run(run())


def test_failed_telegram_delivery_is_not_recorded_as_an_assistant_turn():
    async def run():
        brain = FakeBrain(ready=True)
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "Как рассчитать прибыль?")
        bot.session.side_effect = OSError("delivery failed")
        try:
            buffer.add(incoming)
            with pytest.raises(OSError, match="delivery failed"):
                await buffer.wait_idle()
            assert contents(buffer.store.history(key_for(incoming))) == [("user", incoming.text)]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_only_successfully_delivered_chunks_enter_assistant_history():
    async def run():
        reply = "а" * 5000
        brain = FakeBrain(ready=True, reply=reply)
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "Как рассчитать прибыль?")
        bot.session.side_effect = [None, OSError("second chunk failed")]
        try:
            buffer.add(incoming)
            with pytest.raises(OSError, match="second chunk failed"):
                await buffer.wait_idle()
            delivered = [row["content"] for row in buffer.store.history(key_for(incoming)) if row["role"] == "assistant"]
            assert delivered == ["а" * 4096]
        finally:
            await buffer.close()
    asyncio.run(run())


def test_summary_compacts_prompt_without_deleting_original_dialogue():
    async def run():
        store = ConversationStore()
        key = (42, None, 42, None)
        brain = FakeBrain()
        brain.summarize.return_value = "Пользователь владеет кофейней Север."
        try:
            for index in range(8):
                store.append_user(key, f"Факт {index}: " + "кофейня " * 40, message_id=index + 1)
                store.append_assistant(key, f"Уточнение {index}: " + "обсудили " * 30)
            current_text = "Как называется моя кофейня?"
            current_id = store.append_user(key, current_text, message_id=99)
            raw_before = store.history(key)
            context = DialogueContext(store, brain, max_chars=1200)
            history = await context.prepare(key, {current_id}, current_text, lambda: True)
            brain.summarize.assert_awaited()
            assert store.history(key) == raw_before
            summary, cursor = store.summary(key)
            assert summary == "Пользователь владеет кофейней Север."
            assert cursor > 0
            assert any(summary in row["content"] for row in history)
            assert all(row["content"] != current_text for row in history)
            summarized_rows = brain.summarize.call_args.args[0]
            assert all(row["content"] != current_text for row in summarized_rows)
        finally:
            store.close()
    asyncio.run(run())


def test_obsolete_summary_does_not_repopulate_cleared_memory():
    async def run():
        store = ConversationStore()
        key = (42, None, 42, None)
        brain = FakeBrain()
        started, release = asyncio.Event(), asyncio.Event()
        valid = True

        async def summarize(history, previous_summary=""):
            started.set()
            await release.wait()
            return "Устаревшая сводка"

        brain.summarize.side_effect = summarize
        try:
            for index in range(8):
                store.append_user(key, "Старый факт " * 70, message_id=index + 1)
            current_text = "Что теперь делать?"
            current_id = store.append_user(key, current_text, message_id=99)
            context = DialogueContext(store, brain, max_chars=1200)
            preparing = asyncio.create_task(context.prepare(key, {current_id}, current_text, lambda: valid))
            await asyncio.wait_for(started.wait(), timeout=1)
            valid = False
            store.clear(key)
            release.set()
            try:
                await preparing
            except asyncio.CancelledError:
                pass
            assert store.history(key) == []
            assert store.summary(key) == ("", 0)
        finally:
            release.set()
            store.close()
    asyncio.run(run())


def test_reset_followed_immediately_by_a_question_does_not_create_a_mixed_stub():
    async def run():
        store = ConversationStore()
        brain = FakeBrain(ready=True, reply="Новый ответ после сброса")
        unused_buffer, bot = make_buffer(brain, memory=store)
        dispatcher = create_dispatcher(group_delay_seconds=0.05, llm=brain, memory=store)
        buffer = dispatcher["answer_buffer"]
        reset = message(bot, "/reset", message_id=2)
        store.append_user(key_for(reset), "Старый магазин", message_id=1)
        question = message(bot, "Как рассчитать прибыль?", message_id=3)
        try:
            await dispatcher.feed_update(bot, Update(update_id=2, message=reset))
            await dispatcher.feed_update(bot, Update(update_id=3, message=question))
            await asyncio.wait_for(buffer.wait_idle(), timeout=1)
            brain.answer_with_history.assert_awaited_once_with(question.text, [])
            assert len(sent_texts(bot)) == 2
            assert sent_texts(bot)[-1] == "Новый ответ после сброса"
            assert all("группу сообщений" not in text for text in sent_texts(bot))
            assert contents(store.history(key_for(reset))) == [
                ("user", question.text), ("assistant", "Новый ответ после сброса"),
            ]
        finally:
            await buffer.close()
            await unused_buffer.close()
            store.close()
    asyncio.run(run())


def test_long_wait_backlog_can_be_compacted_without_losing_the_actual_assistant_question():
    async def run():
        store = ConversationStore()
        key = (42, None, 42, None)
        brain = FakeBrain()
        brain.summarize.return_value = "Обсуждаем кофейню Север; бот спросил месячную выручку."
        last_question = "Какова месячная выручка вашей кофейни?"
        try:
            store.append_user(key, "Рассчитай прибыль кофейни Север", message_id=1)
            question_id = store.append_assistant(key, last_question)
            for index in range(20):
                store.append_user(key, f"Вводная {index}. " + "Факты о кофейне и расходах. " * 40,
                                  message_id=index + 2)
            current_text = "100000 рублей"
            current_id = store.append_user(key, current_text, message_id=99)
            original = store.history(key)
            context = DialogueContext(store, brain, max_chars=12000)
            for _ in range(2):
                history = await context.prepare(key, {current_id}, current_text, lambda: True)
                assert [row["content"] for row in history if row["role"] == "assistant"] == [last_question]
                assert sum(len(row["content"]) for row in history) <= 12000
                assert all(row["content"] != current_text for row in history)
            brain.summarize.assert_awaited()
            assert store.summary(key)[1] >= question_id
            assert store.history(key) == original
        finally:
            store.close()
    asyncio.run(run())


def test_reset_during_a_cancellation_resistant_send_stops_remaining_chunks():
    async def run():
        brain = FakeBrain(ready=True, reply="Ответ " * 2000)
        buffer, bot = make_buffer(brain)
        incoming = message(bot, "Как рассчитать прибыль?")
        send_started = asyncio.Event()
        cancellation_seen = asyncio.Event()

        async def resistant_send(*args, **kwargs):
            send_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                return None

        bot.session.side_effect = resistant_send
        try:
            buffer.add(incoming)
            await asyncio.wait_for(send_started.wait(), timeout=1)
            buffer.reset(incoming)
            await asyncio.wait_for(buffer.wait_idle(), timeout=1)
            assert cancellation_seen.is_set()
            assert bot.session.await_count == 1
            assert buffer.store.history(key_for(incoming)) == []
        finally:
            await buffer.close()
    asyncio.run(run())
