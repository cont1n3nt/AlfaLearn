"""Conversation isolation and durable memory, without external services."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from alfalearn.memory import ConversationStore

KEY = (42, None, 42, None)


def test_history_and_summary_survive_reopening_nested_database(tmp_path):
    database = tmp_path / "nested" / "conversations.sqlite3"
    store = ConversationStore(database)
    user_id = store.append_user(KEY, "У меня магазин 🛍️\nВыручка 100 000 ₽", message_id=11)
    assistant_id = store.append_assistant(KEY, "Какие расходы?")
    store.set_summary(KEY, "Магазин, выручка 100 000 ₽", user_id)
    store.close()

    reopened = ConversationStore(database)
    try:
        assert reopened.history(KEY) == [
            {"id": user_id, "role": "user", "content": "У меня магазин 🛍️\nВыручка 100 000 ₽"},
            {"id": assistant_id, "role": "assistant", "content": "Какие расходы?"},
        ]
        assert reopened.summary(KEY) == ("Магазин, выручка 100 000 ₽", user_id)
    finally:
        reopened.close()


@pytest.mark.parametrize("other", [
    (43, None, 42, None), (42, 1, 42, None), (42, None, 43, None),
    (42, None, 42, 1), (42, 0, 42, None), (42, None, 42, 0),
])
def test_conversation_ids_do_not_share_history_or_summary(other):
    store = ConversationStore()
    try:
        first = store.append_user(KEY, "Первый диалог", message_id=1)
        second = store.append_user(other, "Другой диалог", message_id=1)
        store.set_summary(KEY, "Первый контекст", first)
        store.set_summary(other, "Другой контекст", second)
        assert [row["content"] for row in store.history(KEY)] == ["Первый диалог"]
        assert [row["content"] for row in store.history(other)] == ["Другой диалог"]
        assert store.summary(KEY) == ("Первый контекст", first)
        assert store.summary(other) == ("Другой контекст", second)
    finally:
        store.close()


def test_chronology_cursor_and_duplicate_telegram_delivery():
    store = ConversationStore()
    try:
        first = store.append_user(KEY, "Первая часть", message_id=0)
        store.append_user((99, None, 99, None), "Другой пользователь", message_id=0)
        second = store.append_user(KEY, "Вторая часть", message_id=1)
        third = store.append_assistant(KEY, "Ответ")
        assert store.append_user(KEY, "Изменённый повтор", message_id=0) == first
        assert store.history(KEY) == [
            {"id": first, "role": "user", "content": "Первая часть"},
            {"id": second, "role": "user", "content": "Вторая часть"},
            {"id": third, "role": "assistant", "content": "Ответ"},
        ]
        assert store.history(KEY, after_id=second) == [
            {"id": third, "role": "assistant", "content": "Ответ"},
        ]
        assert store.history(KEY, after_id=third) == []
    finally:
        store.close()


def test_messages_without_telegram_ids_and_assistant_replies_are_not_deduplicated():
    store = ConversationStore()
    try:
        ids = [store.append_user(KEY, "Да"), store.append_user(KEY, "Да")]
        ids += [store.append_assistant(KEY, "Понял"), store.append_assistant(KEY, "Понял")]
        assert len(set(ids)) == 4
        assert [row["content"] for row in store.history(KEY)] == ["Да", "Да", "Понял", "Понял"]
    finally:
        store.close()


def test_summary_updates_never_regress_or_delete_original_messages():
    store = ConversationStore()
    try:
        first = store.append_user(KEY, "Выручка 100 000", message_id=1)
        second = store.append_user(KEY, "Расходы 70 000", message_id=2)
        original = store.history(KEY)
        assert store.summary(KEY) == ("", 0)
        store.set_summary(KEY, "Полный контекст", second)
        store.set_summary(KEY, "Устаревший контекст", first)
        assert store.summary(KEY) == ("Полный контекст", second)
        store.set_summary(KEY, "Уточнённый контекст", second)
        assert store.summary(KEY) == ("Уточнённый контекст", second)
        assert store.history(KEY) == original
    finally:
        store.close()


def test_clear_forgets_only_one_conversation_and_allows_reusing_message_id():
    other = (42, 7, 42, None)
    store = ConversationStore()
    try:
        first = store.append_user(KEY, "Удалить", message_id=1)
        second = store.append_user(other, "Сохранить", message_id=1)
        store.set_summary(KEY, "Удалить", first)
        store.set_summary(other, "Сохранить", second)
        store.clear(KEY)
        assert store.history(KEY) == []
        assert store.summary(KEY) == ("", 0)
        assert store.history(other) == [{"id": second, "role": "user", "content": "Сохранить"}]
        assert store.summary(other) == ("Сохранить", second)
        assert store.append_user(KEY, "Новый диалог", message_id=1) > first
    finally:
        store.close()


def test_shared_connection_supports_concurrent_duplicate_deliveries():
    store = ConversationStore()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            ids = list(executor.map(lambda _: store.append_user(KEY, "Сообщение", message_id=77), range(20)))
        assert len(set(ids)) == 1
        assert store.history(KEY) == [{"id": ids[0], "role": "user", "content": "Сообщение"}]
    finally:
        store.close()


def test_close_is_idempotent_and_closed_store_cannot_be_reused():
    store = ConversationStore()
    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.history(KEY)
    with pytest.raises(RuntimeError, match="closed"):
        store.append_user(KEY, "Сообщение")
