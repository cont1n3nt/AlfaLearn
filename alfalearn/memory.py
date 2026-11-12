"""Durable, isolated conversation history and rolling summaries in SQLite."""

import json
import sqlite3
from pathlib import Path
from threading import RLock

ConversationKey = tuple[int, int | None, int | None, int | None]


def _encoded_key(key: ConversationKey) -> str:
    if (
        not isinstance(key, tuple) or len(key) != 4
        or any(value is not None and type(value) is not int for value in key)
    ):
        raise ValueError("A conversation key must contain four integer or null IDs")
    return json.dumps(key, separators=(",", ":"))


class ConversationStore:
    """Keep original messages even after they have been incorporated in a summary.

    Each operation commits a short transaction. A lock protects the connection
    when application lifespan and tests use different threads; SQLite also
    serializes writes across separate store instances pointing to the same file.
    """

    def __init__(self, path: str | Path = ":memory:"):
        database = str(path)
        if database != ":memory:":
            database_path = Path(path).expanduser()
            database_path.parent.mkdir(parents=True, exist_ok=True)
            database = str(database_path)
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = sqlite3.connect(
            database, timeout=5, check_same_thread=False,
        )
        try:
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if database != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_key TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    telegram_message_id INTEGER
                );
                CREATE INDEX IF NOT EXISTS conversation_messages_by_key_id
                    ON conversation_messages(conversation_key, id);
                CREATE UNIQUE INDEX IF NOT EXISTS conversation_user_message_id
                    ON conversation_messages(conversation_key, telegram_message_id)
                    WHERE role = 'user' AND telegram_message_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    through_id INTEGER NOT NULL CHECK(through_id >= 0)
                );
            """)
        except BaseException:
            self._connection.close()
            self._connection = None
            raise

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Conversation store is closed")
        return self._connection

    def append_user(
        self, key: ConversationKey, text: str, message_id: int | None = None,
    ) -> int:
        encoded = _encoded_key(key)
        if not isinstance(text, str):
            raise TypeError("Conversation text must be a string")
        if message_id is not None and type(message_id) is not int:
            raise ValueError("Message ID must be an integer or null")
        with self._lock:
            connection = self._db()
            with connection:
                cursor = connection.execute(
                    """INSERT INTO conversation_messages
                           (conversation_key, role, content, telegram_message_id)
                       VALUES (?, 'user', ?, ?) ON CONFLICT DO NOTHING""",
                    (encoded, text, message_id),
                )
                if cursor.rowcount:
                    return int(cursor.lastrowid)
                row = connection.execute(
                    """SELECT id FROM conversation_messages
                       WHERE conversation_key = ? AND role = 'user'
                           AND telegram_message_id = ?""",
                    (encoded, message_id),
                ).fetchone()
                return int(row[0])

    def append_assistant(self, key: ConversationKey, text: str) -> int:
        encoded = _encoded_key(key)
        if not isinstance(text, str):
            raise TypeError("Conversation text must be a string")
        with self._lock:
            connection = self._db()
            with connection:
                cursor = connection.execute(
                    """INSERT INTO conversation_messages
                           (conversation_key, role, content)
                       VALUES (?, 'assistant', ?)""",
                    (encoded, text),
                )
                return int(cursor.lastrowid)

    def history(self, key: ConversationKey, after_id: int = 0) -> list[dict]:
        encoded = _encoded_key(key)
        if type(after_id) is not int or after_id < 0:
            raise ValueError("History cursor must be a nonnegative integer")
        with self._lock:
            rows = self._db().execute(
                """SELECT id, role, content FROM conversation_messages
                   WHERE conversation_key = ? AND id > ? ORDER BY id""",
                (encoded, after_id),
            ).fetchall()
        return [{"id": row[0], "role": row[1], "content": row[2]} for row in rows]

    def summary(self, key: ConversationKey) -> tuple[str, int]:
        encoded = _encoded_key(key)
        with self._lock:
            row = self._db().execute(
                "SELECT content, through_id FROM conversation_summaries WHERE conversation_key = ?",
                (encoded,),
            ).fetchone()
        return (row[0], row[1]) if row is not None else ("", 0)

    def last_assistant(self, key: ConversationKey) -> dict | None:
        with self._lock:
            row = self._db().execute(
                """SELECT id, role, content FROM conversation_messages
                   WHERE conversation_key = ? AND role = 'assistant' ORDER BY id DESC LIMIT 1""",
                (_encoded_key(key),),
            ).fetchone()
        return {"id": row[0], "role": row[1], "content": row[2]} if row else None

    def set_summary(self, key: ConversationKey, text: str, through_id: int) -> None:
        encoded = _encoded_key(key)
        if not isinstance(text, str):
            raise TypeError("Conversation summary must be a string")
        if type(through_id) is not int or through_id < 0:
            raise ValueError("Summary cursor must be a nonnegative integer")
        with self._lock:
            connection = self._db()
            with connection:
                connection.execute(
                    """INSERT INTO conversation_summaries (conversation_key, content, through_id)
                       VALUES (?, ?, ?)
                       ON CONFLICT(conversation_key) DO UPDATE SET
                           content = excluded.content, through_id = excluded.through_id
                       WHERE excluded.through_id >= conversation_summaries.through_id""",
                    (encoded, text, through_id),
                )

    def clear(self, key: ConversationKey) -> None:
        encoded = _encoded_key(key)
        with self._lock:
            connection = self._db()
            with connection:
                connection.execute(
                    "DELETE FROM conversation_messages WHERE conversation_key = ?", (encoded,),
                )
                connection.execute(
                    "DELETE FROM conversation_summaries WHERE conversation_key = ?", (encoded,),
                )

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
