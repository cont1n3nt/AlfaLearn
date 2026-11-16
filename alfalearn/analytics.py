"""Local product events without message text or Telegram identifiers."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock

EVENT_TYPES = {"answer_shown", "material_opened", "feedback_positive", "feedback_negative", "reminder_created", "reminder_delivered", "reminder_cancelled"}


class EventStore:
    def __init__(self, path: str | Path = ":memory:", retention_days: int = 90):
        if type(retention_days) is not int or not 1 <= retention_days <= 3650:
            raise ValueError("Срок хранения событий должен быть от 1 до 3650 дней.")
        database = str(path)
        if database != ":memory:":
            resolved = Path(path).expanduser(); resolved.parent.mkdir(parents=True, exist_ok=True); database = str(resolved)
        self.retention_days, self._lock = retention_days, RLock()
        self._connection = sqlite3.connect(database, timeout=5, check_same_thread=False)
        self._connection.execute("CREATE TABLE IF NOT EXISTS product_events (event_key TEXT PRIMARY KEY, event_type TEXT NOT NULL, created_at_utc TEXT NOT NULL)")

    def record_once(self, event_key: str, event_type: str, *, now: datetime | None = None) -> bool:
        if not isinstance(event_key, str) or not 1 <= len(event_key) <= 100 or event_type not in EVENT_TYPES:
            raise ValueError("Некорректное событие.")
        now = now or datetime.now(UTC)
        if now.tzinfo is None: raise ValueError("Время события должно содержать часовой пояс.")
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM product_events WHERE created_at_utc < ?", ((now - timedelta(days=self.retention_days)).astimezone(UTC).isoformat(),))
            cursor = self._connection.execute("INSERT INTO product_events VALUES (?, ?, ?) ON CONFLICT DO NOTHING", (event_key, event_type, now.astimezone(UTC).isoformat()))
            return bool(cursor.rowcount)

    def counts(self) -> dict[str, int]:
        with self._lock: rows = self._connection.execute("SELECT event_type, COUNT(*) FROM product_events GROUP BY event_type").fetchall()
        return {kind: int(count) for kind, count in rows}

    def summary(self) -> dict[str, object]:
        """Return aggregates only; suitable for local developer inspection."""
        counts = self.counts()
        return {
            "total": sum(counts.values()),
            "by_type": {kind: counts.get(kind, 0) for kind in sorted(EVENT_TYPES)},
            "retention_days": self.retention_days,
        }

    def close(self) -> None:
        with self._lock: self._connection.close()
