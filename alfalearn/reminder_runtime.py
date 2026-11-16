"""Durable Telegram reminders and an explicit confirmation dialogue."""

import asyncio
import json
import re
import secrets
import sqlite3
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

from alfa_ai import AIRequest
from alfa_ai.contracts import ReminderProposal
from alfa_ai.reminders import plan_reminder, unambiguous_time
from alfalearn.memory import ConversationKey, _encoded_key

UTC = timezone.utc
CONFIRM = {"да", "подтверждаю", "подтвердить"}
DECLINE = {"нет", "отмена", "отменить"}
TAX = re.compile(r"\b(?:налог|взнос|ндс|усн|патент)\w*\b", re.I)
TAX_REGIME = re.compile(r"\b(?:усн|осн|осно|патент|псн|нпд|самозанят\w*)\b", re.I)
NEW_TOPIC = re.compile(r"^(?:как|почему|зачем|сколько|что|чем|какой|какая|какие)\b", re.I)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Expected an aware datetime")
    return value.astimezone(UTC)


def _next_occurrence(local_at: str, zone_name: str, repeat: str, weekday: int | None,
                     after: datetime) -> tuple[str, datetime]:
    """Find the next valid wall-clock occurrence, skipping DST gaps/folds."""
    zone = ZoneInfo(zone_name)
    original = datetime.fromisoformat(local_at)
    clock = time(original.hour, original.minute)
    day = max(original.date(), _utc(after).astimezone(zone).date())
    for _ in range(370):
        if repeat != "weekly" or day.weekday() == weekday:
            local = datetime.combine(day, clock)
            if unambiguous_time(local, zone, _utc(after)):
                due = local.replace(tzinfo=zone).astimezone(UTC)
                return local.isoformat(timespec="minutes"), due
        day += timedelta(days=1)
    raise ValueError("No valid occurrence in the next year")


class ReminderStore:
    def __init__(self, path: str | Path = ":memory:"):
        database = str(path)
        if database != ":memory:":
            resolved = Path(path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            database = str(resolved)
        self._lock = RLock()
        self._connection = sqlite3.connect(database, timeout=5, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if database != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                owner_key TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                thread_id INTEGER,
                subject TEXT NOT NULL,
                timezone TEXT NOT NULL,
                local_at TEXT NOT NULL,
                repeat TEXT NOT NULL CHECK(repeat IN ('once','daily','weekly')),
                weekday INTEGER,
                next_due_utc TEXT NOT NULL,
                retry_at_utc TEXT,
                status TEXT NOT NULL CHECK(status IN ('active','sending','done','failed','cancelled')),
                attempts INTEGER NOT NULL DEFAULT 0,
                claim_token TEXT,
                lease_until_utc TEXT
            );
            CREATE INDEX IF NOT EXISTS reminders_due
                ON reminders(status, next_due_utc, retry_at_utc);
            CREATE TABLE IF NOT EXISTS reminder_drafts (
                owner_key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                proposal TEXT,
                phase TEXT NOT NULL CHECK(phase IN ('collect','confirm'))
            );
        """)

    def close(self):
        with self._lock:
            self._connection.close()

    def draft(self, key: ConversationKey) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT query, proposal, phase FROM reminder_drafts WHERE owner_key = ?",
                (_encoded_key(key),),
            ).fetchone()
        return dict(row) if row else None

    def save_draft(self, key: ConversationKey, query: str,
                   proposal: ReminderProposal | None, phase: str):
        with self._lock, self._connection:
            self._connection.execute("""
                INSERT INTO reminder_drafts(owner_key, query, proposal, phase)
                VALUES (?, ?, ?, ?) ON CONFLICT(owner_key) DO UPDATE SET
                    query=excluded.query, proposal=excluded.proposal, phase=excluded.phase
            """, (_encoded_key(key), query,
                  json.dumps(asdict(proposal), ensure_ascii=False) if proposal else None, phase))

    def clear_draft(self, key: ConversationKey):
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM reminder_drafts WHERE owner_key = ?",
                                     (_encoded_key(key),))

    def get(self, key: ConversationKey, reminder_id: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM reminders WHERE owner_key = ? AND id = ?",
                (_encoded_key(key), reminder_id),
            ).fetchone()
        return dict(row) if row else None

    def list_active(self, key: ConversationKey) -> list[dict]:
        with self._lock:
            rows = self._connection.execute("""
                SELECT * FROM reminders WHERE owner_key = ? AND status IN ('active','sending')
                ORDER BY next_due_utc LIMIT 20
            """, (_encoded_key(key),)).fetchall()
        return [dict(row) for row in rows]

    def list_visible(self, key: ConversationKey) -> list[dict]:
        with self._lock:
            rows = self._connection.execute("""
                SELECT * FROM reminders
                WHERE owner_key = ? AND status IN ('active','sending','failed')
                ORDER BY next_due_utc LIMIT 20
            """, (_encoded_key(key),)).fetchall()
        return [dict(row) for row in rows]

    def apply(self, key: ConversationKey, proposal: ReminderProposal, now: datetime) -> str:
        """Apply a confirmed operation, with owner checks in the write transaction."""
        owner = _encoded_key(key)
        now = _utc(now)
        with self._lock, self._connection:
            if proposal.action == "create_reminder":
                if not all((proposal.subject, proposal.local_datetime, proposal.timezone,
                            proposal.repeat)):
                    raise ValueError("Incomplete reminder")
                reminder_id = secrets.token_hex(4)
                local_at = proposal.local_datetime
                due = datetime.fromisoformat(local_at).replace(
                    tzinfo=ZoneInfo(proposal.timezone)).astimezone(UTC)
                if due <= now:
                    if proposal.repeat == "once":
                        raise ValueError("Reminder time has passed; specify a new date and time")
                    local_at, due = _next_occurrence(
                        local_at, proposal.timezone, proposal.repeat, proposal.weekday, now)
                self._connection.execute("""
                    INSERT INTO reminders(id, owner_key, chat_id, thread_id, subject, timezone,
                                          local_at, repeat, weekday, next_due_utc, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """, (reminder_id, owner, key[0], key[1], proposal.subject,
                      proposal.timezone, local_at, proposal.repeat, proposal.weekday,
                      due.isoformat()))
                return f"Напоминание #{reminder_id} сохранено: «{proposal.subject}», {local_at.replace('T', ' ')} ({proposal.timezone})."

            row = self._connection.execute(
                "SELECT * FROM reminders WHERE owner_key = ? AND id = ? AND status IN ('active','sending','failed')",
                (owner, proposal.reminder_id),
            ).fetchone()
            if row is None:
                raise LookupError("Напоминание не найдено или уже отключено.")
            if proposal.action == "cancel_reminder":
                self._connection.execute("""
                    UPDATE reminders SET status='cancelled', claim_token=NULL, lease_until_utc=NULL
                    WHERE id=? AND owner_key=?
                """, (proposal.reminder_id, owner))
                return f"Напоминание #{proposal.reminder_id} отключено."

            repeat = proposal.repeat or row["repeat"]
            zone_name = proposal.timezone or row["timezone"]
            local_at = proposal.local_datetime or row["local_at"]
            weekday = proposal.weekday if proposal.repeat == "weekly" else (
                None if proposal.repeat else row["weekday"])
            if repeat == "once" and not proposal.local_datetime:
                raise ValueError("Для разового напоминания укажите новую дату и время.")
            if repeat == "once":
                due = datetime.fromisoformat(local_at).replace(tzinfo=ZoneInfo(zone_name)).astimezone(UTC)
                if due <= now:
                    raise ValueError("Время уже прошло; укажите новую дату и время.")
            else:
                local_at, due = _next_occurrence(local_at, zone_name, repeat, weekday, now)
            self._connection.execute("""
                UPDATE reminders SET timezone=?, local_at=?, repeat=?, weekday=?, next_due_utc=?,
                    retry_at_utc=NULL, status='active', attempts=0, claim_token=NULL,
                    lease_until_utc=NULL WHERE id=? AND owner_key=?
            """, (zone_name, local_at, repeat, weekday, due.isoformat(),
                  proposal.reminder_id, owner))
            return f"Напоминание #{proposal.reminder_id} изменено. Следующий раз: {local_at.replace('T', ' ')} ({zone_name})."

    def claim_due(self, now: datetime) -> dict | None:
        now = _utc(now)
        token = secrets.token_hex(12)
        lease = (now + timedelta(seconds=90)).isoformat()
        with self._lock, self._connection:
            row = self._connection.execute("""
                UPDATE reminders SET status='sending', claim_token=?, lease_until_utc=?
                WHERE id = (
                    SELECT id FROM reminders
                    WHERE (status='active' OR (status='sending' AND lease_until_utc <= ?))
                      AND COALESCE(retry_at_utc, next_due_utc) <= ?
                    ORDER BY COALESCE(retry_at_utc, next_due_utc) LIMIT 1
                ) RETURNING *
            """, (token, lease, now.isoformat(), now.isoformat())).fetchone()
        return dict(row) if row is not None else None

    def delivered(self, reminder_id: str, token: str, now: datetime) -> bool:
        now = _utc(now)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM reminders WHERE id=? AND status='sending' AND claim_token=?",
                (reminder_id, token),
            ).fetchone()
            if row is None:
                return False
            if row["repeat"] == "once":
                self._connection.execute("""
                    UPDATE reminders SET status='done', claim_token=NULL, lease_until_utc=NULL,
                        attempts=0, retry_at_utc=NULL WHERE id=? AND claim_token=?
                """, (reminder_id, token))
            else:
                local_at, due = _next_occurrence(
                    row["local_at"], row["timezone"], row["repeat"], row["weekday"], now)
                self._connection.execute("""
                    UPDATE reminders SET status='active', local_at=?, next_due_utc=?,
                        claim_token=NULL, lease_until_utc=NULL, attempts=0, retry_at_utc=NULL
                    WHERE id=? AND claim_token=?
                """, (local_at, due.isoformat(), reminder_id, token))
            return True

    def failed(self, reminder_id: str, token: str, now: datetime) -> bool:
        now = _utc(now)
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT attempts FROM reminders WHERE id=? AND status='sending' AND claim_token=?",
                (reminder_id, token),
            ).fetchone()
            if row is None:
                return False
            attempts = row["attempts"] + 1
            status = "failed" if attempts >= 3 else "active"
            retry = (now + timedelta(minutes=2 ** (attempts - 1))).isoformat()
            self._connection.execute("""
                UPDATE reminders SET status=?, attempts=?, retry_at_utc=?, claim_token=NULL,
                    lease_until_utc=NULL WHERE id=? AND claim_token=?
            """, (status, attempts, retry, reminder_id, token))
            return True


class ReminderDialogue:
    def __init__(self, store: ReminderStore, clock=None):
        self.store = store
        self.clock = clock or (lambda: datetime.now(UTC))

    def handle(self, key: ConversationKey, text: str) -> str | None:
        now = _utc(self.clock())
        normalized = text.strip().casefold().rstrip(".! ")
        draft = self.store.draft(key)
        if draft and normalized in DECLINE:
            self.store.clear_draft(key)
            return "Действие отменено; напоминания не изменены."
        if draft and normalized in CONFIRM:
            if draft["phase"] != "confirm" or not draft["proposal"]:
                return "Сначала уточним недостающие параметры напоминания."
            proposal = ReminderProposal(**json.loads(draft["proposal"]))
            try:
                result = self.store.apply(key, proposal, now)
            except (ValueError, LookupError) as error:
                self.store.clear_draft(key)
                return str(error)
            self.store.clear_draft(key)
            return result
        if text.startswith("/"):
            return None
        if draft and (text.strip().endswith("?") or NEW_TOPIC.match(text.strip())):
            self.store.clear_draft(key)
            return None
        if draft and not re.match(r"^(?:напомни|напоминай|перенеси|измени|поменяй|отмени|удали|отключи)\b", text.strip(), re.I):
            query = f"{draft['query']} {text.strip()}"
        else:
            query = text.strip()
        if len(query) > 1000:
            self.store.clear_draft(key)
            return "Сократите запрос о напоминании до 1000 символов."
        planned = plan_reminder(AIRequest(query=query, now=now, timezone=None))
        if planned is None:
            return None
        proposal, answer = planned
        if proposal.action != "create_reminder" and proposal.reminder_id:
            item = self.store.get(key, proposal.reminder_id)
            if item is None or item["status"] not in ("active", "sending", "failed"):
                self.store.clear_draft(key)
                return "Напоминание не найдено или уже отключено. Посмотрите свои номера командой /reminders."
        if proposal.action == "create_reminder" and TAX.search(query) and not TAX_REGIME.search(query):
            self.store.save_draft(key, query, proposal, "collect")
            return "Уточните налог и режим налогообложения. Дату срока я не определяю за вас: укажите подтверждённую вами дату, время и часовой пояс."
        if proposal.missing_fields:
            self.store.save_draft(key, query, proposal, "collect")
            return answer
        self.store.save_draft(key, query, proposal, "confirm")
        return f"{answer} Ответьте «да» для сохранения или «нет» для отказа."


class ReminderWorker:
    def __init__(self, store: ReminderStore, send, clock=None, poll_seconds: float = 5):
        self.store = store
        self.send = send
        self.clock = clock or (lambda: datetime.now(UTC))
        self.poll_seconds = poll_seconds

    async def tick(self) -> bool:
        row = self.store.claim_due(self.clock())
        if row is None:
            return False
        try:
            await self.send(row)
        except Exception:
            self.store.failed(row["id"], row["claim_token"], self.clock())
        else:
            self.store.delivered(row["id"], row["claim_token"], self.clock())
        return True

    async def run(self):
        while True:
            if not await self.tick():
                await asyncio.sleep(self.poll_seconds)
