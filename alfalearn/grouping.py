"""Debounce replies independently for each chat, sender and topic."""

import asyncio
import logging
from dataclasses import dataclass, field

from aiogram.types import Message

from alfalearn.answers import MessageKind, build_answer, classify_message

logger = logging.getLogger(__name__)
GroupKey = tuple[int, int | None, int | None, int | None]


@dataclass
class PendingGroup:
    message: Message
    last_received: float
    kinds: set[MessageKind] = field(default_factory=set)
    count: int = 0
    command_answer: str | None = None


class AnswerBuffer:
    def __init__(self, delay_seconds: float = 2.0):
        if delay_seconds <= 0:
            raise ValueError("Grouping delay must be positive")
        self.delay_seconds = delay_seconds
        self.pending: dict[GroupKey, PendingGroup] = {}
        self.tasks: set[asyncio.Task] = set()
        self.closed = False

    def add(self, message: Message, command_answer: str | None = None) -> None:
        if self.closed:
            return
        key = (
            message.chat.id, message.message_thread_id,
            message.from_user.id if message.from_user else None,
            message.sender_chat.id if message.sender_chat else None,
        )
        now = asyncio.get_running_loop().time()
        group = self.pending.get(key)
        if group is None:
            group = PendingGroup(message=message, last_received=now)
            self.pending[key] = group
            task = asyncio.create_task(self._reply_after_pause(key), name="telegram-answer-group")
            self.tasks.add(task)
            task.add_done_callback(self._task_done)
        group.message = message
        group.last_received = now
        group.kinds.update(classify_message(message, bot_id=message.bot.id))
        group.count += 1
        group.command_answer = command_answer

    async def _reply_after_pause(self, key: GroupKey) -> None:
        while True:
            group = self.pending[key]
            remaining = group.last_received + self.delay_seconds - asyncio.get_running_loop().time()
            if remaining <= 0:
                # Remove before sending: incoming messages start a fresh group and
                # cannot cancel an in-flight Telegram send.
                self.pending.pop(key)
                answer = (
                    group.command_answer if group.count == 1 and group.command_answer
                    else build_answer(group.kinds, group.count)
                )
                await group.message.answer(answer)
                return
            await asyncio.sleep(remaining)

    def _task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # Do not expose exception text or user content in logs.
            logger.error("Grouped Telegram reply failed (%s)", type(task.exception()).__name__)

    async def wait_idle(self) -> None:
        """Wait for current groups, including delivery errors (useful in tests)."""
        while self.tasks:
            await asyncio.gather(*tuple(self.tasks))

    async def close(self) -> None:
        self.closed = True
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
