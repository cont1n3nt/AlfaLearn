"""Reply to complete questions promptly; debounce fragments per chat/sender/topic."""

import asyncio
import logging
import re
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from alfa_ai import AIRequest, AIService, LearningProfile
from alfalearn.answers import MessageKind, build_answer, classify_message
from alfalearn.analytics import EventStore
from alfalearn.calculator_cards import CARDS_BY_SLUG
from alfalearn.calculator_menu import TAX_CALCULATOR_QUERY, card_keyboard
from alfalearn.conversation import DialogueContext
from alfalearn.knowledge import (
    TOO_LONG,
    format_answer,
    format_telegram_answer,
    format_thinking,
    without_custom_emoji,
)
from alfalearn.llm import OllamaClient, OllamaError
from alfalearn.memory import ConversationStore
from alfalearn.reminder_runtime import ReminderDialogue

logger = logging.getLogger(__name__)
GroupKey = tuple[int, int | None, int | None, int | None]
OLLAMA_UNAVAILABLE = "Сейчас не удалось получить ответ от Ollama. Попробуйте ещё раз чуть позже."
GROUP_TOO_LONG = "Сообщение слишком длинное. Сократите вопрос до 16 000 символов и 100 сообщений."
ALBUM_GROUP_DELAY_SECONDS = 0.25
TEXT_KINDS = {
    MessageKind.TEXT, MessageKind.LINK, MessageKind.REPLY,
    MessageKind.QUOTE, MessageKind.FORWARDED, MessageKind.EMOJI,
}


def is_short_clarification(text: str) -> bool:
    """Only a short answer to an open question may extend its stored topic."""
    value = text.strip()
    if len(value) > 120 or "?" in value or "\n" in value:
        return False
    words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
    if not words or len(words) > 7:
        return False
    return not bool(re.match(
        r"^(?:как|почему|зачем|сколько|что|чем|какой|какая|какие|покажи|найди|расскажи)\b",
        value, re.I,
    ))


def is_topic_follow_up(text: str) -> bool:
    """Recognize an explicit reference to the last completed topic."""
    value = text.strip()
    return len(value) <= 120 and bool(re.fullmatch(
        r"(?:а\s+)?(?:как|что|почему|зачем|чем)\s+(?:это|такое|так)"
        r"(?:\s+[^?!.\n]+)?[?!.]*",
        value,
        re.I,
    ))


def is_clear_question(text: str) -> bool:
    """Conservative fast path; ambiguous questions are left to the model."""
    text = text.strip().lower()
    if text.count("?") != 1 or not text.endswith("?") or "\n" in text:
        return False
    if any(mark in text for mark in ('"', '«', '»', '…', '...')):
        return False
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    if len(words) < 3 or words[0] not in {"как", "почему", "зачем", "сколько", "что", "чем", "какой", "какая", "какие"}:
        return False
    if set(words) & {
        "если", "это", "этот", "эта", "эти", "тогда", "там", "он", "она", "они",
        "его", "её", "ее", "их", "я", "ты", "вы", "мы", "так", "же",
    }:
        return False
    return words[-1] not in {"и", "или", "но", "чтобы", "когда"}


def telegram_chunks(text: str):
    """Telegram counts UTF-16 units, including two units for astral characters."""
    chunk: list[str] = []
    size = 0
    for char in text:
        units = 2 if ord(char) > 0xFFFF else 1
        if size + units > 4096:
            yield "".join(chunk)
            chunk, size = [], 0
        chunk.append(char)
        size += units
    if chunk:
        yield "".join(chunk)


def assistant_requested_answer(history: list[dict]) -> bool:
    """Do not let a model mislabel acknowledgements as answers to no question."""
    last = next((row["content"] for row in reversed(history) if row["role"] == "assistant"), "")
    if not last:
        return False
    if re.search(r'\?[\s*_`"»”)\].!]*$', last):
        return True
    # Models sometimes phrase a clarification as "Укажите доходы и расходы."
    # Accept an actual request for input at the end, not an old example question.
    ending = re.split(r"[.!?\n]+", last.rstrip(".!?\n "))[-1].strip(" *_`«»\"").lower()
    return bool(re.match(
        r"(?:пожалуйста[, ]+)?(?:укажите|напишите|назовите|сообщите|уточните|выберите|скажите|подскажите|введите)\b",
        ending,
    ))


@dataclass
class PendingGroup:
    message: Message
    last_received: float
    kinds: set[MessageKind] = field(default_factory=set)
    count: int = 0
    command_answer: str | None = None
    texts: list[str] = field(default_factory=list)
    text_length: int = 0
    has_command: bool = False
    overflow: bool = False
    task: asyncio.Task | None = None
    history_ids: set[int] = field(default_factory=set)
    epoch: int = 0


class AnswerBuffer:
    def __init__(
        self, delay_seconds: float = 1.0, *, llm: OllamaClient | None = None,
        classification_timeout_seconds: float = 0.8,
        memory: ConversationStore | None = None, history_context_chars: int = 12_000,
        ai: AIService | None = None, reminders: ReminderDialogue | None = None,
        analytics: EventStore | None = None,
    ):
        if delay_seconds <= 0:
            raise ValueError("Grouping delay must be positive")
        if classification_timeout_seconds <= 0:
            raise ValueError("Classification timeout must be positive")
        self.delay_seconds = delay_seconds
        self.llm = llm
        self.ai = ai
        self.reminders = reminders
        self.analytics = analytics
        self.classification_timeout_seconds = classification_timeout_seconds
        self.store = memory if memory is not None else ConversationStore()
        self._owns_store = memory is None
        self.context = DialogueContext(self.store, llm, history_context_chars) if llm else None
        self.locks: dict[GroupKey, asyncio.Lock] = {}
        self.epochs: dict[GroupKey, int] = {}
        self.key_tasks: dict[GroupKey, set[asyncio.Task]] = {}
        self.pending: dict[GroupKey, PendingGroup] = {}
        self.queued: dict[GroupKey, deque[tuple[Message, str | None]]] = {}
        # Telegram may redeliver an update after an answer was already sent.
        # Keep only opaque message IDs in process memory; no content is retained.
        self.seen_message_ids: dict[GroupKey, set[tuple[int, tuple[str, ...], str | None]]] = {}
        self.tasks: set[asyncio.Task] = set()
        self.closed = False

    @staticmethod
    def conversation_key(message: Message) -> GroupKey:
        return (
            message.chat.id, message.message_thread_id,
            message.from_user.id if message.from_user else None,
            message.sender_chat.id if message.sender_chat else None,
        )

    def reset(self, message: Message) -> None:
        key = self.conversation_key(message)
        self.epochs[key] = self.epochs.get(key, 0) + 1
        self.pending.pop(key, None)
        self.queued.pop(key, None)
        self.seen_message_ids.pop(key, None)
        for task in tuple(self.key_tasks.get(key, ())):
            task.cancel()
        self.store.clear(key)
        if self.reminders is not None:
            self.reminders.store.clear_draft(key)

    def add(
        self, message: Message, command_answer: str | None = None, *,
        _from_queue: bool = False,
    ) -> None:
        if self.closed:
            return
        key = self.conversation_key(message)
        kinds = classify_message(message, bot_id=message.bot.id)
        if not _from_queue:
            seen = self.seen_message_ids.setdefault(key, set())
            # Telegram guarantees a message ID within a chat; tests and some
            # imported fixtures may reuse an ID for different media.  Treat
            # only the identical delivered payload as a retry.
            delivery = (message.message_id, tuple(sorted(kinds)), message.media_group_id)
            if delivery in seen:
                return
            seen.add(delivery)
            if len(seen) > 2048:
                # Message IDs are monotonically increasing within a chat; retaining
                # the newest window is enough to defeat delivery retries.
                self.seen_message_ids[key] = set(sorted(seen)[-1024:])
        now = asyncio.get_running_loop().time()
        group = self.pending.get(key)
        queued = self.queued.get(key)
        if queued is not None and any(item.message_id == message.message_id for item, _ in queued):
            return
        if queued and not _from_queue:
            queued.append((message, command_answer))
            return
        if group is not None and self._starts_separate_turn(group, message, command_answer, kinds):
            self.queued.setdefault(key, deque()).append((message, command_answer))
            return
        history_id = None
        if (
            (self.llm is not None or self.ai is not None) and message.text and command_answer is None
            and not message.text.startswith("/") and kinds <= TEXT_KINDS
        ):
            history_id = self.store.append_user(key, message.text, message.message_id)
        if group is not None and history_id is not None and history_id in group.history_ids:
            return
        if group is None:
            group = PendingGroup(message=message, last_received=now, epoch=self.epochs.get(key, 0))
            self.pending[key] = group
        elif group.task is not None:
            # Continuations invalidate analysis/generation, but never cancel
            # an in-flight Telegram send (removed from pending before sending).
            group.task.cancel()
        group.message = message
        group.last_received = now
        group.kinds.update(kinds)
        group.count += 1
        group.command_answer = command_answer
        group.has_command |= command_answer is not None or bool(message.text and message.text.startswith("/"))
        if history_id is not None:
            group.history_ids.add(history_id)
        if message.text and not group.overflow:
            group.text_length += len(message.text) + (2 if group.texts else 0)
            group.texts.append(message.text)
        if group.text_length > 16_000 or group.count > 100:
            group.overflow = True
            group.texts.clear()
        task = asyncio.create_task(self._reply_after_pause(key, group), name="telegram-answer-group")
        group.task = task
        self.tasks.add(task)
        self.key_tasks.setdefault(key, set()).add(task)
        task.add_done_callback(lambda done: self._key_task_done(key, done))
        task.add_done_callback(self._task_done)

    def _starts_separate_turn(
        self, group: PendingGroup, message: Message,
        command_answer: str | None, kinds: set[MessageKind],
    ) -> bool:
        """Keep complete rapid-fire questions as ordered dialogue turns."""
        if (
            (self.llm is None and self.ai is None) or command_answer is not None
            or not message.text or message.text.startswith("/") or not kinds <= TEXT_KINDS
            or group.has_command or group.overflow or not group.texts
        ):
            return False
        previous = group.texts[-1].strip()
        incoming = message.text.strip()
        previous_is_request = previous.endswith("?") or bool(re.match(
            r"^(?:расскажи|объясни|подскажи|покажи|найди|посоветуй)\b",
            previous,
            re.I,
        ))
        incoming_is_request = incoming.endswith("?") or bool(re.match(
            r"^(?:расскажи|объясни|подскажи|покажи|найди|посоветуй)\b",
            incoming,
            re.I,
        ))
        return (previous_is_request and incoming_is_request) or is_topic_follow_up(incoming)

    async def _reply_after_pause(self, key: GroupKey, group: PendingGroup) -> None:
        # Keep generation and delivery in order. A message received during an
        # in-flight send must see that successfully delivered assistant reply.
        async with self.locks.setdefault(key, asyncio.Lock()):
            await self._reply_locked(key, group)

    async def _reply_locked(self, key: GroupKey, group: PendingGroup) -> None:
        # Snapshot input; identity checks also reject cancellation-resistant
        # providers returning an obsolete result.
        current = asyncio.current_task()
        def is_current():
            return (
                group.task is current and self.pending.get(key) is group
                and self.epochs.get(key, 0) == group.epoch and not self.closed
            )
        if not is_current():
            return
        text = "\n\n".join(group.texts)
        use_llm = bool(text) and not group.has_command and group.kinds <= TEXT_KINDS and not group.overflow
        ready = use_llm and is_clear_question(text)
        history: list[dict] = []
        context_error = False
        if use_llm and self.ai is None and self.context is not None:
            try:
                history = await self.context.prepare(key, group.history_ids, text, is_current)
            except OllamaError:
                context_error = True
        if use_llm and not ready and self.ai is None and self.llm is not None:
            try:
                if not context_error:
                    decision = await asyncio.wait_for(
                        self.llm.classify_turn(text, history),
                        timeout=self.classification_timeout_seconds,
                    )
                    ready = decision == "QUESTION" or (
                        decision == "ANSWER" and assistant_requested_answer(history)
                    )
            except (OllamaError, TimeoutError):
                logger.info("Turn analysis unavailable; waiting for an explicit question")
        if not is_current():
            return
        if not ready:
            pause = max(
                self.delay_seconds,
                ALBUM_GROUP_DELAY_SECONDS if MessageKind.ALBUM in group.kinds else 0,
            )
            remaining = group.last_received + pause - asyncio.get_running_loop().time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        if not is_current():
            return
        if use_llm and self.ai is None and self.llm is not None and not ready:
            # A pause is not permission to answer. The original text is already
            # durable and will accompany a later question, even after restart.
            self.pending.pop(key)
            return
        record_answer = False
        consultation = None
        rich_answer = None
        thinking_message = None
        reply_markup = None
        if group.overflow:
            answer = GROUP_TOO_LONG
        elif group.count == 1 and group.command_answer:
            answer = group.command_answer
        elif use_llm and self.reminders is not None and (
            reminder_answer := self.reminders.handle(key, text)
        ) is not None:
            answer = reminder_answer
            record_answer = True
        elif use_llm and TAX_CALCULATOR_QUERY.search(text):
            calculator = CARDS_BY_SLUG["kalkulyator-sistem-nalogooblozheniya"]
            answer = (
                "Для выбора налогового режима есть калькулятор Альфа-Курса. "
                "Он сравнивает режимы по параметрам бизнеса, но не рассчитывает сумму налога "
                "к уплате именно на УСН."
            )
            reply_markup = card_keyboard(calculator)
            rich_answer = None
        elif use_llm and self.ai is not None:
            if len(text.strip()) > 1000:
                answer = TOO_LONG
            else:
                profile, previous_query, pending_query = self.store.consultation_state(key)
                effective_text = text.strip()
                if pending_query and is_short_clarification(effective_text):
                    combined = f"{pending_query}\nУточнение пользователя: {effective_text}"
                    if len(combined) <= 1000:
                        effective_text = combined
                elif (
                    previous_query
                    and is_topic_follow_up(effective_text)
                    and not is_topic_follow_up(previous_query)
                ):
                    combined = (
                        f"Текущий вопрос: {effective_text}\n"
                        f"Контекст предыдущей темы: {previous_query}"
                    )
                    if len(combined) <= 1000:
                        effective_text = combined
                try:
                    thinking_message = await _send_thinking(group.message)
                    response = await self.ai.answer(
                        AIRequest(
                            query=effective_text, profile=LearningProfile(**profile),
                            previous_query=previous_query,
                        )
                    )
                except asyncio.CancelledError:
                    await _remove_thinking(thinking_message)
                    raise
                answer = format_answer(response)
                rich_answer = format_telegram_answer(response)
                record_answer = response.status != "error"
                if record_answer:
                    if response.profile_patch is not None:
                        profile.update(response.to_dict()["profile_patch"])
                    last_query = effective_text if response.proposed_action is None else previous_query
                    pending = effective_text if response.status == "clarify" else None
                    consultation = profile, last_query, pending
        elif use_llm and self.llm is not None:
            try:
                if context_error:
                    raise OllamaError("Conversation context unavailable")
                answer = await self.llm.answer_with_history(text, history)
                record_answer = True
            except (OllamaError, TimeoutError):
                logger.warning("Ollama answer unavailable")
                answer = OLLAMA_UNAVAILABLE
        else:
            answer = build_answer(group.kinds, group.count)
        if not is_current():
            await _remove_thinking(thinking_message)
            return
        # Incoming messages now start a new group and cannot cancel delivery.
        self.pending.pop(key)
        if rich_answer is not None:
            try:
                await _deliver_rich_answer(group.message, thinking_message, rich_answer)
            except TelegramBadRequest:
                logger.info("Telegram rejected premium formatting; retrying with fallback emoji")
                await _deliver_rich_answer(
                    group.message, thinking_message, without_custom_emoji(rich_answer),
                )
            if self.epochs.get(key, 0) != group.epoch or self.closed:
                return
            if record_answer and self.epochs.get(key, 0) == group.epoch:
                self.store.append_assistant(key, answer)
                if self.analytics is not None and response.status == "answer":
                    self.analytics.record_once(f"shown:{response.trace_id}", "answer_shown")
        else:
            for index, chunk in enumerate(telegram_chunks(answer)):
                if index == 0 and isinstance(thinking_message, Message):
                    await thinking_message.edit_text(chunk)
                else:
                    await group.message.answer(chunk, reply_markup=reply_markup if index == 0 else None)
                if self.epochs.get(key, 0) != group.epoch or self.closed:
                    return
                if record_answer and self.epochs.get(key, 0) == group.epoch:
                    self.store.append_assistant(key, chunk)
        if consultation is not None:
            self.store.save_consultation(key, *consultation)

    def _key_task_done(self, key: GroupKey, task: asyncio.Task) -> None:
        tasks = self.key_tasks.get(key)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                self.key_tasks.pop(key, None)
        if self.pending.get(key) is None and (queued := self.queued.get(key)):
            message, command_answer = queued.popleft()
            if not queued:
                self.queued.pop(key, None)
            self.add(message, command_answer, _from_queue=True)

    def _task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # Do not expose exception text or user content in logs.
            logger.error("Grouped Telegram reply failed (%s)", type(task.exception()).__name__)

    async def wait_idle(self) -> None:
        """Wait for current groups, ignoring superseded tasks' cancellations."""
        while self.tasks:
            tasks = tuple(self.tasks)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.tasks.difference_update(tasks)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                    raise result

    async def close(self) -> None:
        self.closed = True
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        self.queued.clear()
        self.seen_message_ids.clear()
        if self._owns_store:
            self.store.close()


async def _send_rich_answer(message, answer):
    return await message.answer(
        answer.text,
        entities=list(answer.entities),
        link_preview_options=answer.link_preview_options,
        reply_markup=answer.reply_markup,
    )


async def _send_thinking(message):
    thinking = format_thinking()
    try:
        return await _send_rich_answer(message, thinking)
    except TelegramBadRequest:
        logger.info("Telegram rejected premium loading emoji; using fallback emoji")
        return await _send_rich_answer(message, without_custom_emoji(thinking))


async def _deliver_rich_answer(message, thinking_message, answer):
    if isinstance(thinking_message, Message):
        return await thinking_message.edit_text(
            answer.text,
            entities=list(answer.entities),
            link_preview_options=answer.link_preview_options,
            reply_markup=answer.reply_markup,
        )
    return await _send_rich_answer(message, answer)


async def _remove_thinking(thinking_message):
    if isinstance(thinking_message, Message):
        with suppress(TelegramBadRequest):
            await thinking_message.delete()
