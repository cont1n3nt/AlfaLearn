"""Reply to complete questions promptly; debounce fragments per chat/sender/topic."""

import asyncio
import logging
import re
from dataclasses import dataclass, field

from aiogram.types import Message

from alfa_ai import AIRequest, AIService, LearningProfile
from alfalearn.answers import MessageKind, build_answer, classify_message
from alfalearn.knowledge import TOO_LONG, format_answer
from alfalearn.llm import OllamaClient, OllamaError
from alfalearn.conversation import DialogueContext
from alfalearn.memory import ConversationStore

logger = logging.getLogger(__name__)
GroupKey = tuple[int, int | None, int | None, int | None]
OLLAMA_UNAVAILABLE = "Сейчас не удалось получить ответ от Ollama. Попробуйте ещё раз чуть позже."
GROUP_TOO_LONG = "Сообщение слишком длинное. Сократите вопрос до 16 000 символов и 100 сообщений."
TEXT_KINDS = {
    MessageKind.TEXT, MessageKind.LINK, MessageKind.REPLY,
    MessageKind.QUOTE, MessageKind.FORWARDED, MessageKind.EMOJI,
}


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
        ai: AIService | None = None,
    ):
        if delay_seconds <= 0:
            raise ValueError("Grouping delay must be positive")
        if classification_timeout_seconds <= 0:
            raise ValueError("Classification timeout must be positive")
        self.delay_seconds = delay_seconds
        self.llm = llm
        self.ai = ai
        self.classification_timeout_seconds = classification_timeout_seconds
        self.store = memory if memory is not None else ConversationStore()
        self._owns_store = memory is None
        self.context = DialogueContext(self.store, llm, history_context_chars) if llm else None
        self.locks: dict[GroupKey, asyncio.Lock] = {}
        self.epochs: dict[GroupKey, int] = {}
        self.key_tasks: dict[GroupKey, set[asyncio.Task]] = {}
        self.pending: dict[GroupKey, PendingGroup] = {}
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
        for task in tuple(self.key_tasks.get(key, ())):
            task.cancel()
        self.store.clear(key)

    def add(
        self, message: Message, command_answer: str | None = None,
    ) -> None:
        if self.closed:
            return
        key = self.conversation_key(message)
        now = asyncio.get_running_loop().time()
        kinds = classify_message(message, bot_id=message.bot.id)
        history_id = None
        if (
            (self.llm is not None or self.ai is not None) and message.text and command_answer is None
            and not message.text.startswith("/") and kinds <= TEXT_KINDS
        ):
            history_id = self.store.append_user(key, message.text, message.message_id)
        group = self.pending.get(key)
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
        ready = use_llm and (self.ai is not None or is_clear_question(text))
        history: list[dict] = []
        context_error = False
        if use_llm and self.ai is None and self.context is not None:
            try:
                history = await self.context.prepare(key, group.history_ids, text, is_current)
            except OllamaError:
                context_error = True
        if use_llm and not ready and self.llm is not None:
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
            remaining = group.last_received + self.delay_seconds - asyncio.get_running_loop().time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        if not is_current():
            return
        if use_llm and self.llm is not None and not ready:
            # A pause is not permission to answer. The original text is already
            # durable and will accompany a later question, even after restart.
            self.pending.pop(key)
            return
        record_answer = False
        consultation = None
        if group.overflow:
            answer = GROUP_TOO_LONG
        elif group.count == 1 and group.command_answer:
            answer = group.command_answer
        elif use_llm and self.ai is not None:
            if len(text.strip()) > 1000:
                answer = TOO_LONG
            else:
                profile, previous_query = self.store.consultation(key)
                response = await self.ai.answer(
                    AIRequest(query=text, profile=LearningProfile(**profile), previous_query=previous_query)
                )
                answer = format_answer(response)
                record_answer = response.status != "error"
                if record_answer:
                    if response.profile_patch is not None:
                        profile.update(response.to_dict()["profile_patch"])
                    pending_query = (
                        text.strip()
                        if response.status == "clarify" and response.proposed_action is None
                        else None
                    )
                    consultation = profile, pending_query
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
            return
        # Incoming messages now start a new group and cannot cancel delivery.
        self.pending.pop(key)
        for chunk in telegram_chunks(answer):
            await group.message.answer(chunk)
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

    def _task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # Do not expose exception text or user content in logs.
            logger.error("Grouped Telegram reply failed (%s)", type(task.exception()).__name__)

    async def wait_idle(self) -> None:
        """Wait for current groups, ignoring superseded tasks' cancellations."""
        while self.tasks:
            results = await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
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
        if self._owns_store:
            self.store.close()
