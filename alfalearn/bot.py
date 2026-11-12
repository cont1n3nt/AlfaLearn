from aiogram import Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from alfalearn.grouping import AnswerBuffer
from alfalearn.llm import OllamaClient
from alfalearn.memory import ConversationStore

HELLO_MESSAGE = "Hello, world!"
START_MESSAGE = (
    "Привет! Я АльфаНавигатор — прототип помощника по материалам Альфа-Курса.\n\n"
    "Пока не умею читать сообщения. Для проверки связи напиши /hello."
)
OLLAMA_START_MESSAGE = (
    "Привет! Я АльфаНавигатор — тестовый помощник с Ollama.\n\n"
    "Сохраняю контекст нашего разговора. Задай вопрос или ответь на моё уточнение — "
    "тогда отвечу. Вводные сообщения запомню и молча подожду продолжения.\n"
    "Материалы Альфа-Курса ещё не подключены, вложения пока не читаю.\n"
    "Сброс истории: /reset."
)


def create_dispatcher(
    group_delay_seconds: float = 1.0,
    *, llm: OllamaClient | None = None, classification_timeout_seconds: float = 0.8,
    memory: ConversationStore | None = None, history_context_chars: int = 12_000,
) -> Dispatcher:
    router = Router(name="answers")
    buffer = AnswerBuffer(
        group_delay_seconds, llm=llm,
        classification_timeout_seconds=classification_timeout_seconds,
        memory=memory, history_context_chars=history_context_chars,
    )

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        buffer.add(
            message, command_answer=OLLAMA_START_MESSAGE if llm else START_MESSAGE,
        )

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        buffer.reset(message)
        await message.answer("История этого диалога очищена. Начнём заново.")

    @router.message(Command("hello", "help"))
    async def hello(message: Message) -> None:
        buffer.add(message, command_answer=HELLO_MESSAGE)

    # Register last so commands retain their dedicated replies.
    @router.message()
    async def answer(message: Message) -> None:
        buffer.add(message)

    dispatcher = Dispatcher()
    dispatcher["answer_buffer"] = buffer
    dispatcher.shutdown.register(buffer.close)
    dispatcher.include_router(router)
    return dispatcher
