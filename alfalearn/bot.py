from aiogram import Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from alfalearn.answers import ANSWER_MESSAGE
from alfalearn.grouping import AnswerBuffer

HELLO_MESSAGE = "Hello, world!"
START_MESSAGE = (
    "Привет! Я АльфаНавигатор — прототип помощника по материалам Альфа-Курса.\n\n"
    "Пока не умею читать сообщения. Для проверки связи напиши /hello."
)


def create_dispatcher(group_delay_seconds: float = 2.0) -> Dispatcher:
    router = Router(name="answers")
    buffer = AnswerBuffer(group_delay_seconds)

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        buffer.add(message, command_answer=START_MESSAGE)

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
