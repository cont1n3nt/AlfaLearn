from aiogram import Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from alfalearn.answers import ANSWER_MESSAGE
from alfalearn.calculator_menu import create_calculator_router
from alfalearn.grouping import AnswerBuffer
from alfalearn.vat_bot import create_vat_router

HELLO_MESSAGE = "Hello, world!"
START_MESSAGE = (
    "Привет! Я АльфаНавигатор — прототип помощника по материалам Альфа-Курса.\n\n"
    "Пока не умею читать сообщения. Калькуляторы Альфа-Курса — /calc, "
    "вопрос про НДС можно задать прямо в чате. Для проверки связи напиши /hello."
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
    # Calculators and VAT questions go first: the answers router ends with a catch-all handler.
    dispatcher.include_router(create_calculator_router())
    dispatcher.include_router(create_vat_router())
    dispatcher.include_router(router)
    return dispatcher
