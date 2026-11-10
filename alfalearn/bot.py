from aiogram import Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

HELLO_MESSAGE = "Hello, world!"
START_MESSAGE = (
    "Привет! Я АльфаНавигатор — прототип помощника по материалам Альфа-Курса.\n\n"
    "Пока умею отвечать Hello, world! Напиши /hello."
)


def create_dispatcher() -> Dispatcher:
    router = Router(name="hello")

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(START_MESSAGE)

    @router.message(Command("hello", "help"))
    async def hello(message: Message) -> None:
        await message.answer(HELLO_MESSAGE)

    @router.message(F.text)
    async def text(message: Message) -> None:
        await message.answer(HELLO_MESSAGE)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher
