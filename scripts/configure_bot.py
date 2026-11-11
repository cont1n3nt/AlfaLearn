"""Configure public bot metadata. Run from the project root."""

import asyncio

from aiogram import Bot
from aiogram.types import BotCommand

from alfalearn.calculator_cards import CARDS
from alfalearn.config import Settings


async def main() -> None:
    settings = Settings()
    if not settings.bot_token or not settings.bot_token.get_secret_value():
        raise RuntimeError("Set BOT_TOKEN in .env first.")
    async with Bot(token=settings.bot_token.get_secret_value()) as bot:
        await bot.set_my_name(name="АльфаНавигатор")
        await bot.set_my_description(
            description=(
                "АльфаНавигатор — прототип помощника по обучающим материалам Альфа-Курса. "
                "Создан для хакатона. Сейчас доступны калькуляторы Альфа-Курса (/calc), "
                "ответы на вопросы про НДС и проверка связи (/hello)."
            )
        )
        await bot.set_my_short_description(
            short_description="Прототип помощника по материалам Альфа-Курса. Проект команды хакатона."
        )
        await bot.set_my_commands(commands=[
            BotCommand(command="start", description="Знакомство с АльфаНавигатором"),
            BotCommand(command="calc", description="Все калькуляторы Альфа-Курса"),
            *[BotCommand(command=card.command, description=card.title) for card in CARDS],
            BotCommand(command="hello", description="Hello, world! — проверка связи"),
            BotCommand(command="help", description="Помощь"),
        ])
        me = await bot.get_me()
        name = await bot.get_my_name()
        print(f"Configured @{me.username}: {name.name}")


if __name__ == "__main__":
    asyncio.run(main())
