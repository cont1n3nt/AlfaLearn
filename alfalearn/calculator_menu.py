"""Telegram commands for the calculators: /calc menu and one command per calculator."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
)

from alfalearn.calculator_cards import CARDS, CARDS_BY_SLUG, MENU_TEXT, CalculatorCard

NO_PREVIEW = LinkPreviewOptions(is_disabled=True)
MENU_COMMANDS = ("calc", "calculators")


class CalculatorChoice(CallbackData, prefix="calc"):
    slug: str  # Calculator slug, or "menu" for the list of all calculators


def menu_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(text=card.button, callback_data=CalculatorChoice(slug=card.slug).pack())
        for card in CARDS
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[i:i + 2] for i in range(0, len(buttons), 2)])


def card_keyboard(card: CalculatorCard) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"🧮 Открыть: {card.button}", url=card.url)],
        [InlineKeyboardButton(text="Все калькуляторы", callback_data=CalculatorChoice(slug="menu").pack())],
    ])


async def send_menu(message: Message) -> None:
    await message.answer(
        MENU_TEXT, parse_mode="HTML", reply_markup=menu_keyboard(), link_preview_options=NO_PREVIEW,
    )


async def send_card(message: Message, card: CalculatorCard) -> None:
    await message.answer(
        card.text, parse_mode="HTML", reply_markup=card_keyboard(card), link_preview_options=NO_PREVIEW,
    )


def _card_handler(card: CalculatorCard):
    async def handler(message: Message) -> None:
        await send_card(message, card)

    handler.__name__ = f"calculator_{card.command}"
    return handler


def create_calculator_router() -> Router:
    router = Router(name="calculators")
    router.message.register(send_menu, Command(*MENU_COMMANDS))
    for card in CARDS:
        router.message.register(_card_handler(card), Command(card.command))

    @router.callback_query(CalculatorChoice.filter())
    async def choose(callback: CallbackQuery, callback_data: CalculatorChoice) -> None:
        await callback.answer()
        card = CARDS_BY_SLUG.get(callback_data.slug)
        if card is None:
            await send_menu(callback.message)
        else:
            await send_card(callback.message, card)

    return router
