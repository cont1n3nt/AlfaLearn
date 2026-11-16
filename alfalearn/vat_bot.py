"""Telegram handlers for VAT questions: the answer card and the regime buttons."""

from decimal import Decimal, InvalidOperation

from aiogram import F, Router
from aiogram.filters.callback_data import CallbackData
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
)

from alfalearn.vat_answer import (
    MARKETPLACE_COURSE_URL,
    TAX_REGIME_CALCULATOR_URL,
    VAT_CALCULATOR_URL,
    TaxRegime,
    build_regime_answer,
    build_vat_answer,
    extract_amount,
    is_vat_question,
)

NO_PREVIEW = LinkPreviewOptions(is_disabled=True)


class VatRegimeChoice(CallbackData, prefix="vatr"):
    regime: TaxRegime
    amount: str = ""  # The amount from the question, so follow-ups can show numbers.


def _regime_button(text: str, regime: TaxRegime, amount: Decimal | None) -> InlineKeyboardButton:
    data = VatRegimeChoice(regime=regime, amount="" if amount is None else str(amount))
    return InlineKeyboardButton(text=text, callback_data=data.pack())


def _calculator_button() -> InlineKeyboardButton:
    return InlineKeyboardButton(text="🧮 Посчитать в калькуляторе НДС", url=VAT_CALCULATOR_URL)


def vat_answer_keyboard(amount: Decimal | None) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [_regime_button("Я на УСН", TaxRegime.USN, amount), _regime_button("Я на ОСН", TaxRegime.OSN, amount)],
        [_regime_button("Не знаю свой режим", TaxRegime.UNKNOWN, amount)],
        [_calculator_button()],
        [InlineKeyboardButton(text="📘 Курс «УСН на маркетплейсах» — подробнее", url=MARKETPLACE_COURSE_URL)],
    ])


def regime_answer_keyboard(regime: TaxRegime, amount: Decimal | None) -> InlineKeyboardMarkup:
    if regime is TaxRegime.UNKNOWN:
        return InlineKeyboardMarkup(inline_keyboard=[
            [_regime_button("Я на УСН", TaxRegime.USN, amount), _regime_button("Я на ОСН", TaxRegime.OSN, amount)],
            [InlineKeyboardButton(text="🧭 Калькулятор выбора режима", url=TAX_REGIME_CALCULATOR_URL)],
        ])
    return InlineKeyboardMarkup(inline_keyboard=[[_calculator_button()]])


def _parse_amount(value: str) -> Decimal | None:
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount > 0 else None


def create_vat_router() -> Router:
    router = Router(name="vat")

    @router.message(F.text.func(is_vat_question))
    async def vat_question(message: Message) -> None:
        amount = extract_amount(message.text)
        await message.answer(
            build_vat_answer(amount),
            parse_mode="HTML",
            reply_markup=vat_answer_keyboard(amount),
            link_preview_options=NO_PREVIEW,
        )

    @router.callback_query(VatRegimeChoice.filter())
    async def vat_regime(callback: CallbackQuery, callback_data: VatRegimeChoice) -> None:
        await callback.answer()
        amount = _parse_amount(callback_data.amount) if callback_data.amount else None
        await callback.message.answer(
            build_regime_answer(callback_data.regime, amount),
            parse_mode="HTML",
            reply_markup=regime_answer_keyboard(callback_data.regime, amount),
            link_preview_options=NO_PREVIEW,
        )

    return router
