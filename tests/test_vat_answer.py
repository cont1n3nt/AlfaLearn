import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from itertools import count
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from alfalearn.answers import ANSWER_MESSAGE
from alfalearn.bot import create_dispatcher
from alfalearn.vat_answer import (
    MARKETPLACE_COURSE_URL,
    TAX_REGIME_CALCULATOR_URL,
    USN_OSN_ARTICLE_URL,
    VAT_CALCULATOR_URL,
    TaxRegime,
    build_regime_answer,
    build_vat_answer,
    extract_amount,
    is_vat_question,
)
from alfalearn.vat_bot import VatRegimeChoice

D = Decimal
NBSP = " "
CHAT = Chat(id=42, type="private")
USER = User(id=42, is_bot=False, first_name="Test")
_ids = count(1)


def plain(text: str) -> str:
    return text.replace(NBSP, " ")


@pytest.mark.parametrize(("text", "expected"), [
    ("Сколько НДС заплатить с 100 тысяч?", True),
    ("ндс 22%", True),
    ("Как считать налог на добавленную стоимость?", True),
    ("Сколько платить НДФЛ?", False),
    ("Привет", False),
    ("", False),
    (None, False),
])
def test_is_vat_question(text, expected):
    assert is_vat_question(text) is expected


@pytest.mark.parametrize(("text", "expected"), [
    ("Сколько НДС заплатить с 100 тысяч?", "100000"),
    ("НДС с 100 тыс. рублей", "100000"),
    ("ндс с 100к", "100000"),
    ("НДС 22% с 87 500 ₽", "87500"),
    ("выделить НДС из 87500,50", "87500.50"),
    ("НДС с 1,5 млн", "1500000"),
    ("НДС с 2 миллионов", "2000000"),
    ("какой НДС в 2026 году с 300 000 руб", "300000"),
    ("НДС 10 процентов с 5000", "5000"),
    ("сколько НДС?", None),
    ("ставка НДС 22%", None),
])
def test_extract_amount(text, expected):
    assert extract_amount(text) == (None if expected is None else D(expected))


def test_vat_answer_matches_the_card():
    answer = plain(build_vat_answer(D("100000")))
    for fragment in (
        "Сумма НДС зависит от трёх вещей",
        "<b>1. Вы вообще платите НДС?</b>",
        "порог освобождения 20 млн ₽ включительно",
        "Основная ставка с 2026 года — 22% (до 2026 года была 20%)",
        "специальные ставки 5% и 7%",
        "<b>3. 100 000 ₽ — это сумма с НДС или без?</b>",
        "НДС сверху: 100 000 × 22 / 100 = 22 000 ₽, итого к оплате 122 000 ₽.",
        "НДС уже внутри суммы: 100 000 × 22 / 122 = 18 032,79 ₽.",
        f'<a href="{VAT_CALCULATOR_URL}">«Калькулятор НДС», Альфа-Курс</a>',
        f'<a href="{USN_OSN_ARTICLE_URL}">Как перейти с УСН на ОСН и обратно</a> — статья, бесплатно',
        "платный курс, 3990 ₽",
    ):
        assert fragment in answer


def test_vat_answer_without_amount_asks_for_it():
    answer = plain(build_vat_answer(None))
    assert "Сумма × 22 / 122" in answer
    assert "Напишите сумму" in answer


def test_vat_numbers_match_site_example():
    # Example from the calculator page: 87 500 ₽ at 22%.
    answer = plain(build_vat_answer(D("87500")))
    assert "= 19 250 ₽, итого к оплате 106 750 ₽" in answer
    assert "= 15 778,69 ₽" in answer


def test_regime_answers():
    usn = plain(build_regime_answer(TaxRegime.USN, D("100000")))
    assert "НДС на УСН в 2026 году" in usn
    assert "5%: сверху 5 000 ₽ (итого 105 000 ₽), внутри суммы 4 761,90 ₽" in usn
    assert "7%: сверху 7 000 ₽ (итого 107 000 ₽), внутри суммы 6 542,06 ₽" in usn

    osn = plain(build_regime_answer(TaxRegime.OSN, D("100000")))
    assert "22%: сверху 22 000 ₽ (итого 122 000 ₽), внутри суммы 18 032,79 ₽" in osn
    assert "10%: сверху 10 000 ₽ (итого 110 000 ₽), внутри суммы 9 090,91 ₽" in osn

    assert "Напишите сумму" in build_regime_answer(TaxRegime.OSN, None)
    unknown = build_regime_answer(TaxRegime.UNKNOWN, D("100000"))
    assert "применяется по умолчанию" in unknown
    assert TAX_REGIME_CALCULATOR_URL in unknown


# --- Telegram ----------------------------------------------------------------

def message_update(text: str) -> Update:
    return Update(update_id=next(_ids), message=Message(
        message_id=next(_ids), date=datetime.now(timezone.utc), chat=CHAT, from_user=USER, text=text,
    ))


def callback_update(data: str) -> Update:
    bot_message = Message(
        message_id=next(_ids), date=datetime.now(timezone.utc), chat=CHAT,
        from_user=User(id=1, is_bot=True, first_name="Bot"), text="Ответ про НДС",
    )
    return Update(update_id=next(_ids), callback_query=CallbackQuery(
        id=str(next(_ids)), from_user=USER, chat_instance="test", message=bot_message, data=data,
    ))


async def feed(dispatcher, bot, update) -> list:
    bot.session.reset_mock()
    await dispatcher.feed_update(bot, update)
    await dispatcher["answer_buffer"].wait_idle()
    return [call.args[1] for call in bot.session.call_args_list
            if not isinstance(call.args[1], AnswerCallbackQuery)]


def buttons(method: SendMessage) -> list:
    return [button for row in method.reply_markup.inline_keyboard for button in row]


def test_bot_answers_vat_question_with_buttons():
    async def run():
        bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
        bot.session = AsyncMock()
        dispatcher = create_dispatcher(group_delay_seconds=0.01)

        [reply] = await feed(dispatcher, bot, message_update("Сколько НДС заплатить с 100 тысяч?"))
        assert isinstance(reply, SendMessage)
        assert reply.parse_mode == "HTML"
        assert "18 032,79 ₽" in plain(reply.text)
        by_text = {button.text: button for button in buttons(reply)}
        assert by_text["🧮 Посчитать в калькуляторе НДС"].url == VAT_CALCULATOR_URL
        assert by_text["📘 Курс «УСН на маркетплейсах» — подробнее"].url == MARKETPLACE_COURSE_URL
        usn_data = by_text["Я на УСН"].callback_data
        assert VatRegimeChoice.unpack(usn_data) == VatRegimeChoice(regime=TaxRegime.USN, amount="100000.00")

        [usn] = await feed(dispatcher, bot, callback_update(usn_data))
        assert "4 761,90 ₽" in plain(usn.text)
        assert buttons(usn)[0].url == VAT_CALCULATOR_URL

        [unknown] = await feed(dispatcher, bot, callback_update(by_text["Не знаю свой режим"].callback_data))
        assert "применяется по умолчанию" in unknown.text
        assert {button.text for button in buttons(unknown)} >= {"Я на УСН", "Я на ОСН"}

        # Other messages still get the usual placeholder.
        [other] = await feed(dispatcher, bot, message_update("Как начать обучение?"))
        assert other.text == ANSWER_MESSAGE

    asyncio.run(run())


def test_broken_amount_in_button_is_ignored():
    async def run():
        bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
        bot.session = AsyncMock()
        dispatcher = create_dispatcher(group_delay_seconds=0.01)
        data = VatRegimeChoice(regime=TaxRegime.OSN, amount="abc").pack()
        [reply] = await feed(dispatcher, bot, callback_update(data))
        assert "Напишите сумму" in reply.text

    asyncio.run(run())
