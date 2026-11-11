import asyncio
import re
from datetime import datetime, timezone
from itertools import count
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.methods import AnswerCallbackQuery, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from alfalearn.bot import ANSWER_MESSAGE, create_dispatcher
from alfalearn.calculator_cards import CARDS, MENU_TEXT, NDFL, NDS
from alfalearn.calculator_menu import CalculatorChoice

CHAT = Chat(id=42, type="private")
USER = User(id=42, is_bot=False, first_name="Test")
_ids = count(1)

# Addresses of all seven calculators in the Alfa-Kurs database.
EXPECTED = {
    "nds": "https://kurs.alfabank.ru/calculators/nds/",
    "procent": "https://kurs.alfabank.ru/calculators/raschet-procentov/",
    "propis": "https://kurs.alfabank.ru/calculators/summa-propisyu-online/",
    "dni": "https://kurs.alfabank.ru/calculators/rabochie-dni/",
    "nalog": "https://kurs.alfabank.ru/calculators/kalkulyator-sistem-nalogooblozheniya/",
    "zarplata": "https://kurs.alfabank.ru/calculators/zarplata/",
    "ndfl": "https://kurs.alfabank.ru/calculators/ndfl/",
}


def test_seven_calculators_with_unique_commands():
    assert {card.command: card.url for card in CARDS} == EXPECTED
    for card in CARDS:
        # Telegram allows only lowercase latin letters, digits and _ in commands.
        assert re.fullmatch(r"[a-z0-9_]{1,32}", card.command)
        assert f"/{card.command} — {card.title}" in MENU_TEXT
        assert len(card.text) < 4096
        assert f'<a href="{card.url}">' in card.text


def test_card_examples_match_site_pages():
    assert "НДС 15 778,69 ₽" in NDS.text
    assert "Налог 492 000 ₽, на руки за год 3 108 000 ₽" in NDFL.text


def message_update(text: str) -> Update:
    return Update(update_id=next(_ids), message=Message(
        message_id=next(_ids), date=datetime.now(timezone.utc), chat=CHAT, from_user=USER, text=text,
    ))


def callback_update(data: str) -> Update:
    bot_message = Message(
        message_id=next(_ids), date=datetime.now(timezone.utc), chat=CHAT,
        from_user=User(id=1, is_bot=True, first_name="Bot"), text="Калькуляторы",
    )
    return Update(update_id=next(_ids), callback_query=CallbackQuery(
        id=str(next(_ids)), from_user=USER, chat_instance="test", message=bot_message, data=data,
    ))


def buttons(method: SendMessage) -> list:
    return [button for row in method.reply_markup.inline_keyboard for button in row]


def run_bot(scenario):
    async def run():
        bot = Bot(token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
        bot.session = AsyncMock()
        dispatcher = create_dispatcher(group_delay_seconds=0.01)

        async def feed(update: Update) -> SendMessage:
            bot.session.reset_mock()
            await dispatcher.feed_update(bot, update)
            await dispatcher["answer_buffer"].wait_idle()
            replies = [call.args[1] for call in bot.session.call_args_list
                       if not isinstance(call.args[1], AnswerCallbackQuery)]
            assert len(replies) == 1, replies
            return replies[0]

        await scenario(feed)

    asyncio.run(run())


@pytest.mark.parametrize("card", CARDS, ids=lambda card: card.command)
def test_each_command_sends_card_with_link(card):
    async def scenario(feed):
        reply = await feed(message_update(f"/{card.command}"))
        assert reply.text == card.text
        assert reply.parse_mode == "HTML"
        link, back = buttons(reply)
        assert link.url == card.url
        assert CalculatorChoice.unpack(back.callback_data).slug == "menu"

    run_bot(scenario)


def test_menu_lists_all_calculators_and_opens_cards():
    async def scenario(feed):
        menu = await feed(message_update("/calc"))
        assert menu.text == MENU_TEXT
        menu_buttons = buttons(menu)
        assert [CalculatorChoice.unpack(b.callback_data).slug for b in menu_buttons] == [c.slug for c in CARDS]

        ndfl_button = next(b for b in menu_buttons if b.text == "НДФЛ")
        card = await feed(callback_update(ndfl_button.callback_data))
        assert card.text == NDFL.text

        back = await feed(callback_update(buttons(card)[1].callback_data))
        assert back.text == MENU_TEXT

        # Unknown slug from an old keyboard falls back to the menu.
        assert (await feed(callback_update(CalculatorChoice(slug="old").pack()))).text == MENU_TEXT

    run_bot(scenario)


def test_other_messages_keep_old_answers():
    async def scenario(feed):
        assert (await feed(message_update("Привет"))).text == ANSWER_MESSAGE
        assert "18 032,79" in (await feed(message_update("Сколько НДС с 100 тысяч?"))).text.replace(" ", " ")

    run_bot(scenario)
