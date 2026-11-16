"""Telegram commands for demo accounts: /login, /logout and /profile."""

import asyncio
import logging
from contextlib import suppress
from html import escape

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import LinkPreviewOptions, Message

from alfalearn.accounts.service import AccessProfile, AccountService

logger = logging.getLogger(__name__)
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)

ASK_LOGIN = "Введите логин:"
ASK_PASSWORD = "Теперь введите пароль. Сообщение с паролем я сразу удалю из чата."
WRONG_CREDENTIALS = "Неверный логин или пароль. Попробуйте ещё раз: /login"
CANCELLED = "Вход отменён."
LOGGED_OUT = "Вы вышли из аккаунта. Теперь доступны только бесплатные курсы. Войти снова — /login"
NOT_LOGGED_IN = "Вы не входили в аккаунт. Войти — /login"


class LoginFlow(StatesGroup):
    login = State()
    password = State()


def format_profile(profile: AccessProfile) -> str:
    if profile.user is None:
        lines = ["👤 <b>Гость</b> — вход не выполнен.", "Доступны только бесплатные курсы."]
    else:
        user = profile.user
        lines = [f"👤 <b>{escape(user.name)}</b> (логин {escape(user.login)})", escape(user.access_note)]

    lines += ["", "<b>Продукты</b>"]
    for product in profile.products:
        if product.owned:
            lines.append(f"✅ {escape(product.title)} — есть")
        else:
            lines.append(
                f'❌ <a href="{product.url}">{escape(product.title)}</a> — {escape(product.price_note)}'
            )

    lines += ["", "<b>Платные курсы</b>"]
    for course in profile.courses:
        mark = "✅" if course.available else "🔒"
        lines.append(f'{mark} <a href="{course.url}">{escape(course.title)}</a>')

    available = sum(course.available for course in profile.courses)
    lines += ["", f"Открыто {available} из {len(profile.courses)} платных курсов."]
    lines.append("Войти — /login" if profile.user is None else "Сменить аккаунт — /login, выйти — /logout")
    return "\n".join(lines)


async def _delete(message: Message) -> None:
    with suppress(TelegramAPIError):
        await message.delete()


def create_accounts_router(service: AccountService) -> Router:
    router = Router(name="accounts")

    async def finish_login(message: Message, state: FSMContext, login: str, password: str) -> None:
        await state.clear()
        profile = await asyncio.to_thread(service.login, message.from_user.id, login, password)
        if profile is None:
            await message.answer(WRONG_CREDENTIALS)
            return
        await message.answer(
            f"Вход выполнен.\n\n{format_profile(profile)}",
            parse_mode="HTML", link_preview_options=NO_PREVIEW,
        )

    @router.message(Command("login"))
    async def login(message: Message, command: CommandObject, state: FSMContext) -> None:
        parts = (command.args or "").split()
        if len(parts) >= 2:
            # "/login boris password": the password is visible in the chat, remove it.
            await _delete(message)
            await finish_login(message, state, parts[0], " ".join(parts[1:]))
        elif parts:
            await state.set_state(LoginFlow.password)
            await state.update_data(login=parts[0])
            await message.answer(ASK_PASSWORD)
        else:
            await state.set_state(LoginFlow.login)
            await message.answer(ASK_LOGIN)

    @router.message(Command("cancel"), StateFilter(LoginFlow))
    async def cancel(message: Message, state: FSMContext) -> None:
        await state.clear()
        await message.answer(CANCELLED)

    @router.message(Command("logout"))
    async def logout(message: Message, state: FSMContext) -> None:
        await state.clear()
        logged_out = await asyncio.to_thread(service.logout, message.from_user.id)
        await message.answer(LOGGED_OUT if logged_out else NOT_LOGGED_IN)

    @router.message(Command("profile"))
    async def profile(message: Message, state: FSMContext) -> None:
        await state.clear()
        data = await asyncio.to_thread(service.profile, message.from_user.id)
        await message.answer(format_profile(data), parse_mode="HTML", link_preview_options=NO_PREVIEW)

    @router.message(LoginFlow.login, F.text, ~F.text.startswith("/"))
    async def got_login(message: Message, state: FSMContext) -> None:
        await state.update_data(login=message.text.strip())
        await state.set_state(LoginFlow.password)
        await message.answer(ASK_PASSWORD)

    @router.message(LoginFlow.password, F.text, ~F.text.startswith("/"))
    async def got_password(message: Message, state: FSMContext) -> None:
        # The password never reaches the conversation history or the LLM.
        await _delete(message)
        data = await state.get_data()
        await finish_login(message, state, data.get("login", ""), message.text)

    return router
