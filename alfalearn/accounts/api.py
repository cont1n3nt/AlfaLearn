"""HTTP endpoints for demo accounts. The bot calls the same `AccountService` methods."""

from fastapi import APIRouter, HTTPException, Path, Request, Response
from pydantic import BaseModel, Field

from alfalearn.accounts.service import AccessProfile, AccountService

TAG = "Демо-юзеры"
TelegramId = Path(description="Числовой id пользователя Telegram (message.from_user.id)", examples=[42])


class LoginRequest(BaseModel):
    login: str = Field(min_length=1, max_length=64, examples=["boris"])
    password: str = Field(min_length=1, max_length=256, examples=["Boris-demo-2026"])


def _service(request: Request) -> AccountService:
    return request.app.state.accounts


def create_accounts_router() -> APIRouter:
    router = APIRouter(prefix="/telegram/{telegram_id}", tags=[TAG])

    @router.post(
        "/login", response_model=AccessProfile, summary="Войти демо-юзером",
        responses={401: {"description": "Неверный логин или пароль"}},
    )
    def login(request: Request, body: LoginRequest, telegram_id: int = TelegramId) -> AccessProfile:
        """Проверить логин и пароль и привязать Telegram-аккаунт к демо-юзеру.

        Повторный вход другим логином переключает аккаунт. То же делает команда /login в боте.
        """
        profile = _service(request).login(telegram_id, body.login, body.password)
        if profile is None:
            raise HTTPException(status_code=401, detail="Неверный логин или пароль")
        return profile

    @router.delete("/login", status_code=204, summary="Выйти")
    def logout(request: Request, telegram_id: int = TelegramId) -> Response:
        """Отвязать Telegram-аккаунт от демо-юзера. Повторный выход ничего не ломает."""
        _service(request).logout(telegram_id)
        return Response(status_code=204)

    @router.get(
        "/user", response_model=AccessProfile, summary="Кто пишет и что ему доступно",
        responses={404: {"description": "С этого Telegram-аккаунта никто не вошёл"}},
    )
    def user(request: Request, telegram_id: int = TelegramId) -> AccessProfile:
        """Юзер, его продукты и все платные курсы с отметкой `available`.

        Для закрытых курсов `url` продукта из `products` — страница покупки.
        Если вход не выполнен, ответ 404: такой человек видит только бесплатные курсы.
        """
        profile = _service(request).profile(telegram_id)
        if profile.user is None:
            raise HTTPException(status_code=404, detail="Пользователь не вошёл: /login в боте")
        return profile

    return router
