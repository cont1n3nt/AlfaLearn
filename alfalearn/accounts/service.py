"""Account logic shared by the Telegram bot and the HTTP endpoints."""

from pydantic import BaseModel, Field

from alfalearn.accounts import demo_data
from alfalearn.accounts.passwords import verify_password
from alfalearn.accounts.store import AccountStore


class UserInfo(BaseModel):
    id: int
    login: str
    name: str
    access_note: str = Field(description="Короткое описание доступа демо-юзера")


class ProductInfo(BaseModel):
    id: str = Field(examples=["kurs_plus"])
    kind: str = Field(description="subscription — подписка, course — разовая покупка курса")
    title: str
    url: str = Field(description="Страница продукта на Альфа-Курсе")
    price_rub: int
    price_note: str
    owned: bool = Field(description="Есть ли продукт у юзера")


class CourseAccess(BaseModel):
    id: str
    title: str
    url: str = Field(description="Страница курса на Альфа-Курсе")
    description: str
    lessons: int
    duration: str
    product_id: str = Field(description="Продукт, который открывает курс")
    available: bool = Field(description="Открыт ли курс юзеру")


class AccessProfile(BaseModel):
    """Что юзеру доступно из платного. Для гостя `user` пуст и всё закрыто."""

    user: UserInfo | None
    products: list[ProductInfo]
    courses: list[CourseAccess]


class AccountService:
    def __init__(self, store: AccountStore):
        self.store = store

    def seed_demo(self) -> list[str]:
        return self.store.seed(demo_data.COURSES, demo_data.PRODUCTS, demo_data.DEMO_USERS)

    def reset_demo(self) -> list[str]:
        """Recreate demo users: drop their Telegram links and restore their access."""
        self.store.delete_users(user.login for user in demo_data.DEMO_USERS)
        return self.seed_demo()

    def login(self, telegram_id: int, login: str, password: str) -> AccessProfile | None:
        """Check the password and link the Telegram account. None for wrong credentials."""
        user = self.store.user_by_login(login.strip().lower())
        if user is None or not verify_password(password, user["password_hash"]):
            return None
        self.store.link_telegram(telegram_id, user["id"])
        return self.profile(telegram_id)

    def logout(self, telegram_id: int) -> bool:
        return self.store.unlink_telegram(telegram_id)

    def profile(self, telegram_id: int) -> AccessProfile:
        user = self.store.user_by_telegram(telegram_id)
        owned = self.store.user_product_ids(user["id"]) if user else set()
        return AccessProfile(
            user=UserInfo(**dict(user)) if user else None,
            products=[
                ProductInfo(**dict(row), owned=row["id"] in owned) for row in self.store.products()
            ],
            courses=[
                CourseAccess(**dict(row), available=row["product_id"] in owned)
                for row in self.store.paid_courses()
            ],
        )
