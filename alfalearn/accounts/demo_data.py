"""Demo catalog and fictional demo users.

Courses, prices and page addresses come from the Alfa-Kurs database dump: only public
card fields, no lesson content. The users are invented and exist only for the demo;
their passwords are published in README on purpose so reviewers can log in.
"""

from dataclasses import dataclass

SITE = "https://kurs.alfabank.ru"


@dataclass(frozen=True)
class Course:
    id: str  # Courses.Id in the dump
    title: str
    slug: str
    description: str
    lessons: int
    duration: str

    @property
    def url(self) -> str:
        return f"{SITE}/courses/{self.slug}/"


@dataclass(frozen=True)
class Product:
    id: str
    kind: str  # "subscription" or "course"
    title: str
    url: str
    price_rub: int
    price_note: str
    course_ids: tuple[str, ...]


@dataclass(frozen=True)
class DemoUser:
    login: str
    password: str
    name: str
    access_note: str
    product_ids: tuple[str, ...]


MARKETING = Course(
    "8be9c4dc-d301-4af3-9548-0a091b8cf441", "Интернет-маркетинг: разбираемся с экспертами",
    "internet-marketing-dlya-malogo-biznesa", "Как продвигать свои товары и услуги в интернете",
    17, "2 часа",
)
GOOGLE_SHEETS = Course(
    "b3b52649-3f00-4c66-9d0f-8def7c0391b5", "Google Таблицы для бизнеса: упрощаем работу с данными",
    "google-tablicy-dlya-biznesa-uproshchaem-rabotu-s-dannymi",
    "Ваши таблицы могут больше: автоматизируйте рутинные задачи бизнеса в Google Таблицах",
    24, "2 часа",
)
FINANCE_BASICS = Course(
    "b2dea4ba-b84b-4b07-9f81-f2fcc2cc6f58", "Азы финансовой грамотности", "azy-finansovoj-gramotnosti",
    "Курс по основам личных финансов: от защиты сбережений до понимания принципов налогообложения",
    8, "1 час 25 минут",
)
TEAM_MANAGEMENT = Course(
    "a25c8282-94c7-458e-abf6-9fbd7b996ef7", "Как управлять командой", "kak-upravlyat-komandoi",
    "Команду собрали, а дальше? Расскажем, как сделать, чтобы сотрудники были довольны, "
    "а бизнес развивался",
    5, "25 минут",
)
MARKETPLACE_USN = Course(
    "28e2a868-0b58-4fd6-af4d-7aa268ae91b4", "УСН на маркетплейсах: особенности налогового учёта",
    "usn-na-marketplejsah-osobennosti-nalogovogo-uchyota",
    "Узнайте, как вести учёт продаж на маркетплейсах и не допускать ошибок при расчёте налогов",
    7, "4 часа",
)

COURSES = (MARKETING, GOOGLE_SHEETS, FINANCE_BASICS, TEAM_MANAGEMENT, MARKETPLACE_USN)

KURS_PLUS = Product(
    "kurs_plus", "subscription", "Курс+", f"{SITE}/subscription/", 990,
    "990 ₽ в месяц вместо 2990 ₽",
    (MARKETING.id, GOOGLE_SHEETS.id, FINANCE_BASICS.id, TEAM_MANAGEMENT.id),
)
USN_COURSE = Product(
    "usn_marketplaces", "course", MARKETPLACE_USN.title, MARKETPLACE_USN.url, 3990,
    "3990 ₽, разовая покупка", (MARKETPLACE_USN.id,),
)

PRODUCTS = (KURS_PLUS, USN_COURSE)

DEMO_USERS = (
    DemoUser("anna", "Anna-demo-2026", "Анна", "без платного доступа", ()),
    DemoUser("boris", "Boris-demo-2026", "Борис", "одна покупка: курс «УСН на маркетплейсах»",
             (USN_COURSE.id,)),
    DemoUser("vera", "Vera-demo-2026", "Вера", "все продукты: подписка «Курс+» и курс «УСН на маркетплейсах»",
             (KURS_PLUS.id, USN_COURSE.id)),
)
