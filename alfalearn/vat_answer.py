"""Answer to VAT questions like "сколько НДС заплатить с 100 тысяч?".

Facts and formulas come from the "Калькулятор НДС" page on kurs.alfabank.ru
(and the tax regime calculator page for the "don't know my regime" branch).
Texts use Telegram HTML formatting.
"""

import re
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

SITE = "https://kurs.alfabank.ru"
VAT_CALCULATOR_URL = f"{SITE}/calculators/nds/"
TAX_REGIME_CALCULATOR_URL = f"{SITE}/calculators/kalkulyator-sistem-nalogooblozheniya/"
USN_OSN_ARTICLE_URL = f"{SITE}/articles/ne-nalomat-drov-perekhodim-s-usn-na-osn-i-obratno/"
MARKETPLACE_COURSE_URL = f"{SITE}/courses/usn-na-marketplejsah-osobennosti-nalogovogo-uchyota/"

NBSP = " "
KOPECK = Decimal("0.01")
MAX_AMOUNT = Decimal("1000000000000")
BASE_RATE = 22


class TaxRegime(StrEnum):
    USN = "usn"
    OSN = "osn"
    UNKNOWN = "unknown"


# --- Recognising the question -------------------------------------------------

_VAT_WORDS = re.compile(r"(?<![а-яёa-z])ндс(?![а-яёa-z])|налог\w*\s+на\s+добавленн\w*\s+стоимость", re.IGNORECASE)
_AMOUNT = re.compile(
    r"(?<![\d.,])(?P<number>\d{1,3}(?:[   ]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"\s*(?P<suffix>млрд\.?|миллиард(?:а|ов)?|млн\.?|миллион(?:а|ов)?|тыс(?:яч[аи]?)?\.?|т\.?\s?р\.?|к|k)?"
    r"(?![а-яёa-z\d])(?P<tail>\s*(?:%|процент|год|г\.))?",
    re.IGNORECASE,
)
_PREPOSITION = re.compile(r"(?:^|[^а-яё])(?:с|со|из|от|на|за)\s*$", re.IGNORECASE)
_MULTIPLIERS = (("млрд", 10**9), ("миллиард", 10**9), ("млн", 10**6), ("миллион", 10**6),
                ("тыс", 10**3), ("т", 10**3), ("к", 10**3), ("k", 10**3))


def is_vat_question(text: str | None) -> bool:
    return bool(text and _VAT_WORDS.search(text))


def extract_amount(text: str) -> Decimal | None:
    """Find the ruble amount in a question: "с 100 тысяч", "87 500 ₽", "1,5 млн".

    Percentages and years are skipped; a number after "с/из/от/на/за" wins.
    """
    candidates: list[tuple[bool, Decimal]] = []
    for match in _AMOUNT.finditer(text):
        if match["tail"]:
            continue
        number = re.sub(r"[   ]", "", match["number"]).replace(",", ".")
        value = Decimal(number)
        suffix = (match["suffix"] or "").lower()
        for prefix, multiplier in _MULTIPLIERS:
            if suffix.startswith(prefix):
                value *= multiplier
                break
        if 0 < value <= MAX_AMOUNT:
            after_preposition = bool(_PREPOSITION.search(text[:match.start()]))
            candidates.append((after_preposition, value.quantize(KOPECK, rounding=ROUND_HALF_UP)))
    if not candidates:
        return None
    preferred = [value for after_preposition, value in candidates if after_preposition]
    return (preferred or [value for _, value in candidates])[0]


# --- Numbers -------------------------------------------------------------------

def vat_on_top(amount: Decimal, rate: int) -> Decimal:
    """НДС сверху: Сумма × Ставка / 100."""
    return (amount * rate / 100).quantize(KOPECK, rounding=ROUND_HALF_UP)


def vat_inside(amount: Decimal, rate: int) -> Decimal:
    """НДС внутри суммы: Сумма × Ставка / (100 + Ставка)."""
    return (amount * rate / (100 + rate)).quantize(KOPECK, rounding=ROUND_HALF_UP)


def format_number(value: Decimal) -> str:
    whole, fraction = f"{value:,.2f}".split(".")
    whole = whole.replace(",", NBSP)
    return whole if fraction == "00" else f"{whole},{fraction}"


def rub(value: Decimal) -> str:
    return f"{format_number(value)}{NBSP}₽"


# --- Texts ---------------------------------------------------------------------

SOURCE_LINE = f'📎 Источник: <a href="{VAT_CALCULATOR_URL}">«Калькулятор НДС», Альфа-Курс</a>'
MORE_TO_READ = (
    "💡 <b>Что ещё посмотреть:</b>\n"
    f'• <a href="{USN_OSN_ARTICLE_URL}">Как перейти с УСН на ОСН и обратно</a> — статья, бесплатно\n'
    f'• <a href="{MARKETPLACE_COURSE_URL}">УСН на маркетплейсах: особенности налогового учёта</a> — '
    "платный курс, 3990 ₽. Поможет, если вы продаёте на маркетплейсах и разбираетесь с налогами на УСН."
)
ASK_AMOUNT = "Напишите сумму — посчитаю, например: «НДС с 100 тысяч»."


def _amount_section(amount: Decimal | None) -> str:
    if amount is None:
        return (
            "<b>3. Сумма указана с НДС или без?</b>\n"
            f"Для ставки {BASE_RATE}%:\n"
            f"• НДС сверху: Сумма × {BASE_RATE} / 100.\n"
            f"• НДС уже внутри суммы: Сумма × {BASE_RATE} / {100 + BASE_RATE}.\n"
            f"{ASK_AMOUNT}"
        )
    on_top = vat_on_top(amount, BASE_RATE)
    return (
        f"<b>3. {format_number(amount)}{NBSP}₽ — это сумма с НДС или без?</b>\n"
        f"Для ставки {BASE_RATE}%:\n"
        f"• НДС сверху: {format_number(amount)} × {BASE_RATE} / 100 = {rub(on_top)}, "
        f"итого к оплате {rub(amount + on_top)}.\n"
        f"• НДС уже внутри суммы: {format_number(amount)} × {BASE_RATE} / {100 + BASE_RATE} = "
        f"{rub(vat_inside(amount, BASE_RATE))}."
    )


def build_vat_answer(amount: Decimal | None) -> str:
    return "\n\n".join([
        "Сумма НДС зависит от трёх вещей. Давайте уточню, чтобы не ошибиться 👇",
        "<b>1. Вы вообще платите НДС?</b>\n"
        "Платят компании и ИП на ОСН и ЕСХН, а также бизнес на УСН с доходом от 20 млн ₽ в год. "
        "Если вы на УСН и доход меньше 20 млн ₽, НДС не начисляется, "
        "и заявление на освобождение подавать не нужно.",
        "<b>2. Какая у вас ставка?</b>\n"
        "• Основная ставка с 2026 года — 22% (до 2026 года была 20%).\n"
        "• 10% — продукты питания, книги, детские и медицинские товары.\n"
        "• Для УСН есть пониженные ставки: 5% при доходе 20–272,5 млн ₽ "
        "и 7% при доходе 272,5–490,5 млн ₽.",
        _amount_section(amount),
        SOURCE_LINE,
        MORE_TO_READ,
    ])


def _rates_section(amount: Decimal | None, rates: tuple[int, ...]) -> str:
    if amount is None:
        return ASK_AMOUNT
    lines = [f"<b>Для {rub(amount)}:</b>"]
    for rate in rates:
        on_top = vat_on_top(amount, rate)
        lines.append(
            f"• {rate}%: сверху {rub(on_top)} (итого {rub(amount + on_top)}), "
            f"внутри суммы {rub(vat_inside(amount, rate))}"
        )
    return "\n".join(lines)


def build_regime_answer(regime: TaxRegime, amount: Decimal | None) -> str:
    if regime is TaxRegime.USN:
        return "\n\n".join([
            "<b>НДС на УСН зависит от дохода за год:</b>\n"
            "• до 20 млн ₽ — НДС не платите;\n"
            "• 20–272,5 млн ₽ — ставка 5%;\n"
            "• 272,5–490,5 млн ₽ — ставка 7%;\n"
            "• больше 490,5 млн ₽ — право на УСН теряется, налоги платят по ОСН "
            "с начала квартала, в котором превышен лимит.",
            "Можно платить НДС и по стандартной ставке 22%. Бизнес на УСН «Доходы минус расходы» "
            "тогда имеет право на вычет входящего НДС. Вернуться с пониженной ставки на стандартную "
            "можно не раньше чем через 4 квартала, а если пониженную ставку уже применяли — через 12.",
            _rates_section(amount, (5, 7)),
            SOURCE_LINE,
        ])
    if regime is TaxRegime.OSN:
        return "\n\n".join([
            "<b>На ОСН и ЕСХН НДС платят по общим ставкам:</b>\n"
            "• 22% — основная ставка с 2026 года;\n"
            "• 10% — продукты питания, книги, детские и медицинские товары;\n"
            "• 0% — экспорт и международные грузоперевозки.",
            _rates_section(amount, (22, 10)),
            SOURCE_LINE,
        ])
    return "\n\n".join([
        "<b>Как понять свой режим</b>\n"
        "Если при регистрации бизнеса или в течение 30 дней после неё вы не подавали заявление "
        "на УСН, патент или АУСН — вы на ОСН: она применяется по умолчанию.\n"
        "Если подавали заявление на УСН — вы на упрощёнке, и НДС зависит от дохода за год.",
        f'Подобрать подходящий режим поможет <a href="{TAX_REGIME_CALCULATOR_URL}">калькулятор '
        "выбора системы налогообложения</a>. Результат носит рекомендательный характер — "
        "перед окончательным выбором лучше посоветоваться с бухгалтером.",
        "Выберите свой режим, и я покажу расчёт:",
    ])
