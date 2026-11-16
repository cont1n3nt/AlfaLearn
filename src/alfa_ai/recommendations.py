"""Rank a curated catalog of public cards with explicit access constraints."""

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import AIRequest, LearningProfile, Recommendation
from .reviewed import (
    CUSTOMER_CONFLICT,
    FEEDBACK,
    HIRING,
    LOCAL_DELIVERY,
    LOYALTY,
    PEST_ANALYSIS,
    PRINTER,
    PRODUCT_SKU,
    REMOTE_TEAM,
    REPEAT_SALES,
    RESALE,
    REVIEWED,
    SALON_RETURN,
    SERVICE_QUALITY,
)

TOPICS = {
    "retention": (
        "возврат клиентов",
        r"верн\w*\s+клиент\w*|возврат\w*\s+клиент\w*|спящ\w*\s+(?:баз|клиент)\w*",
    ),
    "support": (
        "работа поддержки",
        r"служб\w*\s+поддержк\w*|баз\w*\s+знани\w*|faq|скорост\w*\s+обслуживан\w*",
    ),
    "marketplaces": (
        "продажи на маркетплейсах",
        r"маркетплейс\w*|селлер\w*|wildberries|вайлдберриз|ozon|озон",
    ),
    "resale": ("перепродажа вещей", r"ресейл\w*|перепродаж\w*|винтаж\w*|секонд\W*хенд\w*"),
    "labels": (
        "печать этикеток",
        r"термопринтер\w*|термотрансфер\w*|риббон\w*|этикет\w*|принтер\w*",
    ),
    "loyalty": ("программа лояльности", r"лояльност\w*|бонус\w*|накоплен\w*\s+балл\w*"),
    "salon": ("работа салона", r"салон\w*|маникюр\w*"),
    "service": (
        "обслуживание клиентов",
        r"клиентск\w*\s+сервис\w*|обслуживан\w*\s+клиент\w*|отзыв\w*|чаев\w*|удерж\w*\s+клиент\w*|сервис\w*\s+в\s+салон\w*",
    ),
    "feedback": (
        "обратная связь сотрудникам",
        r"обратн\w*\s+связ\w*\s+сотрудник\w*|(?:по)?хвал\w*\s+сотрудник\w*|крити\w*\s+сотрудник\w*|ошибк\w*\s+подчиненн\w*",
    ),
    "cashflow": (
        "кассовый разрыв",
        r"кассов\w*\s+разрыв\w*|денег\s+на\s+счете\s+нет|нехватк\w*\s+денег\s+на\s+счете",
    ),
    "funding": (
        "поиск финансирования",
        r"финансирован\w*|деньги\s+на\s+(?:запуск|бизнес)|взять\s+деньги",
    ),
    "tax": ("учёт на УСН", r"усн|налог\w*\s+учет\w*"),
    "delivery": (
        "доставка по району",
        r"гиперлокальн\w*\s+доставк\w*|доставк\w*\s+(?:по|внутри|в\s+одном)\s+район\w*",
    ),
    "pest": ("PEST-анализ", r"pest|пест[\s-]анализ\w*"),
    "conflict": (
        "работа с жалобами клиентов",
        r"конфликт\w*\s+с\s+клиент\w*|жалоб\w*\s+(?:от\s+)?клиент\w*|недовольн\w*\s+(?:покупател|клиент)\w*",
    ),
    "remote": (
        "управление удалённой командой",
        r"удален\w*\s+(?:команд|сотрудник|работник)\w*|распределенн\w*\s+команд\w*|онбординг\w*",
    ),
    "hiring": (
        "поиск сотрудников",
        r"наем|найм\w*|рекрутинг\w*|(?:найти|искать|привлечь|подбор)\s+(?:нов\w*\s+)?(?:сотрудник|кандидат|персонал|работник)\w*",
    ),
    "sku": ("артикулы товаров", r"артикул\w*|sku"),
    "repeat_sales": ("повторные продажи", r"повторн\w*\s+(?:продаж|покуп)\w*|rfm|ltv|когорт\w*"),
}


@dataclass(frozen=True, slots=True)
class Card:
    material_id: str
    title: str
    kind: str
    access: str
    url: str
    benefit: str
    topics: tuple[str, ...]
    stage: str | None = None


def article(id_, title, benefit, *topics):
    return Card(id_, title, "article", "free", REVIEWED[id_][0], benefit, topics)


CATALOG = (
    article(
        SALON_RETURN,
        "Как вернуть клиентов в салон красоты",
        "Подскажет, как продумать предложение для тех, кто давно не записывался.",
        "retention",
        "salon",
    ),
    article(
        SERVICE_QUALITY,
        "Клиентский сервис: как улучшить качество обслуживания",
        "Поможет выбрать, с чего начать работу над скоростью обслуживания и базой знаний.",
        "support",
        "service",
    ),
    article(
        LOCAL_DELIVERY,
        "Гиперлокальная доставка: как работает и когда выгодна предпринимателю",
        "Поможет оценить спрос в районе и сравнить условия доставки у конкурентов.",
        "delivery",
    ),
    article(
        PEST_ANALYSIS,
        "Что такое PEST-анализ и как его провести",
        "Показывает, как оценить влияние внешних факторов и расставить приоритеты.",
        "pest",
    ),
    article(
        CUSTOMER_CONFLICT,
        "Точка кипения: как вести себя в конфликте с клиентом",
        "Поможет выстроить разговор с недовольным покупателем и согласовать решение.",
        "conflict",
    ),
    article(
        LOYALTY,
        "Как создать программу лояльности для клиентов салона красоты",
        "Поможет подобрать правила начисления бонусов и показатели для оценки программы.",
        "loyalty",
        "salon",
    ),
    article(
        RESALE,
        "Что такое ресейл и как начать заниматься им с нуля",
        "Поможет разобраться в форматах перепродажи и площадках.",
        "resale",
    ),
    article(
        PRINTER,
        "Как выбрать принтер этикеток для маркетплейсов",
        "Поможет сравнить типы печати для разных условий хранения и доставки.",
        "labels",
        "marketplaces",
    ),
    article(
        FEEDBACK,
        "Итоговая обратная связь: инструкция для руководителя",
        "Поможет подготовить разговор о работе сотрудника.",
        "feedback",
    ),
    article(
        REMOTE_TEAM,
        "Распределённая команда: что такое, как собрать и как управлять",
        "Поможет организовать адаптацию, коммуникацию и оценку работы удалённых сотрудников.",
        "remote",
    ),
    article(
        HIRING,
        "Как привлечь сотрудников в вашу компанию",
        "Поможет подобрать каналы поиска и продумать рекомендации кандидатов через коллег.",
        "hiring",
    ),
    article(
        PRODUCT_SKU,
        "Как придумать артикул и присвоить его товару",
        "Поможет выбрать структуру артикула и избежать дубликатов в ассортименте.",
        "sku",
        "marketplaces",
    ),
    article(
        REPEAT_SALES,
        "Как увеличить повторные продажи",
        "Поможет сгруппировать клиентов по покупкам и определить, когда они перестают возвращаться.",
        "repeat_sales",
        "retention",
    ),
    Card(
        "course:d0aa4598-3da2-478b-bbf8-bda81b231b58",
        "Маркетплейсы: с чего начать и как преуспеть",
        "course",
        "free",
        "https://kurs.alfabank.ru/courses/marketpleysy-s-chego-nachat-kak-preuspet/",
        "Подходит для знакомства с продажами на маркетплейсах и основами аналитики.",
        ("marketplaces",),
        "starting",
    ),
    Card(
        "course:55e903e5-a65f-40b0-83db-17fc050e565a",
        "Где взять деньги на бизнес",
        "course",
        "free",
        "https://kurs.alfabank.ru/courses/gde-vzyat-dengi-na-biznes/",
        "Поможет изучить варианты финансирования бизнеса.",
        ("funding",),
        "starting",
    ),
    Card(
        "podcast:3ec43e4d-43b5-4b5e-8b9a-9f6d0b5ffde6",
        "Сервис в малом бизнесе: как удержать клиентов и заработать больше",
        "podcast",
        "free",
        "https://kurs.alfabank.ru/podcasts/servis-v-malom-biznese-kak-uderzhat-klientov-i-zarabotat-bolshe/",
        "В выпуске разбирают улучшение обслуживания и сбор отзывов.",
        ("service", "salon"),
        "running",
    ),
    Card(
        "podcast:296e7be7-e4ff-4ed9-9302-7f3dacad8d8b",
        "Кассовый разрыв — не приговор: как управлять финансами бизнеса",
        "podcast",
        "subscription",
        "https://kurs.alfabank.ru/podcasts/kassovyj-razryv-ne-prigovor-kak-upravlyat-finansami-biznesa/",
        "Выпуск объясняет признаки и причины кассового разрыва.",
        ("cashflow",),
        "running",
    ),
    Card(
        "course:28e2a868-0b58-4fd6-af4d-7aa268ae91b4",
        "УСН на маркетплейсах: особенности налогового учёта",
        "course",
        "paid",
        "https://kurs.alfabank.ru/courses/usn-na-marketplejsah-osobennosti-nalogovogo-uchyota/",
        "Курс посвящён налоговому учёту на УСН у продавцов маркетплейсов.",
        ("tax", "marketplaces"),
        "running",
    ),
)

CATALOG_SHA256 = "10d9b7b96fd02cd382b56a7ac62b71ddec66971684198a74db698693093f6fc3"


def verify_catalog() -> None:
    payload = json.dumps(
        [asdict(card) for card in CATALOG],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if sha256(payload).hexdigest() != CATALOG_SHA256:
        raise ValueError("Проверенный каталог материалов изменён.")


verify_catalog()


TOPIC_PATTERNS = {key: re.compile(rf"\b(?:{pattern})\b") for key, (_, pattern) in TOPICS.items()}
TOPIC_TYPO_STEMS = {
    "retention": ("возврат",),
    "support": ("поддержк",),
    "marketplaces": ("маркетплейс", "вайлдберриз"),
    "resale": ("ресейл", "перепродаж"),
    "labels": ("термопринтер", "термотрансфер", "этикет"),
    "loyalty": ("лояльност", "бонус"),
    "service": ("обслуживан",),
    "feedback": ("обратн",),
    "cashflow": ("кассов",),
    "funding": ("финансирован",),
    "delivery": ("гиперлокальн",),
    "conflict": ("конфликт", "жалоб"),
    "remote": ("распределенн", "онбординг"),
    "hiring": ("рекрутинг",),
    "repeat_sales": ("когорт",),
}
FREE_ONLY = re.compile(r"\b(?:бесплатн\w*|без\s+оплаты|не\s+предлагай\s+платн\w*)\b")
MIXED_ACCESS = re.compile(
    r"\b(?:не\s+только\s+бесплатн\w*|платн\w*\s+и\s+бесплатн\w*|бесплатн\w*\s+и\s+платн\w*)\b"
)
NO_SUBSCRIPTION = re.compile(r"\b(?:без\s+подписки|не\s+хочу\s+подписку)\b")
FORMAT_ONLY = re.compile(r"\bтолько\s+(подкаст|курс|стать)\w*\b")
FORMAT_NAMES = {"подкаст": "podcast", "курс": "course", "стать": "article"}
MATERIAL_LOOKUP = re.compile(
    r"(?:где (?:найти|посмотреть)|есть ли|есть|посоветуй(?:те)?|подбери(?:те)?)\s+"
    r"(?:(?:мне|только)\s+)*(?:(?P<access>бесплатн\w*|платн\w*)\s+)?"
    r"(?:(?P<media>подкаст|курс)ы?|(?P<article>стать)[юияи])"
    r"(?:\s+(?:по|о|об|про|для|без)\s+[^?!.\n]+)?[?!.]?"
)
ADDITIONAL_QUESTION = re.compile(r"\bи\s+(?:как|что|где|почему|сколько|когда)\b")


def lookup_requested_materials(request: AIRequest):
    """Handle a single explicit catalog request without asking the model."""
    query = request.query.casefold().replace("ё", "е")
    match = MATERIAL_LOOKUP.fullmatch(query)
    if match is None or ADDITIONAL_QUESTION.search(query):
        return None
    media = match["media"] or match["article"]
    paid = bool(match["access"] and match["access"].startswith("платн"))
    return recommend_materials(request, required_format=FORMAT_NAMES[media], paid_only=paid)


def recommend_materials(
    request: AIRequest, *, excluded_ids=(), required_format=None, paid_only=False
):
    query = request.query.casefold().replace("ё", "е")
    profile = request.profile or LearningProfile()
    free_only = profile.free_only or bool(
        FREE_ONLY.search(query) and not MIXED_ACCESS.search(query)
    )
    no_subscription = bool(NO_SUBSCRIPTION.search(query))
    format_match = FORMAT_ONLY.search(query)
    if format_match:
        required_format = FORMAT_NAMES[format_match.group(1)]
    matched = {
        key for key, pattern in TOPIC_PATTERNS.items()
        if pattern.search(query) or _has_one_edit_topic_typo(query, key)
    }
    candidates = []
    for card in CATALOG:
        if card.material_id in excluded_ids or (free_only and card.access != "free"):
            continue
        if paid_only and card.access == "free":
            continue
        if no_subscription and card.access == "subscription":
            continue
        if required_format and card.kind != required_format:
            continue
        topics = [key for key in card.topics if key in matched]
        if card.topics[0] not in matched:
            continue
        if "tax" in matched and ("tax" not in card.topics or "marketplaces" not in matched):
            continue
        rank = (
            -len(topics),
            card.kind != profile.preferred_format,
            profile.stage is None or card.stage != profile.stage,
            profile.industry != "salon" or "salon" not in card.topics,
            card.access != "free",
            card.material_id,
        )
        subject = ", ".join(TOPICS[key][0] for key in topics)
        candidates.append(
            (
                rank,
                Recommendation(
                    card.material_id,
                    card.title,
                    card.kind,
                    card.access,
                    card.url,
                    f"По теме «{subject}». {card.benefit}",
                ),
            )
        )
    return tuple(item for _, item in sorted(candidates, key=lambda row: row[0])[:2])


def _has_one_edit_topic_typo(query: str, topic: str) -> bool:
    """Tolerate one typo in distinctive catalog terms, without fuzzy free text."""
    words = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
    return any(
        _prefix_within_one_edit(word, stem)
        for word in words
        for stem in TOPIC_TYPO_STEMS.get(topic, ())
    )


def _prefix_within_one_edit(word: str, stem: str) -> bool:
    if len(word) < len(stem) - 1:
        return False
    return any(
        _within_one_edit(word[:size], stem)
        for size in (len(stem) - 1, len(stem), len(stem) + 1)
        if size <= len(word)
    )


def _within_one_edit(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        mismatches = [
            index for index, pair in enumerate(zip(left, right))
            if pair[0] != pair[1]
        ]
        if len(mismatches) <= 1:
            return True
        return (
            len(mismatches) == 2
            and mismatches[1] == mismatches[0] + 1
            and left[mismatches[0]] == right[mismatches[1]]
            and left[mismatches[1]] == right[mismatches[0]]
        )
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index = 0
    while index < len(shorter) and shorter[index] == longer[index]:
        index += 1
    return shorter[index:] == longer[index + 1:]
