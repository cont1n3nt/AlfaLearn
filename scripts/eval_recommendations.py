"""Run a small labeled evaluation of the standalone recommender."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from alfa_ai import AIRequest, LearningProfile, recommend_materials
from alfa_ai.reviewed import (
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
    SALON_RETURN,
    SERVICE_QUALITY,
)
from scripts.corpus import write_jsonl

MARKETPLACE = "course:d0aa4598-3da2-478b-bbf8-bda81b231b58"
FUNDING = "course:55e903e5-a65f-40b0-83db-17fc050e565a"
SERVICE = "podcast:3ec43e4d-43b5-4b5e-8b9a-9f6d0b5ffde6"
CASHFLOW = "podcast:296e7be7-e4ff-4ed9-9302-7f3dacad8d8b"
TAX = "course:28e2a868-0b58-4fd6-af4d-7aa268ae91b4"

CASES = (
    (
        "market-start",
        "Выхожу на Ozon впервые, с чего учиться?",
        MARKETPLACE,
        LearningProfile(stage="starting"),
    ),
    ("label", "Надо печатать этикетки для коробок на складе.", PRINTER, None),
    ("secondhand", "Хочу перепродавать винтажные вещи", RESALE, None),
    ("feedback", "Как хвалить сотрудника в общем чате?", FEEDBACK, None),
    ("funding", "Где искать деньги на запуск своей мастерской?", FUNDING, None),
    ("loyalty", "В салоне запускаем бонусную программу", LOYALTY, None),
    ("service", "Посоветуй только подкаст про сбор отзывов", SERVICE, None),
    ("paid", "Есть курс по УСН для селлеров на Ozon?", TAX, None),
    ("subscription", "Что послушать про кассовый разрыв?", CASHFLOW, None),
    ("restricted-text", "Хочу бесплатный курс по УСН на маркетплейсах", None, None),
    ("restricted-profile", "Подкаст про кассовые разрывы", None, LearningProfile(free_only=True)),
    ("format-filter", "Только подкаст про этикетки", None, None),
    ("irrelevant", "Сколько спутников у Марса?", None, None),
    ("industry-only", "У меня салон", None, None),
    (
        "prefer-podcast",
        "В салоне хочу наладить отзывы и бонусы",
        SERVICE,
        LearningProfile(preferred_format="podcast"),
    ),
    (
        "local-delivery",
        "Что почитать перед запуском доставки в одном районе?",
        LOCAL_DELIVERY,
        None,
    ),
    ("strategy", "Нужна бесплатная статья по PEST-анализу", PEST_ANALYSIS, None),
    ("complaint", "Есть статья про общение с недовольным покупателем?", CUSTOMER_CONFLICT, None),
    ("delivery-format", "Нужен только подкаст про доставку по району", None, None),
    ("delivery-general", "Сколько стоит международная доставка?", None, None),
    ("remote", "Хочу организовать онбординг сотрудников на удалёнке", REMOTE_TEAM, None),
    ("recruitment", "Где искать новых кандидатов в свою компанию?", HIRING, None),
    ("sku", "Как присваивать артикулы товарам на Ozon?", PRODUCT_SKU, None),
    (
        "cohort",
        "Посоветуйте бесплатный материал по когортам и повторным покупкам",
        REPEAT_SALES,
        None,
    ),
)

TRANSFER_CASES = (
    (
        "dormant",
        "Что почитать про спящую базу салона, если у меня мало времени?",
        SALON_RETURN,
        LearningProfile(industry="salon", preferred_format="article"),
    ),
    (
        "feedback-video",
        "Только подкаст о том, как хвалить сотрудника",
        None,
        LearningProfile(stage="running"),
    ),
    (
        "repeat-shop",
        "В моём магазине хочу понять, какие клиенты покупают повторно: есть материал про RFM?",
        REPEAT_SALES,
        LearningProfile(industry="retail"),
    ),
    (
        "free-podcast",
        "Есть что послушать про кассовые разрывы без подписки?",
        None,
        LearningProfile(preferred_format="podcast"),
    ),
    (
        "pay-once",
        "Учу налоговый учёт на маркетплейсе. Есть курс без подписки?",
        TAX,
        LearningProfile(stage="running", preferred_format="course"),
    ),
    (
        "specific-format",
        "Только курс об артикулах товаров",
        None,
        LearningProfile(preferred_format="course"),
    ),
    (
        "wrong-domain",
        "Как найти размер иглы для швейной машины?",
        None,
        LearningProfile(industry="retail"),
    ),
    (
        "read-service",
        "Хочу материал, чтобы настроить базу знаний поддержки",
        SERVICE_QUALITY,
        LearningProfile(preferred_format="article"),
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("dev", "transfer"), default="dev")
    args = parser.parse_args()
    results = []
    cases = TRANSFER_CASES if args.split == "transfer" else CASES
    for id_, question, expected, profile in cases:
        started = perf_counter()
        cards = recommend_materials(AIRequest(question, profile))
        milliseconds = round((perf_counter() - started) * 1000, 3)
        actual = cards[0].material_id if cards else None
        results.append(
            {
                "id": id_,
                "expected_top1": expected,
                "actual_top1": actual,
                "pass": actual == expected,
                "milliseconds": milliseconds,
                "recommendations": [asdict(card) for card in cards],
            }
        )
        print(f"{id_}: {'PASS' if actual == expected else 'FAIL'}")
    if args.output:
        write_jsonl(args.output, results)
    passed = sum(row["pass"] for row in results)
    print(
        f"{passed}/{len(results)} проверок. Макс. {max(row['milliseconds'] for row in results):.3f} мс на подбор."
    )
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
