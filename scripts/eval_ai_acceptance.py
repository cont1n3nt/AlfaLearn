"""Check the installed AIService contract and save the outcomes of local scenarios."""

import argparse
import asyncio
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from alfa_ai import AIRequest, AIService, LearningProfile, Settings
from alfa_ai.reviewed import FEEDBACK, SALON_RETURN
from scripts.corpus import LOCAL_DATA, write_jsonl

NOW = datetime.fromisoformat("2026-11-14T12:00:00+03:00")
ZONE = "Europe/Moscow"
FEEDBACK_QUERY = "Нужно обсудить ошибку подчинённого: наедине или при коллегах?"

CASES = (
    ("grounded_answer", AIRequest(FEEDBACK_QUERY), {"status": "answer", "document_id": FEEDBACK}),
    (
        "unknown_fact",
        AIRequest("Какой точный процент бонусов начислять клиентам салона за покупку?"),
        {"status": "no_answer"},
    ),
    ("ambiguous", AIRequest("Как это оформить?"), {"status": "clarify"}),
    (
        "follow_up",
        AIRequest(
            "А для моего салона?",
            previous_query="Как вернуть клиентов, которые давно не приходили?",
        ),
        {"status": "answer", "document_id": SALON_RETURN, "profile_patch": {"industry": "salon"}},
    ),
    ("without_history", AIRequest("А для моего салона?"), {"status": "clarify"}),
    (
        "business_onboarding",
        AIRequest("Управляю своей кофейней."),
        {"status": "clarify", "profile_patch": {"industry": "cafe", "stage": "running"}},
    ),
    (
        "topic_switch",
        AIRequest(
            FEEDBACK_QUERY,
            LearningProfile(industry="salon"),
            previous_query="Как вернуть клиентов салона?",
        ),
        {"status": "answer", "document_id": FEEDBACK, "profile_patch": {}},
    ),
    (
        "paid_card",
        AIRequest("Где найти курс по УСН на маркетплейсах?"),
        {"status": "no_answer", "recommendation_access": ["paid"]},
    ),
    (
        "free_only",
        AIRequest("Что послушать про кассовый разрыв?", LearningProfile(free_only=True)),
        {"recommendation_access": []},
    ),
    (
        "missing_schedule",
        AIRequest("Напомни пройти курс", now=NOW),
        {"status": "clarify", "action": {"missing_fields": ["timezone", "time", "date"]}},
    ),
    (
        "time_follow_up",
        AIRequest("18:00", previous_query="Напомни завтра пройти курс", now=NOW, timezone=ZONE),
        {
            "status": "action_proposal",
            "action": {
                "action": "create_reminder",
                "repeat": "once",
                "local_datetime": "2026-11-15T18:00",
                "needs_confirmation": True,
            },
        },
    ),
    (
        "weekly",
        AIRequest("Напоминай каждый понедельник в 09:00 учиться", now=NOW, timezone=ZONE),
        {
            "status": "action_proposal",
            "action": {"repeat": "weekly", "weekday": 0, "local_datetime": "2026-11-16T09:00"},
        },
    ),
    (
        "multiple_weekdays",
        AIRequest("Напоминай по понедельникам и средам в 09:00 учиться", now=NOW, timezone=ZONE),
        {
            "status": "clarify",
            "action": {
                "repeat": None,
                "local_datetime": None,
                "missing_fields": ["repeat", "date"],
            },
        },
    ),
    (
        "change",
        AIRequest("Измени напоминание #r42, повторяй ежедневно", now=NOW, timezone=ZONE),
        {
            "status": "action_proposal",
            "action": {"action": "update_reminder", "reminder_id": "r42", "repeat": "daily"},
        },
    ),
    (
        "cancel",
        AIRequest("Отмени напоминание #r42", now=NOW, timezone=ZONE),
        {
            "status": "action_proposal",
            "action": {
                "action": "cancel_reminder",
                "reminder_id": "r42",
                "needs_confirmation": True,
            },
        },
    ),
    (
        "tax_missing_date",
        AIRequest("Напомни заплатить налог", now=NOW, timezone=ZONE),
        {
            "status": "clarify",
            "action": {"local_datetime": None, "missing_fields": ["time", "date"]},
        },
    ),
)


def verify(result, expected):
    failures = []
    if result["status"] == "error" or (
        "status" in expected and result["status"] != expected["status"]
    ):
        failures.append("status")
    if "document_id" in expected and not any(
        source["document_id"] == expected["document_id"] for source in result["sources"]
    ):
        failures.append("source")
    if "profile_patch" in expected and result["profile_patch"] != expected["profile_patch"]:
        failures.append("profile_patch")
    if (
        "recommendation_access" in expected
        and [card["access"] for card in result["recommendations"]]
        != expected["recommendation_access"]
    ):
        failures.append("recommendations")
    action = result.get("proposed_action") or {}
    for field, value in expected.get("action", {}).items():
        actual = action.get(field)
        if field == "missing_fields":
            actual = list(actual) if actual is not None else None
        if actual != value:
            failures.append(f"proposed_action.{field}")
    return failures


async def evaluate(output):
    rows = []
    settings = Settings.from_env()
    metadata = {
        "run": {
            "model": settings.model,
            "model_digest": settings.model_digest,
            "corpus_sha256": sha256(settings.corpus_path.read_bytes()).hexdigest(),
            "podcast_corpus_sha256": settings.podcast_corpus_sha256,
            "cases": len(CASES),
        }
    }
    async with await AIService.create(settings) as service:
        for case_id, request, expected in CASES:
            started = perf_counter()
            result = (await service.answer(request)).to_dict()
            failures = verify(result, expected)
            rows.append(
                {
                    "case": case_id,
                    "query": request.query,
                    "expected": expected,
                    "result": result,
                    "failures": failures,
                    "seconds": round(perf_counter() - started, 3),
                }
            )
            write_jsonl(output, [metadata, *rows])
            verdict = f"FAIL ({', '.join(failures)})" if failures else "PASS"
            print(f"{case_id}: {verdict}", flush=True)
    passed = sum(not row["failures"] for row in rows)
    print(f"{passed}/{len(rows)} проверок API. Отчёт — {output}")
    return int(passed != len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LOCAL_DATA / "ai-acceptance.jsonl")
    return asyncio.run(evaluate(parser.parse_args().output))


if __name__ == "__main__":
    raise SystemExit(main())
