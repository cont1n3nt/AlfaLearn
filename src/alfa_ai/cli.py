"""Command-line interface for AIService and catalog lookup."""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import datetime

from .contracts import AIRequest, LearningProfile
from .dialogue import resolve_dialogue
from .recommendations import recommend_materials
from .service import AIService
from .settings import Settings


async def run(requests: list[AIRequest]) -> int:
    async with await AIService.create(Settings.from_env()) as service:
        failed = False
        for request in requests:
            response = await service.answer(request)
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2), flush=True)
            failed |= response.status == "error"
        return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument(
        "--recommend-only",
        action="store_true",
        help="Show related catalog cards without starting RAG.",
    )
    parser.add_argument("--stage", choices=("starting", "running"))
    parser.add_argument("--prefer-format", choices=("article", "course", "podcast"))
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--industry", choices=("salon", "cafe", "retail"))
    parser.add_argument(
        "--previous-query", help="Previous substantive question for a short follow-up."
    )
    parser.add_argument("--timezone", help="Explicit IANA timezone for reminder proposals.")
    parser.add_argument(
        "--now", help="Override the clock for replay, e.g. 2026-11-12T12:00:00+03:00."
    )
    args = parser.parse_args()
    try:
        profile = LearningProfile(
            stage=args.stage,
            preferred_format=args.prefer_format,
            free_only=args.free_only or None,
            industry=args.industry,
        )
        now = datetime.fromisoformat(args.now) if args.now else None
        requests = [
            AIRequest(
                query=query,
                profile=profile,
                previous_query=args.previous_query,
                now=now,
                timezone=args.timezone,
            )
            for query in args.query
        ]
    except ValueError as exc:
        parser.error(str(exc))
    if args.recommend_only:
        for request in requests:
            effective_request, profile_patch, clarification = resolve_dialogue(request)
            cards = () if clarification else recommend_materials(effective_request)
            patch_fields = asdict(profile_patch) if profile_patch is not None else {}
            payload = {
                "recommendations": [asdict(card) for card in cards],
                "profile_patch": {
                    key: value for key, value in patch_fields.items() if value is not None
                },
            }
            if clarification:
                payload.update(status="clarify", text=clarification)
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    try:
        return asyncio.run(run(requests))
    except (OSError, ValueError, KeyError, TypeError):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "startup_error",
                    "text": "Не удалось запустить AI-модуль. Проверьте настройки и файл корпуса.",
                    "sources": [],
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
