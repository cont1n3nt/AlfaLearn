"""Resolve explicit short follow-ups without storing user dialogue in the service."""

import re
from dataclasses import replace

from .contracts import AIRequest, LearningProfile

INDUSTRY_TERMS = {"салон": "salon", "кофейн": "cafe", "магазин": "retail"}
INDUSTRY_LABELS = {"salon": "салон красоты", "cafe": "кофейня", "retail": "магазин"}
SELF_INDUSTRY = re.compile(
    r"\b(?:у меня|у нас|веду|управляю|открываю|запускаю)\s+"
    r"(?:(?:свой|свою|своим|своей)\s+)?(салон|кофейн|магазин)\w*\b"
)
FOLLOW_UP = re.compile(
    r"(?:(?:а\s+)?(?:как|что)\s+(?:у меня|в моем случае)|(?:а\s+)?для\s+(?:меня|(?:моего|моей|нашего|нашей)\s+(?:салон|кофейн|магазин)\w*))\s*[?!.]*"
)
DECLARATION = re.compile(
    r"(?:у меня|у нас|(?:я\s+)?(?:веду|управляю|открываю|запускаю))\s+"
    r"(?:(?:свой|свою|своим|своей)\s+)?(?:салон\w*(?: красоты)?|кофейн\w*|магазин\w*)\s*[.!]*"
)
FORMATS = {"подкаст": "podcast", "стать": "article", "курс": "course"}
FORMAT_PREFERENCE = re.compile(
    r"\b(?:я предпочитаю|мне удобнее)\s+(?:(?:слушать|читать|проходить)\s+)?(подкаст|стать|курс)\w*\b"
)


def resolve_dialogue(request: AIRequest) -> tuple[AIRequest, LearningProfile | None, str | None]:
    """Return the effective request, explicit profile changes, and any clarification."""
    query = request.query.casefold().replace("ё", "е")
    follow_up = FOLLOW_UP.fullmatch(query)
    explicit = SELF_INDUSTRY.search(query)
    changes = {}
    if explicit:
        changes["industry"] = INDUSTRY_TERMS[explicit.group(1)]
    elif follow_up:
        for stem, industry in INDUSTRY_TERMS.items():
            if re.search(rf"\b{stem}\w*", query):
                changes["industry"] = industry
                break
    starting = re.match(r"^(?:я\s+)?(?:открываю|запускаю|только начинаю)\b", query)
    running = re.match(r"^(?:я\s+(?:уже\s+)?)?(?:веду|управляю)\b", query)
    business_object = r"\s+(?:(?:свой|своим|свое)\s+)?(?:бизнес|бизнесом|дело|делом)\b"
    starts_own_business = starting and (
        (explicit and explicit.start() < starting.end())
        or re.match(business_object, query[starting.end() :])
    )
    runs_own_business = running and (
        (explicit and explicit.start() < running.end())
        or re.match(business_object, query[running.end() :])
    )
    if starts_own_business:
        changes["stage"] = "starting"
    elif runs_own_business:
        changes["stage"] = "running"
    preference = FORMAT_PREFERENCE.search(query)
    if preference:
        changes["preferred_format"] = FORMATS[preference.group(1)]
    patch = LearningProfile(**changes) if changes else None
    profile = replace(request.profile or LearningProfile(), **changes)

    if DECLARATION.fullmatch(query):
        return request, patch, "Какую задачу для бизнеса вы хотите решить?"
    if not follow_up:
        return replace(request, profile=profile), patch, None
    if not request.previous_query or FOLLOW_UP.fullmatch(
        request.previous_query.casefold().replace("ё", "е")
    ):
        return request, patch, "Уточните, какой вопрос хотите применить к своему бизнесу?"
    if not profile.industry:
        return request, patch, "Чем занимается ваш бизнес? Тогда я подберу пример для вашей сферы."
    effective = f"{request.previous_query}\nСфера бизнеса: {INDUSTRY_LABELS[profile.industry]}."
    if len(effective) > 1000:
        return request, patch, "Сформулируйте вопрос о вашем бизнесе одним коротким сообщением."
    return AIRequest(query=effective, profile=profile), patch, None
