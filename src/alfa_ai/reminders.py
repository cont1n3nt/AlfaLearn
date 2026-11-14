"""Parse explicit reminder requests into proposals for confirmation."""

import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .contracts import ReminderProposal

COMMAND = re.compile(
    r"^(?:пожалуйста,?\s+)?(?:(напомни|напоминай)\b|"
    r"(перенеси|измени|поменяй|отмени|удали|отключи)\s+(?:частоту\s+)?напоминани\w*\b)",
    re.I,
)
DATE = r"(?<![\d.])(?:\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}|сегодня|завтра|послезавтра)\b"
TIME = r"(?<!\d)\d{1,2}:\d{2}(?!\d)"
ZONE = r"\b(?:UTC|[A-Za-z_]+(?:/[A-Za-z_+-]+)+)\b"
REMINDER_ID = r"(?<!\w)#[A-Za-z0-9_-]{1,64}\b"
SLOT_REPLY = re.compile(rf"(?:\s|[,.;!?]|\b(?:в|на)\b|{DATE}|{TIME}|{ZONE}|{REMINDER_ID})+", re.I)
WEEKDAYS = (
    ("понедельник", "понедельникам"),
    ("вторник", "вторникам"),
    ("среду", "средам"),
    ("четверг", "четвергам"),
    ("пятницу", "пятницам"),
    ("субботу", "субботам"),
    ("воскресенье", "воскресеньям"),
)
DAILY = r"\b(?:ежедневно|каждый\s+день)\b"
ONCE = r"\b(?:один\s+раз|разово)\b"
WEEKDAY_NAMES = "|".join(day for forms in WEEKDAYS for day in forms)
WEEKLY = rf"\b(?:кажд(?:ый|ую|ое)|по)\s+({WEEKDAY_NAMES})\b"
OTHER_REPEAT = r"\b(?:еженедельно|ежемесячно|по\s+будням|раз\s+в|кажд\w*)\b"
RELATIVE_OFFSET = re.compile(
    rf"\b(?:через\s+\d|за\s+\d+\s+(?:минут|час|дн|ден)|(?:до|после)\s+{TIME})", re.I
)
QUESTIONS = {
    "reminder_id": "Укажите номер напоминания в формате #номер.",
    "subject": "О чём вам напомнить?",
    "date": "На какую дату поставить напоминание? Укажите её в формате ГГГГ-ММ-ДД.",
    "time": "В какое время напомнить? Укажите время в формате ЧЧ:ММ.",
    "timezone": "В каком часовом поясе напомнить? Например, Europe/Moscow.",
    "repeat": "Как повторять напоминание? Можно выбрать ежедневно, конкретный день недели или разово.",
    "change": "Уточните новое расписание или частоту напоминания.",
}


def next_date(local_now, at, weekday):
    today = local_now.date()
    step = 1 if weekday is None else 7
    target = today if weekday is None else today + timedelta(days=(weekday - today.weekday()) % 7)
    if datetime.combine(target, at) <= local_now.replace(tzinfo=None):
        target += timedelta(days=step)
    return target


def unambiguous_time(value, zone, now):
    valid = set()
    for fold in (0, 1):
        instant = value.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) == value:
            valid.add(instant)
    return len(valid) == 1 and next(iter(valid)) > now


def plan_reminder(request):
    text = request.query
    command = COMMAND.match(text)
    if not command and request.previous_query and SLOT_REPLY.fullmatch(text):
        text = request.previous_query + " " + text
        command = COMMAND.match(text)
    if not command:
        return None

    body = text[command.end() :].strip(" ,.!?")
    verb = (command[2] or "").casefold()
    action = (
        "create_reminder"
        if not verb
        else "cancel_reminder"
        if verb in {"отмени", "удали", "отключи"}
        else "update_reminder"
    )
    missing = []
    ids = re.findall(REMINDER_ID, body)
    reminder_id = ids[0][1:] if len(ids) == 1 and verb else None
    if verb and not reminder_id:
        missing.append("reminder_id")
    if action == "cancel_reminder":
        if re.search(DATE, body, re.I) or re.search(TIME, body):
            proposal = ReminderProposal(
                action, reminder_id=reminder_id, missing_fields=(*missing, "change")
            )
            return (
                proposal,
                "Для отмены сейчас укажите только номер напоминания, без будущей даты или времени.",
            )
        proposal = ReminderProposal(action, reminder_id=reminder_id, missing_fields=tuple(missing))
        return proposal, QUESTIONS[
            "reminder_id"
        ] if missing else f"Подтвердите отмену напоминания #{reminder_id}."

    lower = body.casefold()
    weeks = list(re.finditer(WEEKLY, lower))
    week = weeks[0] if weeks else None
    day = re.search(DAILY, lower)
    once = re.search(ONCE, lower)
    repeat = "weekly" if week else "daily" if day else "once" if once or not verb else None
    weekday = next((i for i, forms in enumerate(WEEKDAYS) if week and week[1] in forms), None)
    repeat_tail = re.sub(rf"{WEEKLY}|{DAILY}|{ONCE}", " ", lower)
    other_weekday = week and re.search(rf"\b(?:{WEEKDAY_NAMES})\b", repeat_tail)
    if (
        len(weeks) > 1
        or other_weekday
        or sum(bool(value) for value in (week, day, once)) > 1
        or re.search(OTHER_REPEAT, repeat_tail)
        or (command[1] and command[1].casefold() == "напоминай" and not (week or day or once))
    ):
        missing.append("repeat")
        repeat, weekday = None, None

    dates = re.findall(DATE, lower)
    times = re.findall(TIME, lower)
    zones = re.findall(ZONE, body)
    zone_name = zones[0] if len(zones) == 1 else request.timezone if not zones else None
    subject = None
    if not verb:
        subject = re.sub(
            rf"(?:\b(?:в|на)\s+)?(?:{DATE}|{TIME}|{ZONE}|{WEEKLY}|{DAILY}|{ONCE})",
            " ",
            body,
            flags=re.I,
        )
        subject = re.sub(r"\b(?:вечером|утром|днем|днём|ночью)\b", " ", subject, flags=re.I)
        subject = " ".join(subject.strip(" ,.!?").split()) or None
        if not subject:
            missing.append("subject")

    local_datetime = None
    if not verb or dates or times:
        now = (request.now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        try:
            zone = ZoneInfo(zone_name) if zone_name else None
        except (ValueError, ZoneInfoNotFoundError):
            zone = None
        if zone is None:
            zone_name = None
            missing.append("timezone")
        try:
            at = time(*map(int, times[0].split(":"))) if len(times) == 1 else None
        except ValueError:
            at = None
        if at is None or RELATIVE_OFFSET.search(body):
            missing.append("time")
            at = None
        local_now = now.astimezone(zone) if zone else None
        target = None
        if len(dates) == 1:
            token = dates[0]
            relative = {"сегодня": 0, "завтра": 1, "послезавтра": 2}
            if token in relative:
                if local_now:
                    target = local_now.date() + timedelta(days=relative[token])
            else:
                try:
                    target = (
                        date.fromisoformat(token)
                        if "-" in token
                        else datetime.strptime(token, "%d.%m.%Y").date()
                    )
                except ValueError:
                    missing.append("date")
        elif not dates and repeat in {"daily", "weekly"}:
            if local_now and at:
                target = next_date(local_now, at, weekday)
        else:
            missing.append("date")
        if target and weekday is not None and target.weekday() != weekday:
            missing.append("date")
            target = None
        if zone and target and at:
            value = datetime.combine(target, at)
            if unambiguous_time(value, zone, now):
                local_datetime = value.isoformat(timespec="minutes")
            else:
                missing.append("time")
    elif repeat is None and "repeat" not in missing:
        missing.append("change")

    proposal = ReminderProposal(
        action,
        subject,
        reminder_id,
        local_datetime,
        zone_name,
        repeat,
        weekday,
        tuple(dict.fromkeys(missing)),
    )
    if missing:
        return proposal, " ".join(QUESTIONS[field] for field in proposal.missing_fields)
    operation = (
        f"создание напоминания «{subject}»" if subject else f"изменение напоминания #{reminder_id}"
    )
    when = f" на {local_datetime.replace('T', ' ')} ({zone_name})" if local_datetime else ""
    cadence = " Повтор ежедневно." if repeat == "daily" else ""
    if repeat == "weekly":
        cadence = f" Повтор раз в неделю ({WEEKDAYS[weekday][0]})."
    elif repeat == "once":
        cadence = " Однократно."
    return proposal, f"Подтвердите {operation}{when}.{cadence}"
