"""Map consultation responses to plain HTTP text and rich Telegram messages."""

import re
from dataclasses import dataclass

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    MessageEntity,
)
from aiogram.utils.formatting import (
    Bold,
    CustomEmoji,
    ExpandableBlockQuote,
    Italic,
    Text,
    TextLink,
)

from alfa_ai import AIResponse

TOO_LONG = "Сократите вопрос до 1000 символов."
MASCOT_EMOJI_ID = "5229206885713848024"
ALFA_HEART_EMOJI_ID = "5231147734190269621"
THINKING_EMOJI_ID = "5386367538735104399"
CITATION = re.compile(r"\[(S\d+)\]")


@dataclass(frozen=True, slots=True)
class TelegramAnswer:
    text: str
    entities: tuple[MessageEntity, ...]
    link_preview_options: LinkPreviewOptions | None = None
    reply_markup: InlineKeyboardMarkup | None = None


def without_custom_emoji(answer: TelegramAnswer) -> TelegramAnswer:
    """Keep the visible fallback emoji when a bot owner cannot use premium entities."""
    keyboard = answer.reply_markup
    if keyboard is not None:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [button.model_copy(update={"icon_custom_emoji_id": None}) for button in row]
            for row in keyboard.inline_keyboard
        ])
    return TelegramAnswer(
        answer.text,
        tuple(entity for entity in answer.entities if entity.type != "custom_emoji"),
        answer.link_preview_options,
        keyboard,
    )


def format_thinking() -> TelegramAnswer:
    rendered = Text(
        CustomEmoji("⌛", custom_emoji_id=THINKING_EMOJI_ID),
        " ", Italic("Думаю…"),
    ).as_kwargs(replace_parse_mode=True)
    return TelegramAnswer(rendered["text"], tuple(rendered["entities"]))


def format_answer(response: AIResponse) -> str:
    sections = [response.text]
    if response.proposed_action is not None:
        sections = ["Напоминания создаются в Telegram-диалоге после подтверждения. HTTP API не сохраняет задания."]
    if response.sources:
        sources = []
        for source in response.sources:
            interval = ""
            if source.start_sec is not None and source.end_sec is not None:
                interval = f" ({format_timestamp(source.start_sec)}–{format_timestamp(source.end_sec)})"
            sources.append(f"[{source.source_id}] {source.title}{interval}\n{source.url}")
        sections.append("Источники\n" + "\n".join(sources))
    if response.recommendations:
        access_labels = {"free": "бесплатно", "paid": "платно", "subscription": "по подписке"}
        cards = [
            f"{card.title} ({access_labels[card.access]})\n{card.reason}\n{card.url}"
            for card in response.recommendations
        ]
        sections.append("Материалы\n" + "\n\n".join(cards))
    return "\n\n".join(sections)


def format_telegram_answer(response: AIResponse) -> TelegramAnswer | None:
    """Build an entity-safe compact answer; return None for the plain long-message fallback."""
    if response.proposed_action is not None:
        return None

    source_by_id = {source.source_id: source for source in response.sources}
    body = response.text
    if body.startswith("В проверенных материалах:"):
        body = body.removeprefix("В проверенных материалах:").lstrip()
    headings = {
        "answer": "Коротко",
        "clarify": "Уточню один момент",
        "no_answer": "Не нашёл подтверждения",
        "error": "Временно не могу ответить",
        "action_proposal": "Проверьте действие",
    }
    heading = headings[response.status]
    if response.status == "no_answer" and response.recommendations:
        heading = "Подобрал материал" if len(response.recommendations) == 1 else "Подобрал материалы"
    content = [
        CustomEmoji("😸", custom_emoji_id=MASCOT_EMOJI_ID),
        " ", Bold(heading), "\n\n",
        _linked_citations(body, source_by_id),
    ]

    source_nodes = _source_nodes(response)
    if source_nodes:
        content.extend((
            "\n\n", "🔎 ", Bold(f"Источники · {len(source_nodes)}"), "\n",
            ExpandableBlockQuote(*_join_nodes(source_nodes, "\n")),
        ))

    if response.recommendations:
        content.extend((
            "\n\n", CustomEmoji("❤️", custom_emoji_id=ALFA_HEART_EMOJI_ID),
            " ", Bold("Следующий шаг"), "\n",
        ))
        for index, card in enumerate(response.recommendations):
            if index:
                content.append("\n\n")
            access = {
                "free": "🟢 Бесплатно",
                "paid": "💳 Платно",
                "subscription": "⭐ По подписке",
            }[card.access]
            content.extend((
                TextLink(Bold(card.title), url=card.url), "\n",
                Italic(access), "\n", card.reason,
            ))

    rendered = Text(*content).as_kwargs(replace_parse_mode=True)
    if _utf16_length(rendered["text"]) > 4096:
        return None

    preview = None
    keyboard = None
    if response.recommendations:
        preview = LinkPreviewOptions(
            url=response.recommendations[0].url,
            prefer_large_media=True,
            show_above_text=False,
        )
        multiple = len(response.recommendations) > 1
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(
                text=f"Открыть материал {index}" if multiple else "Открыть материал",
                icon_custom_emoji_id=ALFA_HEART_EMOJI_ID,
                style="primary",
                url=card.url,
            )]
            for index, card in enumerate(response.recommendations, 1)
        ])
    if response.status == "answer":
        feedback = [
            InlineKeyboardButton(text="👍 Полезно", callback_data=f"feedback:+:{response.trace_id}"),
            InlineKeyboardButton(text="👎 Не помогло", callback_data=f"feedback:-:{response.trace_id}"),
        ]
        keyboard = InlineKeyboardMarkup(inline_keyboard=[*(keyboard.inline_keyboard if keyboard else []), feedback])
    return TelegramAnswer(
        rendered["text"], tuple(rendered["entities"]), preview, keyboard,
    )


def _linked_citations(text, sources):
    nodes = []
    cursor = 0
    for match in CITATION.finditer(text):
        nodes.append(text[cursor:match.start()])
        source = sources.get(match[1])
        nodes.append(TextLink("(источник ↗)", url=source.url) if source else "(источник)")
        cursor = match.end()
    nodes.append(text[cursor:])
    return Text(*nodes)


def _source_nodes(response):
    groups = {}
    for source in response.sources:
        timed = source.start_sec is not None and source.end_sec is not None
        key = (source.title, source.url, source.start_sec, source.end_sec) if timed else (
            source.title, source.url, None, None
        )
        groups.setdefault(key, []).append(source.source_id)
    result = []
    for index, ((title, url, start, end), _labels) in enumerate(groups.items(), 1):
        interval = (
            f" · {format_timestamp(start)}–{format_timestamp(end)}"
            if start is not None and end is not None else ""
        )
        proof = next(
            (source.evidence_text for source in response.sources
             if source.title == title and source.url == url and source.start_sec == start
             and source.end_sec == end and source.evidence_text),
            None,
        )
        label = TextLink(Bold(f"{index}. {title}"), url=url)
        # The raw, hash-reviewed wording remains available for audit without
        # making the normal answer read like an extracted paragraph.
        result.append(Text(label, interval, "\n", proof) if proof else Text(label, interval))
    return result


def _join_nodes(nodes, separator):
    result = []
    for index, node in enumerate(nodes):
        if index:
            result.append(separator)
        result.append(node)
    return result


def _utf16_length(text):
    return len(text.encode("utf-16-le")) // 2


def format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remainder:02d}"
