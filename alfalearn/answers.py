"""Shared placeholder replies and Telegram message classification."""

import re
from collections.abc import Iterable
from enum import StrEnum

from aiogram.types import Message


class MessageKind(StrEnum):
    TEXT = "text"
    VOICE = "voice"
    PHOTO = "photo"
    DOCUMENT = "document"
    LINK = "link"
    VIDEO = "video"
    VIDEO_NOTE = "video-note"
    AUDIO = "audio"
    STICKER = "sticker"
    GIF = "gif"
    EMOJI = "emoji"
    CONTACT = "contact"
    LOCATION = "location"
    FORWARDED = "forwarded"
    REPLY = "reply"
    QUOTE = "quote"
    ALBUM = "album"
    CAPTION = "caption"


ANSWER_MESSAGE = "Извините я пока-что не могу читать ваши сообщения"

# Labels also name the individual endpoints in Swagger.
KIND_LABELS = {
    MessageKind.TEXT: "текстовое сообщение",
    MessageKind.VOICE: "голосовое сообщение",
    MessageKind.PHOTO: "фото или скриншот",
    MessageKind.DOCUMENT: "документ или файл",
    MessageKind.LINK: "ссылка",
    MessageKind.VIDEO: "видео",
    MessageKind.VIDEO_NOTE: "видеокружок",
    MessageKind.AUDIO: "аудиофайл",
    MessageKind.STICKER: "стикер",
    MessageKind.GIF: "GIF",
    MessageKind.EMOJI: "эмодзи",
    MessageKind.CONTACT: "контакт",
    MessageKind.LOCATION: "геолокация",
    MessageKind.FORWARDED: "пересланное сообщение",
    MessageKind.REPLY: "ответ на сообщение бота",
    MessageKind.QUOTE: "сообщение с цитатой",
    MessageKind.ALBUM: "альбом фотографий или видео",
    MessageKind.CAPTION: "вложение с текстовой подписью",
}

KIND_REPLIES = {
    MessageKind.TEXT: ANSWER_MESSAGE,
    MessageKind.VOICE: "Извините, я пока не могу прослушивать голосовые сообщения.",
    MessageKind.PHOTO: "Извините, я пока не могу просматривать фотографии и скриншоты.",
    MessageKind.DOCUMENT: "Извините, я пока не могу читать документы и файлы.",
    MessageKind.LINK: "Извините, я пока не могу открывать и читать ссылки.",
    MessageKind.VIDEO: "Извините, я пока не могу просматривать видео.",
    MessageKind.VIDEO_NOTE: "Извините, я пока не могу просматривать видеокружки.",
    MessageKind.AUDIO: "Извините, я пока не могу прослушивать аудиофайлы.",
    MessageKind.STICKER: "Извините, я пока не могу распознавать стикеры.",
    MessageKind.GIF: "Извините, я пока не могу просматривать GIF-анимации.",
    MessageKind.EMOJI: "Извините, я пока не могу интерпретировать эмодзи.",
    MessageKind.CONTACT: "Извините, я пока не могу обрабатывать контакты.",
    MessageKind.LOCATION: "Извините, я пока не могу обрабатывать геолокацию.",
    MessageKind.FORWARDED: "Извините, я пока не могу читать пересланные сообщения и вложения.",
    MessageKind.REPLY: "Извините, я пока не могу обрабатывать ответы на мои сообщения.",
    MessageKind.QUOTE: "Извините, я пока не могу читать сообщения с цитатами.",
    MessageKind.ALBUM: "Извините, я пока не могу просматривать альбомы фотографий и видео.",
    MessageKind.CAPTION: "Извините, я пока не могу обрабатывать вложения с текстовыми подписями.",
}


def build_answer(kinds: Iterable[MessageKind], message_count: int = 1) -> str:
    unique = set(kinds)
    if not unique:
        return ANSWER_MESSAGE
    if len(unique) == 1 and message_count == 1:
        return KIND_REPLIES[next(iter(unique))]
    subject = "эту группу сообщений" if message_count > 1 or MessageKind.ALBUM in unique else "это сообщение"
    labels = ", ".join(KIND_LABELS[kind] for kind in MessageKind if kind in unique)
    return f"Извините, я пока не могу обработать {subject}: {labels}."


_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF"
    "\u2B00-\u2BFF\u2194-\u2199\u21A9\u21AA\u3030\u303D\u3297\u3299"
    "\u00A9\u00AE\u203C\u2049\u2122\u2139]+"
)


def is_emoji_only(text: str) -> bool:
    # Handle keycaps, ZWJ sequences, variation selectors and subdivision flags.
    text = re.sub(r"[0-9#*]\ufe0f?\u20e3", "😀", text)
    text = "".join(
        char for char in text
        if not char.isspace() and char not in "\u200d\ufe0e\ufe0f"
        and not 0xE0020 <= ord(char) <= 0xE007F
    )
    return bool(_EMOJI.fullmatch(text))


def classify_message(message: Message, bot_id: int | None = None) -> set[MessageKind]:
    kinds: set[MessageKind] = set()
    # Telegram can include document alongside animation; classify GIF first.
    for field, kind in (
        ("animation", MessageKind.GIF), ("voice", MessageKind.VOICE),
        ("photo", MessageKind.PHOTO), ("video", MessageKind.VIDEO),
        ("video_note", MessageKind.VIDEO_NOTE), ("audio", MessageKind.AUDIO),
        ("sticker", MessageKind.STICKER), ("document", MessageKind.DOCUMENT),
        ("contact", MessageKind.CONTACT), ("location", MessageKind.LOCATION),
        ("venue", MessageKind.LOCATION),
    ):
        if getattr(message, field):
            kinds.add(kind)
            break

    entities = [*(message.entities or []), *(message.caption_entities or [])]
    if any(entity.type in {"url", "text_link"} for entity in entities):
        kinds.add(MessageKind.LINK)
    if message.text and not kinds:
        kinds.add(MessageKind.EMOJI if is_emoji_only(message.text) else MessageKind.TEXT)
    if message.caption is not None:
        kinds.add(MessageKind.CAPTION)
    if message.forward_origin:
        kinds.add(MessageKind.FORWARDED)
    original = message.reply_to_message
    if (original and original.from_user and original.from_user.is_bot
            and (bot_id is None or original.from_user.id == bot_id)):
        kinds.add(MessageKind.REPLY)
    if message.quote or any(entity.type in {"blockquote", "expandable_blockquote"} for entity in entities):
        kinds.add(MessageKind.QUOTE)
    if message.media_group_id:
        kinds.add(MessageKind.ALBUM)
    return kinds
