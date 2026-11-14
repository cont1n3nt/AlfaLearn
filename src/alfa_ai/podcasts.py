"""Build searchable passages from reviewed podcast cues with original timestamps."""

import json
import re
from hashlib import sha256
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

SENTENCE_END = re.compile(r'[.!?…]["»”)]*$')
QUOTED_NAME = re.compile(r'«([^»]+)»|"([^"]+)"|“([^”]+)”')
EPISODE_REFERENCE = re.compile(
    r"\b(?:в|из)\s+(?:(?:этом|этом же|этом самом|этого|того|учебном|учебного|"
    r"предыдущем|предыдущего|прошлом|прошлого|последнем|последнего)\s+)"
    r"(?:выпуске|выпуска|подкасте|подкаста)\b"
)
PODCAST_LINK = re.compile(r'https://[^\s<>"«»]+', re.IGNORECASE)


def podcast_name_key(text):
    return " ".join(re.findall(r"[^\W_]+", text.casefold().replace("ё", "е")))


def podcast_names(title):
    """Use the catalog title and names the title explicitly quotes."""
    quoted = {
        podcast_name_key(next(group for group in match.groups() if group))
        for match in QUOTED_NAME.finditer(title)
    }
    full_name = podcast_name_key(title)
    if " " in full_name:
        quoted.add(full_name)
    return quoted - {""}


def named_podcast_episodes(question, episodes, *, exact=False):
    """Match full names at word boundaries or exact catalog links."""
    words = f" {podcast_name_key(question)} "
    urls = [question.strip()] if exact else PODCAST_LINK.findall(question)
    links = {url.rstrip(".,;:!?)/") for url in urls}
    return {
        document_id
        for document_id, (names, url) in episodes.items()
        if url.rstrip("/") in links
        or any(words == f" {name} " if exact else f" {name} " in words for name in names)
    }


def podcast_search_text(question, episodes, episode_ids):
    """Exclude the known title or link from ranking within an episode."""
    text = PODCAST_LINK.sub(" ", question)
    names = {name for episode_id in episode_ids for name in episodes[episode_id][0]}
    for name in sorted(names, key=lambda item: (-len(item), item)):
        pattern = r"[\W_]+".join(re.escape(word).replace("е", "[её]") for word in name.split())
        text = re.sub(rf"(?<!\w){pattern}(?!\w)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:уточнение выпуска|выпуск)\s*[«»“”\".:\s]*$", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip(' «»“”"') or question


def validate_interval(start, end):
    if (
        any(type(value) not in (int, float) or not 0 <= value <= 86400 for value in (start, end))
        or end <= start
    ):
        raise ValueError("Нужен корректный интервал в секундах: 0 <= start_sec < end_sec <= 86400.")


def catalog_from_media_export(export, *, episode_id, title, url, use_speakers=False):
    """Adapt the podcast-transcription v1 export, whose times already use playback seconds."""
    if (
        not isinstance(export, dict)
        or type(export.get("schema_version")) is not int
        or export["schema_version"] != 1
    ):
        raise ValueError("Ожидается экспорт медиа-сервиса версии 1.")
    if export.get("time_unit") != "seconds":
        raise ValueError("Экспорт должен содержать таймкоды в секундах.")
    external_id = export.get("external_id")
    if external_id is not None and external_id != episode_id:
        raise ValueError("external_id экспорта не совпадает с идентификатором выпуска.")
    segments = export.get("speaker_segments" if use_speakers else "segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("В выбранной дорожке экспорта нет реплик.")
    cues = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Некорректный сегмент экспорта.")
        cue = {
            "start_sec": segment.get("t_start"),
            "end_sec": segment.get("t_end"),
            "text": segment.get("text"),
        }
        if use_speakers and segment.get("speaker") is not None:
            cue["speaker"] = segment["speaker"]
        cues.append(cue)
    return {
        "version": 1,
        "episodes": [
            {
                "episode_id": episode_id,
                "title": title,
                "url": url,
                "cues": cues,
            }
        ],
    }


def podcast_passages(document):
    if (
        not isinstance(document, dict)
        or type(document.get("version")) is not int
        or document["version"] != 1
    ):
        raise ValueError("Ожидается каталог расшифровок версии 1.")
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Каталог должен содержать хотя бы один выпуск.")
    seen = set()
    passages = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("Некорректная запись выпуска.")
        episode_id, title, url = (episode.get(key) for key in ("episode_id", "title", "url"))
        if not isinstance(episode_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", episode_id):
            raise ValueError("Некорректный episode_id.")
        if episode_id in seen:
            raise ValueError(f"Выпуск {episode_id} повторяется.")
        seen.add(episode_id)
        if not isinstance(title, str) or not title.strip() or not isinstance(url, str):
            raise ValueError("Выпуску нужны название и ссылка.")
        link = urlsplit(url)
        if link.scheme != "https" or not link.hostname or link.username is not None:
            raise ValueError("Ссылка на выпуск должна быть HTTPS без учётных данных.")
        cues = episode.get("cues")
        if not isinstance(cues, list) or not cues:
            raise ValueError(f"Нет реплик выпуска {episode_id}.")

        # Prefer a sentence boundary near 30 seconds. Bound the wait for one.
        group = []
        previous_end = 0
        for cue in cues:
            if not isinstance(cue, dict):
                raise ValueError("Некорректная реплика.")
            start, end, text = (cue.get(key) for key in ("start_sec", "end_sec", "text"))
            speaker = cue.get("speaker")
            search_terms = cue.get("search_terms", "")
            if speaker is not None and (not isinstance(speaker, str) or not speaker.strip()):
                raise ValueError("Метка голоса должна быть непустой строкой.")
            if not isinstance(search_terms, str) or len(search_terms) > 240:
                raise ValueError("Поисковые термины реплики должны быть строкой до 240 символов.")
            if speaker is not None:
                speaker = speaker.strip()
            validate_interval(start, end)
            if start < previous_end or not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Реплики должны содержать текст и идти по времени без пересечений."
                )
            if group:
                span = end - group[0]["start_sec"]
                if (
                    speaker != group[-1]["speaker"]
                    or start - previous_end >= 5
                    or span > 45
                    or (span > 30 and SENTENCE_END.search(group[-1]["text"]))
                ):
                    passages.append(make_passage(episode_id, title, url, group))
                    group = []
            group.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "text": text.strip(),
                    "speaker": speaker,
                    "search_terms": search_terms.strip(),
                }
            )
            previous_end = end
        passages.append(make_passage(episode_id, title, url, group))
    return passages


def make_passage(episode_id, title, url, cues):
    start, end = cues[0]["start_sec"], cues[-1]["end_sec"]
    paragraphs, sentence = [], []
    for cue in cues:
        sentence.append(cue["text"])
        if SENTENCE_END.search(cue["text"]):
            paragraphs.append(" ".join(sentence))
            sentence = []
    if sentence:
        paragraphs.append(" ".join(sentence))
    text = "\n\n".join(paragraphs)
    document_id = f"podcast:{episode_id}"
    chunk_id = str(
        uuid5(NAMESPACE_URL, json.dumps([document_id, start, end, text], ensure_ascii=False))
    )
    keywords = " ".join(dict.fromkeys(cue["search_terms"] for cue in cues if cue["search_terms"]))
    index_text = "\n\n".join(part for part in (title.strip(), text, keywords) if part)
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "title": title.strip(),
        "url": url,
        "text": text,
        "start_sec": start,
        "end_sec": end,
        "source_block_orders": [],
        "embedding_text": f"passage: {index_text}",
    }


def load_podcast_passages(path, expected_sha256):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Файл расшифровок изменился. Сверьте его содержимое и SHA-256.")
    return podcast_passages(json.loads(data))
