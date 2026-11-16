"""Retrieve reviewed passages and assemble cited answers."""

import json
import re
from hashlib import sha256

from .evidence import EvidenceError
from .generation import GenerationTruncated, generate_with_context, pinned_model_available
from .podcasts import (
    EPISODE_REFERENCE,
    load_podcast_passages,
    named_podcast_episodes,
    podcast_name_key,
    podcast_names,
    podcast_search_text,
)
from .retrieval import LIMIT, NAMED_EPISODE_LIMIT, LexicalIndex, select_context
from .reviewed import approved_passages
from .semantic import SemanticIndex
from .settings import DEFAULT_CORPUS_PATH, PACKAGED_CORPUS_SHA256, Settings

BARE_REFERENCE = re.compile(
    r"(?:как\s+(?:это|такое)\s+(?:настроить|сделать|оформить|посчитать|подключить|проверить)"
    r"|что\s+(?:тут|здесь)\s+(?:лучше\s+)?выбрать)"
)
PODCAST_CLARIFICATION = {
    "status": "clarify",
    "text": "Какой выпуск вы имеете в виду? Укажите его название.",
    "sources": [],
}
COMPOUND_PART = re.compile(
    r"\s+и\s+(?=(?:как|что|чем|почему|зачем|сколько|где|когда|какой|какая|какие)\b)",
    re.I,
)
PODCAST_QUERY = re.compile(r"\b(?:подкаст|выпуск)\w*\b", re.I)


def compound_parts(question):
    parts = [part.strip(" ,") for part in COMPOUND_PART.split(question) if part.strip(" ,")]
    return parts if len(parts) == 2 and all(len(part) >= 12 for part in parts) else []


def merge_contexts(contexts, limit=4):
    merged = {}
    seen = set()
    for context in contexts:
        for source in context.values():
            key = source["document_id"], source["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            source_id = f"S{len(merged) + 1}"
            merged[source_id] = {**source, "source_id": source_id}
            if len(merged) == limit:
                return merged
    return merged


class RagEngine:
    """Keep the corpus and search indexes in memory for the service lifetime."""

    def __init__(self, chunks=None, *, settings=None):
        self.settings = settings or Settings.from_env()
        if chunks is None:
            data = self.settings.corpus_path.read_bytes()
            if (
                self.settings.corpus_path == DEFAULT_CORPUS_PATH
                and sha256(data).hexdigest() != PACKAGED_CORPUS_SHA256
            ):
                raise ValueError("Упакованный корпус изменён или повреждён.")
            chunks = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        self.passages = approved_passages(chunks)
        if self.settings.podcast_corpus_path is not None:
            self.passages.extend(
                load_podcast_passages(
                    self.settings.podcast_corpus_path,
                    self.settings.podcast_corpus_sha256,
                )
            )
        self.episodes = {}
        for passage in self.passages:
            document_id = passage["document_id"]
            if document_id.startswith("podcast:") and document_id not in self.episodes:
                self.episodes[document_id] = podcast_names(passage["title"]), passage["url"]
        self.index = LexicalIndex(self.passages)
        self.semantic_index = (
            SemanticIndex(self.passages, self.settings.e5_model_dir)
            if self.settings.e5_model_dir is not None and self.episodes
            else None
        )

    def readiness(self):
        if not self.passages:
            return False
        if self.settings.corpus_path == DEFAULT_CORPUS_PATH and (
            sha256(self.settings.corpus_path.read_bytes()).hexdigest() != PACKAGED_CORPUS_SHA256
        ):
            return False
        return pinned_model_available(self.settings)

    def retrieve(self, question, *, document_ids=None, limit=None):
        index = self.index
        focused_question = None
        limit = LIMIT if limit is None else limit
        supplemental_index = None
        if document_ids and self.episodes.keys() >= document_ids:
            focused = podcast_search_text(question, self.episodes, document_ids)
            if len(document_ids) == 1:
                focused_question = focused
                limit = NAMED_EPISODE_LIMIT
                supplemental_index = self.semantic_index
            elif self.semantic_index is not None:
                index = self.semantic_index
                question = focused
            else:
                focused_question = focused
        return select_context(
            question,
            self.passages,
            index,
            document_ids=document_ids,
            focused_question=focused_question,
            limit=limit,
            semantic_index=supplemental_index,
        )

    def answer(self, question, *, trace=None, previous_query=None):
        question = question.strip()
        if not question or len(question) > 1000:
            raise ValueError("Вопрос должен содержать от 1 до 1000 символов.")
        if trace is not None:
            trace.clear()
        episode_ids = named_podcast_episodes(question, self.episodes)
        if (
            previous_query
            and len(episode_ids) == 1
            and named_podcast_episodes(question, self.episodes, exact=True) == episode_ids
            and not named_podcast_episodes(previous_query, self.episodes)
        ):
            clarified_question = f"{previous_query}\nУточнение выпуска: {question}"
            if len(clarified_question) <= 1000:
                question = clarified_question
        normalized_question = podcast_name_key(question)
        if BARE_REFERENCE.fullmatch(normalized_question):
            return {
                "status": "clarify",
                "text": "Уточните, о чём именно вы спрашиваете?",
                "sources": [],
            }
        if not self.episodes and PODCAST_QUERY.search(question):
            return {
                "status": "no_answer",
                "text": "Проверенных расшифровок подкастов пока нет в базе ответов.",
                "sources": [],
            }
        if (
            not episode_ids
            and self.episodes
            and EPISODE_REFERENCE.search(normalized_question)
        ):
            if trace is not None:
                trace["episode_clarification"] = "unnamed_reference"
            return {**PODCAST_CLARIFICATION, "sources": []}
        parts = compound_parts(question) if not episode_ids else []
        context = (
            merge_contexts([self.retrieve(part, limit=2) for part in parts])
            if parts else self.retrieve(question, document_ids=episode_ids)
        )
        if trace is not None:
            trace["context"] = list(context.values())
        try:
            result = generate_with_context(question, context, trace=trace, settings=self.settings)
        except (EvidenceError, GenerationTruncated) as exc:
            retry_evidence = isinstance(exc, EvidenceError) and bool(context)
            retry_named_truncation = (
                isinstance(exc, GenerationTruncated)
                and len(episode_ids) == 1
                and len(context) > 1
            )
            if retry_evidence or retry_named_truncation:
                if trace is not None:
                    trace["citation_retry"] = str(exc)
                shorter = bool(episode_ids and len(context) > NAMED_EPISODE_LIMIT)
                retry_context = dict(list(context.items())[:NAMED_EPISODE_LIMIT]) if shorter else context
                try:
                    return generate_with_context(
                        question,
                        retry_context,
                        trace=trace,
                        settings=self.settings,
                        brief=retry_evidence and not shorter,
                    )
                except (EvidenceError, GenerationTruncated) as retry_error:
                    exc = retry_error
            if isinstance(exc, GenerationTruncated):
                raise exc
            if trace is not None:
                trace["evidence_error"] = str(exc)
            return {
                "status": "no_answer",
                "text": "Не удалось подтвердить ответ по проверенным источникам.",
                "sources": [],
            }
        if (
            result["status"] == "no_answer"
            and not parts
            and not episode_ids
            and len(context) > 1
            and max(self.index.score([question])[0], default=0) >= 0.25
        ):
            focused_context = {"S1": next(iter(context.values()))}
            if trace is not None:
                trace["focused_retry"] = focused_context["S1"]["document_id"]
            try:
                focused = generate_with_context(
                    question, focused_context, trace=trace, settings=self.settings, brief=True
                )
                if focused["status"] == "answer":
                    result = focused
            except (EvidenceError, GenerationTruncated):
                pass
        if not episode_ids and len(self.episodes) > 1:
            cited_episodes = {
                source["document_id"]
                for source in result["sources"]
                if source["document_id"] in self.episodes
            }
            if len(cited_episodes) > 1:
                if trace is not None:
                    trace["episode_clarification"] = "mixed_citations"
                return {**PODCAST_CLARIFICATION, "sources": []}
        return result
