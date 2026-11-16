"""Render source paragraphs instead of trusting a free-form model paraphrase."""

import re

from .reviewed import PARAGRAPH_GROUPS


class EvidenceError(ValueError):
    """A draft could not be tied unambiguously to the supplied sources."""


def normalized_text(value):
    return " ".join(value.casefold().split())


def locate_paragraphs(source_text, quote):
    """Find the anchor, then restore whole original paragraphs and their qualifiers.

    Whitespace/case normalization is only used for matching, never for output.
    This verifies textual provenance, not relevance or the truth of the source.
    """
    paragraphs = tuple(p.strip() for p in re.split(r"\n\s*\n", source_text) if p.strip())
    normalized = [normalized_text(p) for p in paragraphs]
    body = " ".join(normalized)
    anchor_text = normalized_text(quote)
    anchors = [anchor_text]
    if anchor_text not in body:
        anchors = [part.strip() for part in re.split(r"(?<!\.)(?:\.{3}|…)(?!\.)", anchor_text)]
    spans = []
    for anchor in anchors:
        start = body.find(anchor)
        if len(anchor) < 12 or start < 0:
            raise EvidenceError("Цитата отсутствует в исходном тексте или слишком коротка.")
        if body.find(anchor, start + 1) != -1:
            raise EvidenceError("Цитата неоднозначна: повторяется в разных местах источника.")
        if spans and start < spans[-1][1]:
            raise EvidenceError("Части цитаты не идут в порядке исходного текста.")
        spans.append((start, start + len(anchor)))
    # An omission marker can leave out an important qualifier. Restore the
    # original passage between the first and last verified anchors as well.
    start, end = spans[0][0], spans[-1][1]
    selected = set()
    offset = 0
    for index, paragraph in enumerate(normalized):
        if offset < end and offset + len(paragraph) > start:
            selected.add(index)
        offset += len(paragraph) + 1
    return paragraphs, selected


def render_evidence(evidence, context):
    selected = {}
    for item in evidence:
        if not isinstance(item, dict) or item.keys() != {"source_id", "quote"}:
            raise ValueError("Некорректная структура доказательства.")
        source_id, quote = item["source_id"], item["quote"]
        if not isinstance(source_id, str) or source_id not in context or not isinstance(quote, str):
            raise EvidenceError("Модель сослалась на источник вне проверенного контекста.")
        try:
            paragraphs, indexes = locate_paragraphs(context[source_id]["text"], quote)
        except EvidenceError:
            # The model can return several anchors for one answer. Keep only
            # anchors independently found in the reviewed source; generated
            # prose is never rendered, and all-invalid evidence still fails.
            continue
        if source_id not in selected:
            selected[source_id] = (paragraphs, set())
        selected[source_id][1].update(indexes)

    sections, sources = [], []
    for source_id, (paragraphs, indexes) in selected.items():
        source = context[source_id]
        key = (source.get("document_id"), tuple(source.get("source_block_orders", ())))
        for group in PARAGRAPH_GROUPS.get(key, ()):
            if indexes.intersection(group):
                indexes.update(group)
        if source.get("document_id", "").startswith("podcast:"):
            # Include the antecedent when a transcript paragraph begins with a
            # reference to the previous line, e.g. "Этот фильм ...".
            for index in tuple(indexes):
                if index and re.match(r"(?i)^(?:этот|эта|эти|такой|такая)\s", paragraphs[index]):
                    indexes.add(index - 1)
        excerpt = "\n\n".join(paragraphs[index] for index in sorted(indexes))
        sections.append(f"{excerpt} [{source_id}]")
        # Keep the restored original wording separately from the answer.  It is
        # shown only in Telegram's expandable evidence block; presentation text
        # never replaces this provenance record.
        sources.append({
            **{key: value for key, value in source.items() if key != "text"},
            "evidence_text": excerpt,
        })
    if not sections:
        raise EvidenceError("Нет подтверждающих фрагментов.")
    return {
        "status": "answer",
        "text": "В проверенных материалах:\n\n" + "\n\n".join(sections),
        "sources": sources,
    }
