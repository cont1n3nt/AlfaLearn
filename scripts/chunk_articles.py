"""Split the local pilot corpus into passages for multilingual E5."""

import argparse
import json
import re
import sys
from bisect import bisect_right
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from tokenizers import Tokenizer

from alfa_ai.semantic import MODEL_ID, MODEL_LIMIT, MODEL_REVISION
from scripts.corpus import LOCAL_DATA, split_sections, write_jsonl

CHUNKER_VERSION = "e5-sections-v3"
TOKEN_BUDGET = 350
PREFIX_BUDGET = 96
SENTENCE_END = re.compile(r'[.!?…](?:["»”)]*)(?=\s|$)')
WORD_END = re.compile(r"\S+(?=\s|$)")


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True).ids)


def token_boundary(tokenizer, prefix, text, budget):
    """Estimate a cut from full-input offsets. A cut must be re-tokenized."""
    encoding = tokenizer.encode(prefix + text, add_special_tokens=True)
    if len(encoding.ids) <= budget:
        return None
    return max(0, encoding.offsets[budget - 1][1] - len(prefix))


def split_passages(text, paragraph_ends, prefix, tokenizer):
    """Yield contiguous character spans measured with the full E5 input."""
    start = 0
    while start < len(text):
        remaining = text[start:]
        body = remaining.strip()
        leading = len(remaining) - len(remaining.lstrip())
        boundary = token_boundary(tokenizer, prefix, body, TOKEN_BUDGET)
        if boundary is None:
            yield start, len(text)
            return

        # Cutting can change the last token. Validate the cut before using it.
        body_start = start + leading
        fitting_end = body_start + max(1, boundary)
        while fitting_end > body_start:
            if (
                count_tokens(tokenizer, prefix + text[body_start:fitting_end].rstrip())
                <= TOKEN_BUDGET
            ):
                break
            fitting_end -= 1
        if fitting_end == body_start:
            raise ValueError("не удалось уместить начало текста в бюджет фрагмента")

        paragraph_index = bisect_right(paragraph_ends, fitting_end) - 1
        paragraph_end = paragraph_ends[paragraph_index] if paragraph_index >= 0 else start
        window = text[start : fitting_end + 1]
        ends = [paragraph_end]
        for pattern in (SENTENCE_END, WORD_END):
            last_end = max(
                (
                    start + match.end()
                    for match in pattern.finditer(window)
                    if start + match.end() <= fitting_end
                ),
                default=start,
            )
            ends.append(last_end)
        minimum_end = start + (fitting_end - start) // 2
        end = next(
            (
                candidate
                for candidate in ends
                if candidate > max(start, minimum_end)
                and count_tokens(tokenizer, prefix + text[start:candidate].strip()) <= TOKEN_BUDGET
            ),
            fitting_end,
        )
        yield start, end
        start = end


def embedding_prefix(context, tokenizer):
    prefix = f"passage: {context}\n\n"
    if count_tokens(tokenizer, prefix) <= PREFIX_BUDGET:
        return prefix

    boundary = token_boundary(tokenizer, "passage: ", context, PREFIX_BUDGET)
    if boundary is None:
        boundary = len(context)
    words = list(WORD_END.finditer(context[: boundary + 1]))
    for word in reversed(words):
        if word.end() <= boundary:
            candidate = f"passage: {context[: word.end()].rstrip(' /')}\n\n"
            if count_tokens(tokenizer, candidate) <= PREFIX_BUDGET:
                return candidate
    return "passage: "


def chunk_document(document, tokenizer, corpus_sha256):
    document_id = document["document_id"]
    chunks = []
    for section_index, section in enumerate(split_sections(document["blocks"])):
        headings = section["headings"]
        context = " / ".join([document["title"], *headings])
        prefix = embedding_prefix(context, tokenizer)
        paragraphs = section["paragraphs"]
        section_text = "\n\n".join(paragraph["text"] for paragraph in paragraphs)
        ranges = []
        cursor = 0
        for paragraph in paragraphs:
            end = cursor + len(paragraph["text"])
            ranges.append((cursor, end, paragraph["order"]))
            cursor = end + 2

        pieces = []
        for start, end in split_passages(
            section_text, [end for _, end, _ in ranges], prefix, tokenizer
        ):
            piece = section_text[start:end]
            pieces.append(piece)
            body = piece.strip()
            embedding_text = prefix + body
            tokens = count_tokens(tokenizer, embedding_text)
            orders = list(
                dict.fromkeys(
                    order for left, right, order in ranges if left < end and start < right
                )
            )
            if not body or not orders or not 0 < tokens <= TOKEN_BUDGET <= MODEL_LIMIT:
                raise ValueError(f"{document_id}: недопустимый фрагмент ({tokens} токенов)")
            chunk_index = len(chunks)
            passage_hash = sha256(embedding_text.encode("utf-8")).hexdigest()
            identity = (
                f"{CHUNKER_VERSION}:{MODEL_REVISION}:{document_id}:{chunk_index}:{passage_hash}"
            )
            chunks.append(
                {
                    "chunk_id": str(uuid5(NAMESPACE_URL, identity)),
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "section_index": section_index,
                    "section": headings,
                    "title": document["title"],
                    "url": document["url"],
                    "slug": document.get("slug"),
                    "text": piece,
                    "embedding_text": embedding_text,
                    "token_count": tokens,
                    "source_block_orders": orders,
                    "answer_allowed": document["answer_allowed"],
                    "recommend_allowed": document["recommend_allowed"],
                    "external_api_allowed": document["external_api_allowed"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "chunker_version": CHUNKER_VERSION,
                    "token_budget": TOKEN_BUDGET,
                    "corpus_sha256": corpus_sha256,
                }
            )
        if "".join(pieces) != section_text:
            raise ValueError(f"{document_id}: при разбиении изменился текст раздела")
    if not chunks:
        raise ValueError(f"{document_id}: нет текста для разбиения")
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LOCAL_DATA / "pilot_articles.jsonl")
    parser.add_argument("--output", type=Path, default=LOCAL_DATA / "pilot_chunks.jsonl")
    parser.add_argument("--preview", type=UUID, help="Print passages from one article.")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if any(
        not path.is_relative_to(LOCAL_DATA.resolve()) or path.suffix != ".jsonl"
        for path in (source, output)
    ):
        parser.error("--input and --output must be .jsonl files under data/local/")
    if source == output:
        parser.error("--input and --output must be different files")

    try:
        corpus = source.read_bytes()
        corpus_hash = sha256(corpus).hexdigest()
        documents = [
            json.loads(line) for line in corpus.decode("utf-8").split("\n") if line.strip()
        ]
        ids = {document["document_id"] for document in documents}
        if not documents or len(ids) != len(documents):
            raise ValueError("входной корпус пуст или содержит повторяющиеся document_id")
        preview_id = f"article:{args.preview}" if args.preview else None
        if preview_id and preview_id not in ids:
            raise ValueError(f"{preview_id} отсутствует во входном корпусе")

        tokenizer_path = hf_hub_download(MODEL_ID, "tokenizer.json", revision=MODEL_REVISION)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.no_truncation()
        tokenizer.no_padding()

        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document, tokenizer, corpus_hash)
        ]
        write_jsonl(output, chunks)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
    ) as exc:
        print(f"Ошибка подготовки фрагментов: {exc}", file=sys.stderr)
        return 1

    for chunk in chunks:
        if chunk["document_id"] == preview_id:
            print(
                f"\n#{chunk['chunk_index']} | токенов={chunk['token_count']} | блоки={chunk['source_block_orders']}"
            )
            print(chunk["embedding_text"])
    print(
        f"Статей={len(documents)} | фрагментов={len(chunks)} | максимум токенов={max(c['token_count'] for c in chunks)}"
    )
    print(f"JSONL: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
