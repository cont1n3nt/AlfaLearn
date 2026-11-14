"""Prepare locally reviewed article candidates from the restored PostgreSQL dump."""

import argparse
import json
import os
import re
import sys
from collections import Counter
from getpass import getpass
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import psycopg
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from psycopg.rows import dict_row

from scripts.corpus import HEADING, LOCAL_DATA, split_sections, write_jsonl

PUBLISHED_STATUS = UUID("61f12dbc-3cd4-4c14-b64b-1709548b904e")
PILOT_IDS = [
    UUID(value)
    for value in """
000ecbf2-ccb2-4305-82e2-f3ed7ce1b22c
33a3a4ca-ab5e-4e58-bbeb-62afae4de0ba
4177c427-c698-4985-a5a2-e18e356bdae5
6169c88f-c100-42f7-8b1c-d4f8b6bf9c1f
e5c8d28a-79c0-4877-80dd-06b68ba67576
255a0449-8167-467a-96d6-7bff5688f73f
0d63697c-8fe9-4b4b-9a66-7cad43aab1ea
9e4e421f-bc27-4bac-aa7a-bc527e7b6539
30c260a0-5f8b-4877-b37c-8b6bb871f0d5
e07e384c-6f11-47c3-b2d6-dfd82e97f730
bfc159df-4a6a-4ec1-9343-9920f1654c92
de95fb1b-35cb-4a3f-9d36-9a5e672bf466
02fd3dab-fecf-4618-ae31-35a931a318ee
f079c693-aafb-46f3-987b-6d071cc5c1f1
ce10a90f-8257-4ac9-9b86-e586d81c5c6c
f3d25c33-90cc-47ab-8d55-c67d9241b049
ef04f756-44bf-478a-b022-51fae04ecee5
33eae91e-6594-43b9-862a-8869c0e25f81
6ae980f7-41d2-4c8f-87f7-3348be389e4d
b6aa0384-1619-4638-8f00-bd1f3dfc25b2
035c49ae-7d41-4aa6-85d6-79e94dfbc00a
d0ce5d7a-655a-49dc-a99e-ca014fefc469
a3743a1b-6752-4ee3-b645-061164169f58
00312ff2-f294-40eb-aaa1-9fb8d2dd78a6
fba0ba5a-45d3-40be-9d57-3872f5ff226a
""".split()
]

ARTICLE_BLOCKS_SQL = """
SELECT a."Id" AS article_id, a."Title"::text AS title, a."Url"::text AS slug,
       e."Id" AS block_id, e."Type" AS block_type, e."Order" AS block_order,
       e."HtmlContent"::text AS html, e."Params"::text AS params
FROM public."Articles" a
JOIN public."ArticleEmbedItems" e ON e."ArticleId" = a."Id"
WHERE a."Id" = ANY(%s) AND a."StatusId" = %s
  AND NOT a."IsDeleted" AND a."IsPublic"
  AND NOT a."IsClosed" AND NOT a."IsLesson"
ORDER BY a."Id", e."Order" NULLS LAST, e."Id"
"""

# Keep real HTML headings distinct from literal "# ..." until paragraphs are assembled.
HTML_HEADING = "\0"
INVISIBLE = str.maketrans("", "", HTML_HEADING + "\ufeff\u200b\u2060")
BLOCK_TAGS = {"p", "div", "section", "article", "blockquote", "figure", "figcaption"}


def render_list(tag: Tag) -> str:
    items = tag.find_all("li", recursive=False)
    ordered = tag.name == "ol"
    step = -1 if tag.has_attr("reversed") else 1
    default_start = len(items) if step == -1 else 1
    number = html_integer(tag.get("start"), default_start)
    lines = []
    for item in items:
        number = html_integer(item.get("value"), number)
        prefix = f"{number}. " if ordered else "- "
        body = "".join(render_node(child) for child in item.children).strip()
        if body:
            lines.append(prefix + body)
        number += step
    return "\n" + "\n".join(lines) + "\n"


def html_integer(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def render_node(node: Tag | NavigableString) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node).translate(INVISIBLE))
    if node.name in {"script", "style", "template"} or "ql-cursor" in node.get("class", []):
        return ""
    if node.name == "br":
        return "\n"
    if node.name in {"ol", "ul"}:
        return render_list(node)
    if node.name == "table":
        rows = (row for row in node.find_all("tr") if row.find_parent("table") is node)
        body = "\n".join(render_node(row).strip() for row in rows)
        return f"\n\n{body}\n\n"
    if node.name == "tr":
        cells = node.find_all(["th", "td"], recursive=False)
        values = [" ".join(render_node(cell).split()) for cell in cells]
        return "\n| " + " | ".join(values) + " |\n" if cells else ""

    body = "".join(render_node(child) for child in node.children)
    if re.fullmatch(r"h[1-6]", node.name or ""):
        heading = " ".join(body.split())
        return f"\n\n{HTML_HEADING}{'#' * int(node.name[1])} {heading}\n\n" if heading else ""
    if node.name in BLOCK_TAGS:
        return f"\n\n{body.strip()}\n\n"
    return body


def clean_html(html: str | None) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    text = render_node(soup)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    paragraphs = []
    for paragraph in text.split("\n\n"):
        if paragraph.startswith(HTML_HEADING):
            paragraph = paragraph.removeprefix(HTML_HEADING)
        elif HEADING.fullmatch(paragraph.lstrip("\\")):
            paragraph = "\\" + paragraph
        paragraphs.append(paragraph.replace(HTML_HEADING, ""))
    return "\n\n".join(paragraphs)


def callout_text(raw: str | None) -> str:
    try:
        params = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise ValueError(f"некорректный JSON в Params: {exc.msg}") from exc
    if not isinstance(params, dict):
        raise ValueError("ожидался JSON-объект в Params")
    title, description = params.get("title"), params.get("description")
    if not isinstance(description, str) or (title is not None and not isinstance(title, str)):
        raise ValueError("ожидались строковый description и необязательный строковый title")
    return "\n\n".join(filter(None, (clean_html(title), clean_html(description))))


def prepare_document(article_id: UUID, rows: list[dict]) -> tuple[dict, Counter, Counter, int]:
    if not rows:
        raise ValueError("нет блоков или статья не прошла фильтры")
    orders = [row["block_order"] for row in rows]
    if None in orders or len(set(orders)) != len(orders):
        raise ValueError("пустой или повторяющийся Order")

    included, skipped = Counter(), Counter()
    blocks = []
    empty = 0
    for row in rows:
        block_type = row["block_type"]
        if block_type == 0:
            text = clean_html(row["html"])
        elif block_type == 2:
            try:
                text = callout_text(row["params"])
            except ValueError as exc:
                raise ValueError(
                    f"блок {row['block_id']} (Order={row['block_order']}): {exc}"
                ) from exc
        else:
            skipped[block_type] += 1
            continue

        if not text:
            empty += 1
            continue
        included[block_type] += 1
        blocks.append(
            {
                "block_id": str(row["block_id"]),
                "order": row["block_order"],
                "type": block_type,
                "text": text,
            }
        )

    if not blocks:
        raise ValueError("после очистки нет текста")

    text = "\n\n".join(block["text"] for block in blocks)
    source_url = (rows[0]["slug"] or "").strip()
    url = slug = None
    try:
        parsed = urlsplit(source_url)
    except ValueError:
        pass
    else:
        if parsed.scheme in {"http", "https"} and parsed.hostname:
            url = source_url
        elif (
            not parsed.scheme
            and not parsed.netloc
            and parsed.path
            and not source_url.startswith("//")
        ):
            slug = source_url
    document = {
        "document_id": f"article:{article_id}",
        "source_id": str(article_id),
        "kind": "article",
        "title": rows[0]["title"],
        "slug": slug,
        "url": url,
        "text": text,
        "blocks": blocks,
        "answer_allowed": False,
        "recommend_allowed": False,
        "external_api_allowed": False,
        "content_hash": sha256(text.encode("utf-8")).hexdigest(),
    }
    return document, included, skipped, empty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--article-id", type=UUID, action="append", help="Repeat to select multiple articles."
    )
    parser.add_argument(
        "--preview", action="store_true", help="Print cleaned blocks with their source orders."
    )
    parser.add_argument(
        "--ask-password", action="store_true", help="Prompt for the PostgreSQL password."
    )
    parser.add_argument(
        "--output", type=Path, help="Optionally write a JSONL file under data/local/."
    )
    args = parser.parse_args()
    output = args.output.resolve() if args.output else None
    if output is not None and (
        not output.is_relative_to(LOCAL_DATA.resolve()) or output.suffix != ".jsonl"
    ):
        parser.error("--output must be a .jsonl file under data/local/")

    article_ids = list(dict.fromkeys(args.article_id or PILOT_IDS))
    if args.preview and len(article_ids) != 1:
        parser.error("--preview requires exactly one --article-id")

    grouped = {article_id: [] for article_id in article_ids}
    connection_options = {
        "host": os.getenv("PGHOST") or "127.0.0.1",
        "port": os.getenv("PGPORT") or "5433",
        "dbname": os.getenv("PGDATABASE") or "alfa_course",
        "user": os.getenv("PGUSER") or "alfa",
    }
    if args.ask_password:
        try:
            connection_options["password"] = getpass("Пароль PostgreSQL: ")
        except (EOFError, KeyboardInterrupt):
            print("\nВвод пароля прерван.", file=sys.stderr)
            return 1
    try:
        with psycopg.connect(row_factory=dict_row, connect_timeout=5, **connection_options) as conn:
            conn.read_only = True
            for row in conn.execute(ARTICLE_BLOCKS_SQL, (article_ids, PUBLISHED_STATUS)):
                grouped[row["article_id"]].append(row)
    except psycopg.Error as exc:
        print(f"PostgreSQL: {exc}", file=sys.stderr)
        return 1

    documents = []
    for article_id, rows in grouped.items():
        try:
            document, included, skipped, empty = prepare_document(article_id, rows)
            sections = split_sections(document["blocks"])
            if not sections:
                raise ValueError("нет текста для разбиения")
        except ValueError as exc:
            print(f"ERROR | {article_id} | {exc}", file=sys.stderr)
            continue

        documents.append(document)
        print(
            f"OK | {document['title']} | {article_id} | символов={len(document['text'])} "
            f"| разделов={len(sections)} | включено={dict(included)} "
            f"| пропущено={dict(skipped)} | пустых={empty}"
        )
        if args.preview:
            for block in document["blocks"]:
                print(f"\nOrder={block['order']} Type={block['type']}\n{block['text']}")

    errors = len(article_ids) - len(documents)
    print(f"Запрошено={len(article_ids)} | собрано={len(documents)} | ошибок={errors}")
    if errors:
        return 1
    if output is not None:
        try:
            write_jsonl(output, documents)
        except OSError as exc:
            print(f"Ошибка записи JSONL: {exc}", file=sys.stderr)
            return 1
        print(f"JSONL: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
