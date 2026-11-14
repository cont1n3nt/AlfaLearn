"""Shared corpus paths, section structure and JSONL output."""

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

HEADING = re.compile(r"^(#{1,6}) (.+)$")
LOCAL_DATA = Path(__file__).resolve().parents[1] / "data" / "local"


def split_sections(blocks: list[dict]) -> list[dict]:
    """Keep source block orders on each paragraph for the next chunking step."""
    sections = [{"headings": [], "paragraphs": []}]
    heading_path = {}
    for block in blocks:
        for paragraph in block["text"].split("\n\n"):
            heading = HEADING.fullmatch(paragraph)
            if heading:
                level = len(heading[1])
                heading_path = {key: value for key, value in heading_path.items() if key < level}
                heading_path[level] = heading[2]
                sections.append({"headings": list(heading_path.values()), "paragraphs": []})
            elif paragraph.strip():
                if paragraph.startswith("\\") and HEADING.fullmatch(paragraph.lstrip("\\")):
                    paragraph = paragraph[1:]
                sections[-1]["paragraphs"].append({"order": block["order"], "text": paragraph})
    return [section for section in sections if section["paragraphs"]]


def write_jsonl(output: Path, documents: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            for document in documents:
                stream.write(json.dumps(document, ensure_ascii=False) + "\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
