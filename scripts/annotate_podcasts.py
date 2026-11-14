"""Add reviewed search terms to local podcast cues using playback seconds."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from alfa_ai.podcasts import podcast_passages
from scripts.corpus import write_jsonl


def annotate(catalog, annotations, corrections=None):
    podcast_passages(catalog)
    episodes = {episode["episode_id"]: episode for episode in catalog["episodes"]}
    if not isinstance(annotations, dict) or not annotations.keys() <= episodes.keys():
        raise ValueError("Разметка содержит неизвестный выпуск.")
    if corrections is not None:
        apply_corrections(episodes, corrections)
    for episode_id, positions in annotations.items():
        if not isinstance(positions, dict):
            raise ValueError("Ожидаются поисковые термины по секундам выпуска.")
        marked = set()
        for second, terms in positions.items():
            time = float(second)
            if not isinstance(terms, str) or not terms.strip() or len(terms) > 240:
                raise ValueError("Поисковые термины должны содержать от 1 до 240 символов.")
            for index, cue in enumerate(episodes[episode_id]["cues"]):
                if cue["start_sec"] <= time < cue["end_sec"]:
                    if index in marked:
                        raise ValueError(
                            f"Две отметки попали в одну реплику: {episode_id}, {second}."
                        )
                    cue["search_terms"] = terms.strip()
                    marked.add(index)
                    break
            else:
                raise ValueError(f"На отметке {second} нет реплики выпуска {episode_id}.")
    return catalog


def apply_corrections(episodes, corrections):
    if not isinstance(corrections, dict) or not corrections.keys() <= episodes.keys():
        raise ValueError("Правки содержат неизвестный выпуск.")
    for episode_id, items in corrections.items():
        cues = {cue["start_sec"]: cue for cue in episodes[episode_id]["cues"]}
        if not isinstance(items, list):
            raise ValueError("Ожидается список исправлений реплик.")
        for item in items:
            if not isinstance(item, dict) or item.keys() != {"start_sec", "before", "after"}:
                raise ValueError("Исправление требует start_sec, before и after.")
            cue = cues.get(item["start_sec"]) if type(item["start_sec"]) in (int, float) else None
            if (
                cue is None
                or cue["text"] != item["before"]
                or not isinstance(item["after"], str)
                or not item["after"].strip()
            ):
                raise ValueError(
                    f"Правка не совпала с исходной репликой: {episode_id}, {item['start_sec']}."
                )
            cue["text"] = item["after"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--terms", required=True, type=Path)
    parser.add_argument(
        "--corrections", type=Path, help="Reviewed cue text corrections guarded by original text."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Выходной файл уже существует. Укажи новый путь.")
    try:
        corrections = json.loads(args.corrections.read_bytes()) if args.corrections else None
        catalog = annotate(
            json.loads(args.input.read_bytes()), json.loads(args.terms.read_bytes()), corrections
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    write_jsonl(args.output, [catalog])
    print(sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
