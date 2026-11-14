"""Transcribe a permitted local recording to a candidate podcast catalog."""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from alfa_ai.podcasts import podcast_passages
from scripts.corpus import write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error("--audio должен указывать на существующий локальный файл.")
    model_dir = args.model_dir.resolve()
    if not all(
        (model_dir / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")
    ):
        parser.error("--model-dir должен содержать локальную модель CTranslate2 с токенизатором.")
    if args.output.exists():
        parser.error("Файл вывода уже существует. Укажи новый путь для кандидата.")

    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)
    segments, _ = model.transcribe(str(args.audio), language=args.language, vad_filter=True)
    cues = []
    next_progress = 300
    for segment in segments:
        text = segment.text.strip()
        if text:
            cues.append({"start_sec": segment.start, "end_sec": segment.end, "text": text})
        if segment.end >= next_progress:
            print(f"Распознано до {segment.end:.0f} с записи.", file=sys.stderr, flush=True)
            next_progress = (int(segment.end) // 300 + 1) * 300
    catalog = {
        "version": 1,
        "episodes": [
            {
                "episode_id": args.episode_id,
                "title": args.title,
                "url": args.url,
                "cues": cues,
            }
        ],
    }
    passages = podcast_passages(catalog)
    write_jsonl(args.output, [catalog])
    print(
        json.dumps(
            {
                "file": str(args.output),
                "cues": len(cues),
                "passages": len(passages),
                "sha256": sha256(args.output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
