"""Convert a local podcast-transcription results.json export to a catalog candidate."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from alfa_ai.podcasts import catalog_from_media_export, podcast_passages
from scripts.corpus import write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--url", required=True, help="Verified public episode page")
    parser.add_argument(
        "--speaker-segments", action="store_true", help="Use the export track split by speaker"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Файл вывода уже существует. Укажи новый путь для кандидата.")
    try:
        catalog = catalog_from_media_export(
            json.loads(args.input.read_bytes()),
            episode_id=args.episode_id,
            title=args.title,
            url=args.url,
            use_speakers=args.speaker_segments,
        )
        passages = podcast_passages(catalog)
        write_jsonl(args.output, [catalog])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "file": str(args.output),
                "cues": len(catalog["episodes"][0]["cues"]),
                "passages": len(passages),
                "sha256": sha256(args.output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
