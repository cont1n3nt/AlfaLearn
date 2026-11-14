"""Transcribe a permitted local recording to a candidate podcast catalog."""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from alfa_ai.podcasts import podcast_passages
from scripts.corpus import write_jsonl
from scripts.podcast_speakers import identify_turns, speaker_cues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    media = parser.add_mutually_exclusive_group(required=True)
    media.add_argument("--audio", type=Path, help="Local audio recording")
    media.add_argument("--video", type=Path, help="Local video with an audio track")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker-segmentation", type=Path, help="Local segmentation ONNX model")
    parser.add_argument("--speaker-embedding", type=Path, help="Local speaker embedding ONNX model")
    parser.add_argument("--speakers", type=int, help="Expected number of speakers, if known")
    args = parser.parse_args()
    recording = args.audio if args.audio is not None else args.video
    if not recording.is_file():
        parser.error("Укажи существующий локальный аудио- или видеофайл.")
    model_dir = args.model_dir.resolve()
    if not all(
        (model_dir / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")
    ):
        parser.error("--model-dir должен содержать локальную модель CTranslate2 с токенизатором.")
    if args.output.exists():
        parser.error("Файл вывода уже существует. Укажи новый путь для кандидата.")
    speaker_models = (args.speaker_segmentation, args.speaker_embedding)
    diarize = all(path is not None for path in speaker_models)
    if any(path is not None for path in speaker_models) and not diarize:
        parser.error("Укажи обе ONNX-модели для разделения голосов.")
    if diarize and not all(path.is_file() for path in speaker_models):
        parser.error("Файлы ONNX-моделей для разделения голосов не найдены.")
    if args.speakers is not None and (not diarize or args.speakers < 1):
        parser.error("--speakers должен быть положительным числом и требует обе ONNX-модели.")

    from faster_whisper import WhisperModel

    offset = 0.0
    if args.video is not None:
        try:
            offset = video_audio_offset(args.video)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", local_files_only=True)
    audio = str(recording)
    turns = None
    if diarize:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(recording))
        if not audio.size:
            parser.error("Аудиодорожка пуста.")
        try:
            turns = identify_turns(audio, *speaker_models, args.speakers or -1)
        except ModuleNotFoundError as exc:
            if exc.name != "sherpa_onnx":
                raise
            parser.error("Для разделения голосов установи зависимости через --group diarization.")
        except ValueError as exc:
            parser.error(str(exc))
    segments, _ = model.transcribe(
        audio, language=args.language, vad_filter=True, word_timestamps=diarize
    )
    tracked_segments = report_progress(segments)
    if turns is not None:
        cues = list(speaker_cues(tracked_segments, turns, offset))
    else:
        cues = [
            {
                "start_sec": float(segment.start) + offset,
                "end_sec": float(segment.end) + offset,
                "text": text,
            }
            for segment in tracked_segments
            if (text := segment.text.strip())
        ]
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
                "speakers": len({cue["speaker"] for cue in cues if "speaker" in cue}),
                "sha256": sha256(args.output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )
    return 0


def report_progress(segments):
    next_progress = 300
    for segment in segments:
        if segment.end >= next_progress:
            print(f"Распознано до {segment.end:.0f} с записи.", file=sys.stderr, flush=True)
            next_progress = (int(segment.end) // 300 + 1) * 300
        yield segment


def video_audio_offset(path: Path) -> float:
    """Translate decoded audio timestamps to the container's playback timeline."""
    import av

    with av.open(str(path)) as recording:
        if not recording.streams.video or not recording.streams.audio:
            raise ValueError("Для --video нужны видеоряд и аудиодорожка.")
        audio = recording.streams.audio[0]
        origin = recording.start_time
        if origin is None or audio.time_base is None:
            raise ValueError("Не удалось определить начало воспроизведения видео.")
        first_frame = next(recording.decode(audio=0), None)
        if first_frame is None or first_frame.pts is None or first_frame.time_base is None:
            raise ValueError("Не удалось определить начало аудиодорожки.")
        audio_start = first_frame.pts * first_frame.time_base
        offset = float(audio_start) - origin / av.time_base
        if offset < -0.001:
            raise ValueError("Аудиодорожка начинается раньше шкалы воспроизведения видео.")
        return max(0.0, offset)


if __name__ == "__main__":
    sys.exit(main())
