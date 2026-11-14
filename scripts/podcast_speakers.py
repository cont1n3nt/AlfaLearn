"""Assign local speaker labels to words recognized by Whisper."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def identify_turns(samples, segmentation: Path, embedding: Path, speaker_count: int):
    import sherpa_onnx as sherpa

    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
            num_threads=4,
        ),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=4),
        clustering=sherpa.FastClusteringConfig(num_clusters=speaker_count, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("Не удалось загрузить локальные ONNX-модели разделения голосов.")
    diarizer = sherpa.OfflineSpeakerDiarization(config)
    if diarizer.sample_rate != 16_000:
        raise ValueError("Модели разделения голосов должны принимать аудио с частотой 16 кГц.")
    labels: dict[int, str] = {}
    turns = []
    for segment in diarizer.process(samples).sort_by_start_time():
        label = labels.setdefault(segment.speaker, f"Speaker {len(labels) + 1}")
        turns.append(SpeakerTurn(float(segment.start), float(segment.end), label))
    return turns


def speaker_cues(segments: Iterable, turns: list[SpeakerTurn], offset: float):
    """Split at changes in the word-level speaker assignment."""

    def overlap(unit, turn):
        return max(0.0, min(unit.end, turn.end) - max(unit.start, turn.start))

    previous_end = offset
    for segment in segments:
        preceding_end = previous_end
        groups: list[dict] = []
        for unit in segment.words or (segment,):
            text = unit.word if segment.words else segment.text
            if not text.strip():
                continue
            best = max(turns, key=lambda turn: overlap(unit, turn), default=None)
            speaker = best.speaker if best is not None and overlap(unit, best) > 0 else None
            start = max(previous_end, float(unit.start) + offset)
            end = max(start, float(unit.end) + offset)
            if groups and groups[-1].get("speaker") == speaker:
                groups[-1]["text"] += text
                groups[-1]["end_sec"] = end
            else:
                cue = {"start_sec": start, "end_sec": end, "text": text}
                if speaker is not None:
                    cue["speaker"] = speaker
                groups.append(cue)
            previous_end = end
        if any(cue["start_sec"] == cue["end_sec"] for cue in groups):
            # Fall back to the segment's interval if a word has no usable duration.
            start = max(preceding_end, float(segment.start) + offset)
            previous_end = max(previous_end, float(segment.end) + offset)
            if previous_end <= start:
                raise ValueError("У реплики отсутствует корректный интервал воспроизведения.")
            yield {"start_sec": start, "end_sec": previous_end, "text": segment.text.strip()}
        else:
            for cue in groups:
                cue["text"] = cue["text"].strip()
                yield cue
