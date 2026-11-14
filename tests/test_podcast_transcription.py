from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from scripts.podcast_speakers import SpeakerTurn, speaker_cues


class SpeakerCueTests(TestCase):
    def test_numpy_timestamps_become_json_numbers(self):
        segment = SimpleNamespace(
            start=np.float64(0.5), end=np.float64(1.5), text="Реплика.", words=None
        )
        [cue] = speaker_cues([segment], [], offset=2.0)
        self.assertIs(type(cue["start_sec"]), float)
        self.assertIs(type(cue["end_sec"]), float)
        self.assertEqual((cue["start_sec"], cue["end_sec"]), (2.5, 3.5))

    def test_keeps_timestamps_text_and_speaker_changes(self):
        segment = SimpleNamespace(
            start=0.0,
            end=2.0,
            text="Вопрос? Ответ!",
            words=[
                SimpleNamespace(start=0.0, end=1.0, word=" Вопрос?"),
                SimpleNamespace(start=1.0, end=2.0, word=" Ответ!"),
            ],
        )
        turns = [SpeakerTurn(0, 1, "Speaker 1"), SpeakerTurn(1, 2, "Speaker 2")]
        self.assertEqual(
            list(speaker_cues([segment], turns, offset=3)),
            [
                {"start_sec": 3, "end_sec": 4, "text": "Вопрос?", "speaker": "Speaker 1"},
                {"start_sec": 4, "end_sec": 5, "text": "Ответ!", "speaker": "Speaker 2"},
            ],
        )

    def test_speech_without_overlapping_turn_stays_unlabeled(self):
        segment = SimpleNamespace(start=2, end=3, text="Реплика.", words=None)
        self.assertEqual(
            list(speaker_cues([segment], [SpeakerTurn(0, 2, "Speaker 1")], offset=0)),
            [{"start_sec": 2, "end_sec": 3, "text": "Реплика."}],
        )

    def test_word_without_duration_preserves_entire_segment(self):
        segment = SimpleNamespace(
            start=0,
            end=2,
            text="Без потерь.",
            words=[
                SimpleNamespace(start=0, end=0, word=" Без"),
                SimpleNamespace(start=1, end=2, word=" потерь."),
            ],
        )
        self.assertEqual(
            list(speaker_cues([segment], [SpeakerTurn(1, 2, "Speaker 1")], offset=0)),
            [{"start_sec": 0, "end_sec": 2, "text": "Без потерь."}],
        )
