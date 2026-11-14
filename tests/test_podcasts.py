import json
import unittest
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from alfa_ai import AIResponse, Settings, Source
from alfa_ai.engine import RagEngine
from alfa_ai.generation import validate_response
from alfa_ai.podcasts import catalog_from_media_export, load_podcast_passages, podcast_passages

FIXTURE = Path(__file__).with_name("fixtures") / "podcast-demo.json"
MULTI_EPISODE = FIXTURE.with_name("podcast-eval-extra.json")


class PodcastTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(FIXTURE.read_bytes())

    def test_original_bounds_and_pause(self):
        passages = podcast_passages(self.document)
        self.assertEqual([(p["start_sec"], p["end_sec"]) for p in passages], [(12.5, 25), (41, 49)])
        self.assertIn("журнал остатков", passages[0]["text"])
        self.assertEqual(passages, podcast_passages(self.document))

    def test_search_terms_do_not_change_transcript_or_source_id(self):
        [original, *_] = podcast_passages(self.document)
        cue = self.document["episodes"][0]["cues"][0]
        cue["search_terms"] = "проверка наличия"
        [annotated, *_] = podcast_passages(self.document)
        self.assertIn("проверка наличия", annotated["embedding_text"])
        self.assertEqual(annotated["text"], original["text"])
        self.assertEqual(annotated["chunk_id"], original["chunk_id"])
        cue["search_terms"] = {"bad": "shape"}
        with self.assertRaises(ValueError):
            podcast_passages(self.document)

    def test_window_keeps_sentence_spanning_two_cues(self):
        self.document["episodes"][0]["cues"] = [
            {"start_sec": 0, "end_sec": 28, "text": "Покупатель говорит,"},
            {"start_sec": 28, "end_sec": 34, "text": "что хочет вернуться."},
            {"start_sec": 34, "end_sec": 40, "text": "Смена темы."},
        ]
        first, second = podcast_passages(self.document)
        self.assertEqual(first["text"], "Покупатель говорит, что хочет вернуться.")
        self.assertEqual(first["end_sec"], 34)
        self.assertEqual(second["start_sec"], 34)

    def test_media_export_preserves_playback_times_and_splits_on_voice_change(self):
        export = {
            "schema_version": 1,
            "time_unit": "seconds",
            "external_id": "e01",
            "source_start_seconds": 10,
            "source_url": "/api/videos/test/source",
            "segments": [{"t_start": 12, "t_end": 16, "text": "Исходная реплика."}],
            "speaker_segments": [
                {"t_start": 12, "t_end": 14, "text": "Вопрос ведущего", "speaker": "Speaker 1"},
                {"t_start": 14, "t_end": 16, "text": "Ответ гостя.", "speaker": "Speaker 2"},
            ],
        }
        metadata = {"episode_id": "e01", "title": "Выпуск", "url": "https://example.org/e01"}
        [raw] = podcast_passages(catalog_from_media_export(export, **metadata))
        self.assertEqual(raw["text"], "Исходная реплика.")
        passages = podcast_passages(
            catalog_from_media_export(export, **metadata, use_speakers=True)
        )
        self.assertEqual([(p["start_sec"], p["end_sec"]) for p in passages], [(12, 14), (14, 16)])
        self.assertEqual([p["text"] for p in passages], ["Вопрос ведущего", "Ответ гостя."])
        self.assertEqual(passages[0]["url"], metadata["url"])

    def test_media_export_rejects_mismatched_episode_or_time_unit(self):
        export = {"schema_version": 1, "time_unit": "seconds", "external_id": "wrong"}
        metadata = {"episode_id": "e01", "title": "Выпуск", "url": "https://example.org/e01"}
        with self.assertRaisesRegex(ValueError, "external_id"):
            catalog_from_media_export(export, **metadata)
        with self.assertRaisesRegex(ValueError, "секундах"):
            catalog_from_media_export({**export, "time_unit": "ms"}, **metadata)

    def test_catalog_hash_covers_link_and_timestamps(self):
        original = FIXTURE.read_bytes()
        for key, value in (("url", "https://example.org/another"), ("start_sec", 1)):
            modified = deepcopy(self.document)
            if key == "url":
                modified["episodes"][0][key] = value
            else:
                modified["episodes"][0]["cues"][0][key] = value
            with self.subTest(key=key), TemporaryDirectory() as directory:
                path = Path(directory) / "edited.json"
                path.write_text(json.dumps(modified), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "изменился"):
                    load_podcast_passages(path, sha256(original).hexdigest())

    def test_invalid_time_and_duplicate_episode_are_rejected(self):
        for time in (float("nan"), -1, True, 30):
            modified = deepcopy(self.document)
            modified["episodes"][0]["cues"][0]["start_sec"] = time
            with self.subTest(time=time), self.assertRaises(ValueError):
                podcast_passages(modified)
        self.document["episodes"] *= 2
        with self.assertRaisesRegex(ValueError, "повторяется"):
            podcast_passages(self.document)

    def test_retrieval_citation_and_api_keep_timestamps(self):
        settings = Settings(
            podcast_corpus_path=FIXTURE,
            podcast_corpus_sha256=sha256(FIXTURE.read_bytes()).hexdigest(),
        )
        with patch("alfa_ai.reviewed.REVIEWED", {}):
            pilot = RagEngine([], settings=settings)
        question = "Когда в мастерской «Лазурный енот» проводят инвентаризацию?"
        context = pilot.retrieve(question)
        selected = context["S1"]
        self.assertIn("четверг", selected["text"])
        draft = {
            "status": "answer",
            "text": "Ответ.",
            "evidence": [
                {"source_id": "S1", "quote": "инвентаризацию проводят каждый четверг."},
            ],
        }
        result = validate_response(draft, context)
        response = AIResponse(
            result["status"],
            result["text"],
            "test",
            tuple(Source(**source) for source in result["sources"]),
        )
        source = response.to_dict()["sources"][0]
        self.assertEqual((source["start_sec"], source["end_sec"]), (12.5, 25))
        self.assertEqual(source["url"], self.document["episodes"][0]["url"])
        self.assertIn("каждый четверг", response.text)

    def test_article_wire_format_and_optional_config(self):
        source = Source("S1", "article:a", "b", "Статья", (1,), "https://example.org/")
        result = AIResponse("answer", "Текст [S1]", "test", (source,)).to_dict()
        self.assertNotIn("start_sec", result["sources"][0])
        with self.assertRaises(ValueError):
            Settings(podcast_corpus_path=FIXTURE)


class PodcastScopeTests(unittest.TestCase):
    def setUp(self):
        settings = Settings(
            podcast_corpus_path=MULTI_EPISODE,
            podcast_corpus_sha256=sha256(MULTI_EPISODE.read_bytes()).hexdigest(),
        )
        with patch("alfa_ai.reviewed.REVIEWED", {}):
            self.pilot = RagEngine([], settings=settings)

    def test_explicit_episode_limits_context_before_generation(self):
        for query in (
            "Когда Коралловая лиса оформляет заказ на упаковку?",
            "Когда оформляют заказ на упаковку в https://example.org/podcasts/demo-coral/?",
        ):
            with (
                self.subTest(query=query),
                patch(
                    "alfa_ai.engine.generate_with_context",
                    return_value={"status": "no_answer", "sources": []},
                ) as generate,
            ):
                self.pilot.answer(query)
                context = generate.call_args.args[1]
                self.assertEqual(
                    {source["document_id"] for source in context.values()}, {"podcast:demo-coral"}
                )
                self.assertEqual((context["S1"]["start_sec"], context["S1"]["end_sec"]), (41, 49))

    def test_unspecified_episode_reference_asks_before_generation(self):
        trace = {}
        with patch("alfa_ai.engine.generate_with_context") as generate:
            result = self.pilot.answer(
                "Когда в мастерской из учебного выпуска пересчитывают запасы?", trace=trace
            )
        generate.assert_not_called()
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(trace["episode_clarification"], "unnamed_reference")

    def test_unscoped_mixed_citations_ask_which_episode(self):
        query = "В какой день мастерская заказывает упаковку?"
        sources = [{"document_id": "podcast:demo-coral"}, {"document_id": "podcast:demo-mint"}]
        with patch(
            "alfa_ai.engine.generate_with_context",
            return_value={"status": "answer", "sources": sources},
        ):
            result = self.pilot.answer(query)
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["sources"], [])

    def test_explicitly_named_episodes_can_be_compared(self):
        query = "Сравни, когда Мятный кит и Коралловая лиса заказывают упаковку."
        with patch(
            "alfa_ai.engine.generate_with_context",
            return_value={"status": "no_answer", "sources": []},
        ) as generate:
            self.pilot.answer(query)
        self.assertEqual(
            {source["document_id"] for source in generate.call_args.args[1].values()},
            {"podcast:demo-mint", "podcast:demo-coral"},
        )

    def test_short_selection_recovers_previous_question_and_keeps_scope(self):
        previous = "В какой день в мастерской из учебного выпуска пересчитывают запасы?"
        with patch(
            "alfa_ai.engine.generate_with_context",
            return_value={"status": "no_answer", "sources": []},
        ) as generate:
            self.pilot.answer("«Мятный кит»", previous_query=previous)
        question, context = generate.call_args.args
        self.assertIn(previous, question)
        self.assertEqual(
            {source["document_id"] for source in context.values()}, {"podcast:demo-mint"}
        )

    def test_semantic_index_only_ranks_named_episode_without_its_title(self):
        query = "Когда оформляют заказ на упаковку? Выпуск «Учебный выпуск: мастерская «Коралловая лиса»»."
        self.pilot.semantic_index = Mock(wraps=self.pilot.index)
        with patch(
            "alfa_ai.engine.generate_with_context",
            return_value={"status": "no_answer", "sources": []},
        ) as generate:
            self.pilot.answer(query)
            self.pilot.answer("Когда оформляют заказ на упаковку?")
        self.pilot.semantic_index.score.assert_called_once_with(
            ["Когда оформляют заказ на упаковку?"]
        )
        self.assertEqual(generate.call_args_list[0].args[0], query)
        self.assertEqual(
            {s["document_id"] for s in generate.call_args_list[0].args[1].values()},
            {"podcast:demo-coral"},
        )

    def test_semantic_supplements_named_lexical_results_without_duplicates(self):
        import numpy as np

        from alfa_ai.retrieval import select_context

        passage = self.pilot.passages[0]
        candidates = [{**passage, "chunk_id": str(i)} for i in range(7)]
        candidates[-1]["document_id"] = "podcast:other"
        lexical = Mock()
        lexical.score.return_value = np.array([[9, 8, 7, 6, 5, 4, 3]])
        semantic = Mock()
        semantic.score.return_value = np.array([[9, 4, 3, 2, 7, 8, 10]])
        context = select_context(
            "вопрос",
            candidates,
            lexical,
            document_ids={passage["document_id"]},
            limit=4,
            semantic_index=semantic,
        )
        self.assertEqual(
            [source["chunk_id"] for source in context.values()], ["0", "1", "2", "3", "5", "4"]
        )

    def test_named_episode_retries_invalid_evidence_with_shorter_quotes(self):
        from alfa_ai.evidence import EvidenceError

        expected = {"status": "no_answer", "sources": []}
        trace = {}
        with patch(
            "alfa_ai.engine.generate_with_context",
            side_effect=[EvidenceError("неверная цитата"), expected],
        ) as generate:
            result = self.pilot.answer(
                "Когда оформляют заказ на упаковку? Выпуск «Коралловая лиса».", trace=trace
            )
        self.assertEqual(result, expected)
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(generate.call_args.kwargs["brief"])
        self.assertEqual(trace["citation_retry"], "неверная цитата")

    def test_named_episode_retry_drops_semantic_supplements_after_truncation(self):
        from alfa_ai.generation import GenerationTruncated

        context = {f"S{i}": {"document_id": "podcast:demo-coral"} for i in range(1, 7)}
        expected = {"status": "no_answer", "sources": []}
        with (
            patch.object(self.pilot, "retrieve", return_value=context),
            patch(
                "alfa_ai.engine.generate_with_context",
                side_effect=[GenerationTruncated("обрезано"), expected],
            ) as generate,
        ):
            result = self.pilot.answer(
                "Когда оформляют заказ на упаковку? Выпуск «Коралловая лиса»."
            )
        self.assertEqual(result, expected)
        self.assertEqual(list(generate.call_args.args[1]), ["S1", "S2", "S3", "S4"])

    def test_named_lexical_search_combines_query_versions_and_limits_episode(self):
        plain = "Когда оформляют заказ на упаковку?"
        query = plain + " Выпуск «Коралловая лиса»."
        self.pilot.index = Mock(wraps=self.pilot.index)
        context = self.pilot.retrieve(query, document_ids={"podcast:demo-coral"})
        self.pilot.index.score.assert_called_once_with([query, plain])
        self.assertEqual(
            {source["document_id"] for source in context.values()}, {"podcast:demo-coral"}
        )
        self.assertEqual(context["S1"]["start_sec"], 41)

        self.pilot.retrieve(query)
        self.pilot.index.score.assert_called_with([query])


if __name__ == "__main__":
    unittest.main()
