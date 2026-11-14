import unittest
from unittest.mock import patch

from alfa_ai.evidence import EvidenceError, locate_paragraphs
from alfa_ai.generation import generate_with_context, validate_response
from alfa_ai.settings import Settings


def answer(body, quote, draft_text="Подтверждено источниками."):
    context = {"S1": {"source_id": "S1", "text": body, "document_id": "synthetic"}}
    draft = {
        "status": "answer",
        "text": draft_text,
        "evidence": [{"source_id": "S1", "quote": quote}],
    }
    return validate_response(draft, context)


class EvidenceRenderingTests(unittest.TestCase):
    def test_reviewed_group_keeps_reason_with_recommendation(self):
        reason = "Изображение на этикетке менее долговечно."
        recommendation = "При долгом хранении лучше выбрать другой способ печати."
        context = {
            "S1": {
                "source_id": "S1",
                "document_id": "synthetic",
                "source_block_orders": [1, 2],
                "text": f"{reason}\n\n{recommendation}\n\nДругой раздел.",
            }
        }
        draft = {
            "status": "answer",
            "text": "Ответ.",
            "evidence": [{"source_id": "S1", "quote": recommendation}],
        }
        with patch("alfa_ai.evidence.PARAGRAPH_GROUPS", {("synthetic", (1, 2)): ((0, 1),)}):
            result = validate_response(draft, context)
        self.assertIn(f"{reason}\n\n{recommendation}", result["text"])
        self.assertNotIn("Другой раздел.", result["text"])

    def test_true_quote_does_not_license_invented_paraphrase(self):
        body = "Позитивную обратную связь можно сделать публичной."
        result = answer(body, body, "Обязательно хвалите всех сотрудников публично.")
        self.assertIn(body, result["text"])
        self.assertNotIn("Обязательно", result["text"])

    def test_negation_and_condition_outside_anchor_are_preserved(self):
        body = "При длительном хранении не используйте такие этикетки: изображение выцветает."
        result = answer(body, "используйте такие этикетки")
        self.assertIn(body, result["text"])

    def test_property_is_not_transferred_between_objects(self):
        body = "Магазин А продаёт одежду.\n\nМагазин Б продаёт только украшения."
        result = answer(body, "Магазин Б продаёт", "Оба магазина продают только украшения.")
        self.assertIn("Магазин Б продаёт только украшения.", result["text"])
        self.assertNotIn("Оба магазина", result["text"])
        self.assertNotIn("Магазин А", result["text"])

    def test_original_case_whitespace_and_list_markers_are_preserved(self):
        body = "Показатели:\n\n- Средний чек.\n- Частота посещений.\n\nСравните периоды."
        result = answer(body, "средний   чек.")
        self.assertIn("- Средний чек.\n- Частота посещений.", result["text"])
        self.assertNotIn("средний   чек", result["text"])

    def test_multiline_anchor_restores_all_intersecting_paragraphs(self):
        body = "Первый абзац с условием.\n\nВторой абзац с оговоркой."
        paragraphs, indexes = locate_paragraphs(body, "с условием. Второй абзац")
        self.assertEqual(indexes, {0, 1})
        self.assertEqual(paragraphs[1], "Второй абзац с оговоркой.")

    def test_podcast_reference_includes_preceding_sentence(self):
        title = "Гость называет фильм «Северный ветер»."
        quote = "Этот фильм мы разбирали на учебной программе."
        context = {
            "S1": {"source_id": "S1", "document_id": "podcast:demo", "text": f"{title}\n\n{quote}"}
        }
        draft = {
            "status": "answer",
            "text": "ok",
            "evidence": [{"source_id": "S1", "quote": quote}],
        }
        self.assertIn(title, validate_response(draft, context)["text"])

    def test_explicit_omission_requires_each_quote_anchor_in_source_order(self):
        body = "Первый абзац с условием.\n\nВажное исключение.\n\nПоследний абзац с оговоркой."
        for quote in (
            "Первый абзац с условием. ... Последний абзац с оговоркой.",
            "Первый абзац с условием... Последний абзац с оговоркой.",
        ):
            with self.subTest(quote=quote):
                _, indexes = locate_paragraphs(body, quote)
                self.assertEqual(indexes, {0, 1, 2})
        for quote in (
            "Первый абзац с условием. … Подставленный текст.",
            "Последний абзац с оговоркой. ... Первый абзац с условием.",
        ):
            with self.subTest(quote=quote), self.assertRaises(EvidenceError):
                locate_paragraphs(body, quote)

    def test_duplicate_evidence_does_not_duplicate_paragraph(self):
        body = "Подтверждённое предложение для проверки."
        draft = {
            "status": "answer",
            "text": "Ответ.",
            "evidence": [{"source_id": "S1", "quote": body}] * 2,
        }
        result = validate_response(draft, {"S1": {"source_id": "S1", "text": body}})
        self.assertEqual(result["text"].count(body), 1)
        self.assertEqual(len(result["sources"]), 1)

    def test_ambiguous_anchor_is_rejected(self):
        body = "Общее условие здесь. Первый вариант.\n\nОбщее условие здесь. Второй вариант."
        with self.assertRaises(EvidenceError):
            locate_paragraphs(body, "Общее условие здесь.")

    def test_fabricated_quote_is_still_rejected(self):
        with self.assertRaises(EvidenceError):
            answer("Цена не указана в источнике.", "Цена составляет 100 рублей.")

    def test_truncated_json_records_length_before_parsing(self):
        settings = Settings()
        replies = [
            {"models": [{"name": settings.model, "digest": settings.model_digest}]},
            {"message": {"content": '{"status": "answer", "text": "'}, "done_reason": "length"},
        ]
        trace = {}
        with patch("alfa_ai.generation.local_request", side_effect=replies):
            with self.assertRaisesRegex(ValueError, "обрезан лимитом"):
                generate_with_context("вопрос", {}, trace=trace, settings=settings)
        self.assertEqual(trace, {"done_reason": "length"})


if __name__ == "__main__":
    unittest.main()
