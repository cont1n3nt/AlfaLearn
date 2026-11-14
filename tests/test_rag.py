"""Offline regression tests using synthetic passages; no Ollama or corpus files."""

import unittest
from hashlib import sha256
from unittest.mock import patch

import numpy as np

from alfa_ai.engine import RagEngine
from alfa_ai.evidence import EvidenceError
from alfa_ai.generation import generate_with_context, validate_response
from alfa_ai.retrieval import diverse_context
from alfa_ai.reviewed import approved_passages
from alfa_ai.settings import DEFAULT_DIGEST, DEFAULT_MODEL
from scripts.eval_rag import evaluate_case

BODY = "Ошибки сотрудника обсуждайте наедине, без свидетелей."
CHUNK = {
    "document_id": "article:synthetic",
    "chunk_id": "synthetic-chunk",
    "source_block_orders": [30],
    "text": BODY,
    "title": "Разговор с сотрудником",
    "section": [],
    "embedding_text": "untrusted exported input",
    "url": "https://invalid.example/",
}
REVIEWED = {
    CHUNK["document_id"]: (
        "https://example.org/reviewed/",
        {(30,): sha256(BODY.encode()).hexdigest()},
    )
}
CONTEXT = {"S1": {**CHUNK, "source_id": "S1", "url": REVIEWED[CHUNK["document_id"]][0]}}


def draft():
    return {
        "status": "answer",
        "text": "Обсудите ошибку наедине.",
        "evidence": [{"source_id": "S1", "quote": BODY}],
    }


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.review_patch = patch("alfa_ai.reviewed.REVIEWED", REVIEWED)
        self.review_patch.start()
        self.addCleanup(self.review_patch.stop)

    def test_unknown_passage_is_excluded_and_url_is_trusted(self):
        unknown = {**CHUNK, "document_id": "article:unreviewed"}
        approved = approved_passages([CHUNK, unknown])
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["url"], CONTEXT["S1"]["url"])
        self.assertNotIn("untrusted", approved[0]["embedding_text"])

    def test_changed_or_missing_body_fails_closed(self):
        for chunks in ([], [{**CHUNK, "text": BODY + " изменено"}]):
            with self.subTest(chunks=chunks), self.assertRaises(ValueError):
                approved_passages(chunks)

    def test_duplicate_passage_is_rejected(self):
        with self.assertRaises(ValueError):
            approved_passages([CHUNK, CHUNK])

    def test_paraphrase_without_old_topic_keywords_reaches_generation(self):
        pilot = RagEngine([CHUNK])
        with patch(
            "alfa_ai.engine.generate_with_context", return_value={"status": "answer"}
        ) as generate:
            pilot.answer("Нужно обсудить ошибку подчинённого: наедине или при коллегах?")
        self.assertEqual(generate.call_args.args[1]["S1"]["chunk_id"], CHUNK["chunk_id"])

    def test_evidence_failure_abstains_and_records_reason(self):
        trace = {}
        with patch("alfa_ai.engine.generate_with_context", side_effect=EvidenceError("bad quote")):
            result = RagEngine([CHUNK]).answer("Как обсудить ошибку?", trace=trace)
        self.assertEqual(result["status"], "no_answer")
        self.assertEqual(result["sources"], [])
        self.assertEqual(trace["evidence_error"], "bad quote")

    def test_invalid_query_rejected_before_generation(self):
        pilot = RagEngine([CHUNK])
        for query in (" ", "a" * 1001):
            with self.subTest(query=query), self.assertRaises(ValueError):
                pilot.answer(query)

    def test_retrieval_only_replaces_duplicate_with_close_distinct_source(self):
        passages = [{"document_id": doc} for doc in ("a", "a", "b", "c")]
        self.assertEqual(
            diverse_context(range(4), np.array([0.20, 0.16, 0.15, 0.145]), passages, 3), [0, 2, 3]
        )
        self.assertEqual(
            diverse_context(range(4), np.array([0.20, 0.16, 0.15, 0.13]), passages, 3), [0, 1, 2]
        )

    def test_bare_reference_does_not_let_retrieval_invent_a_topic(self):
        pilot = RagEngine([CHUNK])
        with patch("alfa_ai.engine.generate_with_context") as generate:
            self.assertEqual(pilot.answer("Как это настроить?")["status"], "clarify")
            generate.assert_not_called()
            pilot.answer("Как это настроить в CRM?")
            generate.assert_called_once()


class EvidenceTests(unittest.TestCase):
    def test_valid_quote_binds_trusted_source(self):
        result = validate_response(draft(), CONTEXT)
        self.assertTrue(result["text"].endswith("[S1]"))
        self.assertEqual(result["sources"][0]["url"], CONTEXT["S1"]["url"])
        self.assertNotIn("text", result["sources"][0])

    def test_unknown_source_or_fabricated_quote_rejected(self):
        for evidence in (
            {"source_id": "S99", "quote": BODY},
            {"source_id": "S1", "quote": "Публичная критика всегда полезна."},
            {"source_id": "S1", "quote": "Ошибки"},
        ):
            response = {**draft(), "evidence": [evidence]}
            with self.subTest(evidence=evidence), self.assertRaises(EvidenceError):
                validate_response(response, CONTEXT)

    def test_model_cannot_supply_url_or_label(self):
        for text in (
            "Ссылка https://example.org",
            "Ответ [S99]",
            "Ответ (S1)",
            "Ответ S1",
            "[ссылка](https://example.org)",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_response({**draft(), "text": text}, CONTEXT)

    def test_status_evidence_mismatch_rejected(self):
        for response in ({**draft(), "evidence": []}, {**draft(), "status": "no_answer"}):
            with self.subTest(response=response), self.assertRaises(EvidenceError):
                validate_response(response, CONTEXT)

    def test_clarification_has_no_citations(self):
        response = {"status": "clarify", "text": "Что настроить?", "evidence": []}
        self.assertEqual(validate_response(response, CONTEXT)["sources"], [])

    def test_non_answer_does_not_repeat_unsupported_facts_or_guessed_topics(self):
        for status in ("clarify", "no_answer"):
            response = {"status": status, "text": "Принтер стоит 1000 рублей.", "evidence": []}
            with self.subTest(status=status):
                result = validate_response(response, CONTEXT)
                self.assertNotIn("1000", result["text"])
                self.assertNotIn("Принтер", result["text"])
                self.assertEqual(result["status"], status)

    def test_malformed_schema_rejected(self):
        for response in (None, {**draft(), "extra": 1}, {**draft(), "text": ""}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                validate_response(response, CONTEXT)

    def test_truncated_generation_is_error(self):
        import json

        replies = [
            {"models": [{"name": DEFAULT_MODEL, "digest": DEFAULT_DIGEST}]},
            {"message": {"content": json.dumps(draft())}, "done_reason": "length"},
        ]
        with (
            patch("alfa_ai.generation.local_request", side_effect=replies),
            self.assertRaises(ValueError),
        ):
            generate_with_context("Как обсудить ошибку?", CONTEXT)


class EvaluationTests(unittest.TestCase):
    def test_missing_second_source_fails_even_when_status_matches(self):
        source = {"document_id": "a", "source_block_orders": [1]}
        second = {"document_id": "a", "source_block_orders": [2]}
        case = {"expected_status": "answer", "required_sources": [source, second]}
        result = {"status": "answer", "sources": [source]}
        checks = evaluate_case(case, result, {"context": [source, second]})
        self.assertTrue(checks["retrieval_ok"])
        self.assertFalse(checks["automatic_pass"])

    def test_automatic_pass_never_means_manual_review(self):
        case = {"expected_status": "no_answer", "required_sources": []}
        checks = evaluate_case(case, {"status": "no_answer", "sources": []}, {})
        self.assertTrue(checks["automatic_pass"])
        self.assertIsNone(checks["manual_review"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
