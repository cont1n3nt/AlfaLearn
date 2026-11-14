"""Regression tests for HTML normalization and E5 passage boundaries."""

import unittest

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from scripts.chunk_articles import (
    MODEL_ID,
    MODEL_REVISION,
    PREFIX_BUDGET,
    TOKEN_BUDGET,
    chunk_document,
    count_tokens,
    embedding_prefix,
    split_passages,
)
from scripts.corpus import split_sections
from scripts.prepare_articles import clean_html


class HtmlTests(unittest.TestCase):
    def test_blank_lines_and_headings(self):
        html = (
            r"<p>A<br><br><br>B</p><h2>Section</h2><p><b>#</b> 123</p><p>\# 456</p><p>\\# 789</p>"
        )
        sections = split_sections([{"order": 1, "text": clean_html(html)}])
        self.assertEqual([p["text"] for p in sections[0]["paragraphs"]], ["A", "B"])
        self.assertEqual(sections[1]["headings"], ["Section"])
        self.assertEqual(
            [p["text"] for p in sections[1]["paragraphs"]],
            ["# 123", r"\# 456", r"\\# 789"],
        )

    def test_heading_without_body(self):
        self.assertEqual(split_sections([{"order": 1, "text": clean_html("<h2>Title</h2>")}]), [])


class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = hf_hub_download(MODEL_ID, "tokenizer.json", revision=MODEL_REVISION)
        cls.tokenizer = Tokenizer.from_file(path)
        cls.tokenizer.no_truncation()
        cls.tokenizer.no_padding()

    def test_exact_budget_and_one_token_over(self):
        prefix = "passage: "
        for delta in (0, 1):
            with self.subTest(delta=delta):
                target = TOKEN_BUDGET + delta
                body = " ".join(["word"] * (target - count_tokens(self.tokenizer, prefix)))
                self.assertEqual(count_tokens(self.tokenizer, prefix + body), target)
                spans = list(split_passages(body, [len(body)], prefix, self.tokenizer))
                self.assertEqual(len(spans), delta + 1)
                self.assertEqual("".join(body[start:end] for start, end in spans), body)
                for start, end in spans:
                    self.assertLessEqual(
                        count_tokens(self.tokenizer, prefix + body[start:end].strip()),
                        TOKEN_BUDGET,
                    )

    def test_long_title_is_shortened_at_word_boundary(self):
        title = ("Длинный заголовок с английскими словами " * 150).strip()
        prefix = embedding_prefix(title, self.tokenizer)
        shortened = prefix.removeprefix("passage: ").strip()
        self.assertTrue(shortened)
        self.assertLess(len(shortened), len(title))
        self.assertTrue(title.startswith(shortened + " "))
        self.assertLessEqual(count_tokens(self.tokenizer, prefix), PREFIX_BUDGET)

        heading = "Раздел " * 300 + "конец"
        body = "Проверка текста 😀 с числами 123 и аббревиатурами. " * 150 + "Последнее слово."
        article = {
            "document_id": "article:test",
            "title": title,
            "url": None,
            "slug": "test",
            "blocks": [{"order": 1, "text": f"## {heading}\n\n{body}"}],
            "answer_allowed": False,
            "recommend_allowed": False,
            "external_api_allowed": False,
        }
        chunks = chunk_document(article, self.tokenizer, "test")
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), body)
        for chunk in chunks:
            self.assertEqual(chunk["title"], title)
            self.assertEqual(chunk["section"], [heading])
            self.assertEqual(chunk["slug"], "test")
            self.assertLessEqual(chunk["token_count"], TOKEN_BUDGET)

    def test_long_word_without_separators(self):
        body = "абвгд😀" * 400
        prefix = "passage: "
        spans = list(split_passages(body, [len(body)], prefix, self.tokenizer))
        self.assertGreater(len(spans), 1)
        self.assertEqual("".join(body[start:end] for start, end in spans), body)
        for start, end in spans:
            self.assertLessEqual(
                count_tokens(self.tokenizer, prefix + body[start:end]), TOKEN_BUDGET
            )


if __name__ == "__main__":
    unittest.main()
