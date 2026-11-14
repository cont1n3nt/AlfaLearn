import asyncio
import json
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.error import URLError

from alfa_ai import AIRequest, AIResponse, AIService, Settings, Source
from alfa_ai.generation import ModelUnavailable, generate_with_context

SOURCE = Source("S1", "article:test", "chunk:test", "Тест", (1,), "https://example.org/test/")


class FakeEngine:
    def __init__(self, error=None):
        self.error = error

    def answer(self, query):
        if self.error:
            raise self.error
        return {"status": "clarify", "text": query, "sources": []}


class BlockingEngine:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def answer(self, query):
        self.calls.append(query)
        self.entered.set()
        if not self.release.wait(3):
            raise TimeoutError("test safety deadline")
        return {"status": "clarify", "text": query, "sources": []}


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def make_service(self, engine, **settings):
        service = AIService(engine, Settings(**settings))
        self.addAsyncCleanup(service.aclose)
        # Async cleanups run LIFO: release a blocked fake before awaiting shutdown.
        if isinstance(engine, BlockingEngine):
            self.addAsyncCleanup(asyncio.to_thread, engine.release.set)
        return service

    async def wait_entered(self, engine):
        self.assertTrue(await asyncio.to_thread(engine.entered.wait, 1))

    async def test_explicit_course_lookup_returns_cards_without_model(self):
        engine = FakeEngine(AssertionError("catalog lookup must not call the model"))
        service = await self.make_service(engine)
        result = await service.answer(AIRequest("Где найти курс по УСН на маркетплейсах?"))
        self.assertEqual(result.status, "no_answer")
        self.assertEqual([card.access for card in result.recommendations], ["paid"])
        free = await service.answer(AIRequest("Где найти бесплатный курс по УСН на маркетплейсах?"))
        self.assertEqual(free.status, "no_answer")
        self.assertEqual(free.recommendations, ())
        paid = await service.answer(AIRequest("Где найти платный курс по маркетплейсам?"))
        self.assertEqual(paid.recommendations, ())

    async def test_course_lookup_with_factual_question_goes_to_model(self):
        service = await self.make_service(FakeEngine())
        for query in (
            "Где найти курс по УСН на маркетплейсах и как выбрать налоговый режим?",
            "Где найти курс валют для продавцов на Ozon?",
        ):
            with self.subTest(query=query):
                result = await service.answer(AIRequest(query))
                self.assertEqual(result.text, query)

    async def test_event_loop_runs_while_generation_is_blocked(self):
        engine = BlockingEngine()
        service = await self.make_service(engine)
        task = asyncio.create_task(service.answer(AIRequest("первый")))
        await self.wait_entered(engine)
        self.assertFalse(task.done())
        engine.release.set()
        self.assertEqual((await task).text, "первый")

    async def test_two_requests_do_not_share_results_or_trace_ids(self):
        service = await self.make_service(FakeEngine(), max_workers=2)
        first, second = await asyncio.gather(
            service.answer(AIRequest("первый")), service.answer(AIRequest("второй"))
        )
        self.assertEqual((first.text, second.text), ("первый", "второй"))
        self.assertNotEqual(first.trace_id, second.trace_id)

    async def test_timeout_does_not_free_running_capacity(self):
        engine = BlockingEngine()
        service = await self.make_service(engine, request_timeout=0.05, max_pending=1)
        first = await service.answer(AIRequest("первый"))
        self.assertEqual(first.error_code, "timeout")
        self.assertTrue(engine.entered.is_set())
        second = await service.answer(AIRequest("второй"))
        self.assertEqual(second.error_code, "busy")
        self.assertEqual(engine.calls, ["первый"])
        engine.release.set()
        await service.aclose()

    async def test_cancellation_propagates_and_retains_running_capacity(self):
        engine = BlockingEngine()
        service = await self.make_service(engine, max_pending=1)
        first = asyncio.create_task(service.answer(AIRequest("первый")))
        await self.wait_entered(engine)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual((await service.answer(AIRequest("второй"))).error_code, "busy")
        engine.release.set()

    async def test_cancelled_queued_request_never_reaches_engine(self):
        engine = BlockingEngine()
        service = await self.make_service(engine, max_pending=2)
        first = asyncio.create_task(service.answer(AIRequest("первый")))
        await self.wait_entered(engine)
        second = asyncio.create_task(service.answer(AIRequest("второй")))
        await asyncio.sleep(0)  # Let second enter the executor queue.
        self.assertEqual((await service.answer(AIRequest("третий"))).error_code, "busy")
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        engine.release.set()
        await first
        await service.aclose()
        self.assertEqual(engine.calls, ["первый"])

    async def test_shutdown_waits_for_running_request_and_rejects_new_ones(self):
        engine = BlockingEngine()
        service = await self.make_service(engine)
        task = asyncio.create_task(service.answer(AIRequest("первый")))
        await self.wait_entered(engine)
        closing = asyncio.create_task(service.aclose())
        await asyncio.sleep(0)
        self.assertFalse(closing.done())
        self.assertEqual((await service.answer(AIRequest("второй"))).error_code, "closed")
        engine.release.set()
        await asyncio.gather(task, closing)

    async def test_errors_are_sanitized_and_not_reported_as_no_answer(self):
        failures = [
            (URLError("secret service URL"), "unavailable"),
            (TimeoutError("private query"), "timeout"),
            (ModelUnavailable("local model path"), "model_unavailable"),
            (ValueError("private context"), "invalid_response"),
            (RuntimeError("secret data"), "internal_error"),
        ]
        for error, expected in failures:
            with self.subTest(expected=expected):
                service = await self.make_service(FakeEngine(error))
                with self.assertLogs("alfa_ai.service") as logs:
                    result = await service.answer(AIRequest("private query"))
                self.assertEqual(result.status, "error")
                self.assertEqual(result.error_code, expected)
                self.assertNotIn(str(error), json.dumps(result.to_dict()))
                self.assertNotIn("private query", " ".join(logs.output))

    async def test_corrupt_output_is_not_exposed(self):
        engine = FakeEngine()
        with patch.object(
            engine, "answer", return_value={"status": "answer", "text": "fake [S9]", "sources": []}
        ):
            service = await self.make_service(engine)
            with self.assertLogs("alfa_ai.service"):
                result = await service.answer(AIRequest("вопрос"))
        self.assertEqual(result.error_code, "invalid_response")

    async def test_missing_corpus_fails_at_startup(self):
        with self.assertRaises(FileNotFoundError):
            await AIService.create(Settings(corpus_path="missing-test-corpus.jsonl"))


class ContractTests(unittest.TestCase):
    def test_invalid_query_or_unsupported_history_rejected(self):
        for query in (None, " ", "x" * 1001, 123):
            with self.subTest(query=query), self.assertRaises(ValueError):
                AIRequest(query)
        with self.assertRaises(TypeError):
            AIRequest("вопрос", history=[])

    def test_response_serializes_without_private_context(self):
        response = AIResponse("answer", "Ответ [S1]", "trace", (SOURCE,))
        output = json.loads(json.dumps(response.to_dict()))
        self.assertEqual(output["sources"][0]["source_block_orders"], [1])
        self.assertNotIn("evidence", output)
        self.assertNotIn("text", output["sources"][0])

    def test_invalid_source_or_citation_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(SOURCE, url="javascript:alert(1)")
        with self.assertRaises(ValueError):
            AIResponse("answer", "Ответ [S99]", "trace", (SOURCE,))

    def test_only_local_ollama_allowed(self):
        for url in (
            "https://example.org",
            "http://localhost:11434",
            "http://127.0.0.1.evil/",
            "http://127.0.0.1:11434/api",
            "http://user:password@127.0.0.1",
            "http://127.0.0.1?url=external",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Settings(ollama_url=url)

    def test_invalid_limits_and_cloud_model_rejected(self):
        for config in (
            {"request_timeout": float("nan")},
            {"http_timeout": 0},
            {"max_workers": 0},
            {"max_workers": 2, "max_pending": 1},
            {"model": "test:cloud"},
            {"model_digest": "unknown"},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                Settings(**config)

    def test_cloud_metadata_cannot_satisfy_local_model_check(self):
        settings = Settings()
        tags = {
            "models": [
                {
                    "name": settings.model,
                    "digest": settings.model_digest,
                    "remote_host": "https://example.org",
                }
            ]
        }
        with patch("alfa_ai.generation.local_request", return_value=tags) as request:
            with self.assertRaises(ModelUnavailable):
                generate_with_context("вопрос", {}, settings=settings)
        self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
