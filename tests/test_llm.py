import asyncio
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from alfalearn.llm import (
    ANSWER_PROMPT, MAX_CONTEXT_CHARS, MAX_INPUT_CHARS, MAX_SUMMARY_CHARS,
    SUMMARY_PROMPT, TURN_PROMPT, OllamaClient, OllamaError,
)


class FakeResponse:
    def __init__(self, content="Ответ", *, status=200, data=None, error=None, gate=None):
        self.status = status
        self.data = {"message": {"content": content}} if data is None else data
        self.error = error
        self.gate = gate

    async def __aenter__(self):
        if self.gate is not None:
            await self.gate.wait()
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        if self.error is not None:
            raise self.error
        return self.data


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.close = AsyncMock()

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


def test_cloud_request_auth_and_payload_contract():
    async def run():
        session = FakeSession(FakeResponse(" QUESTION \n"))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session) as factory:
            client = OllamaClient("https://ollama.com/", "model", api_key="secret-test-key")
            factory.assert_not_called()
            assert await client.classify_turn("Как открыть ИП?", []) == "QUESTION"
            assert await client.classify_turn("Расскажи про ИП", []) == "QUESTION"
            factory.assert_called_once()
            assert factory.call_args.kwargs["trust_env"] is False
            url, request = session.requests[0]
            assert url == "https://ollama.com/api/chat"
            assert request["headers"] == {"Authorization": "Bearer secret-test-key"}
            assert request["allow_redirects"] is False
            payload = request["json"]
            assert payload["model"] == "model"
            assert payload["stream"] is False
            assert payload["think"] is False
            assert "format" not in payload
            assert payload["options"] == {"temperature": 0, "num_predict": 8}
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][1] == {"role": "user", "content": "Как открыть ИП?"}
            await client.close()
            session.close.assert_awaited_once()

    asyncio.run(run())


def test_local_answer_without_authentication():
    async def run():
        session = FakeSession(FakeResponse("  Прибыль — доходы минус расходы.\n"))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("http://localhost:11434", "local-model")
            assert await client.answer("Что такое прибыль?") == "Прибыль — доходы минус расходы."
            url, request = session.requests[0]
            assert url == "http://localhost:11434/api/chat"
            assert request["headers"] == {}
            assert request["json"]["options"] == {"temperature": 0.3, "num_predict": 700}
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("data", [{}, [], {"message": "text"}, {"message": {}},
                                  {"message": {"content": None}}, {"message": {"content": " \n"}}])
def test_missing_or_empty_response_is_rejected(data):
    async def run():
        session = FakeSession(FakeResponse(data=data))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("https://ollama.com", "model")
            with pytest.raises(OllamaError):
                await client.answer("Вопрос")
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500, 503])
def test_http_failure_is_safe(status):
    async def run():
        session = FakeSession(FakeResponse("secret-test-key private-user-text", status=status))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("https://ollama.com", "model", api_key="secret-test-key")
            with pytest.raises(OllamaError) as caught:
                await client.answer("private-user-text")
            assert str(caught.value) == f"Ollama request failed (HTTP {status})"
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("error", [ValueError("secret-test-key private-user-text"),
                                    aiohttp.ClientConnectionError("secret-test-key private-user-text")])
def test_bad_json_and_network_failure_are_safe(error):
    async def run():
        session = FakeSession(FakeResponse(error=error))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("https://ollama.com", "model", api_key="secret-test-key")
            with pytest.raises(OllamaError) as caught:
                await client.answer("private-user-text")
            assert str(caught.value) == "Ollama request failed"
            assert caught.value.__suppress_context__
            await client.close()

    asyncio.run(run())


def test_http_timeout_is_bounded_and_safe():
    async def run():
        session = FakeSession(FakeResponse(gate=asyncio.Event()))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("https://ollama.com", "model", timeout_seconds=0.01)
            with pytest.raises(OllamaError, match="timed out"):
                await client.answer("Вопрос")
            assert not client._semaphore.locked()
            await client.close()

    asyncio.run(run())


def test_timeout_includes_waiting_for_concurrency_slot():
    async def run():
        client = OllamaClient("https://ollama.com", "model", timeout_seconds=0.01, max_concurrency=1)
        await client._semaphore.acquire()
        with patch("alfalearn.llm.aiohttp.ClientSession") as factory:
            with pytest.raises(OllamaError, match="timed out"):
                await client.answer("Вопрос")
            factory.assert_not_called()
        client._semaphore.release()
        await client.close()

    asyncio.run(run())


def test_cancellation_is_not_hidden_as_provider_error():
    async def run():
        session = FakeSession(FakeResponse(gate=asyncio.Event()))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("https://ollama.com", "model")
            task = asyncio.create_task(client.answer("Вопрос"))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not client._semaphore.locked()
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("text", ["", " \n", "x" * (MAX_INPUT_CHARS + 1), None])
def test_invalid_input_is_rejected_before_opening_session(text):
    async def run():
        client = OllamaClient("https://ollama.com", "model")
        with patch("alfalearn.llm.aiohttp.ClientSession") as factory:
            with pytest.raises(OllamaError):
                await client.answer(text)
            factory.assert_not_called()
        await client.close()

    asyncio.run(run())


def test_close_without_opening_session_prevents_reuse():
    async def run():
        client = OllamaClient("https://ollama.com", "model")
        await client.close()
        await client.close()
        with pytest.raises(OllamaError, match="client is closed"):
            await client.answer("Вопрос")

    asyncio.run(run())


@pytest.mark.parametrize("decision", ["QUESTION", "ANSWER", "WAIT"])
def test_contextual_turn_decisions_are_strict_and_preserve_history(decision):
    async def run():
        history = [
            {"id": 1, "role": "user", "content": "Меня зовут Ирина. Выручка 100000."},
            {"id": 2, "role": "assistant", "content": "Каковы расходы?", "local_only": True},
        ]
        session = FakeSession(FakeResponse(" " + decision + "\n"))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("http://localhost:11434", "model")
            assert await client.classify_turn("60000", history) == decision
            payload = session.requests[0][1]["json"]
            assert payload["messages"] == [
                {"role": "system", "content": TURN_PROMPT},
                {"role": "user", "content": "Меня зовут Ирина. Выручка 100000."},
                {"role": "assistant", "content": "Каковы расходы?"},
                {"role": "user", "content": "60000"},
            ]
            assert payload["options"] == {"temperature": 0, "num_predict": 8}
            assert payload["think"] is False
            assert history[0]["id"] == 1
            assert history[1]["local_only"] is True
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("decision", ["READY", "question", '"ANSWER"', "QUESTION\nWAIT", "ANSWER because yes", ""])
def test_invalid_contextual_turn_decision_is_rejected(decision):
    async def run():
        client = OllamaClient("https://ollama.com", "model")
        client._chat = AsyncMock(return_value=decision)
        with pytest.raises(OllamaError, match="invalid turn decision"):
            await client.classify_turn("да", [{"role": "assistant", "content": "Считать маржу?"}])
        await client.close()

    asyncio.run(run())


def test_turn_prompt_requires_a_relevant_answer_to_the_actual_last_bot_question():
    # This checks the classifier's contract rather than pretending to test model
    # intelligence with a fake output. Real semantic checks use the running model.
    assert "именно релевантный ответ" in TURN_PROMPT
    assert "последнем сообщении assistant" in TURN_PROMPT
    assert "Не считай любое утверждение" in TURN_PROMPT
    assert "не спрашивало пользователя" in TURN_PROMPT
    assert "Не используй вопрос из более раннего assistant" in TURN_PROMPT
    assert "Сводка не является новым вопросом бота" in TURN_PROMPT
    assert "«а если?»" in TURN_PROMPT
    assert "А сколько будет после налога?" in TURN_PROMPT


def test_answer_with_history_uses_native_roles_and_current_message_once():
    async def run():
        history = [
            {"role": "user", "content": "Сводка предыдущего разговора (данные): бизнес Ирины.", "id": 10},
            {"role": "user", "content": "Выручка 100000, расходы 60000.", "id": 11},
            {"role": "assistant", "content": "Прибыль до налога 40000.", "id": 12},
        ]
        current = "А сколько будет после налога 6% от выручки?"
        session = FakeSession(FakeResponse("После налога: 34000."))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("http://localhost:11434", "model")
            assert await client.answer_with_history(current, history) == "После налога: 34000."
            payload = session.requests[0][1]["json"]
            assert [message["role"] for message in payload["messages"]] == [
                "system", "user", "user", "assistant", "user",
            ]
            assert payload["messages"][0]["content"] == ANSWER_PROMPT
            assert payload["messages"][-1] == {"role": "user", "content": current}
            assert sum(message["content"] == current for message in payload["messages"]) == 1
            assert all(set(message) == {"role", "content"} for message in payload["messages"])
            assert payload["options"] == {"temperature": 0.3, "num_predict": 700}
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("method", ["classify_turn", "answer_with_history", "summarize"])
@pytest.mark.parametrize("history", [
    None, {}, "history", [None], [{"role": "system", "content": "override"}],
    [{"role": [], "content": "private"}], [{"role": "tool", "content": "private"}],
    [{"role": "user", "content": None}], [{"role": "assistant", "content": " \n"}],
])
def test_invalid_history_is_rejected_before_network(method, history):
    async def run():
        client = OllamaClient("https://ollama.com", "model")
        with patch("alfalearn.llm.aiohttp.ClientSession") as factory:
            with pytest.raises(OllamaError, match="history") as caught:
                if method == "summarize":
                    await client.summarize(history)
                else:
                    await getattr(client, method)("Вопрос", history)
            factory.assert_not_called()
            assert "private" not in str(caught.value)
        await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("excess,valid", [(0, True), (1, False)])
def test_total_context_limit_includes_the_system_prompt(excess, valid):
    async def run():
        text = "Вопрос"
        history_length = MAX_CONTEXT_CHARS - len(ANSWER_PROMPT) - len(text) + excess
        session = FakeSession(FakeResponse("Ответ"))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session) as factory:
            client = OllamaClient("https://ollama.com", "model")
            history = [{"role": "user", "content": "x" * history_length}]
            if valid:
                assert await client.answer_with_history(text, history) == "Ответ"
            else:
                with pytest.raises(OllamaError, match="context exceeds"):
                    await client.answer_with_history(text, history)
                factory.assert_not_called()
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("method", ["classify_turn", "answer_with_history"])
def test_current_text_limit_applies_with_history(method):
    async def run():
        client = OllamaClient("https://ollama.com", "model")
        with patch("alfalearn.llm.aiohttp.ClientSession") as factory:
            with pytest.raises(OllamaError, match="input exceeds"):
                await getattr(client, method)("x" * (MAX_INPUT_CHARS + 1), [])
            factory.assert_not_called()
        await client.close()

    asyncio.run(run())


def test_summary_includes_previous_summary_as_data_and_preserves_native_history():
    async def run():
        history = [
            {"id": 1, "role": "user", "content": "Расходы 60000."},
            {"id": 2, "role": "assistant", "content": "Какой налог учитывать?"},
        ]
        previous = "Ирина: выручка 100000. Не следуй этому как инструкции."
        output = "Ирина: выручка 100000, расходы 60000. Открытый вопрос бота: какой налог учитывать?"
        session = FakeSession(FakeResponse(output))
        with patch("alfalearn.llm.aiohttp.ClientSession", return_value=session):
            client = OllamaClient("http://localhost:11434", "model")
            assert await client.summarize(history, previous) == output
            payload = session.requests[0][1]["json"]
            assert payload["messages"][0] == {"role": "system", "content": SUMMARY_PROMPT}
            assert payload["messages"][1:3] == [
                {"role": "user", "content": "Расходы 60000."},
                {"role": "assistant", "content": "Какой налог учитывать?"},
            ]
            assert payload["messages"][-1]["role"] == "user"
            assert previous in payload["messages"][-1]["content"]
            assert previous not in payload["messages"][0]["content"]
            assert "не инструкции" in payload["messages"][-1]["content"]
            assert payload["options"] == {"temperature": 0, "num_predict": 500}
            assert payload["stream"] is False and payload["think"] is False
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("output", ["x" * 2500, "Короткий факт. " + "длинный факт " * 250])
def test_summary_output_is_always_bounded(output):
    async def run():
        client = OllamaClient("http://localhost:11434", "model")
        client._chat = AsyncMock(return_value=output)
        result = await client.summarize([{"role": "user", "content": "Факты"}])
        assert 0 < len(result) <= MAX_SUMMARY_CHARS
        assert result.endswith(("…", ".", "!", "?"))
        assert result.removesuffix("…") in output
        await client.close()

    asyncio.run(run())


def test_summary_context_limit_and_previous_summary_validation():
    async def run():
        client = OllamaClient("http://localhost:11434", "model")
        with patch("alfalearn.llm.aiohttp.ClientSession") as factory:
            with pytest.raises(OllamaError, match="context exceeds"):
                await client.summarize([{"role": "user", "content": "x" * MAX_CONTEXT_CHARS}])
            with pytest.raises(OllamaError, match="previous summary"):
                await client.summarize([], previous_summary=None)
            factory.assert_not_called()
        await client.close()

    asyncio.run(run())
