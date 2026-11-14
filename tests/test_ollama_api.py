"""HTTP integration and lifecycle checks with no external model calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from alfalearn.api import create_app
from alfalearn.config import Settings
from alfalearn.grouping import OLLAMA_UNAVAILABLE
from alfalearn.llm import OllamaError


TEST_KEY = "test-only-api-key"


def settings(**overrides):
    values = {
        "_env_file": None,
        "bot_enabled": False,
        "bot_token": None,
        "ollama_enabled": True,
        "ollama_base_url": "https://ollama.com",
        "ollama_api_key": TEST_KEY,
        "ollama_model": "test-model",
        "ollama_timeout_seconds": 30,
        "ollama_max_concurrency": 4,
    }
    return Settings(**(values | overrides))


@pytest.fixture
def model(monkeypatch):
    brain = SimpleNamespace(
        answer=AsyncMock(return_value="Прибыль — выручка минус расходы."),
        classify_turn=AsyncMock(return_value="QUESTION"),
        close=AsyncMock(),
    )
    factory = Mock(return_value=brain)
    monkeypatch.setattr("alfalearn.api.OllamaClient", factory)
    return brain, factory


def test_answers_route_uses_ollama_interface_and_closes_on_shutdown(model):
    brain, factory = model
    with TestClient(create_app(settings())) as client:
        response = client.post("/answers", json={"message": "  Что такое прибыль?  "})
        assert response.status_code == 200
        assert response.json() == {"answer": "Прибыль — выручка минус расходы."}
        brain.answer.assert_awaited_once_with("Что такое прибыль?")
        brain.classify_turn.assert_not_awaited()
        brain.close.assert_not_awaited()
        factory.assert_called_once_with(
            base_url="https://ollama.com/", model="test-model", api_key=TEST_KEY,
            timeout_seconds=30, max_concurrency=4,
        )
    brain.close.assert_awaited_once()


@pytest.mark.parametrize("error", [OllamaError("upstream-private-detail"), TimeoutError()])
def test_upstream_failure_returns_safe_503_and_keeps_server_running(model, caplog, error):
    brain, _ = model
    brain.answer.side_effect = error
    with TestClient(create_app(settings())) as client:
        response = client.post("/answers", json={"message": "Как рассчитать прибыль?"})
        assert response.status_code == 503
        assert response.json() == {"answer": OLLAMA_UNAVAILABLE}
        assert TEST_KEY not in response.text
        assert "upstream-private-detail" not in response.text
        assert "upstream-private-detail" not in caplog.text
        assert TEST_KEY not in caplog.text
        assert client.get("/hello").status_code == 200
    brain.close.assert_awaited_once()


@pytest.mark.parametrize("message", ["", " ", "\n\t\r", "x" * 16_001])
def test_invalid_text_is_rejected_before_model_request(model, message):
    brain, _ = model
    with TestClient(create_app(settings())) as client:
        assert client.post("/answers", json={"message": message}).status_code == 422
        brain.answer.assert_not_awaited()
        brain.classify_turn.assert_not_awaited()


def test_health_shows_only_model_configuration_without_upstream_call(model):
    brain, _ = model
    with TestClient(create_app(settings())) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok", "bot": "disabled", "username": None,
            "llm": {"provider": "ollama", "model": "test-model"},
        }
        assert TEST_KEY not in response.text
        assert "api_key" not in response.text
        brain.answer.assert_not_awaited()
        brain.classify_turn.assert_not_awaited()


def test_model_client_is_closed_when_telegram_startup_fails(model):
    brain, factory = model
    with pytest.raises(RuntimeError, match="Set BOT_TOKEN"):
        with TestClient(create_app(settings(bot_enabled=True, bot_token=None))):
            pytest.fail("Startup should require the Telegram token")
    factory.assert_called_once()
    brain.close.assert_awaited_once()
    brain.answer.assert_not_awaited()


@pytest.mark.parametrize("key", [None, ""])
def test_cloud_requires_key_before_creating_client(model, key):
    brain, factory = model
    with pytest.raises(RuntimeError, match="Set OLLAMA_API_KEY"):
        with TestClient(create_app(settings(ollama_api_key=key))):
            pytest.fail("Cloud startup should require an API key")
    factory.assert_not_called()
    brain.answer.assert_not_awaited()


def test_local_ollama_works_without_api_key(model):
    brain, factory = model
    with TestClient(create_app(settings(
        ollama_base_url="http://127.0.0.1:11434", ollama_api_key=None,
        ollama_model="qwen3.5:4b",
    ))) as client:
        assert client.post("/answers", json={"message": "Объясни прибыль"}).status_code == 200
        assert client.get("/health").json()["llm"] == {
            "provider": "ollama", "model": "qwen3.5:4b",
        }
        assert factory.call_args.kwargs["api_key"] is None
        assert factory.call_args.kwargs["base_url"] == "http://127.0.0.1:11434/"
    brain.answer.assert_awaited_once_with("Объясни прибыль")
    brain.close.assert_awaited_once()


@pytest.mark.parametrize("origin", [
    "https://user:password@ollama.com",
    "https://user@ollama.com",
    "https://ollama.com?api_key=forbidden",
    "https://ollama.com/#fragment",
    "https://ollama.com/api",
    "https://ollama.com/api/chat",
    "http://ollama.com",
    "http://example.com:11434",
    "ftp://localhost:11434",
])
def test_ollama_origin_rejects_credentials_paths_queries_and_insecure_remote_http(origin):
    with pytest.raises(ValidationError):
        settings(ollama_base_url=origin)


@pytest.mark.parametrize("origin", [
    "https://ollama.com", "https://model.example.com:11434",
    "http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434",
])
def test_ollama_origin_accepts_https_and_local_http(origin):
    assert str(settings(ollama_base_url=origin).ollama_base_url).rstrip("/") == origin
