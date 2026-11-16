"""Preload the pinned local Ollama model before starting the HTTP/bot process."""

from alfa_ai.generation import local_request, pinned_model_available
from alfa_ai.settings import Settings


def main():
    settings = Settings.from_env()
    if not pinned_model_available(settings):
        raise RuntimeError("Pinned local Ollama model is missing")
    response = local_request(
        "/api/generate",
        {"model": settings.model, "prompt": "", "stream": False, "keep_alive": -1},
        settings=settings,
    )
    if response.get("model") != settings.model or not response.get("done"):
        raise RuntimeError("Ollama did not preload the pinned model")
    print("Pinned local model is loaded in memory")


if __name__ == "__main__":
    main()
