"""Fetch the pinned public GGUF into the Docker model volume if needed."""

import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

MODEL_NAME = "qwen35-unc-local:latest"
MODEL_DIGEST = "cd68126d1fec3ada81728c06f6649901d5ca209ac7662641bc1f5dd581947c96"
FILE_SHA256 = "2ca636d9e81d3d23ca9b60c234fe185d30ec082eeba69ce770fdb0c76559a4f5"
FILE_SIZE = 5_627_044_224
FILE_URL = (
    "https://huggingface.co/HauhauCS/"
    "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive/resolve/"
    "0a41c68809d375475f954be12ba7c40efa56c2a9/"
    "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
)
MODEL_PATH = Path("/models/model.gguf")
OLLAMA_TAGS = "http://ollama:11434/api/tags"


def model_ready() -> bool:
    with urlopen(OLLAMA_TAGS, timeout=10) as response:
        models = json.load(response)["models"]
    return any(
        row.get("name") == MODEL_NAME
        and row.get("digest") == MODEL_DIGEST
        and not row.get("remote_host")
        for row in models
    )


def verified_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != FILE_SIZE:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == FILE_SHA256


def fetch_model() -> None:
    if model_ready():
        print("Pinned Ollama model is already available", flush=True)
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if verified_file(MODEL_PATH):
        print("Pinned GGUF is already available", flush=True)
        return
    partial = MODEL_PATH.with_suffix(".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > FILE_SIZE:
        partial.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urlopen(Request(FILE_URL, headers=headers), timeout=120) as response:
        if offset and response.status != 206:
            raise RuntimeError("Model server did not support the requested resume range")
        mode = "ab" if offset else "wb"
        with partial.open(mode) as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
    if not verified_file(partial):
        raise RuntimeError("Downloaded GGUF size or SHA-256 does not match the pinned file")
    os.replace(partial, MODEL_PATH)
    print("Pinned GGUF verified", flush=True)


if __name__ == "__main__":
    fetch_model()
