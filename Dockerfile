FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
RUN python -m pip install --no-cache-dir uv==0.12.19
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --group bot-runtime --no-install-project
COPY src/alfa_ai ./src/alfa_ai
COPY alfalearn ./alfalearn
COPY scripts/__init__.py scripts/fetch_model.py scripts/warm_model.py ./scripts/
COPY assets/reviewed_articles.jsonl ./assets/reviewed_articles.jsonl
COPY assets/reviewed_podcasts.json ./assets/reviewed_podcasts.json
COPY assets/corpus_manifest.json ./assets/corpus_manifest.json
RUN uv sync --locked --no-dev --group bot-runtime

CMD ["python", "-m", "alfalearn"]
