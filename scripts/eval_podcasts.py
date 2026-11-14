"""Compare timed retrieval against reference intervals across podcast episodes."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from urllib.error import URLError

import numpy as np

from alfa_ai.engine import RagEngine
from alfa_ai.evidence import normalized_text
from alfa_ai.generation import SYSTEM_PROMPT
from alfa_ai.podcasts import validate_interval
from alfa_ai.retrieval import (
    LIMIT,
    NAMED_EPISODE_LIMIT,
    NAMED_EPISODE_SEMANTIC_LIMIT,
    reciprocal_rank_fusion,
)
from alfa_ai.semantic import MODEL_REVISION
from scripts.corpus import write_jsonl

MIN_REFERENCE_COVERAGE = 0.5


def reference_coverage(passage, case):
    if passage["document_id"] != f"podcast:{case['episode_id']}":
        return 0.0
    start, end = case["start_sec"], case["end_sec"]
    overlap = min(passage["end_sec"], end) - max(passage["start_sec"], start)
    return max(0.0, overlap) / (end - start)


def print_metrics(label, ranks):
    print(
        f"{label}: Hit@1={ranks.count(1)}/{len(ranks)}, "
        f"Hit@{LIMIT}={sum(r is not None for r in ranks)}/{len(ranks)}, "
        f"MRR@{LIMIT}={sum(1 / r for r in ranks if r) / len(ranks):.3f}",
        flush=True,
    )


def evaluate_answers(pilot, cases, rows, output, *, name_episodes=False):
    titles = {passage["document_id"]: passage["title"] for passage in pilot.passages}
    passed = 0
    retrieved = 0
    for case in cases:
        trace = {}
        started = perf_counter()
        question = case["question"]
        if name_episodes:
            title = titles[f"podcast:{case['episode_id']}"]
            question = f"{question} Выпуск «{title}»."
        try:
            result = pilot.answer(question, trace=trace)
        except (OSError, ValueError, KeyError) as exc:
            result = {"status": "error", "sources": []}
            trace["error_type"] = type(exc).__name__
            if isinstance(exc, URLError) and isinstance(exc.reason, ConnectionRefusedError):
                print("Ollama недоступна. Запустите её и повторите проверку.", flush=True)
                break
        citations = result["sources"]
        context = trace.get("context", [])
        answer_text = normalized_text(result.get("text", "")).replace("ё", "е")
        missing_excerpts = [
            excerpt
            for excerpt in case.get("required_excerpts", [])
            if normalized_text(excerpt).replace("ё", "е") not in answer_text
        ]
        coverage = max((reference_coverage(source, case) for source in citations), default=0.0)
        wrong_source = any(
            source["document_id"] != f"podcast:{case['episode_id']}" for source in citations
        )
        retrieval_rank = next(
            (
                rank
                for rank, source in enumerate(context, 1)
                if reference_coverage(source, case) >= MIN_REFERENCE_COVERAGE
            ),
            None,
        )
        retrieved += retrieval_rank is not None
        if result["status"] == "error":
            failure = "runtime"
        elif retrieval_rank is None:
            failure = "retrieval"
        elif wrong_source or coverage < MIN_REFERENCE_COVERAGE:
            failure = "citation"
        elif missing_excerpts:
            failure = "content"
        else:
            failure = None
        passed += failure is None
        rows.append(
            {
                "case": case["id"],
                "method": "answer",
                "status": result["status"],
                "failure": failure,
                "retrieval_hit": retrieval_rank is not None,
                "retrieval_rank": retrieval_rank,
                "citation_coverage": round(coverage, 4),
                "sources": citations,
                "error_type": trace.get("error_type"),
                "evidence_error": trace.get("evidence_error"),
                "citation_retry": trace.get("citation_retry"),
                "missing_excerpts": missing_excerpts,
                "question": question,
                "answer_text": result.get("text", ""),
                "seconds": round(perf_counter() - started, 3),
            }
        )
        write_jsonl(output, rows)
        print(f"{case['id']}: {failure or 'PASS'} ({result['status']})", flush=True)
    print(f"Контекст ответа с нужным интервалом: {retrieved}/{len(cases)}.", flush=True)
    print(
        f"Ответы с нужным выпуском, интервалом и обязательными выдержками: {passed}/{len(cases)}.",
        flush=True,
    )
    return int(passed != len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e5", action="store_true", help="Also compare multilingual E5 and RRF.")
    parser.add_argument(
        "--answers", action="store_true", help="Also validate answer citations using local Ollama."
    )
    parser.add_argument(
        "--name-episodes", action="store_true", help="Name the reference episode in answer queries."
    )
    args = parser.parse_args()
    if args.name_episodes and not args.answers:
        parser.error("--name-episodes используется вместе с --answers.")
    pilot = RagEngine()
    if not pilot.settings.podcast_corpus_path:
        parser.error(
            "Укажи проверенную расшифровку через AI_PODCAST_CORPUS_PATH и AI_PODCAST_CORPUS_SHA256."
        )
    data = args.cases.read_bytes()
    cases = json.loads(data)
    if not isinstance(cases, list) or not cases:
        parser.error("--cases должен содержать непустой массив вопросов с интервалами.")
    episode_ids = {
        p["document_id"].removeprefix("podcast:")
        for p in pilot.passages
        if p["document_id"].startswith("podcast:")
    }
    best_coverage = {}
    for case in cases:
        validate_interval(case["start_sec"], case["end_sec"])
        excerpts = case.get("required_excerpts", [])
        if not isinstance(excerpts, list) or any(
            not isinstance(item, str) or not item.strip() for item in excerpts
        ):
            parser.error(f"required_excerpts должен содержать непустые строки: {case['id']}.")
        if case["episode_id"] not in episode_ids:
            parser.error(f"В индексе нет выпуска для {case['id']}.")
        if case["id"] in best_coverage:
            parser.error(f"Повторяющийся ID вопроса: {case['id']}.")
        best_coverage[case["id"]] = max(reference_coverage(p, case) for p in pilot.passages)

    questions = [case["question"] for case in cases]
    started = perf_counter()
    lexical = pilot.index.score(questions)
    batch_seconds = {"TF-IDF": round(perf_counter() - started, 4)}
    methods = {"TF-IDF": lexical}
    e5_index_seconds = None
    if args.e5:
        started = perf_counter()
        if pilot.semantic_index is None:
            parser.error("--e5 требует локальный снимок модели в AI_E5_MODEL_DIR.")
        encoder = pilot.semantic_index.encoder
        texts = [passage["embedding_text"] for passage in pilot.passages]
        vectors = np.concatenate(
            [encoder.encode(texts[i : i + 16]) for i in range(0, len(texts), 16)]
        )
        e5_index_seconds = round(perf_counter() - started, 4)
        started = perf_counter()
        dense = encoder.encode([f"query: {q}" for q in questions]) @ vectors.T
        batch_seconds["E5"] = round(perf_counter() - started, 4)
        methods["E5"] = dense
        methods["RRF"] = [
            reciprocal_rank_fusion(dense_row, lexical_row)
            for dense_row, lexical_row in zip(dense, lexical, strict=True)
        ]

    rows = [
        {
            "run": {
                "cases_sha256": sha256(data).hexdigest(),
                "podcast_corpus_sha256": pilot.settings.podcast_corpus_sha256,
                "passages_sha256": sha256(
                    json.dumps(pilot.passages, sort_keys=True).encode()
                ).hexdigest(),
                "e5_revision": MODEL_REVISION if args.e5 else None,
                "top_k": LIMIT,
                "answer_episode_named": args.name_episodes,
                "answer_semantic_enabled": pilot.semantic_index is not None,
                "answer_context_limit": (
                    NAMED_EPISODE_LIMIT
                    + (NAMED_EPISODE_SEMANTIC_LIMIT if pilot.semantic_index else 0)
                    if args.name_episodes
                    else LIMIT
                )
                if args.answers
                else None,
                "answer_model_digest": pilot.settings.model_digest if args.answers else None,
                "answer_prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest()
                if args.answers
                else None,
                "episodes": sorted(episode_ids),
                "batch_score_seconds": batch_seconds,
                "e5_index_seconds": e5_index_seconds,
                "article_passages": sum(
                    p["document_id"].startswith("article:") for p in pilot.passages
                ),
            }
        }
    ]
    covered = sum(value >= MIN_REFERENCE_COVERAGE for value in best_coverage.values())
    print(f"Интервал покрыт хотя бы наполовину одним фрагментом у {covered}/{len(cases)} вопросов.")
    for name, scores in methods.items():
        ranks_by_episode = {}
        for case, scores_row in zip(cases, scores, strict=True):
            order = np.argsort(-scores_row, kind="stable")
            hits = [pilot.passages[i] for i in order[:LIMIT]]
            coverage = [reference_coverage(hit, case) for hit in hits]
            full_rank = next(
                (
                    i
                    for i, index in enumerate(order, 1)
                    if reference_coverage(pilot.passages[index], case) >= MIN_REFERENCE_COVERAGE
                ),
                None,
            )
            rank = full_rank if full_rank is not None and full_rank <= LIMIT else None
            ranks_by_episode.setdefault(case["episode_id"], []).append(rank)
            rows.append(
                {
                    "method": name,
                    "case": case["id"],
                    "rank": rank,
                    "full_rank": full_rank,
                    "reference": {key: case[key] for key in ("episode_id", "start_sec", "end_sec")},
                    "best_index_coverage": round(best_coverage[case["id"]], 4),
                    "hits": [
                        {
                            **{
                                key: hit[key]
                                for key in ("document_id", "chunk_id", "start_sec", "end_sec")
                                if key in hit
                            },
                            "reference_coverage": round(value, 4),
                        }
                        for hit, value in zip(hits, coverage, strict=True)
                    ],
                }
            )
        print_metrics(name, [rank for ranks in ranks_by_episode.values() for rank in ranks])
        if len(ranks_by_episode) > 1:
            for episode_id, ranks in ranks_by_episode.items():
                print_metrics(f"  {episode_id}", ranks)
    write_jsonl(args.output, rows)
    return (
        evaluate_answers(pilot, cases, rows, args.output, name_episodes=args.name_episodes)
        if args.answers
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
