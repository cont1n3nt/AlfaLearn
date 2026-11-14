"""Evaluate answer statuses and sources, then save evidence for review."""

import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.error import URLError

from alfa_ai.engine import RagEngine
from alfa_ai.generation import ANSWER_MODE, SYSTEM_PROMPT
from alfa_ai.retrieval import LIMIT
from scripts.corpus import LOCAL_DATA, write_jsonl
from scripts.rag_cases import (
    ACCEPTANCE_CASES,
    BUSINESS_CASES,
    CASES,
    CONTROL_CASES,
    EXPANDED_CASES,
    REGRESSION_CASES,
    TRANSFER_CASES,
)


def source_keys(sources):
    return {(s["document_id"], tuple(s["source_block_orders"])) for s in sources}


def evaluate_case(case, result, trace):
    required = source_keys(case["required_sources"])
    status_ok = result["status"] == case["expected_status"]
    cited = source_keys(result["sources"])
    is_error = result["status"] == "error"
    sources_ok = (
        None
        if is_error
        else (required <= cited if case["expected_status"] == "answer" else not cited)
    )
    unexpected = sorted(
        {document_id for document_id, _ in cited} & set(case.get("forbidden_sources", ()))
    )
    missing_excerpts = [
        excerpt
        for excerpt in case.get("required_excerpts", ())
        if not is_error and excerpt.casefold() not in result["text"].casefold()
    ]
    return {
        "status_ok": status_ok,
        "required_sources_ok": sources_ok,
        "retrieval_ok": required <= source_keys(trace.get("context", [])) if required else None,
        "missing_excerpts": missing_excerpts,
        "forbidden_citations": unexpected,
        "automatic_pass": status_ok
        and sources_ok is True
        and not missing_excerpts
        and not unexpected,
        # Correct ID/quotation is not proof of entailment or completeness.
        "manual_review": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    splits = {
        "dev": CASES,
        "regression": REGRESSION_CASES,
        "control": CONTROL_CASES,
        "business": BUSINESS_CASES,
        "acceptance": ACCEPTANCE_CASES,
        "expanded": EXPANDED_CASES,
        "transfer": TRANSFER_CASES,
    }
    parser.add_argument("--split", choices=list(splits), default="dev")
    parser.add_argument(
        "--case", action="append", choices=[c["id"] for cases in splits.values() for c in cases]
    )
    args = parser.parse_args()
    cases = splits[args.split]
    if args.case and not set(args.case) <= {c["id"] for c in cases}:
        parser.error("Все --case должны принадлежать выбранному --split.")
    selected = [c for c in cases if not args.case or c["id"] in args.case]
    try:
        engine = RagEngine()
    except (OSError, ValueError, KeyError) as exc:
        print(f"Не удалось загрузить проверенный корпус: {exc}", file=sys.stderr)
        return 1
    started = datetime.now(timezone.utc)
    metadata = {
        "started_at_utc": started.isoformat(),
        "model": engine.settings.model,
        "model_digest": engine.settings.model_digest,
        "prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "answer_mode": ANSWER_MODE,
        "corpus_sha256": sha256(
            json.dumps(engine.passages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "cases_sha256": sha256(
            json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("eval_rag.py", "rag_cases.py")
        },
        "retrieval": "TF-IDF char_wb 3-5, reviewed passages and search metadata",
        "limit": LIMIT,
        "split": args.split,
    }
    package_root = Path(__file__).resolve().parents[1] / "src" / "alfa_ai"
    metadata["code_sha256"].update(
        {
            f"alfa_ai/{path.name}": sha256(path.read_bytes()).hexdigest()
            for path in sorted(package_root.glob("*.py"))
        }
    )
    output = LOCAL_DATA / f"rag_eval_{started.strftime('%Y%m%dT%H%M%S%fZ')}.jsonl"
    rows = []
    for case in selected:
        trace = {}
        begin = perf_counter()
        try:
            result = engine.answer(case["question"], trace=trace)
        except URLError as exc:
            message = str(exc)
            if isinstance(exc.reason, ConnectionRefusedError):
                message = f"Ollama недоступна по адресу {engine.settings.ollama_url}. Запустите ollama serve в отдельном терминале."
            result = {"status": "error", "text": message, "sources": []}
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            result = {"status": "error", "text": str(exc), "sources": []}
        checks = evaluate_case(case, result, trace)
        rows.append(
            {
                "case": case,
                "result": result,
                "checks": checks,
                "seconds": round(perf_counter() - begin, 3),
                "trace": trace,
            }
        )
        # Preserve completed cases even if a later call is interrupted.
        write_jsonl(output, [{"run": metadata}, *rows])
        verdict = "PASS" if checks["automatic_pass"] else "FAIL"
        print(
            f"{case['id']} {verdict}: {result['status']} (ожидалось {case['expected_status']}); "
            f"источники={checks['required_sources_ok']}, поиск={checks['retrieval_ok']}",
            flush=True,
        )
        if result["status"] == "error":
            print(f"  Ошибка: {result['text']}", flush=True)
        elif checks["missing_excerpts"]:
            print("  Пропущены выдержки: " + ", ".join(checks["missing_excerpts"]), flush=True)
        if checks["forbidden_citations"]:
            print(
                "  Посторонние источники: " + ", ".join(checks["forbidden_citations"]), flush=True
            )
    review = [
        "# Ручная проверка RAG-пилота",
        "",
        f"Режим ответа: {ANSWER_MODE}.",
        "",
        "PASS означает совпадение статуса, необходимых источников и обязательных выдержек.",
        "Ответ собран из исходных абзацев. Проверь их уместность, полноту и сохранение контекста; происхождение текста не гарантирует правильного выбора.",
        "",
    ]
    for row in rows:
        case, result = row["case"], row["result"]
        review.extend(
            [
                f"## {case['id']}: {case['question']}",
                "",
                f"Проверить: {case['review']}",
                "",
                f"Статус: {result['status']}. Автопроверка: {row['checks']['automatic_pass']}.",
                "",
                result["text"],
                "",
                "Доказательства модели:",
                "",
                "```json",
                json.dumps(
                    row["trace"].get("draft", {}).get("evidence", []), ensure_ascii=False, indent=2
                ),
                "```",
                "",
            ]
        )
        for source in row["trace"].get("context", []):
            review.extend(
                [
                    f"### {source['source_id']}: {source['title']} — блоки {source['source_block_orders']}",
                    "",
                    source["text"],
                    "",
                ]
            )
        review.extend(["Ручной итог: НЕ ПРОВЕРЕНО. Замечания:", ""])
    review_path = output.with_suffix(".md")
    review_path.write_text("\n".join(review), encoding="utf-8")
    passed = sum(row["checks"]["automatic_pass"] for row in rows)
    retrieval = [
        row["checks"]["retrieval_ok"] for row in rows if row["checks"]["retrieval_ok"] is not None
    ]
    if retrieval:
        print(
            f"Поиск: все обязательные фрагменты в top-{LIMIT} у {sum(retrieval)}/{len(retrieval)} вопросов."
        )
    seconds = [row["seconds"] for row in rows]
    print(f"Время запроса: медиана {median(seconds):.2f} с, максимум {max(seconds):.2f} с.")
    print(f"Автопроверки: {passed}/{len(rows)}. Смысл и полнота требуют ручной проверки.")
    print(f"JSONL: {output}\nРевью: {review_path}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
