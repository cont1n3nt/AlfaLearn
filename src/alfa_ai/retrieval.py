"""Lexical scoring, source diversity, and rank fusion."""

from itertools import islice

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

LIMIT = 3
NAMED_EPISODE_LIMIT = 4
NAMED_EPISODE_SEMANTIC_LIMIT = 2
SCOPED_LEXICAL_QUERY_WEIGHT = 0.75
RRF_K = 60
COMPARABLE_SCORE = 0.9


class LexicalIndex:
    def __init__(self, chunks):
        passages = [chunk["embedding_text"].removeprefix("passage: ") for chunk in chunks]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), norm="l2")
        self.vectors = self.vectorizer.fit_transform(passages)

    def score(self, queries):
        return (self.vectorizer.transform(queries) @ self.vectors.T).toarray()


def diverse_context(ranking, scores, passages, limit):
    selected = list(ranking[:limit])
    documents = [passages[index]["document_id"] for index in selected]
    repeated = next(
        (
            position
            for position in range(len(documents) - 1, 0, -1)
            if documents[position] in documents[:position]
        ),
        None,
    )
    if repeated is None:
        return selected
    threshold = scores[selected[repeated]] * COMPARABLE_SCORE
    for index in ranking[limit:]:
        if scores[index] <= 0 or scores[index] < threshold:
            break
        if passages[index]["document_id"] not in documents:
            return selected[:repeated] + selected[repeated + 1 :] + [index]
    return selected


def reciprocal_rank_fusion(dense_scores, lexical_scores):
    fused = np.zeros(len(dense_scores), dtype=np.float64)
    # A zero TF-IDF score contributes no lexical rank.
    lexical_matches = np.flatnonzero(lexical_scores > 0)
    rankings = (
        np.argsort(-dense_scores, kind="stable"),
        lexical_matches[np.argsort(-lexical_scores[lexical_matches], kind="stable")],
    )
    for ranking in rankings:
        fused[ranking] += 1 / (RRF_K + np.arange(1, len(ranking) + 1))
    return fused


def select_context(
    question,
    candidates,
    index=None,
    *,
    document_ids=None,
    focused_question=None,
    limit=LIMIT,
    semantic_index=None,
):
    index = index if index is not None else LexicalIndex(candidates)
    if focused_question and focused_question != question:
        full, focused = index.score([question, focused_question])
        scores = (1 - SCOPED_LEXICAL_QUERY_WEIGHT) * full + SCOPED_LEXICAL_QUERY_WEIGHT * focused
    else:
        scores = index.score([question])[0]
    ranking = np.argsort(-scores, kind="stable")
    if document_ids:
        scoped_ranking = (i for i in ranking if candidates[i]["document_id"] in document_ids)
        selected = list(islice(scoped_ranking, limit))
        if semantic_index is not None:
            semantic_scores = semantic_index.score([focused_question or question])[0]
            semantic_ranking = np.argsort(-semantic_scores, kind="stable")
            supplemental = (
                i
                for i in semantic_ranking
                if i not in selected and candidates[i]["document_id"] in document_ids
            )
            selected.extend(islice(supplemental, NAMED_EPISODE_SEMANTIC_LIMIT))
    else:
        selected = diverse_context(ranking, scores, candidates, limit)
    sources = {}
    for rank, candidate_index in enumerate(selected, 1):
        source_id = f"S{rank}"
        chunk = candidates[candidate_index]
        sources[source_id] = {
            "source_id": source_id,
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "title": chunk["title"],
            "source_block_orders": chunk["source_block_orders"],
            "url": chunk["url"],
            "text": chunk["text"].strip(),
        }
        if chunk["document_id"].startswith("podcast:"):
            sources[source_id].update(start_sec=chunk["start_sec"], end_sec=chunk["end_sec"])
    return sources
