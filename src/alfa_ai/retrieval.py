"""Lexical retrieval, local second-stage reranking, and source diversity."""

import re
from itertools import islice

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

LIMIT = 3
RERANK_CANDIDATES = 12
NAMED_EPISODE_LIMIT = 4
NAMED_EPISODE_SEMANTIC_LIMIT = 2
SCOPED_LEXICAL_QUERY_WEIGHT = 0.75
RRF_K = 60
# With the expanded corpus, a second passage from the same article should not
# crowd out a nearly equal passage that covers a distinct part of the question.
COMPARABLE_SCORE = 0.89


class LexicalIndex:
    def __init__(self, chunks):
        passages = [chunk["embedding_text"].removeprefix("passage: ") for chunk in chunks]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), norm="l2")
        self.vectors = self.vectorizer.fit_transform(passages)

    def score(self, queries):
        return (self.vectorizer.transform(queries) @ self.vectors.T).toarray()


def _terms(text):
    """Keep meaningful Cyrillic/Latin terms; no external model or network required."""
    return tuple(term for term in re.findall(r"[\w-]{3,}", text.casefold()) if not term.isdigit())


def rerank(question, candidates, base_scores, ranking, *, candidate_limit=RERANK_CANDIDATES):
    """Rerank only first-stage candidates using query coverage in trusted headings.

    The candidate set is fixed by TF-IDF first, so this function never admits a
    passage that the primary retriever did not select. It is intentionally local
    and deterministic: release quality does not depend on an additional API or
    a downloaded cross-encoder.
    """
    shortlist = list(ranking[:candidate_limit])
    terms = set(_terms(question))
    if not shortlist or not terms:
        return shortlist + list(ranking[candidate_limit:])
    rescored = []
    for position, index in enumerate(shortlist):
        candidate = candidates[index]
        heading = " ".join([candidate["title"], *candidate.get("section", ())]).casefold()
        body = candidate["text"].casefold()
        heading_coverage = sum(term in heading for term in terms) / len(terms)
        body_coverage = sum(term in body for term in terms) / len(terms)
        # The first-stage score remains dominant; headings only resolve close
        # candidates whose topic names match the question more directly.
        score = base_scores[index] + 0.01 * heading_coverage + 0.0025 * body_coverage
        rescored.append((-score, position, index))
    rescored.sort()
    return [index for _, _, index in rescored] + list(ranking[candidate_limit:])


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
    ranking = rerank(question, candidates, scores, ranking)
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
