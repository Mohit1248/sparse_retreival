"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").
"""
import math
from typing import List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

# Same "whiteboard" pattern as boolean_vsm.py: build() writes the index
# here once; score() reads it from here instead of being handed it.
_INDEX: Optional[InvertedIndex] = None


def _idf(term: str) -> float:
    """BM25's smoothed IDF: ln((N - df + 0.5) / (df + 0.5) + 1).
    Stays non-negative even for a term in more than half the corpus."""
    df = _INDEX.document_frequency(term)
    return math.log((_INDEX.N - df + 0.5) / (df + 0.5) + 1)


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything BM25-specific (e.g. cached IDF values
    per term) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX
    _INDEX = index


def _saturated_tf(tf: int, doc_len: int, k1: float, b: float) -> float:
    """tf * (k1+1) / (tf + k1 * length_adjustment) -- diminishing returns
    on repeated terms, scaled by how this doc's length compares to the
    corpus average."""
    length_adjustment = 1 - b + b * (doc_len / _INDEX.avg_doc_len)
    return (tf * (k1 + 1)) / (tf + k1 * length_adjustment)


def score(query: str, k: int, k1: float = 1.2, b: float = 0.75) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked,
    highest score first."""
    terms = tokenize(query)
    if not terms:
        return []

    candidate_docs = set()
    for term in terms:
        candidate_docs.update(_INDEX.postings.get(term, {}).keys())

    scores = []
    for doc_id in candidate_docs:
        doc_len = _INDEX.doc_len[doc_id]
        total = 0.0
        for term in terms:
            tf = _INDEX.postings.get(term, {}).get(doc_id, 0)
            if tf == 0:
                continue
            total += _idf(term) * _saturated_tf(tf, doc_len, k1, b)
        scores.append((doc_id, total))

    # Sort by score desc, doc_id asc as a tie-break. candidate_docs came
    # from a Python set, whose iteration order is hash-randomized per
    # process -- without an explicit tie-break key, equal-score documents
    # would land in an arbitrary order that differs between process runs
    # (build_index/load_index/retrieve each run in a fresh subprocess),
    # making otherwise-identical submissions score slightly differently
    # run to run.
    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return scores[:k]
