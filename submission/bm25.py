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
from typing import Dict, List, Optional, Tuple

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

    # Term-at-a-time accumulation: visit each query term's postings list
    # exactly once and add its contribution directly into a running score
    # dict, instead of first unioning all terms' doc-ids into one
    # candidate set and then, for every candidate, re-checking every query
    # term via a dict lookup that mostly misses. That doc-at-a-time
    # approach does |candidate_docs| * |terms| lookups; a query with one
    # high-df term (e.g. "covid", 58k+ docs) and a couple of rare terms
    # wastes most of those on misses. This does exactly
    # sum(df(term) for term in terms) work -- the minimum possible --
    # with an identical score formula, just accumulated in a different
    # order.
    totals: Dict[str, float] = {}
    for term in terms:
        idf = _idf(term)
        for doc_id, tf in _INDEX.postings.get(term, {}).items():
            doc_len = _INDEX.doc_len[doc_id]
            contribution = idf * _saturated_tf(tf, doc_len, k1, b)
            totals[doc_id] = totals.get(doc_id, 0.0) + contribution

    scores = list(totals.items())

    # Sort by score desc, doc_id asc as a tie-break. Dict iteration order
    # (insertion order in modern Python) isn't itself hash-randomized, but
    # an explicit tie-break still keeps output deterministic regardless of
    # which order terms/postings were visited in.
    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return scores[:k]
