"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""
import math
from typing import List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

# The "whiteboard": build() writes the index here once; every other
# function in this file reads it from here instead of being handed it.
_INDEX: Optional[InvertedIndex] = None

# doc_id -> length of that document's full TF-IDF vector (precomputed
# once in build(), reused by every later vsm_score() call).
_DOC_NORMS: dict = {}


def _idf(term: str) -> float:
    """log(N / df(t)) -- how rare `term` is across the whole corpus.
    0.0 for a term that never appears (avoids dividing by zero)."""
    df = _INDEX.document_frequency(term)
    if df == 0:
        return 0.0
    return math.log(_INDEX.N / df)


def build(index: InvertedIndex) -> None:
    """Optional: precompute anything VSM-specific (e.g. document vector
    norms) from the InvertedIndex built in indexer.py.

    Call this from retrieve.load_index(), not retrieve.build_index() —
    the harness runs those two in separate processes, so any cache this
    creates only needs to exist in the process that also calls
    retrieve(). If you want a precomputed cache to persist across the
    build/load boundary too, write it out via InvertedIndex.save() instead
    (it then counts toward your index-size score) and rebuild the cache
    here from the loaded index."""
    global _INDEX, _DOC_NORMS
    _INDEX = index

    doc_sq_sums: dict = {}  # doc_id -> running total of (weight^2)
    for term, postings in _INDEX.postings.items():
        idf = _idf(term)
        for doc_id, tf in postings.items():
            weight = tf * idf
            doc_sq_sums[doc_id] = doc_sq_sums.get(doc_id, 0.0) + weight * weight

    _DOC_NORMS = {doc_id: math.sqrt(sq_sum) for doc_id, sq_sum in doc_sq_sums.items()}


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return the (unranked) list of doc_ids matching `query`, treating it
    as a conjunction (`mode="and"`) or disjunction (`mode="or"`) of its
    terms."""
    terms = tokenize(query)
    if not terms:
        return []

    doc_sets = [set(_INDEX.postings.get(term, {}).keys()) for term in terms]

    if mode == "and":
        matching_docs = set.intersection(*doc_sets)
    else:  # mode == "or"
        matching_docs = set.union(*doc_sets)

    return list(matching_docs)


def _query_vector(terms: List[str]) -> dict:
    """{term: tf(term, query) * idf(term)} for the given (already
    tokenized) query terms."""
    term_counts: dict = {}
    for term in terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    return {term: count * _idf(term) for term, count in term_counts.items()}


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by
    TF-IDF cosine similarity, highest score first."""
    terms = tokenize(query)
    if not terms:
        return []

    q_vec = _query_vector(terms)                              # step 1
    q_norm = math.sqrt(sum(w * w for w in q_vec.values()))
    if q_norm == 0:
        return []

    candidate_docs = set()                                    # step 2
    for term in q_vec:
        candidate_docs.update(_INDEX.postings.get(term, {}).keys())

    scores = []
    for doc_id in candidate_docs:
        dot = 0.0                                             # step 3
        for term, q_weight in q_vec.items():
            tf = _INDEX.postings.get(term, {}).get(doc_id, 0)
            if tf == 0:
                continue
            d_weight = tf * _idf(term)
            dot += q_weight * d_weight

        d_norm = _DOC_NORMS.get(doc_id, 0.0)
        if d_norm == 0:
            continue
        scores.append((doc_id, dot / (q_norm * d_norm)))       # step 4

    # Score desc, doc_id asc tie-break -- see bm25.py's score() for why
    # (candidate_docs comes from a hash-randomized set, so without this the
    # tie order would vary between process runs).
    scores.sort(key=lambda pair: (-pair[1], pair[0]))           # step 5
    return scores[:k]
