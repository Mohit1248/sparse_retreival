"""
submission/custom_scorer.py — combined BM25 + VSM scorer.

Not required, but explicitly called out in the assignment (Section 4.1)
as "where separation in the leaderboard tends to happen": this combines
the two required signals (BM25, VSM cosine).

Design (see report for the full before/after numbers):

Combine BM25 and VSM cosine scores. Each is min-max normalised to [0, 1]
per query (their raw scales aren't comparable -- BM25 is unbounded,
cosine similarity is naturally in [0, 1]) before being combined as a
weighted sum, weight tuned via a grid search over the full dev set (see
below) rather than left at an untested default.

An earlier version of this file also excluded corpus-adaptive
high-document-frequency terms (e.g. "covid"/"19", each in ~1/3 of this
corpus) from the query before scoring, on the theory that they carry
~no discriminative weight. Measured against the full dev set, that
filtering actively hurt nDCG@10 (0.6638 -> 0.4911 in isolation) because
it also stripped terms that turned out to carry real signal for most
queries, not just the handful of queries it was meant to help -- BM25's
own IDF weighting already discounts high-df terms without needing to
remove them outright. Removed rather than kept as an unused toggle.

Blend weight: swept BM25 weight from 0.0 to 1.0 in steps of 0.05 (finer
near the top) against the full dev set. The curve is a smooth, broad
peak around 0.75 (0.6638 at 1.0 -- pure BM25 -- rising to 0.6735 at
0.75, then gently falling back toward 1.0), not an isolated spike, and
every metric (nDCG@10, MAP@10, MRR, P@10) improves together at that
point versus pure BM25 -- evidence this reflects a real signal rather
than fitting noise in these 50 dev queries.

If you use this, wire it in from submission/retrieve.py's retrieve()
instead of calling a single scorer directly, and describe what you did
and why in your report (Section 7, "one-paragraph description of your
final competition entry").
"""
import math
from typing import Dict, List, Optional, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None

# BM25/VSM blend weight and BM25's k1/b -- kept as defaults here (not
# re-exposed as score() parameters) so retrieve.py's call stays simple;
# see the report for the grid search behind these values.
_BM25_WEIGHT = 0.75
_VSM_WEIGHT = 0.25
_K1 = 2.50
_B = 0.60


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index() — the
    harness runs those two in separate processes. Anything this needs at
    query time either comes from the loaded InvertedIndex or must have
    been written to index_dir by InvertedIndex.save() (which then counts
    toward your index-size score)."""
    global _INDEX
    _INDEX = index


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max scale to [0, 1] so BM25's unbounded scale and VSM's
    already-[0,1] cosine scale contribute comparably to the blend."""
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (v - lo) / (hi - lo) for doc_id, v in scores.items()}


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by a
    normalised BM25+VSM blend, highest score first."""
    terms = tokenize(query)
    if not terms:
        return []

    # Term-at-a-time for both signals: visit each query term's postings
    # list exactly once and accumulate directly into a score dict, rather
    # than unioning all terms' doc-ids into one candidate set and then, for
    # every candidate, re-checking every query term via a mostly-missing
    # dict lookup. See bm25.py's score() for the full rationale -- same
    # O(sum of df(term)) vs O(|candidates| * |terms|) argument applies to
    # both signals computed here.
    bm25_totals: Dict[str, float] = {}
    for term in terms:
        idf = bm25._idf(term)
        for doc_id, tf in _INDEX.postings.get(term, {}).items():
            doc_len = _INDEX.doc_len[doc_id]
            contribution = idf * bm25._saturated_tf(tf, doc_len, _K1, _B)
            bm25_totals[doc_id] = bm25_totals.get(doc_id, 0.0) + contribution

    q_vec = boolean_vsm._query_vector(terms)
    q_norm = math.sqrt(sum(w * w for w in q_vec.values())) if q_vec else 0.0
    vsm_dot: Dict[str, float] = {}
    if q_norm > 0:
        for term, q_weight in q_vec.items():
            idf = boolean_vsm._idf(term)
            for doc_id, tf in _INDEX.postings.get(term, {}).items():
                vsm_dot[doc_id] = vsm_dot.get(doc_id, 0.0) + q_weight * (tf * idf)

    vsm_scores: Dict[str, float] = {}
    for doc_id, dot in vsm_dot.items():
        d_norm = boolean_vsm._DOC_NORMS.get(doc_id, 0.0)
        if d_norm > 0:
            vsm_scores[doc_id] = dot / (q_norm * d_norm)

    bm25_n = _normalize(bm25_totals)
    vsm_n = _normalize(vsm_scores)

    candidate_docs = set(bm25_totals) | set(vsm_scores)
    combined = [
        (doc_id, _BM25_WEIGHT * bm25_n.get(doc_id, 0.0) + _VSM_WEIGHT * vsm_n.get(doc_id, 0.0))
        for doc_id in candidate_docs
    ]
    combined.sort(key=lambda pair: (-pair[1], pair[0]))
    return combined[:k]
