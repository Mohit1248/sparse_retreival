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

Title signal: a third component scores overlap with each document's
title zone (indexer.py's title_postings -- see TITLE_ZONE_WORDS'
docstring there for why that's a leading-word-count heuristic and not a
separate corpus field or punctuation split). Grid-searched jointly with
the title zone's word count against the full dev set: a broad plateau
across word-count 7-9 and title weight 0.15-0.25 (not a single sharp
spike -- e.g. K=8/w=0.20 -> 0.6907, K=9/w=0.15 -> 0.6959, K=9/w=0.20 ->
0.6958, all close together), landing on K=9 words / weight 0.20 as a
point solidly inside that plateau rather than the single best-observed
value (which risks being fine-grained noise on only 50 dev queries).
That takes nDCG@10 from 0.6735 (BM25+VSM alone) to 0.6958 -- a further
real gain on top of the blend weight tuning above.

If you use this, wire it in from submission/retrieve.py's retrieve()
instead of calling a single scorer directly, and describe what you did
and why in your report (Section 7, "one-paragraph description of your
final competition entry").
"""
import heapq
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None

# BM25/VSM/title blend weights and BM25's k1/b -- kept as defaults here
# (not re-exposed as score() parameters) so retrieve.py's call stays
# simple; see the report for the grid search behind these values.
_TITLE_WEIGHT = 0.20
_BM25_WEIGHT = (1 - _TITLE_WEIGHT) * 0.75
_VSM_WEIGHT = (1 - _TITLE_WEIGHT) * 0.25
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


def _minmax(scores: Dict[str, float]) -> Tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    values = scores.values()
    return min(values), max(values)


def _norm(value: Optional[float], lo: float, hi: float) -> float:
    """Min-max scale to [0, 1] so BM25's unbounded scale and VSM's
    already-[0,1] cosine scale contribute comparably to the blend.
    Missing (None) -> 0.0, matching the old dict.get(doc_id, 0.0) default;
    a flat signal (hi == lo) -> 1.0 for every doc that has it, same as
    the old _normalize()'s special case."""
    if value is None:
        return 0.0
    if hi == lo:
        return 1.0
    return (value - lo) / (hi - lo)


def score(query: str, k: int) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, ranked by a
    normalised BM25+VSM+title blend, highest score first."""
    terms = tokenize(query)
    if not terms:
        return []

    # Term-at-a-time, one postings walk per unique query term: BM25 and
    # VSM both read the same _INDEX.postings[term], so they're accumulated
    # together in a single pass instead of two (this used to be two full
    # walks over what can be a near-corpus-size postings list for a
    # high-df term -- COVID-corpus queries hit these constantly). `count`
    # replicates what looping over `terms` with repeats used to do: a
    # repeated query term got its postings walked once per occurrence, so
    # here each unique term's contribution is scaled by how many times it
    # occurred instead.
    term_counts = Counter(terms)

    q_vec = boolean_vsm._query_vector(terms)
    q_norm = math.sqrt(sum(w * w for w in q_vec.values())) if q_vec else 0.0

    bm25_totals: Dict[str, float] = {}
    vsm_dot: Dict[str, float] = {}
    for term, count in term_counts.items():
        postings = _INDEX.postings.get(term)
        if not postings:
            continue
        idf_bm25 = bm25._idf(term)
        # Only worth computing VSM's side if this term actually carries
        # query weight (it always will unless q_norm == 0, i.e. every
        # query term has idf 0) -- avoids a second idf lookup otherwise.
        q_weight = q_vec.get(term, 0.0) if q_norm > 0 else 0.0
        idf_vsm = boolean_vsm._idf(term) if q_weight else 0.0
        for doc_id, tf in postings.items():
            doc_len = _INDEX.doc_len[doc_id]
            bm25_totals[doc_id] = (
                bm25_totals.get(doc_id, 0.0) + count * idf_bm25 * bm25._saturated_tf(tf, doc_len, _K1, _B)
            )
            if q_weight:
                vsm_dot[doc_id] = vsm_dot.get(doc_id, 0.0) + q_weight * (tf * idf_vsm)

    vsm_scores: Dict[str, float] = {}
    for doc_id, dot in vsm_dot.items():
        d_norm = boolean_vsm._DOC_NORMS.get(doc_id, 0.0)
        if d_norm > 0:
            vsm_scores[doc_id] = dot / (q_norm * d_norm)

    # Title signal: simple term-overlap count against each document's
    # title zone (indexer.py's title_postings) -- these postings lists are
    # bounded by TITLE_ZONE_WORDS, so much cheaper than the main index.
    title_totals: Dict[str, float] = {}
    for term, count in term_counts.items():
        tpost = _INDEX.title_postings.get(term)
        if not tpost:
            continue
        for doc_id, tf in tpost.items():
            title_totals[doc_id] = title_totals.get(doc_id, 0.0) + count * tf

    bm25_lo, bm25_hi = _minmax(bm25_totals)
    vsm_lo, vsm_hi = _minmax(vsm_scores)
    title_lo, title_hi = _minmax(title_totals)

    # Normalize inline instead of materializing three full-candidate-size
    # dicts up front, and select the top k with a partial heap instead of
    # sorting the whole (possibly near-corpus-size) candidate set -- same
    # (-score, doc_id) tie-break as a full sort would give, just O(n log k)
    # instead of O(n log n).
    #
    # candidate_docs is just bm25_totals's keys, not a 3-way set union:
    # vsm_dot and title_totals are only ever populated for doc_ids that
    # bm25_totals already has an entry for (same _INDEX.postings[term]
    # walk for VSM; the title zone is always a prefix of the same doc's
    # full text, tokenized the same way, so any term in title_postings[t]
    # is necessarily also in postings[t]) -- so both are always subsets.
    combined = (
        (
            doc_id,
            _BM25_WEIGHT * _norm(raw_bm25, bm25_lo, bm25_hi)
            + _VSM_WEIGHT * _norm(vsm_scores.get(doc_id), vsm_lo, vsm_hi)
            + _TITLE_WEIGHT * _norm(title_totals.get(doc_id), title_lo, title_hi),
        )
        for doc_id, raw_bm25 in bm25_totals.items()
    )
    return heapq.nsmallest(k, combined, key=lambda pair: (-pair[1], pair[0]))
