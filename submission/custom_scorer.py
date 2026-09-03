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
point versus pure BM25 on this corpus.

Same problem as the title weight, though: that 0.75 peak was only ever
checked against this one corpus. Cross-checked against a structurally
different corpus, adding VSM weight *monotonically hurts* it at every
level tested (at the k1/b then in use) -- no peak, no compensating
benefit anywhere, pure BM25 was that corpus's best point in that sweep.
First response to that was a compromise: 0.90/0.10, the point that
scored best summed across both rather than either one's own optimum --
still a real trade-off (this corpus's own peak was 0.75, the other
corpus's was near 1.0, so 0.90 gave up some of both).

k1/b, then blend weight, were tuned sequentially (one axis fixed while
sweeping the other) up to that point -- a full joint grid over k1 x b x
blend weight (125 combinations, screened cheaply on this corpus then the
top 20 validated against the other) found a point that isn't a
trade-off at all: k1=1.20, b=0.50, blend=0.80/0.20 improves BOTH
corpora on BOTH nDCG@10 and MAP@10 simultaneously versus the sequential
compromise (this corpus's nDCG@10 0.6658->0.6789; the other corpus's
nDCG@10 0.2168->0.2191 and MAP@10 0.1197->0.1221, both measured on its
full 2,426-query set, not just the smaller one used for the initial
125-combo screen). Confirmed this isn't a hidden fit to either corpus by
checking it against each one's own independently-found optimum: this
corpus's own peak favors b=0.55 not 0.50 (0.6802 vs. this point's
0.6789, a 0.2% difference) and the other corpus's own peak favors a much
lower k1 (~0.7-0.9), lower b (~0.35-0.45), and close to pure BM25
(~0.97-1.0 weight) -- this point sits clearly away from both, not
parked at either one's corner.

Title signal (tried, removed): an earlier version added a third component
scoring term overlap with each document's first few words as an
approximate "title" zone. Grid-searched jointly with that word count
against the full dev set, it looked like a real win in isolation (a
broad plateau, not an isolated spike). But that grid search only ever
validated against this one corpus's 50 dev queries -- cross-checked
against a structurally different corpus (short, title-less passages
instead of long titled documents), the same weight sweep reversed sign
entirely: raising the title weight steadily helped this corpus's dev
nDCG@10 and steadily hurt the other one's, the signature of fitting this
corpus's document structure (informative titles) rather than a
transferable ranking signal. Set to weight 0 first as insurance, then
removed outright (the indexing pass that built it, and the on-disk
fields it needed) once it was clear weight 0 was the durable answer, not
a temporary hedge -- keeping a whole second per-document indexing pass
and a whole extra set of persisted fields alive purely to multiply their
output by zero cost real, measurable index-build time and on-disk index
size for no benefit. BM25's k1/b, by contrast, showed no such sign-flip
across corpora and were kept at their tuned values.

If you use this, wire it in from submission/retrieve.py's retrieve()
instead of calling a single scorer directly, and describe what you did
and why in your report (Section 7, "one-paragraph description of your
final competition entry").
"""
import heapq
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None

# BM25/VSM blend weight and BM25's k1/b -- kept as defaults here (not
# re-exposed as score() parameters) so retrieve.py's call stays simple;
# see the report for the grid search behind these values.
#
# k1=1.20, b=0.50, blend=0.80/0.20 -- found via a full joint grid over all
# three parameters together (not swept one at a time), then verified to
# improve nDCG@10 AND MAP@10 on BOTH corpora simultaneously versus the
# earlier sequentially-tuned point, and to sit clearly away from either
# corpus's own independently-found optimum (not fit to either one) -- see
# the module docstring's "Blend weight" section for the full numbers.
_BM25_WEIGHT = 0.80
_VSM_WEIGHT = 0.20
_K1 = 1.20
_B = 0.50


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
    normalised BM25+VSM blend, highest score first."""
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

    # Local bindings for everything read once per posting (millions of
    # times on high-df terms): CPython resolves a local (LOAD_FAST) far
    # cheaper than a module-global or an attribute chain (LOAD_GLOBAL /
    # LOAD_ATTR) re-walked on every one of those iterations.
    postings_get = _INDEX.postings.get
    doc_len_arr = _INDEX.doc_len  # doc_idx -> length, a list (see indexer.py)
    avg_doc_len = _INDEX.avg_doc_len
    k1, b = _K1, _B
    k1_plus_1 = k1 + 1  # same fixed float bm25._saturated_tf recomputed on every call

    # Keyed by doc_idx (int), not doc_id string -- InvertedIndex.postings
    # is doc_idx-keyed (see indexer.py), so every accumulation dict below
    # hashes a small int instead of a ~40-char doc_id string, on the order
    # of millions of times per query for this corpus's high-df terms.
    # defaultdict(float): same accumulation arithmetic as a
    # `d.get(doc_idx, 0.0) + v` / `d[doc_idx] = ...` pair, just without a
    # bound-method .get() call on every posting -- identical floats out,
    # fewer bytecodes to get there.
    bm25_totals: Dict[int, float] = defaultdict(float)
    vsm_dot: Dict[int, float] = defaultdict(float)
    for term, count in term_counts.items():
        postings = postings_get(term)
        if not postings:
            continue
        idf_bm25 = bm25._idf(term)
        # Only worth computing VSM's side if this term actually carries
        # query weight (it always will unless q_norm == 0, i.e. every
        # query term has idf 0) -- avoids a second idf lookup otherwise.
        q_weight = q_vec.get(term, 0.0) if q_norm > 0 else 0.0
        idf_vsm = boolean_vsm._idf(term) if q_weight else 0.0
        # count * idf_bm25 doesn't depend on doc_idx -- the old code
        # recomputed this identical product on every posting in this
        # term's list (up to ~60k times for a high-df COVID term) even
        # though Python never hoists loop-invariant subexpressions for
        # you. Same two floats multiplied once instead of N times is
        # bit-identical, just without the redundant work.
        bm25_coeff = count * idf_bm25
        for doc_idx, tf in postings.items():
            # Inlined bm25._saturated_tf: identical formula, same
            # operation order, just without a Python function-call per
            # posting (this loop body runs millions of times per query
            # on the high-df terms this corpus is full of).
            length_adjustment = 1 - b + b * (doc_len_arr[doc_idx] / avg_doc_len)
            saturated = (tf * k1_plus_1) / (tf + k1 * length_adjustment)
            bm25_totals[doc_idx] += bm25_coeff * saturated
            if q_weight:
                vsm_dot[doc_idx] += q_weight * (tf * idf_vsm)

    vsm_scores: Dict[int, float] = {}
    doc_norms_get = boolean_vsm._DOC_NORMS.get
    for doc_idx, dot in vsm_dot.items():
        d_norm = doc_norms_get(doc_idx, 0.0)
        if d_norm > 0:
            vsm_scores[doc_idx] = dot / (q_norm * d_norm)

    bm25_lo, bm25_hi = _minmax(bm25_totals)
    vsm_lo, vsm_hi = _minmax(vsm_scores)

    # Normalize inline instead of materializing full-candidate-size dicts
    # up front, and select the top k with a partial heap instead of
    # sorting the whole (possibly near-corpus-size) candidate set -- same
    # (-score, doc_id) tie-break as a full sort would give, just O(n log k)
    # instead of O(n log n). doc_idx -> doc_id string conversion happens
    # right here, the one place it's actually needed (the interface
    # contract and the tie-break both need the real doc_id string, not the
    # index) -- everything upstream stays in cheaper int-keyed dicts.
    #
    # candidate_docs is just bm25_totals's keys, not a set union: vsm_dot
    # is only ever populated for doc_idxs that bm25_totals already has an
    # entry for (same _INDEX.postings[term] walk for both), so it's
    # always a subset.
    doc_ids = _INDEX.doc_ids
    combined = (
        (
            doc_ids[doc_idx],
            _BM25_WEIGHT * _norm(raw_bm25, bm25_lo, bm25_hi)
            + _VSM_WEIGHT * _norm(vsm_scores.get(doc_idx), vsm_lo, vsm_hi),
        )
        for doc_idx, raw_bm25 in bm25_totals.items()
    )
    return heapq.nsmallest(k, combined, key=lambda pair: (-pair[1], pair[0]))
