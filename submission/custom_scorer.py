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
level tested -- no peak, no compensating benefit anywhere, pure BM25 is
that corpus's best point in the whole sweep. Not a full sign-flip like
the title weight (VSM never actively helps this corpus, it just costs
it by varying amounts), but a real, one-sided overfitting risk in the
same family. Rather than either extreme -- 0.75 (locally optimal here,
costs the other corpus the most) or dropping VSM entirely to 0.0/1.0
(optimal for the other corpus specifically, which risks the same
mistake in the opposite direction: fitting to that corpus's optimum
instead of this one's), landed on 0.90/0.10 as the point that scores
best summed across both rather than the best on either alone -- keeps
most of this corpus's blend benefit while giving up little of the
other's.

Title signal: a third component (mechanism still present, weight
currently 0 -- see below) scores overlap with each document's title zone
(indexer.py's title_postings -- see TITLE_ZONE_WORDS' docstring there for
why that's a leading-word-count heuristic and not a separate corpus field
or punctuation split). Grid-searched jointly with the title zone's word
count against the full dev set, it looked like a real win in isolation:
a broad plateau across word-count 7-9 and title weight 0.15-0.25 (e.g.
K=9/w=0.20 -> 0.6958 vs. 0.6735 without it), not an isolated spike.

But that grid search only ever validated against this one corpus's 50 dev
queries. Cross-checked against a structurally different corpus (short,
title-less passages instead of long titled documents), the same weight
sweep reverses sign entirely -- raising the title weight steadily
*helped* this corpus's dev nDCG@10 and steadily *hurt* the other one's,
which is the signature of fitting this corpus's document structure
(informative titles) rather than a transferable ranking signal. Since
the held-out evaluation corpus's structure isn't something this can
verify in advance, the title term is kept in the code (score() still
computes it correctly whenever _TITLE_WEIGHT is nonzero) but its weight
is set to 0 by default -- BM25's k1/b, by contrast, showed no such
sign-flip across corpora and were kept at their tuned values.

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

# BM25/VSM/title blend weights and BM25's k1/b -- kept as defaults here
# (not re-exposed as score() parameters) so retrieve.py's call stays
# simple; see the report for the grid search behind these values.
#
# _TITLE_WEIGHT is 0 despite grid-searching better on this corpus's dev
# queries alone -- see the module docstring's "Title signal" section for
# why that plateau didn't survive a cross-corpus check and was judged too
# corpus-specific to trust on an unseen held-out set.
_TITLE_WEIGHT = 0.0
# 0.90/0.10, not this corpus's locally-optimal 0.75/0.25 -- see the module
# docstring's "Blend weight" section: VSM's benefit here doesn't transfer
# to a structurally different corpus (monotonically costs it instead), so
# this is deliberately the best-summed-across-both point, not either
# corpus's individual optimum.
_BM25_WEIGHT = (1 - _TITLE_WEIGHT) * 0.90
_VSM_WEIGHT = (1 - _TITLE_WEIGHT) * 0.10
#
# k1=1.60, b=0.50 (was k1=2.50, b=0.60): b controls how strongly document
# length is normalised, and lowering it costs this corpus's dev nDCG@10
# only ~2% while meaningfully helping a structurally different, more
# uniform-length corpus used as a cross-corpus check -- moving to it was a
# net improvement on the SUM of both corpora's nDCG@10, not just a
# compromise. k1 alone showed no comparable cross-corpus benefit; b is the
# axis that matters here. Kept, unlike the title weight above, because
# this is a smooth trade-off curve (no sign flip), not a fit that reverses
# outright -- 1.60/0.50 is simply a better point on that curve for
# performing acceptably on both rather than optimally on only one.
_K1 = 1.60
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

    # Local bindings for everything read once per posting (millions of
    # times on high-df terms): CPython resolves a local (LOAD_FAST) far
    # cheaper than a module-global or an attribute chain (LOAD_GLOBAL /
    # LOAD_ATTR) re-walked on every one of those iterations.
    postings_get = _INDEX.postings.get
    title_postings_get = _INDEX.title_postings.get
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

    # Title signal: simple term-overlap count against each document's
    # title zone (indexer.py's title_postings) -- these postings lists are
    # bounded by TITLE_ZONE_WORDS, so much cheaper than the main index.
    # Skipped entirely when _TITLE_WEIGHT is 0 (its current default, see
    # module docstring) -- no point walking title_postings for a term
    # whose contribution is about to be multiplied by 0 anyway.
    title_totals: Dict[int, float] = defaultdict(float)
    if _TITLE_WEIGHT:
        for term, count in term_counts.items():
            tpost = title_postings_get(term)
            if not tpost:
                continue
            for doc_idx, tf in tpost.items():
                title_totals[doc_idx] += count * tf

    bm25_lo, bm25_hi = _minmax(bm25_totals)
    vsm_lo, vsm_hi = _minmax(vsm_scores)
    title_lo, title_hi = _minmax(title_totals)

    # Normalize inline instead of materializing three full-candidate-size
    # dicts up front, and select the top k with a partial heap instead of
    # sorting the whole (possibly near-corpus-size) candidate set -- same
    # (-score, doc_id) tie-break as a full sort would give, just O(n log k)
    # instead of O(n log n). doc_idx -> doc_id string conversion happens
    # right here, the one place it's actually needed (the interface
    # contract and the tie-break both need the real doc_id string, not the
    # index) -- everything upstream stays in cheaper int-keyed dicts.
    #
    # candidate_docs is just bm25_totals's keys, not a 3-way set union:
    # vsm_dot and title_totals are only ever populated for doc_idxs that
    # bm25_totals already has an entry for (same _INDEX.postings[term]
    # walk for VSM; the title zone is always a prefix of the same doc's
    # full text, tokenized the same way, so any term in title_postings[t]
    # is necessarily also in postings[t]) -- so both are always subsets.
    doc_ids = _INDEX.doc_ids
    combined = (
        (
            doc_ids[doc_idx],
            _BM25_WEIGHT * _norm(raw_bm25, bm25_lo, bm25_hi)
            + _VSM_WEIGHT * _norm(vsm_scores.get(doc_idx), vsm_lo, vsm_hi)
            + _TITLE_WEIGHT * _norm(title_totals.get(doc_idx), title_lo, title_hi),
        )
        for doc_idx, raw_bm25 in bm25_totals.items()
    )
    return heapq.nsmallest(k, combined, key=lambda pair: (-pair[1], pair[0]))
