"""
submission/custom_scorer.py — combined BM25 + VSM scorer with
corpus-adaptive high-frequency term dampening.

Not required, but explicitly called out in the assignment (Section 4.1)
as "where separation in the leaderboard tends to happen": this combines
the two required signals (BM25, VSM cosine) plus one additional feature.

Design (see report for the actual measured before/after numbers):

1. Corpus-adaptive high-frequency term filtering. A generic English
   stopword list (already applied in indexer.tokenize()) can't catch
   words that are only common *within this specific corpus* -- e.g. on
   trec-covid, "covid" and "19" show up in ~1/3 of the entire 171k-doc
   collection, so they're barely more useful than "the" for telling
   documents apart, even though they're not stopwords in general English.
   Terms whose document frequency exceeds _HIGH_DF_THRESHOLD of the
   corpus are excluded from the query before scoring. This is a
   corpus-level statistic computed from the index, not a pretrained
   model or external resource.
2. Combine BM25 and VSM cosine scores. Each is min-max normalised to
   [0, 1] per query (their raw scales aren't comparable -- BM25 is
   unbounded, cosine similarity is naturally in [0, 1]) before being
   combined as a weighted sum.

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

# Terms appearing in more than this fraction of the corpus are treated as
# corpus-specific "stopwords" for this scorer only -- bm25.py/boolean_vsm.py
# are left untouched so the required base scorers stay simple and this
# experiment is isolated to the optional combined scorer.
_HIGH_DF_THRESHOLD = 0.25
_HIGH_DF_TERMS: set = set()

# BM25/VSM blend weights and BM25's k1/b -- kept as defaults here (not
# re-exposed as score() parameters) so retrieve.py's call stays simple;
# see the report for how these were chosen.
_BM25_WEIGHT = 0.7
_VSM_WEIGHT = 0.3
_K1 = 2.50
_B = 0.60


def build(index: InvertedIndex) -> None:
    """Called from retrieve.load_index(), not retrieve.build_index() — the
    harness runs those two in separate processes. Anything this needs at
    query time either comes from the loaded InvertedIndex or must have
    been written to index_dir by InvertedIndex.save() (which then counts
    toward your index-size score)."""
    global _INDEX, _HIGH_DF_TERMS
    _INDEX = index
    _HIGH_DF_TERMS = {
        term for term, postings in index.postings.items()
        if index.N > 0 and len(postings) / index.N > _HIGH_DF_THRESHOLD
    }


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
    normalised BM25+VSM blend over a high-frequency-term-filtered query,
    highest score first."""
    all_terms = tokenize(query)
    terms = [t for t in all_terms if t not in _HIGH_DF_TERMS]
    if not terms:
        # Every term got filtered (e.g. a query that's *only* generic
        # corpus-wide words) -- fall back to the unfiltered query rather
        # than returning nothing.
        terms = all_terms
    if not terms:
        return []

    candidate_docs = set()
    for term in terms:
        candidate_docs.update(_INDEX.postings.get(term, {}).keys())

    bm25_scores: Dict[str, float] = {}
    for doc_id in candidate_docs:
        doc_len = _INDEX.doc_len[doc_id]
        total = 0.0
        for term in terms:
            tf = _INDEX.postings.get(term, {}).get(doc_id, 0)
            if tf == 0:
                continue
            total += bm25._idf(term) * bm25._saturated_tf(tf, doc_len, _K1, _B)
        bm25_scores[doc_id] = total

    q_vec = boolean_vsm._query_vector(terms)
    q_norm = math.sqrt(sum(w * w for w in q_vec.values())) if q_vec else 0.0
    vsm_scores: Dict[str, float] = {}
    if q_norm > 0:
        for doc_id in candidate_docs:
            dot = 0.0
            for term, q_weight in q_vec.items():
                tf = _INDEX.postings.get(term, {}).get(doc_id, 0)
                if tf == 0:
                    continue
                dot += q_weight * (tf * boolean_vsm._idf(term))
            d_norm = boolean_vsm._DOC_NORMS.get(doc_id, 0.0)
            if d_norm > 0:
                vsm_scores[doc_id] = dot / (q_norm * d_norm)

    bm25_n = _normalize(bm25_scores)
    vsm_n = _normalize(vsm_scores)

    combined = [
        (doc_id, _BM25_WEIGHT * bm25_n.get(doc_id, 0.0) + _VSM_WEIGHT * vsm_n.get(doc_id, 0.0))
        for doc_id in candidate_docs
    ]
    combined.sort(key=lambda pair: (-pair[1], pair[0]))
    return combined[:k]
