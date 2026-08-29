"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.

The grading harness only ever imports and calls the three functions below.
Their names and signatures are fixed by the assignment (Section 5 of the
assignment spec, "Submission Interface & Conformance Checking") — do not
rename them, change their signatures, or move them out of this file.

    build_index(corpus_path: str, index_dir: str) -> None
        Called once, in its own process, with the path to a corpus.jsonl
        file (see data/README.md) and a directory to write your index
        into. Build whatever index and statistics you need, and WRITE
        THEM TO index_dir. The harness runs build_index() and
        load_index()/retrieve() in two SEPARATE processes on purpose (see
        harness/run_harness.py's module docstring) — nothing you only
        hold in memory here survives into load_index(). This call is
        timed as your "index build time" efficiency metric. The harness
        also measures the on-disk byte size of index_dir once this
        returns — that's your "index size" score (assignment Section 7),
        so write only what retrieve() actually needs, and consider
        compressing it.

    load_index(index_dir: str) -> None
        Called once, in a fresh process, before any retrieve() calls.
        Reconstruct everything retrieve() needs by reading index_dir —
        and only index_dir; there is no leftover state from
        build_index() to fall back on. Timed as your "index load time".

    retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]
        Called once per query, only after load_index() has run in the
        same process. Return up to k (doc_id, score) pairs, sorted by
        score descending (highest score = most relevant). This is exactly
        the ranking the harness scores with nDCG@10 / MAP@10. doc_id values
        must be ones that appeared in the corpus passed to build_index().
"""
from typing import List, Optional, Tuple

from submission import bm25, boolean_vsm, custom_scorer
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

# ---------------------------------------------------------------------------
# Module-level state. load_index() populates this; retrieve() reads it.
# build_index() runs in a SEPARATE process and cannot rely on this state
# surviving into load_index()/retrieve() — anything needed at query time
# must be written to index_dir in build_index() and read back in
# load_index().
# ---------------------------------------------------------------------------
_INDEX: Optional[InvertedIndex] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Load the corpus, build the inverted index, and persist it to
    index_dir so load_index() can reconstruct it in a fresh process."""
    corpus = load_corpus(corpus_path)
    index = InvertedIndex()
    index.build(corpus)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Reconstruct the index from index_dir, then hand it to each scorer
    module's build() so they can precompute whatever they need (e.g.
    boolean_vsm's document norms)."""
    global _INDEX
    _INDEX = InvertedIndex.load(index_dir)
    boolean_vsm.build(_INDEX)
    bm25.build(_INDEX)
    custom_scorer.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked."""
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )

    # BM25 alone (k1=2.50, b=0.60 -- re-tuned after adding Porter stemming to
    # tokenize(), see indexer.py) scored nDCG@10=0.6638 on the full dev set.
    # custom_scorer blends that with VSM cosine (weight swept over the full
    # dev set: a smooth, broad peak at 0.75 BM25 / 0.25 VSM, every metric
    # improving together, not just nDCG -- see custom_scorer.py's docstring)
    # for nDCG@10=0.6735, MAP@10, MRR, and P@10 all improving too.
    return custom_scorer.score(query, k)
