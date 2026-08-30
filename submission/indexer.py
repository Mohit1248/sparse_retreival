"""
submission/indexer.py — build your inverted index here.

This is one of the required components (assignment Section 4.1): you must
build the inverted index yourself, without an existing search/indexing
library (Lucene, Elasticsearch, Pyserini, Whoosh, etc.).

A `tokenize()` helper is provided below purely so that tokenization is
consistent across your Boolean/VSM and BM25 scorers —
feel free to replace it (e.g. add stemming or stopword removal), just make
sure every scorer that reads this index was built with the same tokenizer.

Everything else — the postings representation, what per-document and
collection statistics you track, whether you add positions for
proximity/phrase features — is your design decision. `InvertedIndex`
below sketches a minimal, obviously-sufficient shape; you do not have to
use it, but if you do, filling in `build()` and `document_frequency()` is
enough to support Boolean/VSM and BM25.

Persistence (assignment Section 4.1 / Section 7 "index size" scoring):
`build_index()` in retrieve.py runs in one process and `load_index()` runs
in a separate, later one — so whatever this index needs at query time must
round-trip through `save()`/`load()` below, not just live as Python
attributes. The on-disk byte size of what `save()` writes is graded
directly (smaller, relative to the class median, scores better), so a
compact postings encoding is worth more here than in most course
assignments — see the `save()` docstring for concrete starting points.
"""
import array
import functools
import json
import os
import pickle
import re
import zlib
from typing import Dict, List, Tuple

from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()

# array('I') is documented as the platform C `unsigned int`, not a fixed
# width -- 4 bytes on every mainstream x86/x86_64/ARM target (including
# both the Windows dev machine this was built on and the Ubuntu grading
# machine), but save()/load() below depend on that width matching between
# the two, so assert it rather than silently risking corruption if it
# ever doesn't.
assert array.array("I").itemsize == 4, "array('I') is not 4 bytes on this platform"


@functools.lru_cache(maxsize=None)
def _stem(token: str) -> str:
    """Cached wrapper around PorterStemmer.stem(). A 171k-document corpus
    has on the order of tens of millions of token *occurrences* but only a
    few hundred thousand *unique* words -- calling the (comparatively slow,
    pure-Python) stemmer once per occurrence instead of once per unique
    token blew index build time up ~5x (measured: ~100s -> ~510s) for zero
    quality difference, since stem("infection") is always "infect" no
    matter how many times it shows up. This cache is purely a same-process
    runtime speedup; it holds no state that needs to survive save()/load().
    """
    return _STEMMER.stem(token)

# Standard English stopword list (SMART/NLTK-style, ~170 terms). These are
# high-frequency function words that carry ~no discriminative weight for
# ranking but otherwise sit in nearly every posting list, inflating index
# size and diluting IDF for the terms that actually matter. Static list,
# not corpus-trained -- consistent regardless of what corpus this is built
# against.
#
# _TOKEN_RE strips apostrophes, so "don't" tokenizes as "don" + "t" (two
# tokens) before this filter ever sees it -- the plain contraction spelling
# below would never match. Both the whole-word and split-fragment forms are
# listed so contractions are actually caught either way.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren aren't as at
be because been before being below between both but by can can't cannot
could couldn couldn't d did didn didn't do does doesn doesn't doing don
don't down during each few for from further had hadn hadn't has hasn
hasn't have haven haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn
isn't it it's its itself let's ll m me more most mustn mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own re s same shan shan't she she'd she'll she's should shouldn shouldn't
so some such t than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those
through to too under until up ve very was wasn wasn't we we'd we'll we're
we've were weren weren't what what's when when's where where's which
while who who's whom why why's with won won't would wouldn wouldn't you
you'd you'll you're you've your yours yourself yourselves
""".split())


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric-only tokenization with stopword removal and
    Porter stemming. Stemming runs last so word-form variants (vaccine /
    vaccines / vaccinated / vaccination, infect / infection / infectious)
    collapse onto one postings-list entry instead of each fragmenting the
    term statistics across several near-duplicate terms."""
    return [_stem(tok) for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


# Number of leading whitespace-split words of each document's raw text
# treated as an approximate "title" zone (submission/custom_scorer.py
# weights matches here higher). Deliberately a fixed word count, not a
# punctuation/sentence-boundary heuristic or a separate corpus field:
# inspecting data/full/corpus.jsonl directly showed titles are merged
# into `text` with no reliable delimiter (some end mid-sentence with no
# punctuation at all before the abstract begins), and the documented
# schema (data/README.md) guarantees only {"doc_id", "text"} -- the same
# format the held-out corpus is promised to use -- so a separate title
# field can't be assumed to exist there. A leading word count needs
# neither. Value chosen via a grid search over word-count x blend-weight
# against the full dev set (see custom_scorer.py's docstring).
TITLE_ZONE_WORDS = 9


def _encode_postings(
    postings: Dict[str, Dict[str, int]], doc_id_to_idx: Dict[str, int]
) -> Tuple[List[str], List[int], bytes, bytes]:
    """Shared by save() for both self.postings and self.title_postings --
    see save()'s docstring for the encoding scheme. Returns
    (terms, per-term posting counts, gaps bytes, tfs bytes).

    No per-term sort: doc_id_to_idx assigns indices in corpus encounter
    order (see save()), and build() inserts a doc_id into postings[term]
    exactly once -- the first time that term appears in that document --
    walking the corpus in a single forward pass. Since Python dicts
    preserve insertion order, postings[term] is therefore already in
    increasing doc-index order by construction; a linear scan is enough.
    Sorting ~13M+ (doc_idx, tf) pairs across all terms used to be the
    single largest chunk of index-build time. If this invariant is ever
    violated (e.g. build() starts inserting out of corpus order), the
    unsigned array.array("I") construction below fails loudly on the
    resulting negative gap rather than silently corrupting the index.
    """
    terms = sorted(postings.keys())
    gaps: List[int] = []
    tfs: List[int] = []
    term_counts: List[int] = []
    for term in terms:
        term_postings = postings[term]
        term_counts.append(len(term_postings))
        prev_idx = 0
        for doc_id, tf in term_postings.items():
            doc_idx = doc_id_to_idx[doc_id]
            gaps.append(doc_idx - prev_idx)
            tfs.append(tf)
            prev_idx = doc_idx
    return terms, term_counts, array.array("I", gaps).tobytes(), array.array("I", tfs).tobytes()


def _decode_postings(
    terms: List[str], term_counts: List[int], gaps_bytes: bytes, tfs_bytes: bytes, doc_ids: List[str]
) -> Dict[str, Dict[str, int]]:
    """Inverse of _encode_postings."""
    gaps = array.array("I")
    gaps.frombytes(gaps_bytes)
    tfs = array.array("I")
    tfs.frombytes(tfs_bytes)

    postings: Dict[str, Dict[str, int]] = {}
    pos = 0
    for term, count in zip(terms, term_counts):
        term_postings: Dict[str, int] = {}
        doc_idx = 0
        for i in range(pos, pos + count):
            doc_idx += gaps[i]
            term_postings[doc_ids[doc_idx]] = tfs[i]
        postings[term] = term_postings
        pos += count
    return postings


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        # term -> {doc_id: term_freq}, restricted to each document's first
        # TITLE_ZONE_WORDS words -- see TITLE_ZONE_WORDS' docstring above
        # for why this (not a separate corpus field) is the title-zone
        # source. Read by submission/custom_scorer.py for title-weighted
        # scoring; bm25.py/boolean_vsm.py ignore it entirely.
        self.title_postings: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().

        Tokenizes each document, populates self.postings, self.title_postings,
        self.doc_len, self.N, and self.avg_doc_len. Raw document text is not
        retained — BM25/VSM only need term-frequency and length statistics,
        and keeping it around would inflate the graded on-disk index size
        for no query-time benefit.
        """
        total_len = 0

        for doc_id, text in corpus:
            tokens = tokenize(text)

            self.doc_len[doc_id] = len(tokens)
            total_len += len(tokens)

            term_counts: Dict[str, int] = {}
            for tok in tokens:
                term_counts[tok] = term_counts.get(tok, 0) + 1

            for term, count in term_counts.items():
                if term not in self.postings:
                    self.postings[term] = {}
                self.postings[term][doc_id] = count

            # maxsplit stops after TITLE_ZONE_WORDS splits instead of
            # splitting the whole (possibly long) document just to slice
            # off the first few words -- identical result, cheaper for
            # anything longer than the title zone itself.
            title_zone_text = " ".join(text.split(maxsplit=TITLE_ZONE_WORDS)[:TITLE_ZONE_WORDS])
            title_term_counts: Dict[str, int] = {}
            for tok in tokenize(title_zone_text):
                title_term_counts[tok] = title_term_counts.get(tok, 0) + 1

            for term, count in title_term_counts.items():
                if term not in self.title_postings:
                    self.title_postings[term] = {}
                self.title_postings[term][doc_id] = count

        self.N = len(corpus)
        self.avg_doc_len = total_len / self.N if self.N > 0 else 0.0

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        """Persist everything document_frequency() / your scorers need to
        `index_dir`, so `load()` can reconstruct this object in a fresh
        process with no memory of `build()` ever having run. Called from
        retrieve.build_index().

        The on-disk byte size of whatever you write here is graded
        directly (assignment Section 7, "index size", relative to the
        class median), so a naive json.dump of self.postings (which
        repeats every doc_id string, ~8 chars, once per posting -- tens of
        millions of times across the full corpus) is expensive. Instead:
          1. Integerize doc_ids: assign each doc_id a compact int index
             0..N-1 (stored once, as a single doc_ids array), instead of
             repeating the string in every posting.
          2. Within each term's postings, sort by doc index ascending and
             delta-encode (store gaps between consecutive doc indices,
             not absolute values) -- gaps are usually small for common
             terms, so they pack tighter and compress better.
          3. Pack every (gap, tf) as a fixed-width unsigned 4-byte int via
             the stdlib `array` module's C-level tobytes()/frombytes() --
             a spec note (Section 3) warns the held-out corpus could be
             >=500K docs, so gaps need more headroom than a 2-byte field
             gives; a Python-level loop doing this byte-by-byte (e.g. a
             hand-rolled varint) was measured at ~27s of index-build time
             on the full corpus purely from per-value function-call
             overhead -- array.tobytes() does the same packing in C.
          4. zlib-compress the whole thing as a final pass (stdlib, not a
             search/indexing library -- this is generic byte compression,
             not part of the required indexing/scoring logic). Level 9
             (measured: ~1334s -- over 22 minutes -- for this fixed-width
             payload, vs. ~2.7s at level 3 for barely better compression)
             would be catastrophic for index-build-time; level 3 gives
             nearly all of the size win for a small, bounded time cost.
        """
        # Corpus encounter order, not alphabetical -- _encode_postings()
        # relies on this matching the order build() inserted doc_ids into
        # postings[term], so postings entries come out pre-sorted by doc
        # index with no separate sort pass needed. Nothing else depends on
        # doc_ids being alphabetically sorted (scorers tie-break on the
        # doc_id string itself, not array position).
        doc_ids = list(self.doc_len.keys())
        doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        doc_len_arr = [self.doc_len[doc_id] for doc_id in doc_ids]

        terms, term_counts, gaps_bytes, tfs_bytes = _encode_postings(self.postings, doc_id_to_idx)
        title_terms, title_term_counts, title_gaps_bytes, title_tfs_bytes = _encode_postings(
            self.title_postings, doc_id_to_idx
        )

        payload = {
            "doc_ids": doc_ids,
            "doc_len": doc_len_arr,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
            "terms": terms,
            "term_counts": term_counts,
            "gaps_bytes": gaps_bytes,
            "tfs_bytes": tfs_bytes,
            "title_terms": title_terms,
            "title_term_counts": title_term_counts,
            "title_gaps_bytes": title_gaps_bytes,
            "title_tfs_bytes": title_tfs_bytes,
        }
        blob = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), level=3)

        path = os.path.join(index_dir, "index.bin")
        with open(path, "wb") as f:
            f.write(blob)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`. Rebuilds the
        exact same self.postings / self.doc_len shapes (string-keyed) that
        build() produces, so every scorer module is unaffected by this
        on-disk encoding.
        """
        path = os.path.join(index_dir, "index.bin")
        with open(path, "rb") as f:
            blob = f.read()
        payload = pickle.loads(zlib.decompress(blob))

        doc_ids: List[str] = payload["doc_ids"]
        doc_len_arr: List[int] = payload["doc_len"]

        idx = cls()
        idx.doc_len = {doc_id: length for doc_id, length in zip(doc_ids, doc_len_arr)}
        idx.N = payload["N"]
        idx.avg_doc_len = payload["avg_doc_len"]
        idx.postings = _decode_postings(
            payload["terms"], payload["term_counts"], payload["gaps_bytes"], payload["tfs_bytes"], doc_ids
        )
        idx.title_postings = _decode_postings(
            payload["title_terms"],
            payload["title_term_counts"],
            payload["title_gaps_bytes"],
            payload["title_tfs_bytes"],
            doc_ids,
        )

        return idx
