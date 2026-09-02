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
from collections import Counter
from typing import Dict, List, Tuple

from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()

# array('I')/array('H') are documented as the platform C `unsigned int` /
# `unsigned short`, not a fixed width -- 4 and 2 bytes respectively on
# every mainstream x86/x86_64/ARM target (including both the Windows dev
# machine this was built on and the Ubuntu grading machine), but
# save()/load() below depend on those widths matching between the two, so
# assert them rather than silently risking corruption if they ever don't.
assert array.array("I").itemsize == 4, "array('I') is not 4 bytes on this platform"
assert array.array("H").itemsize == 2, "array('H') is not 2 bytes on this platform"
assert array.array("B").itemsize == 1, "array('B') is not 1 byte on this platform"

# Escape sentinels for _pack_gaps/_pack_tfs' cascading encoding, below.
_GAP_ESCAPE_B = 0xFF     # array('B') tier: value 0-254 direct, 255 = escape to H tier
_GAP_ESCAPE_H = 0xFFFF   # array('H') tier: value 0-65534 direct, 65535 = escape to I tier
_TF_ESCAPE_B = 0xFF      # array('B') tier: value 0-254 direct, 255 = escape to I tier


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


def _pack_gaps(gaps: List[int]) -> Tuple[bytes, bytes, bytes]:
    """3-tier cascading pack: array('B') (0-254 direct, 255 = escape to
    the H tier) -> array('H') (0-65534 direct, 65535 = escape to the I
    tier) -> array('I') (anything, no further escape needed -- a spec
    note (Section 3) warns the held-out corpus could be >=500K docs, and
    32 bits comfortably covers any doc_idx gap at that scale).

    Measured on the full corpus: gaps were ~65% of total index size even
    after zlib with a flat 4-byte encoding (17.5MB of ~27MB) -- gaps for
    common terms are mostly small (delta-encoded against a monotonically
    increasing doc_idx: 86.3% fit in a single byte), but a rare term's
    lone occurrence can land anywhere in a 171k+ doc corpus. zlib on a
    flat encoding already exploited most of the redundancy *within* that
    encoding (repeated zero bytes in small values stored wide), so the
    remaining win comes from using fewer bytes per value to begin with,
    not more compression effort on the same bytes (level 9 measured
    "barely better" than level 3 on this data -- see save()'s docstring).
    A byte-by-byte hand-rolled varint was previously measured at ~27s of
    extra build time for a similar goal; this cascade gets most of the
    same win from one or two cheap comparisons per value instead, still
    packed in bulk via array.tobytes()."""
    l0: List[int] = []
    l1: List[int] = []
    l2: List[int] = []
    for g in gaps:
        if g < _GAP_ESCAPE_B:
            l0.append(g)
        else:
            l0.append(_GAP_ESCAPE_B)
            if g < _GAP_ESCAPE_H:
                l1.append(g)
            else:
                l1.append(_GAP_ESCAPE_H)
                l2.append(g)
    return array.array("B", l0).tobytes(), array.array("H", l1).tobytes(), array.array("I", l2).tobytes()


def _unpack_gaps(l0_bytes: bytes, l1_bytes: bytes, l2_bytes: bytes) -> List[int]:
    """Inverse of _pack_gaps."""
    l0 = array.array("B")
    l0.frombytes(l0_bytes)
    l1 = array.array("H")
    l1.frombytes(l1_bytes)
    l2 = array.array("I")
    l2.frombytes(l2_bytes)
    l1_iter = iter(l1)
    l2_iter = iter(l2)
    out: List[int] = []
    for v in l0:
        if v != _GAP_ESCAPE_B:
            out.append(v)
            continue
        v1 = next(l1_iter)
        out.append(v1 if v1 != _GAP_ESCAPE_H else next(l2_iter))
    return out


def _pack_tfs(tfs: List[int]) -> Tuple[bytes, bytes]:
    """2-tier cascading pack: array('B') (0-254 direct, 255 = escape to
    the I tier) -> array('I') (anything -- tf is bounded by a single
    document's token count, but real-world documents could in principle
    repeat one term arbitrarily many times, so this still escapes rather
    than assuming a hard cap). Measured on the full corpus: 99.99998% of
    tfs fit in a single byte (max tf was 495 across ~12.2M postings)."""
    l0: List[int] = []
    l1: List[int] = []
    for t in tfs:
        if t < _TF_ESCAPE_B:
            l0.append(t)
        else:
            l0.append(_TF_ESCAPE_B)
            l1.append(t)
    return array.array("B", l0).tobytes(), array.array("I", l1).tobytes()


def _unpack_tfs(l0_bytes: bytes, l1_bytes: bytes) -> List[int]:
    """Inverse of _pack_tfs."""
    l0 = array.array("B")
    l0.frombytes(l0_bytes)
    l1 = array.array("I")
    l1.frombytes(l1_bytes)
    l1_iter = iter(l1)
    return [v if v != _TF_ESCAPE_B else next(l1_iter) for v in l0]


def _front_code(sorted_terms: List[str]) -> Tuple[bytes, bytes]:
    """Front-coding for save()'s `terms` vocabulary list (already sorted
    alphabetically, see _encode_postings): each term is stored as (length
    of the prefix shared with the previous term, remaining suffix), so
    adjacent stemmed English words sharing a long root (e.g. consecutive
    entries in a sorted vocabulary very often do) don't repeat those
    shared characters. Assumes no shared prefix exceeds 255 characters --
    true for any realistic tokenized term (tokenize() only ever extracts
    [a-z0-9]+ substrings from natural-language text, never anywhere near
    that long) -- and fails loudly via the array('B') construction below
    if it ever isn't, rather than silently truncating. Suffixes are
    joined with a NUL separator, which tokenize() can never itself
    produce.
    """
    prefix_lens: List[int] = []
    suffixes: List[str] = []
    prev = ""
    for term in sorted_terms:
        limit = min(len(prev), len(term))
        plen = 0
        while plen < limit and prev[plen] == term[plen]:
            plen += 1
        prefix_lens.append(plen)
        suffixes.append(term[plen:])
        prev = term
    return array.array("B", prefix_lens).tobytes(), "\x00".join(suffixes).encode("utf-8")


def _front_decode(prefix_lens_bytes: bytes, suffixes_blob: bytes) -> List[str]:
    """Inverse of _front_code."""
    prefix_lens = array.array("B")
    prefix_lens.frombytes(prefix_lens_bytes)
    suffixes = suffixes_blob.decode("utf-8").split("\x00")
    terms: List[str] = []
    prev = ""
    for plen, suf in zip(prefix_lens, suffixes):
        term = prev[:plen] + suf
        terms.append(term)
        prev = term
    return terms


def _encode_postings(
    postings: Dict[str, Dict[int, int]],
) -> Tuple[List[str], List[int], bytes, bytes, bytes, bytes, bytes]:
    """Used by save() to encode self.postings -- see save()'s and
    _pack_gaps'/_pack_tfs' docstrings for the encoding scheme. Returns
    (terms, per-term posting counts, gaps L0/L1/L2 bytes, tfs L0/L1 bytes).

    postings is already {term: {doc_idx: tf}} -- InvertedIndex.build()
    assigns each doc its integer index the moment it's first seen (see
    build()), so there is no separate doc_id-string-to-index translation
    step here at all (there used to be: a doc_id_to_idx[doc_id] dict
    lookup on every one of ~13M+ postings, now gone entirely). No
    per-term sort either: build() inserts a doc_idx into postings[term]
    exactly once -- the first time that term appears in that document --
    walking the corpus in a single forward pass in increasing-doc_idx
    order, and Python dicts preserve insertion order, so postings[term]
    is therefore already in increasing doc-index order by construction; a
    linear scan is enough. If this invariant is ever violated (e.g.
    build() starts inserting out of corpus order), the unsigned array
    construction inside _pack_gaps fails loudly on the resulting negative
    gap rather than silently corrupting the index.
    """
    terms = sorted(postings.keys())
    gaps: List[int] = []
    tfs: List[int] = []
    term_counts: List[int] = []
    for term in terms:
        term_postings = postings[term]
        term_counts.append(len(term_postings))
        prev_idx = 0
        for doc_idx, tf in term_postings.items():
            gaps.append(doc_idx - prev_idx)
            tfs.append(tf)
            prev_idx = doc_idx
    gaps_l0, gaps_l1, gaps_l2 = _pack_gaps(gaps)
    tfs_l0, tfs_l1 = _pack_tfs(tfs)
    return terms, term_counts, gaps_l0, gaps_l1, gaps_l2, tfs_l0, tfs_l1


def _decode_postings(
    terms: List[str],
    term_counts: List[int],
    gaps_l0_bytes: bytes,
    gaps_l1_bytes: bytes,
    gaps_l2_bytes: bytes,
    tfs_l0_bytes: bytes,
    tfs_l1_bytes: bytes,
) -> Dict[str, Dict[int, int]]:
    """Inverse of _encode_postings. Returns {term: {doc_idx: tf}} --
    doc_idx stays an int; scorers convert to the real doc_id string only
    when constructing their final (doc_id, score) results (see
    InvertedIndex.doc_ids), so no doc_ids list is needed here."""
    gaps = _unpack_gaps(gaps_l0_bytes, gaps_l1_bytes, gaps_l2_bytes)
    tfs = _unpack_tfs(tfs_l0_bytes, tfs_l1_bytes)

    postings: Dict[str, Dict[int, int]] = {}
    pos = 0
    for term, count in zip(terms, term_counts):
        term_postings: Dict[int, int] = {}
        doc_idx = 0
        for i in range(pos, pos + count):
            doc_idx += gaps[i]
            term_postings[doc_idx] = tfs[i]
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
        # term -> {doc_idx: term_freq}. Keyed by each doc's integer index
        # (assigned in corpus encounter order, see build()), not the raw
        # doc_id string -- every scorer's hot loop hashes this key on the
        # order of millions of times per query/build, and hashing/
        # comparing a small int is far cheaper than a ~40-char string.
        # self.doc_ids below maps back to the real doc_id string; scorers
        # only need that conversion once, when constructing their final
        # (doc_id, score) results, not on every posting.
        self.postings: Dict[str, Dict[int, int]] = {}
        self.doc_ids: List[str] = []  # doc_idx -> doc_id
        self.doc_len: List[int] = []  # doc_idx -> number of tokens
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().

        Tokenizes each document, populates self.postings, self.doc_ids,
        self.doc_len, self.N, and self.avg_doc_len. Raw document text is
        not retained — BM25/VSM only need term-frequency and length
        statistics, and keeping it around would inflate the graded on-disk
        index size for no query-time benefit.
        """
        total_len = 0
        postings = self.postings
        doc_ids = self.doc_ids
        doc_len = self.doc_len

        for doc_id, text in corpus:
            doc_idx = len(doc_ids)  # next integer index, in encounter order
            doc_ids.append(doc_id)

            tokens = tokenize(text)

            doc_len.append(len(tokens))
            total_len += len(tokens)

            # Counter(tokens) is a C-accelerated bulk count (CPython's
            # Counter.update has a fast path for exactly this) -- same
            # multiset of counts as a manual get()/increment loop, just
            # without a Python-level dict.get() call per token occurrence
            # (tens of millions across the full corpus).
            for term, count in Counter(tokens).items():
                # setdefault does the "look up this term's bucket, create
                # it if missing" step as one dict operation instead of
                # `if term not in postings: postings[term] = {}` (a
                # separate membership check plus, on a new term, a second
                # insert) followed by a third lookup to fetch it back.
                postings.setdefault(term, {})[doc_idx] = count

        self.N = len(doc_ids)
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
          3. Pack every gap through a 3-tier cascade (1 byte, escaping to
             2, escaping to 4) and every tf through a 2-tier cascade (1
             byte, escaping to 4), via the stdlib `array` module's
             C-level tobytes()/frombytes() -- see _pack_gaps'/_pack_tfs'
             docstrings. Measured on the full corpus: 86.3% of gaps and
             99.99998% of tfs fit in a single byte; a spec note (Section
             3) warns the held-out corpus could be >=500K docs, hence the
             cascading escape rather than assuming everything fits. A
             byte-by-byte hand-rolled varint (1-5 bytes depending on
             magnitude) was measured (against the original flat 4-byte
             encoding) at ~27s of extra index-build time from per-value
             function-call overhead -- this cascade gets a comparable
             size win from one or two cheap conditionals per value
             instead, still packed in bulk via array.tobytes().
          4. zlib-compress the whole thing as a final pass (stdlib, not a
             search/indexing library -- this is generic byte compression,
             not part of the required indexing/scoring logic). Level 9
             (measured: ~1334s -- over 22 minutes -- for this fixed-width
             payload, vs. ~2.7s at level 3 for barely better compression)
             would be catastrophic for index-build-time; level 3 gives
             nearly all of the size win for a small, bounded time cost.
        """
        # self.postings is already {term: {doc_idx: tf}} (build() assigns
        # doc_idx directly, see __init__/build()'s docstrings) -- no
        # doc_id-to-index translation step needed here at all.
        terms, term_counts, gaps_l0, gaps_l1, gaps_l2, tfs_l0, tfs_l1 = _encode_postings(self.postings)

        # Front-code `terms` (sorted, so adjacent entries very often share
        # a long stemmed root) instead of storing each string in full --
        # see _front_code's docstring.
        term_prefix_lens, term_suffixes = _front_code(terms)

        payload = {
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
            "term_prefix_lens": term_prefix_lens,
            "term_suffixes": term_suffixes,
            "term_counts": term_counts,
            "gaps_l0": gaps_l0,
            "gaps_l1": gaps_l1,
            "gaps_l2": gaps_l2,
            "tfs_l0": tfs_l0,
            "tfs_l1": tfs_l1,
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
        exact same self.postings / self.doc_ids / self.doc_len shapes
        (doc_idx-keyed) that build() produces, so every scorer module is
        unaffected by this on-disk encoding.
        """
        path = os.path.join(index_dir, "index.bin")
        with open(path, "rb") as f:
            blob = f.read()
        payload = pickle.loads(zlib.decompress(blob))

        idx = cls()
        idx.doc_ids = payload["doc_ids"]
        idx.doc_len = payload["doc_len"]
        idx.N = payload["N"]
        idx.avg_doc_len = payload["avg_doc_len"]

        terms = _front_decode(payload["term_prefix_lens"], payload["term_suffixes"])
        idx.postings = _decode_postings(
            terms,
            payload["term_counts"],
            payload["gaps_l0"],
            payload["gaps_l1"],
            payload["gaps_l2"],
            payload["tfs_l0"],
            payload["tfs_l1"],
        )

        return idx
