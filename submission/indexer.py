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
import functools
import json
import os
import re
from typing import Dict, List, Tuple

from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()


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


class InvertedIndex:
    """A minimal inverted index skeleton. Extend the data structures here
    however your design needs (e.g. term positions for phrase/proximity
    scoring, a more compact postings representation for the efficiency
    bonus) — this is a starting point, not a fixed schema.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """corpus: list of (doc_id, text) pairs, e.g. from
        submission.corpus_utils.load_corpus().

        Tokenizes each document, populates self.postings, self.doc_len,
        self.N, and self.avg_doc_len. Raw document text is not retained —
        BM25/VSM only need term-frequency and length statistics, and
        keeping it around would inflate the graded on-disk index size for
        no query-time benefit.
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
        class median) — some starting points, roughly in order of effort:
          - json/pickle-dump self.postings etc. directly (works, but
            verbose: repeats every doc_id string per posting).
          - drop self.doc_text if your scorers don't need raw text at
            query time (BM25/VSM only need term-frequency and length
            statistics, not the original documents).
          - delta-encode each postings list's doc-ids (sorted ascending,
            store gaps instead of absolute ids) and varint/byte-pack them,
            instead of a naive JSON list of integers.

        Simplest-correct version: dump everything as one JSON file. Not
        size-optimized yet (see docstring above for compression ideas) —
        revisit once the basic pipeline works end-to-end.
        """
        data = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
        }
        path = os.path.join(index_dir, "index.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct an InvertedIndex purely from what save() wrote to
        `index_dir`. Called in a fresh process — do not rely on any state
        other than what's actually on disk in `index_dir`.
        """
        path = os.path.join(index_dir, "index.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls()
        idx.postings = data["postings"]
        idx.doc_len = data["doc_len"]
        idx.N = data["N"]
        idx.avg_doc_len = data["avg_doc_len"]
        return idx
