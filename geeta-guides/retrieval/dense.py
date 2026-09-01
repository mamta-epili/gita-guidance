"""
retrieval/dense.py — dense retriever over the cached verse embeddings.

Implements the interface eval/score.py expects:

    make_retriever(verses) -> Callable[[str, int], list[str]]

so it can be scored against the same eval set as the BM25 baseline:

    python eval/score.py --retriever retrieval/dense.py --show-misses

MAX-POOLING ACROSS LANGUAGES
----------------------------
embed.py wrote one row per rendering, several rows per verse. A verse's score is
the score of its BEST row, not the mean. Mean would punish a verse for having a
Hindi rendering that an English question doesn't match — which is backwards:
having more renderings should only ever help.

MODEL MISMATCH
--------------
The query has to be embedded by the same model that embedded the corpus, or the
dot products are meaningless — not wrong-looking, just quietly meaningless. The
model name is stored in the index and checked on load, so a mismatch is an error
rather than a mysteriously bad score.
"""

from __future__ import annotations

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INDEX = os.path.join(ROOT, "data", "embeddings.npz")

_model_cache: dict[str, object] = {}


def _load_model(name: str, device: str | None = None):
    """Load (and memoize) a sentence-transformers model."""
    if name in _model_cache:
        return _model_cache[name]
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu")
    m = SentenceTransformer(name, device=device)
    _model_cache[name] = m
    return m


class DenseRetriever:
    """Flat exact search over ~1400 normalized vectors.

    No ANN index, deliberately. At this size an exact matrix multiply takes well
    under a millisecond and returns the true top-k; HNSW would add a dependency,
    a build step, and approximation error in exchange for nothing. Revisit at
    ~100k vectors, not 1.4k.
    """

    def __init__(self, index_path: str = INDEX, device: str | None = None):
        if not os.path.exists(index_path):
            raise SystemExit(
                f"missing {index_path}\n"
                f"  Build it first:  python retrieval/embed.py"
            )
        z = np.load(index_path, allow_pickle=False)
        self.vectors: np.ndarray = z["vectors"]
        self.ids: np.ndarray = z["ids"]
        self.langs: np.ndarray = z["langs"]
        self.model_name = str(z["model"])
        self.dim = int(z["dim"])
        self.model = _load_model(self.model_name, device)

        # Renamed in sentence-transformers 6; support both so the code doesn't
        # break on either side of that version boundary.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            getattr(self.model, "get_sentence_embedding_dimension")
        got = get_dim()
        if got != self.dim:
            raise SystemExit(
                f"dimension mismatch: index is {self.dim}-d (built with "
                f"{self.model_name}), loaded model gives {got}-d.\n"
                f"  Rebuild:  python retrieval/embed.py --model {self.model_name}"
            )

        # Precompute the unique verse ids and, for each, the rows belonging to it,
        # so max-pooling is a vectorised scatter rather than a Python loop.
        self.unique_ids, self._inverse = np.unique(self.ids, return_inverse=True)

        self.last_scores: list[float] = []

    def search(self, query: str, k: int = 10) -> list[str]:
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]

        # Cosine similarity for every row, in one matmul.
        row_scores = self.vectors @ q

        # Max-pool rows down to verses.
        verse_scores = np.full(len(self.unique_ids), -np.inf, dtype=np.float32)
        np.maximum.at(verse_scores, self._inverse, row_scores)

        k = min(k, len(self.unique_ids))
        # argpartition for the top-k, then sort just those.
        top = np.argpartition(-verse_scores, k - 1)[:k]
        top = top[np.argsort(-verse_scores[top])]

        self.last_scores = [float(verse_scores[i]) for i in top]
        return [str(self.unique_ids[i]) for i in top]

    def explain(self, query: str, k: int = 5) -> list[dict]:
        """Top-k with the language of the row that matched — useful for debugging
        whether an English question is matching English or Hindi rows."""
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]
        row_scores = self.vectors @ q
        order = np.argsort(-row_scores)
        out, seen = [], set()
        for i in order:
            vid = str(self.ids[i])
            if vid in seen:
                continue
            seen.add(vid)
            out.append({
                "id": vid,
                "score": round(float(row_scores[i]), 4),
                "matched_lang": str(self.langs[i]),
            })
            if len(out) >= k:
                break
        return out


def make_retriever(verses: list[dict]):
    """Entry point for eval/score.py.

    `verses` is accepted and ignored: the index already carries its verse ids,
    and rebuilding it here would hide whether embed.py and the eval set agree.
    """
    r = DenseRetriever()

    def search(query: str, k: int = 10) -> list[str]:
        ids = r.search(query, k)
        search.last_scores = r.last_scores  # type: ignore[attr-defined]
        return ids

    search.last_scores = []  # type: ignore[attr-defined]
    search.retriever = r     # type: ignore[attr-defined]
    return search


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "i am stressed guide me?"
    r = DenseRetriever()
    print(f"model : {r.model_name}  ({r.dim}-d)")
    print(f"index : {len(r.ids)} rows over {len(r.unique_ids)} verses")
    print(f"query : {q!r}\n")
    for row in r.explain(q, 8):
        print(f"  {row['id']:>7}  {row['score']:.4f}  [{row['matched_lang']}]")
