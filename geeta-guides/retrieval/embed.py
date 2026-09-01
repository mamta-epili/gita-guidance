"""
retrieval/embed.py — embed the 700 verses once and cache the vectors.

    python retrieval/embed.py                  # default model (bge-m3)
    python retrieval/embed.py --model <name>   # any sentence-transformers model
    python retrieval/embed.py --langs en,hi    # which renderings to index

Output: data/embeddings.npz

WHY ONE VECTOR PER LANGUAGE, NOT ONE PER VERSE
----------------------------------------------
A verse has an English rendering, a Hindi rendering, and Sanskrit. Concatenating
them into one string and embedding that produces a vector that is a blurred
average of three languages — worse at all three than a dedicated vector at any
one.

So each rendering gets its own row, all rows carry their verse id, and at query
time we take the BEST-scoring row per verse (max-pooling). An English question
matches the English row; a Hindi question matches the Hindi row; the verse id
that comes back is the same either way.

This is why the query side doesn't need to know what language it was given.

NORMALIZATION
-------------
Vectors are L2-normalized at write time, so cosine similarity is a plain dot
product and the whole index is one matrix multiply. Nothing downstream needs to
divide by norms, and there is no chance of forgetting to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VERSES = os.path.join(ROOT, "data", "verses.json")
OUT = os.path.join(ROOT, "data", "embeddings.npz")

# The model the project's original spec named: multilingual, 1024 dimensions.
# Multilingual matters here — the corpus is Devanagari and the queries are
# English or Hinglish, so a monolingual English model cannot see the Hindi rows
# at all.
DEFAULT_MODEL = "BAAI/bge-m3"

DEFAULT_LANGS = "en,hi"


def load_verses() -> list[dict]:
    if not os.path.exists(VERSES):
        raise SystemExit(f"missing {VERSES}\n  Build it first: make verses")
    with open(VERSES, "r", encoding="utf-8") as f:
        return json.load(f)["verses"]


def pick_rendering(renderings: list[dict], prefer: str | None) -> dict | None:
    """One rendering per language: the preferred translator, else the first."""
    if not renderings:
        return None
    if prefer:
        for r in renderings:
            if r["source_key"] == prefer:
                return r
    return renderings[0]


def build_rows(verses: list[dict], langs: list[str],
               prefer_en: str = "purohit",
               prefer_hi: str | None = None) -> tuple[list[str], list[str], list[str]]:
    """Flatten verses into (verse_ids, langs, texts) — one row per rendering."""
    ids: list[str] = []
    row_langs: list[str] = []
    texts: list[str] = []

    for v in verses:
        if "en" in langs:
            r = pick_rendering(v.get("english", []), prefer_en)
            if r:
                ids.append(v["id"]); row_langs.append("en"); texts.append(r["text"])
        if "hi" in langs:
            r = pick_rendering(v.get("hindi", []), prefer_hi)
            if r:
                ids.append(v["id"]); row_langs.append("hi"); texts.append(r["text"])
        if "sa" in langs:
            iast = v["sanskrit"].get("iast")
            if iast:
                ids.append(v["id"]); row_langs.append("sa"); texts.append(iast)
    return ids, row_langs, texts


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed the verse corpus.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--langs", default=DEFAULT_LANGS,
                    help=f"comma list from en,hi,sa (default: {DEFAULT_LANGS})")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto by default)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    verses = load_verses()
    ids, row_langs, texts = build_rows(verses, langs)

    print(f"corpus : {len(verses)} verses")
    print(f"rows   : {len(texts)}  ({', '.join(f'{l}={row_langs.count(l)}' for l in langs)})")
    print(f"model  : {args.model}")

    # Imported here rather than at module top so `--help` works without the
    # 2 GB of torch/transformers import cost.
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device is None:
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu")
    print(f"device : {device}")

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, device=device)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    vecs = model.encode(
        texts,
        batch_size=args.batch_size,
        # L2-normalize now so cosine similarity is a dot product later.
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0

    print(f"\nembedded {len(texts)} rows in {elapsed:.1f}s "
          f"({len(texts) / max(elapsed, 1e-9):.1f}/s)")
    print(f"matrix : {vecs.shape}  dtype {vecs.dtype}")

    norms = np.linalg.norm(vecs, axis=1)
    print(f"norms  : min {norms.min():.4f}  max {norms.max():.4f}  (should be 1.0)")

    np.savez_compressed(
        args.out,
        vectors=vecs,
        ids=np.array(ids),
        langs=np.array(row_langs),
        # Stored so the retriever can refuse to run with a mismatched model
        # instead of silently comparing vectors from two different spaces.
        model=np.array(args.model),
        dim=np.array(vecs.shape[1]),
    )
    size = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out} ({size:.1f} MB)")

    # Sanity check: a verse should be its own nearest neighbour.
    sims = vecs @ vecs[0]
    best = int(np.argmax(sims))
    print(f"\nself-retrieval check: row 0 ({ids[0]}) nearest is row {best} "
          f"({ids[best]}) at {sims[best]:.4f}")
    if best != 0:
        print("  WARNING: a row is not its own nearest neighbour — something is wrong.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
