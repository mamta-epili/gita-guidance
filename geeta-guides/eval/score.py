"""
eval/score.py — score a retriever against eval/questions.json.

    python eval/score.py                          # built-in BM25 baseline
    python eval/score.py --retriever path/to/my_retriever.py

This exists so that PRD milestone 1c — "a baseline number exists, however bad"
— is satisfiable today, before any embeddings. Every later retrieval change is
then judged as a delta against a real number instead of a feeling.

RETRIEVER INTERFACE
-------------------
Your module must expose:

    def make_retriever(verses: list[dict]) -> Callable[[str, int], list[str]]

`verses` is the list from data/verses.json. The returned callable takes
(query, k) and returns up to k verse ids, best first. Nothing else is assumed,
so a dense retriever, a hybrid one, or a reranking pipeline all drop in.

METRICS
-------
hit@k      fraction of questions with AT LEAST ONE target verse in the top k.
           The user-facing number: did the answer become possible at all?
recall@k   mean fraction of a question's target verses found in the top k.
           Harsher, and the one to watch when questions have several targets.
MRR        mean of 1/(rank of first relevant verse). Rewards ranking, not just
           presence — the difference between position 1 and position 9.

SEPARABILITY
------------
Out-of-corpus questions have no target verses, so retrieval metrics don't apply.
What they can tell you is whether retrieval *scores* distinguish answerable from
unanswerable questions. If the score distributions overlap completely, then no
score threshold can drive refusal, and refusal has to be the generator's job.
Knowing that before building the refusal path is worth a lot.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QUESTIONS = os.path.join(HERE, "questions.json")
VERSES = os.path.join(ROOT, "data", "verses.json")

KS = (1, 3, 5, 10, 20)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

TOKEN_RE = re.compile(r"[A-Za-z']+")


def tokens(text: str) -> list[str]:
    out = []
    for w in TOKEN_RE.findall(text.lower()):
        w = w.strip("'")
        if len(w) < 3:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# Built-in baseline: BM25 over the English renderings.
# ---------------------------------------------------------------------------
# Deliberately the dumbest thing that isn't random. It has no idea that
# "detachment" and "renounce the fruit of action" are related, which is exactly
# the gap embeddings are supposed to close. That makes it the right baseline:
# whatever you build next has to beat *this*, and if it doesn't, the embeddings
# aren't earning their complexity.

def make_retriever(verses: list[dict]):
    """BM25. Returns (query, k) -> [verse_id, ...] plus scores via .last_scores."""
    k1, b = 1.5, 0.75

    ids: list[str] = []
    docs: list[list[str]] = []
    for v in verses:
        text = " ".join(r["text"] for r in v.get("english", []))
        ids.append(v["id"])
        docs.append(tokens(text))

    N = len(docs)
    avgdl = sum(len(d) for d in docs) / max(N, 1)
    tfs = [Counter(d) for d in docs]
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def search(query: str, k: int = 10) -> list[str]:
        qt = tokens(query)
        scored: list[tuple[float, str]] = []
        for i, tf in enumerate(tfs):
            dl = len(docs[i]) or 1
            s = 0.0
            for w in qt:
                f = tf.get(w, 0)
                if not f:
                    continue
                s += idf.get(w, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            if s > 0:
                scored.append((s, ids[i]))
        scored.sort(key=lambda t: (-t[0], t[1]))
        search.last_scores = [s for s, _ in scored[:k]]  # type: ignore[attr-defined]
        return [vid for _, vid in scored[:k]]

    search.last_scores = []  # type: ignore[attr-defined]
    return search


def load_retriever(path: str | None, verses: list[dict]):
    if path is None:
        return make_retriever(verses), "built-in BM25 baseline"
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(path):
        raise SystemExit(f"retriever not found: {path}")
    spec = importlib.util.spec_from_file_location("retriever_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["retriever_under_test"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make_retriever"):
        raise SystemExit(
            f"{path} must expose make_retriever(verses) -> Callable[[str, int], list[str]]"
        )
    return mod.make_retriever(verses), os.path.relpath(path, ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a retriever on the eval set.")
    ap.add_argument("--retriever", default=None,
                    help="path to a module exposing make_retriever(verses)")
    ap.add_argument("--show-misses", action="store_true",
                    help="list questions where nothing relevant appeared in the top 10")
    args = ap.parse_args()

    if not os.path.exists(VERSES):
        raise SystemExit(f"missing {VERSES} — run: make verses")
    with open(VERSES, "r", encoding="utf-8") as f:
        verses = json.load(f)["verses"]
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]

    search, label = load_retriever(args.retriever, verses)
    in_c = [q for q in qs if q["kind"] == "in_corpus"]
    out_c = [q for q in qs if q["kind"] == "out_of_corpus"]

    print("=" * 68)
    print("RETRIEVAL SCORE")
    print("=" * 68)
    print(f"  retriever : {label}")
    print(f"  corpus    : {len(verses)} verses")
    print(f"  questions : {len(in_c)} in-corpus, {len(out_c)} out-of-corpus")

    maxk = max(KS)
    in_top_scores: list[float] = []

    def evaluate(group: list[dict]) -> dict:
        """Score one group of questions.

        hit@k and MRR count `verses` + `also_relevant` as relevant — a retriever
        that surfaces a legitimately good verse you didn't happen to list should
        not be marked wrong. recall@k stays strict on the core `verses` only, so
        `also_relevant` can never inflate it.
        """
        hits = {k: 0 for k in KS}
        recalls = {k: 0.0 for k in KS}
        rr_total = 0.0
        misses = []
        for q in group:
            got = search(q["question"], maxk)
            s = getattr(search, "last_scores", [])
            in_top_scores.append(s[0] if s else 0.0)

            core = set(q["verses"])
            relevant = core | set(q.get("also_relevant", []))

            for k in KS:
                topk = set(got[:k])
                if topk & relevant:
                    hits[k] += 1
                recalls[k] += len(topk & core) / max(len(core), 1)

            rr = 0.0
            for rank, vid in enumerate(got, start=1):
                if vid in relevant:
                    rr = 1.0 / rank
                    break
            rr_total += rr
            if rr == 0.0:
                misses.append((q["id"], sorted(relevant), got[:5]))
        n = max(len(group), 1)
        return dict(n=len(group), hits=hits, recalls=recalls,
                    mrr=rr_total / n, misses=misses)

    def report(title: str, r: dict) -> None:
        if not r["n"]:
            return
        n = r["n"]
        print("\n" + "-" * 68)
        print(f"{title}  ({n} questions)")
        print("-" * 68)
        print(f"  {'k':>4}  {'hit@k':>7}  {'recall@k':>9}")
        for k in KS:
            print(f"  {k:>4}  {r['hits'][k] / n:>6.1%}  {r['recalls'][k] / n:>8.1%}")
        print(f"  MRR : {r['mrr']:.3f}")
        if r["misses"]:
            print(f"\n  {YELLOW}{len(r['misses'])}/{n} found nothing relevant "
                  f"in the top {maxk}{RESET}")
            if args.show_misses:
                for qid, want, got in r["misses"]:
                    print(f"    {qid}")
                    print(f"      relevant : {', '.join(want)}")
                    print(f"      got      : {', '.join(got)}")

    lookup = [q for q in in_c if q.get("mode", "lookup") == "lookup"]
    guidance = [q for q in in_c if q.get("mode") == "guidance"]

    report("ALL IN-CORPUS", evaluate(in_c))
    in_top_scores.clear()  # recomputed per group; keep only the final pass
    if lookup and guidance:
        report("LOOKUP questions", evaluate(lookup))
        report("GUIDANCE questions  (vague, many valid answers)",
               evaluate(guidance))
        print(f"\n  {DIM}Guidance questions are the product's real use case. If "
              f"their scores lag\n  the lookup ones, the retriever is good at "
              f"finding facts and bad at\n  finding help — which is the wrong "
              f"way round.{RESET}")
    in_top_scores.clear()
    evaluate(in_c)  # repopulate in_top_scores from the full set

    # ---- separability -------------------------------------------------
    if out_c:
        out_top_scores = []
        for q in out_c:
            search(q["question"], maxk)
            s = getattr(search, "last_scores", [])
            out_top_scores.append(s[0] if s else 0.0)

        print("\n" + "-" * 68)
        print("REFUSAL SEPARABILITY (top-1 retrieval score)")
        print("-" * 68)

        def stats(xs):
            xs = sorted(xs)
            if not xs:
                return (0.0, 0.0, 0.0)
            return (xs[0], xs[len(xs) // 2], xs[-1])

        i_lo, i_mid, i_hi = stats(in_top_scores)
        o_lo, o_mid, o_hi = stats(out_top_scores)
        print(f"  answerable   : min {i_lo:6.2f}  median {i_mid:6.2f}  max {i_hi:6.2f}")
        print(f"  unanswerable : min {o_lo:6.2f}  median {o_mid:6.2f}  max {o_hi:6.2f}")

        if o_hi >= i_mid:
            print(f"\n  {YELLOW}Distributions overlap.{RESET} The best-scoring "
                  f"unanswerable question ({o_hi:.2f}) outscores half the\n"
                  f"  answerable ones. No score threshold can drive refusal — it "
                  f"has to be\n  the generator's job, grounded in whether the "
                  f"retrieved text actually\n  supports an answer. See PRD FR-3.3.")
        else:
            print(f"\n  A threshold between {o_hi:.2f} and {i_mid:.2f} separates them "
                  f"on this set.\n  Treat that cautiously — {len(out_c)} refusal "
                  f"cases is a thin basis for a threshold.")

    print("\n" + "=" * 68)
    if len(in_c) < 60:
        print(f"  {DIM}Note: {len(in_c)} in-corpus questions. These numbers are "
              f"indicative, not\n  trustworthy, until the set reaches ~60. "
              f"See eval/questions.json.{RESET}")
        print("=" * 68)


if __name__ == "__main__":
    main()
