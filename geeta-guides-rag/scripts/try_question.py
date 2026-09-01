"""
scripts/try_question.py — run questions through the real pipeline, fast.

    python scripts/try_question.py "Why do I keep doing the thing I know is wrong?"
    python scripts/try_question.py -f candidates.txt          # one per line
    python scripts/try_question.py "..." --want 3:37          # assert a verse lands

WHY THIS EXISTS
---------------
Whether a question renders as an Arjuna→Krishna exchange is not something the
question list controls. The pairing only appears if the retriever happens to
rank one of the 18 paired Krishna verses into the teaching block (see
`make pairs`). So picking demo questions is iterative: word it, see what comes
back, reword.

`make demo` answers that too, but it embeds every question and rewrites the
page. This loads the model once and prints just what matters, so wording can be
tried in a loop. Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)
sys.path.insert(0, APP_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="*", help="questions to try")
    ap.add_argument("-f", "--file", help="file of questions, one per line")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--context-k", type=int, default=2)
    ap.add_argument("--want", action="append", default=[],
                    help="verse id (e.g. 3:37) that should appear; exit 1 if it does not")
    ap.add_argument("--public-only", action="store_true",
                    help="match the demo build, which drops the copyrighted Hindi")
    args = ap.parse_args()

    questions = list(args.questions)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            questions += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not questions:
        ap.error("give at least one question, or -f FILE")

    from app.guidance import Guidance, NotReady

    try:
        g = Guidance(public_only=args.public_only)
    except NotReady as e:
        raise SystemExit(str(e))

    print(f"model {g.retriever.model_name} · {len(g.verses)} verses · "
          f"pair_within={g.pair_within} ({len(g._answers)} pairable)\n")

    missing = []
    for q in questions:
        res = g.ask(q, k=args.k, context_k=args.context_k)
        route = "CURATED" if res.get("curated") else "retrieved"
        print(f'"{q}"   [{route}]')

        for v in res["teaching"]:
            asked = v.get("asks")
            score = "  ---" if v["score"] is None else f"{v['score']:.3f}"
            mark = "  <<< PAIR" if asked else ""
            if asked:
                print(f"    {asked['id']:>6}  arjuna   (the question)")
            print(f"    {v['id']:>6}  krishna  {score} {v['matched_lang'] or ''}{mark}")
        for v in res["dialogue"]:
            score = "  ---" if v["score"] is None else f"{v['score']:.3f}"
            print(f"    {v['id']:>6}  {v['speaker']:<8} {score} {v['matched_lang'] or ''}   (dialogue)")

        pairs = [v["id"] for v in res["teaching"] if v.get("asks")]
        print(f"    -> {len(pairs)} pair(s)" + (f": {', '.join(pairs)}" if pairs else "")
              + "\n")

        shown = {v["id"] for v in res["teaching"] + res["dialogue"]}
        shown |= {v["asks"]["id"] for v in res["teaching"] if v.get("asks")}
        for w in args.want:
            if w not in shown:
                missing.append((q, w))

    if missing:
        for q, w in missing:
            print(f"MISS  {w} did not appear for: {q}")
        sys.exit(1)


if __name__ == "__main__":
    main()
