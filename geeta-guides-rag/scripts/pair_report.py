"""
scripts/pair_report.py — what automatic Arjuna→Krishna pairing would claim.

    python scripts/pair_report.py            # the shipped threshold
    python scripts/pair_report.py --within 2 # what a wider one would do

The pairing renders as "Arjuna asks / Krishna answers" above a verse, which is
a factual claim about the text. This prints every claim the current threshold
would make, so it can be checked by eye against the book rather than trusted.

Reads only the corpus and the speaker labels — no embeddings, no model, so it
runs in a second without the index built.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)
sys.path.insert(0, APP_ROOT)


def main() -> None:
    # The same build_pairs the app calls, so this report cannot claim one thing
    # and the page render another.
    from app.guidance import VERSES_PATH, PAIR_WITHIN, build_pairs, verse_order

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--within", type=int, default=PAIR_WITHIN,
                    help=f"gap threshold (shipped default: {PAIR_WITHIN})")
    args = ap.parse_args()

    if not os.path.exists(VERSES_PATH):
        raise SystemExit(f"No corpus at {VERSES_PATH}\n  Build it: cd ../geeta-guides && make verses")

    with open(VERSES_PATH, "r", encoding="utf-8") as f:
        verses = {v["id"]: v for v in json.load(f)["verses"]}

    seq = sorted(verses, key=verse_order)
    speaker = {v: verses[v].get("speaker", "krishna") for v in seq}
    krishna = [v for v in seq if speaker[v] == "krishna"]

    # Distance from each Krishna verse back to the nearest Arjuna verse,
    # regardless of threshold — the shape of the corpus, for context.
    hist: collections.Counter = collections.Counter()
    for i, vid in enumerate(seq):
        if speaker[vid] != "krishna":
            continue
        gap = next((i - j for j in range(i - 1, -1, -1) if speaker[seq[j]] == "arjuna"), None)
        hist[gap if gap is not None and gap <= 5 else ">5"] += 1

    print(f"corpus    : {len(verses)} verses, {len(krishna)} spoken by Krishna")
    print(f"threshold : within={args.within}"
          + ("   (pairing OFF)" if args.within < 1 else "") + "\n")

    print("distance back to the nearest Arjuna verse:")
    for key in (1, 2, 3, 4, 5, ">5"):
        n = hist[key]
        print(f"  gap {str(key):>2} : {n:3d}  ({n / len(krishna) * 100:4.1f}%)")

    pairs = build_pairs(verses, args.within)
    print(f"\n{len(pairs)} of {len(krishna)} Krishna verses "
          f"({len(pairs) / len(krishna) * 100:.1f}%) would be captioned as a reply:\n")

    for vid, asked_by in sorted(pairs.items(), key=lambda kv: verse_order(kv[0])):
        i, j = seq.index(vid), seq.index(asked_by)
        between = [seq[m] for m in range(j + 1, i)]
        mid = ("  via " + ", ".join(f"{m.replace(':', '.')} {speaker[m]}" for m in between)) if between else ""
        flag = "  <-- CHECK: answer began earlier" if any(speaker[m] == "krishna" for m in between) else ""
        print(f"  {asked_by.replace(':', '.'):>7} (arjuna) -> {vid.replace(':', '.'):<7}{mid}{flag}")

    if args.within > 1:
        print("\n  Raising the threshold is nearly a no-op by design: build_pairs stops at\n"
              "  the first Krishna verse it walks back over, so a wider window can only\n"
              "  step across a Sanjaya framing verse. Anything flagged CHECK would be\n"
              "  continuing an answer rather than opening one.")
    print("\n  Revert entirely:  GITA_PAIR_WITHIN=0 make demo")


if __name__ == "__main__":
    main()
