"""
retrieval/speakers.py — label every verse with who is speaking.

    python retrieval/speakers.py            # patch data/verses.json in place
    python retrieval/speakers.py --check    # report only, change nothing

WHY THIS MATTERS FOR RETRIEVAL
------------------------------
The Gita is a dialogue. Arjuna asks and laments; Krishna answers; Sanjaya
narrates; Dhritarashtra asks once, at the very start.

Embedding similarity matches a question to text that *resembles a question*.
So "I am in distress, tell me what to do" scores highest against Arjuna's own
words — because Arjuna is saying the same thing. The retriever is working
correctly and returning the problem statement instead of the solution.

Labelling the speaker lets the answer layer put Krishna's verses first and show
Arjuna's separately, which is the distinction a reader actually wants: teaching
versus lament.

HOW THE LABEL IS DERIVED
------------------------
Only 60 of 700 verses carry an explicit marker (श्रीभगवानुवाच, अर्जुन उवाच,
सञ्जय उवाच, धृतराष्ट्र उवाच) — they appear where the speaker CHANGES. Everything
after a marker belongs to that speaker until the next one, so the label is a
forward fill in canonical verse order, across chapter boundaries (the dialogue
does not restart at a chapter break: chapter 2 ends with Krishna, and 3:1 is
Arjuna interrupting).

The result is checked against the traditional distribution — Krishna 574,
Arjuna 85, Sanjaya 40, Dhritarashtra 1 — which is a genuine test: if the fill
logic were wrong, those numbers would not land.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VERSES = os.path.join(ROOT, "data", "verses.json")

# Marker -> speaker key. Both spellings of Sanjaya occur in the wild.
MARKERS = {
    "श्रीभगवानुवाच": "krishna",
    "भगवानुवाच": "krishna",
    "अर्जुन उवाच": "arjuna",
    "सञ्जय उवाच": "sanjaya",
    "संजय उवाच": "sanjaya",
    "धृतराष्ट्र उवाच": "dhritarashtra",
}

DISPLAY = {
    "krishna": "Shri Krishna",
    "arjuna": "Arjuna",
    "sanjaya": "Sanjaya",
    "dhritarashtra": "Dhritarashtra",
}

# Traditional counts, used as a check on the fill rather than as data.
EXPECTED = {"krishna": 574, "arjuna": 85, "sanjaya": 40, "dhritarashtra": 1}
TOLERANCE = 6  # editions differ by a verse or two on where a speech begins


def marker_in(text: str) -> str | None:
    for m, who in MARKERS.items():
        if m in text:
            return who
    return None


def assign(verses: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Forward-fill the speaker in canonical order. Returns (verses, counts)."""
    ordered = sorted(verses, key=lambda v: (v["chapter"], v["verse"]))
    current = "sanjaya"  # 1:1 carries धृतराष्ट्र उवाच and overwrites this
    counts: dict[str, int] = {}

    for v in ordered:
        deva = v["sanskrit"].get("devanagari") or ""
        who = marker_in(deva)
        if who:
            current = who
            v["speaker_marked"] = True
        else:
            v["speaker_marked"] = False
        v["speaker"] = current
        v["speaker_name"] = DISPLAY[current]
        counts[current] = counts.get(current, 0) + 1

    return ordered, counts


def report(counts: dict[str, int], total: int) -> int:
    print("=" * 62)
    print("SPEAKER DISTRIBUTION")
    print("=" * 62)
    problems = 0
    for who, want in EXPECTED.items():
        got = counts.get(who, 0)
        delta = got - want
        flag = "ok" if abs(delta) <= TOLERANCE else "OFF"
        if abs(delta) > TOLERANCE:
            problems += 1
        print(f"  {DISPLAY[who]:16s} {got:4d}   traditional {want:4d}"
              f"   {delta:+d}  {flag}")
    extra = set(counts) - set(EXPECTED)
    for who in extra:
        print(f"  {who:16s} {counts[who]:4d}   UNEXPECTED")
        problems += 1
    print(f"\n  total {sum(counts.values())} / {total}")
    if problems:
        print("\n  The fill disagrees with the traditional counts by more than "
              f"{TOLERANCE} verses.\n  Check the marker list and the ordering.")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Label verses with the speaker.")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    if not os.path.exists(VERSES):
        raise SystemExit(f"missing {VERSES}\n  Build it first: make verses")
    with open(VERSES, "r", encoding="utf-8") as f:
        doc = json.load(f)

    ordered, counts = assign(doc["verses"])
    problems = report(counts, len(ordered))

    # A couple of spot checks a reader can verify by eye.
    by_id = {v["id"]: v for v in ordered}
    print("\n  spot checks:")
    for vid in ("1:1", "1:29", "2:7", "2:47", "6:34", "6:35", "18:78"):
        if vid in by_id:
            v = by_id[vid]
            mark = " (marked)" if v["speaker_marked"] else ""
            print(f"    {vid:>6}  {v['speaker_name']}{mark}")

    if args.check:
        raise SystemExit(1 if problems else 0)

    doc["verses"] = ordered
    doc.setdefault("meta", {})["speaker_labels"] = (
        "Forward-filled from 60 explicit उवाच markers in canonical verse order. "
        "See retrieval/speakers.py."
    )
    tmp = VERSES + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, VERSES)
    print(f"\nwrote {VERSES}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
