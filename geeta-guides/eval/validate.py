"""
eval/validate.py — check the eval set is sound before you trust any score from it.

Run:  python eval/validate.py

Four classes of problem, in increasing subtlety:

1. STRUCTURAL — missing fields, duplicate ids, malformed verse ids.
2. REFERENTIAL — a cited verse id that doesn't exist in data/verses.json.
   A typo'd '2:74' (chapter 2 has 72 verses) would otherwise show up later as
   an unexplainable retrieval miss.
3. BALANCE — are there enough questions, and enough refusal cases, for the
   numbers to mean anything?
4. CIRCULARITY — the subtle one, and the reason this file exists.

On circularity
--------------
If you write a question while reading its target verse, you reuse the verse's
vocabulary without noticing. Retrieval then scores brilliantly because it is
matching strings, and your 0.95 recall evaporates on the first real user.

This script measures, for each in-corpus question, the fraction of the
question's content words that also appear in its target verses. High overlap
does not prove you cheated — some overlap is unavoidable and legitimate — but a
set where most questions overlap heavily is a set that is testing string
matching rather than retrieval. Treat the distribution as the signal.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QUESTIONS = os.path.join(HERE, "questions.json")
VERSES = os.path.join(ROOT, "data", "verses.json")

TARGET_IN_CORPUS = 60
TARGET_OUT_OF_CORPUS = 15
OVERLAP_WARN = 0.50   # per-question: flag above this
OVERLAP_MEDIAN_WARN = 0.30  # whole set: flag if the median is above this

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

# Words too common to count as evidence of vocabulary borrowing.
STOPWORDS = set("""
a an the and or but if then than that this these those of in on at to for from by with
without about into over under again further is are was were be been being am do does did
doing have has had having i me my we our you your he him his she her it its they them their
what which who whom when where why how all any both each few more most other some such no
nor not only own same so too very can will just should now would could may might must shall
say says said tell told does do get got go goes going make makes made take takes taken
""".split())

TOKEN_RE = re.compile(r"[A-Za-z']+")


def content_words(text: str) -> set[str]:
    """Lowercased content words, stopwords removed, crude stemming applied.

    The stemming is deliberately crude — chopping a trailing 's' catches
    work/works and fruit/fruits, which is where most of the accidental overlap
    lives. A real stemmer would be more accurate and no more useful here.
    """
    out = set()
    for w in TOKEN_RE.findall(text.lower()):
        w = w.strip("'")
        if len(w) < 3 or w in STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def load() -> tuple[list[dict], dict[str, dict]]:
    if not os.path.exists(QUESTIONS):
        raise SystemExit(f"missing {QUESTIONS}")
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        qs = json.load(f)["questions"]

    if not os.path.exists(VERSES):
        raise SystemExit(
            f"missing {VERSES}\n  Build the corpus first: make verses"
        )
    with open(VERSES, "r", encoding="utf-8") as f:
        verses = {v["id"]: v for v in json.load(f)["verses"]}
    return qs, verses


def english_of(verse: dict) -> str:
    """All English renderings of a verse, concatenated, for overlap measurement.

    Uses every translation rather than one, because the question could plausibly
    have echoed any of them — measuring against a single translation would
    understate the overlap.
    """
    return " ".join(r["text"] for r in verse.get("english", []))


def main() -> None:
    qs, verses = load()
    problems: list[str] = []
    warnings: list[str] = []

    print("=" * 68)
    print("EVAL SET VALIDATION")
    print("=" * 68)

    # ---- 1. structural -------------------------------------------------
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for i, q in enumerate(qs):
        where = q.get("id") or f"index {i}"
        for req in ("id", "question", "verses", "kind"):
            if req not in q:
                problems.append(f"{where}: missing required field '{req}'")
        if q.get("id") in seen_ids:
            problems.append(f"{where}: duplicate id")
        seen_ids.add(q.get("id"))

        norm = " ".join(q.get("question", "").lower().split())
        if norm in seen_questions:
            problems.append(f"{where}: same question text as '{seen_questions[norm]}'")
        seen_questions[norm] = q.get("id", where)

        if q.get("kind") not in ("in_corpus", "out_of_corpus"):
            problems.append(f"{where}: kind must be in_corpus or out_of_corpus")
        if q.get("mode", "lookup") not in ("lookup", "guidance"):
            problems.append(f"{where}: mode must be lookup or guidance")

        for vid in q.get("verses", []) + q.get("also_relevant", []):
            if not re.fullmatch(r"\d{1,2}:\d{1,3}", vid):
                problems.append(f"{where}: malformed verse id {vid!r}")

        overlap = set(q.get("verses", [])) & set(q.get("also_relevant", []))
        if overlap:
            problems.append(f"{where}: {sorted(overlap)} listed as both a core "
                            f"target and also_relevant")

        if q.get("kind") == "in_corpus" and not q.get("verses"):
            problems.append(f"{where}: in_corpus question has no target verses")
        if q.get("kind") == "out_of_corpus":
            if q.get("verses"):
                problems.append(f"{where}: out_of_corpus question must have verses: []")
            if not q.get("why"):
                warnings.append(f"{where}: out_of_corpus without a 'why' — "
                                f"say why the text can't answer it")

    # ---- 2. referential ------------------------------------------------
    for q in qs:
        for vid in q.get("verses", []) + q.get("also_relevant", []):
            if vid not in verses:
                ch = vid.split(":")[0]
                n = sum(1 for k in verses if k.startswith(f"{ch}:"))
                problems.append(
                    f"{q.get('id')}: verse {vid} does not exist "
                    f"(chapter {ch} has {n} verses)"
                )

    # ---- 3. balance ----------------------------------------------------
    in_c = [q for q in qs if q.get("kind") == "in_corpus"]
    out_c = [q for q in qs if q.get("kind") == "out_of_corpus"]
    lookup = [q for q in in_c if q.get("mode", "lookup") == "lookup"]
    guidance = [q for q in in_c if q.get("mode") == "guidance"]
    print(f"\n  in-corpus questions     : {len(in_c):3d}  (target {TARGET_IN_CORPUS})")
    print(f"    of which lookup       : {len(lookup):3d}")
    print(f"    of which guidance     : {len(guidance):3d}")
    print(f"  out-of-corpus questions : {len(out_c):3d}  (target {TARGET_OUT_OF_CORPUS})")
    if in_c and not guidance:
        warnings.append(
            "no guidance-mode questions — but vague 'help me with X' questions "
            "are what the product is for. A set of pure lookups measures the "
            "wrong thing."
        )
    if len(in_c) < TARGET_IN_CORPUS:
        warnings.append(f"only {len(in_c)}/{TARGET_IN_CORPUS} in-corpus questions — "
                        f"scores on a set this small are noisy")
    if len(out_c) < TARGET_OUT_OF_CORPUS:
        warnings.append(f"only {len(out_c)}/{TARGET_OUT_OF_CORPUS} refusal cases")

    # chapter spread: a set that only probes chapter 2 tests chapter 2
    chapters = sorted({int(v.split(":")[0]) for q in in_c for v in q["verses"]})
    print(f"  chapters covered        : {len(chapters)}/18  {chapters}")
    if len(chapters) < 8 and len(in_c) >= 20:
        warnings.append(f"targets cluster in {len(chapters)} chapters — "
                        f"broaden coverage or you're only testing those")

    # ---- 4. circularity ------------------------------------------------
    print("\n" + "-" * 68)
    print("LEXICAL OVERLAP (question words also present in target verse)")
    print("-" * 68)
    overlaps: list[tuple[float, str, set[str]]] = []
    for q in in_c:
        qw = content_words(q["question"])
        if not qw:
            continue
        vw: set[str] = set()
        for vid in q["verses"]:
            if vid in verses:
                vw |= content_words(english_of(verses[vid]))
        shared = qw & vw
        frac = len(shared) / len(qw)
        overlaps.append((frac, q["id"], shared))

    overlaps.sort(reverse=True)
    for frac, qid, shared in overlaps:
        mark = f"{YELLOW}HIGH{RESET}" if frac >= OVERLAP_WARN else "    "
        words = ", ".join(sorted(shared)[:6]) or "-"
        print(f"  {mark} {frac:5.0%}  {qid[:34]:34s} {DIM}{words}{RESET}")
        if frac >= OVERLAP_WARN:
            warnings.append(
                f"{qid}: {frac:.0%} of question words appear in the target verse "
                f"— may be testing string matching"
            )

    if overlaps:
        fracs = sorted(f for f, _, _ in overlaps)
        median = fracs[len(fracs) // 2]
        print(f"\n  median overlap: {median:.0%}   "
              f"(want below {OVERLAP_MEDIAN_WARN:.0%})")
        if median > OVERLAP_MEDIAN_WARN:
            warnings.append(
                f"median overlap {median:.0%} is high across the whole set — "
                f"questions were probably written while reading the verses. "
                f"Write the next batch before looking."
            )

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 68)
    if problems:
        print(f"{RED}{len(problems)} PROBLEM(S){RESET}")
        for p in problems:
            print(f"  x {p}")
    if warnings:
        print(f"{YELLOW}{len(warnings)} WARNING(S){RESET}")
        for w in warnings:
            print(f"  ! {w}")
    if not problems and not warnings:
        print(f"{GREEN}eval set is sound{RESET}")
    print("=" * 68)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
