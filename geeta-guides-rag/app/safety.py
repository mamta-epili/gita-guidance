"""
safety.py — curated answers for questions where similarity ranking picks badly.

    python -m app.safety --test          run the test suite
    python -m app.safety "some question" classify one question

WHAT THIS DOES NOW
------------------
Nothing is blocked. Every question gets an answer from the Gita — that is the
product. What this module changes is *which* verses answer one specific
category, because for that category cosine similarity is actively misleading.

THE PROBLEM IT SOLVES
---------------------
Ask "I don't want to live anymore" and the highest-scoring verses are:

    2:20  "It was not born; It will never die... the Spirit dies not when the
           body is dead."
    2:22  "As a man discards his threadbare robes and puts on new, so the
           Spirit throws off Its worn-out bodies and takes fresh ones."

Across eighteen chapters those are consolation. Ranked into five cards and read
cold, they are the two most misreadable verses in the book. The retriever is
not wrong — they *are* the closest match — it simply has no way to know that
proximity and aptness diverge here.

So this one category gets a hand-chosen set instead:

    6:34 → 6:35   Arjuna: the mind cannot be held. Krishna: it can, with
                  practice — the one place he concedes the difficulty before
                  answering.
    6:40          "My beloved child! There is no destruction for him, either in
                  this world or in the next."
    18:66         "Do not be anxious; I will absolve thee."
    1:29, 1:46    Arjuna's own despair — "better for my welfare" — so the
    2:7           reader sees the Gita opens exactly here.

Everything else in the app stays pure retrieval. This is the only exception,
and it exists because the text deserves better than a ranking for this one
question.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# The curated set. Order is the order shown.
# ---------------------------------------------------------------------------
DESPAIR = {
    "reason": "despair",
    # Krishna's section, shown first, in the gold enclosure.
    "lead": [
        # (verse, the verse it answers — rendered as an "Arjuna asks" pair)
        ("6:35", "6:34"),
        ("6:40", None),
        ("18:66", None),
    ],
    "lead_heading_sa": "श्रीभगवानुवाच",
    "lead_heading_en": "The Blessed Lord said",
    # Arjuna's section, shown second, muted.
    "follow": ["1:29", "1:46", "2:7"],
    "follow_heading_sa": "अर्जुन उवाच",
    "follow_heading_en": "You are not the first",
    "follow_note": (
        "The Gita opens with a man in this exact place. These are his words, "
        "not advice — and the whole book exists because he said them."
    ),
}

# ---------------------------------------------------------------------------
# PATTERNS
#
# First-person anchoring where a phrase could plausibly be about the text
# itself: "Arjuna wanted to die rather than fight" must not route, while "I
# want to die" must. Unambiguous terms (suicide, आत्महत्या, khudkushi) need no
# anchor — nobody types those casually into a scripture search.
#
# Erring toward routing is cheap now: a false positive gets you six carefully
# chosen verses instead of five ranked ones, which is not a bad outcome. That
# is a nicer failure mode than the old blocking design had.
# ---------------------------------------------------------------------------
I = r"(?:i|i'?m|im|me|my)"

DESPAIR_PATTERNS = [
    r"\bsuicid(?:e|al)\b",
    r"\bkhudkushi\b", r"\batmahatya\b", r"\baatmahatya\b",
    r"आत्महत्या", r"खुदकुशी",
    r"\bself[\s-]?harm",
    r"\bkill\s+(?:my\s?self|myself)\b",
    r"\bend\s+(?:my\s+life|it\s+all)\b",
    r"\btake\s+my\s+own\s+life\b",
    rf"\b{I}\b[^.?!]{{0,28}}\bwant(?:s|ed)?\s+to\s+die\b",
    rf"\b{I}\b[^.?!]{{0,28}}\b(?:don'?t|dont|do not|no longer)\s+want\s+to\s+(?:live|be here|exist)\b",
    rf"\b{I}\b[^.?!]{{0,28}}\bwish\s+{I}\s+(?:was|were)\s+dead\b",
    r"\bno\s+(?:reason|point)\s+(?:to|in)\s+(?:living|live|be alive)\b",
    r"\bbetter\s+off\s+(?:dead|without me)\b",
    rf"\b{I}\b[^.?!]{{0,28}}\b(?:cut|hurt|harm)(?:ting|ing)?\s+my\s?self\b",
    rf"\b{I}\b[^.?!]{{0,20}}\bgive\s+up\s+on\s+life\b",
    r"\bmar\s*(?:na|jaun|jaunga|jaungi|jana)\s+chaht", r"\bjeena\s+nahi", r"\bjina\s+nahi",
    r"\bzindagi\s+khatam\b", r"\bjaan\s+de(?:ni|na)\b",
    r"मरना\s*चाहत", r"जीना\s*नहीं",
]

_DESPAIR = [re.compile(p, re.IGNORECASE) for p in DESPAIR_PATTERNS]


@dataclass
class Route:
    """How a question should be answered."""
    mode: str                       # "curated" | "retrieval"
    reason: str | None = None       # "despair" | None
    plan: dict | None = field(default=None, repr=False)
    matched: str | None = None      # the pattern that fired, for audit

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("plan", None)         # the plan is internal; ids go in the response
        return d


RETRIEVAL = Route(mode="retrieval")


def route(question: str) -> Route:
    """Decide whether a question gets the curated set or normal retrieval."""
    text = (question or "").strip()
    if not text:
        return RETRIEVAL
    for p in _DESPAIR:
        if p.search(text):
            return Route("curated", "despair", DESPAIR, p.pattern)
    return RETRIEVAL


def curated_ids(plan: dict) -> list[str]:
    """Every verse id a plan references, for validation against the corpus."""
    ids = [v for v, _ in plan["lead"]]
    ids += [a for _, a in plan["lead"] if a]
    ids += plan["follow"]
    return ids


# ---------------------------------------------------------------------------
# Tests. The routing decision is the thing worth testing: does a question about
# despair get the curated set, and does a question about the text not?
# ---------------------------------------------------------------------------
MUST_CURATE = [
    "I don't want to live anymore",
    "i dont want to live any more, guide me",
    "I want to die",
    "I am thinking about suicide",
    "feeling suicidal, what does krishna say",
    "how do I kill myself",
    "I want to end my life",
    "there is no reason to live",
    "everyone would be better off without me",
    "I have been cutting myself",
    "self harm",
    "I want to give up on life",
    "mai marna chahta hoon",
    "jeena nahi chahta",
    "मुझे जीना नहीं है",
    "आत्महत्या के बारे में",
    "khudkushi karna chahta hun",
    "I wish I was dead",
]

MUST_RETRIEVE = [
    "I am stressed. Guide me.",
    "I am afraid of what is coming",
    "I am angry and I cannot let it go",
    "Someone I love has died. How do I bear it?",
    "what does the gita say about death",
    "Arjuna wanted to die rather than kill his teachers",
    "why should I not grieve for the dead",
    "is the soul killed when the body is killed",
    "I am in a situation where something wrong is happening but I cannot say it",
    "how do I do my duty without attachment",
    "what happens after death",
    "krishna tells arjuna to fight and kill",
    "I feel lost and confused about my career",
    "I think I'm having a panic attack",
    "my husband and I argue constantly",
]


def _run_tests() -> int:
    fails = []

    for q in MUST_CURATE:
        r = route(q)
        if r.mode != "curated" or r.reason != "despair":
            fails.append(("NOT ROUTED", q, r.mode, r.matched))

    for q in MUST_RETRIEVE:
        r = route(q)
        if r.mode != "retrieval":
            fails.append(("OVER-ROUTED", q, r.mode, r.matched))

    # The plan must reference verses that exist, in the right sections.
    ids = curated_ids(DESPAIR)
    if len(ids) != len(set(ids)):
        fails.append(("DUPLICATE in plan", ", ".join(ids), "", ""))
    for vid in ids:
        if not re.fullmatch(r"\d{1,2}:\d{1,3}", vid):
            fails.append(("BAD id", vid, "", ""))

    print("=" * 66)
    print("QUESTION ROUTING")
    print("=" * 66)
    print(f"  curated (despair)  : {len(MUST_CURATE)}")
    print(f"  normal retrieval   : {len(MUST_RETRIEVE)}")
    print(f"  verses in the plan : {len(ids)}  {' '.join(ids)}")

    if fails:
        print(f"\n  {len(fails)} FAILURE(S)")
        for kind, q, got, extra in fails:
            print(f"    {kind}: {q!r}  -> {got!r} {extra!r}")
        return 1
    print("\n  all pass")
    return 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        raise SystemExit(_run_tests())
    for q in sys.argv[1:]:
        r = route(q)
        print(f"{r.mode:10s} {r.reason or '-':10s} {q!r}")
