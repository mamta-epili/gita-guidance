"""
guidance.py — retrieval-backed verse recommendation.

Question in, shlokas out. No generation: the shloka, its translations and its
citation all come verbatim from data/verses.json. Nothing on this path can
invent scripture, because nothing on this path writes text.

WHERE THE DATA LIVES
--------------------
The corpus and the embedding index live in the sibling `geeta-guides` project,
which is also where the char-GPT checkpoint comes from. That folder is the
data/ML workspace; this one is the app. Same precedent as CHARGPT_CKPT — point
elsewhere with GUIDES_DIR.

Duplicating the corpus into this repo would mean two copies drifting apart, and
the licence metadata is per-field in that file for a reason.
"""

from __future__ import annotations

import json
import os
import re
import sys

from .safety import route, curated_ids

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GUIDES_DIR = os.environ.get(
    "GUIDES_DIR", os.path.join(os.path.dirname(ROOT), "geeta-guides")
)
VERSES_PATH = os.path.join(GUIDES_DIR, "data", "verses.json")
INDEX_PATH = os.path.join(GUIDES_DIR, "data", "embeddings.npz")

# Strip a leading "2.47" / "||2-47||" that some translations carry inline. The
# citation is rendered from the verse id, so repeating it inside the quote is
# noise.
_LEADING_REF = re.compile(r"^\s*\|*\s*\d+[.\-:]\d+\s*\|*\s*")

# How far back build_pairs may look for the Arjuna verse a Krishna verse
# answers.  0 turns the pairing off entirely — the behaviour before this was
# added.  Override without editing code:  GITA_PAIR_WITHIN=0 make demo
#
# WHY 1
# -----
# Measured over the 573 Krishna verses (make pairs):
#
#   gap 1 :  18  (3.1%)   <- previous verse is Arjuna. A real reply.
#   gap 2 :  18  (3.1%)   <- previous verse is KRISHNA, in all 18 cases.
#   gap>5 : 487  (85.0%)
#
# Every gap-2 case is 6.34 -> [6.35 krishna] -> 6.36: the answer began at 6.35,
# so captioning 6.36 as the reply to 6.34 points at the wrong verse. The
# pairing is a factual claim about the text and is made only where the text
# supports it — 18:58 sits 57 verses into an unbroken monologue and gets none.
#
# Raising this is close to a no-op, and deliberately so: build_pairs stops at
# the first Krishna verse it walks back over, so a larger value can only step
# across a Sanjaya framing verse, never across Krishna's own words. On the
# current corpus within=2 yields exactly the same 18 pairs as within=1. The
# threshold is the dial; that stop condition is the actual safeguard.
PAIR_WITHIN = int(os.environ.get("GITA_PAIR_WITHIN", "1"))


class NotReady(RuntimeError):
    """Raised when the corpus or the embedding index hasn't been built yet."""


def verse_order(vid: str) -> tuple[int, int]:
    """Recitation order. '10:2' sorts before '10:10', which string order won't."""
    ch, n = vid.split(":")
    return int(ch), int(n)


def build_pairs(verses: dict[str, dict], within: int) -> dict[str, str]:
    """Map each Krishna verse to the Arjuna verse it answers, if adjacent.

    Derived from the speaker labels — see retrieval/speakers.py, which
    forward-fills them from the उवाच markers.

    Ordering is (chapter, verse) across the whole book rather than per chapter,
    so an exchange spanning a chapter break would still be found. None
    currently do, but the corpus is regenerated from an upstream dataset.

    Module-level rather than a method because scripts/pair_report.py checks the
    same claims without loading the embedding index — two copies of this rule
    would eventually disagree about what the page is asserting.
    """
    if within < 1:
        return {}

    seq = sorted(verses, key=verse_order)
    speaker = [verses[v].get("speaker", "krishna") for v in seq]

    pairs: dict[str, str] = {}
    for i, vid in enumerate(seq):
        if speaker[i] != "krishna":
            continue
        for back in range(1, within + 1):
            j = i - back
            if j < 0:
                break
            if speaker[j] == "arjuna":
                pairs[vid] = seq[j]
                break
            # Hitting Krishna first means the reply already began earlier, so
            # this verse continues an answer rather than opening one.
            if speaker[j] == "krishna":
                break
    return pairs


def _clean(text: str) -> str:
    return _LEADING_REF.sub("", text).strip()


def _pick(renderings: list[dict], prefer: str | None) -> dict | None:
    if not renderings:
        return None
    if prefer:
        for r in renderings:
            if r.get("source_key") == prefer:
                return r
    return renderings[0]


class Guidance:
    """Loads the corpus + dense index once, answers questions from them."""

    def __init__(self, public_only: bool = False, pair_within: int | None = None):
        if not os.path.exists(VERSES_PATH):
            raise NotReady(
                f"No corpus at {VERSES_PATH}\n"
                f"  Build it:  cd {GUIDES_DIR} && make verses"
            )
        if not os.path.exists(INDEX_PATH):
            raise NotReady(
                f"No embedding index at {INDEX_PATH}\n"
                f"  Build it:  cd {GUIDES_DIR} && make embed"
            )

        with open(VERSES_PATH, "r", encoding="utf-8") as f:
            doc = json.load(f)
        self.verses: dict[str, dict] = {v["id"]: v for v in doc["verses"]}
        self.meta = doc.get("meta", {})

        # Import the retriever from the sibling project rather than vendoring it,
        # so eval/score.py and this endpoint are provably scoring the same code.
        if GUIDES_DIR not in sys.path:
            sys.path.insert(0, GUIDES_DIR)
        from retrieval.dense import DenseRetriever  # noqa: E402

        self.retriever = DenseRetriever(INDEX_PATH)
        self.public_only = public_only
        self.pair_within = PAIR_WITHIN if pair_within is None else pair_within
        # Computed once at load: 700 verses, but it runs on every question.
        self._answers = build_pairs(self.verses, self.pair_within)

    # -- shaping ---------------------------------------------------------
    def render(self, vid: str, score: float | None = None,
               matched_lang: str | None = None) -> dict:
        """One verse in display order: shloka first, then renderings.

        Mirrors show.py's contract deliberately — the CLI and the API must not
        drift into quoting verses two different ways (PRD FR-3.5a).
        """
        v = self.verses[vid]
        ch, n = vid.split(":")

        def rendering(items: list[dict], prefer: str | None) -> dict | None:
            r = _pick(items, prefer)
            if r is None:
                return None
            if self.public_only and not r.get("redistributable", False):
                return None
            return {
                "text": _clean(r["text"]),
                "translator": r["translator"],
                "licence": r["licence"],
                "redistributable": r["redistributable"],
            }

        return {
            "id": vid,
            "chapter": int(ch),
            "verse": int(n),
            "citation": f"Bhagavad Gita {ch}.{n}",
            # Who is speaking. See retrieval/speakers.py — this is what lets the
            # answer put Krishna's teaching above Arjuna's lament.
            "speaker": v.get("speaker", "krishna"),
            "speaker_name": v.get("speaker_name", "Shri Krishna"),
            # The shloka, first.
            "sanskrit": v["sanskrit"].get("devanagari"),
            "iast": v["sanskrit"].get("iast"),
            "english": rendering(v.get("english", []), "purohit"),
            "hindi": rendering(v.get("hindi", []), None),
            "score": None if score is None else round(float(score), 4),
            "matched_lang": matched_lang,
        }

    # -- curated route ----------------------------------------------------
    def _curated(self, question: str, plan) -> dict:
        """Answer from a hand-chosen set instead of the retriever.

        Same response shape as a normal answer, so the frontend renders it
        through the same components — Krishna's section leads, Arjuna's
        follows. Two differences: no scores (nothing was ranked), and a verse
        in the lead may carry `asks`, the verse it replies to, so the
        6:34 → 6:35 exchange reads as an exchange.
        """
        p = plan.plan
        missing = [v for v in curated_ids(p) if v not in self.verses]
        if missing:
            # Loud rather than silent: a curated plan pointing at verses that
            # aren't in the corpus means the corpus changed under it.
            raise NotReady(f"curated plan references missing verses: {missing}")

        teaching = []
        for vid, asked_by in p["lead"]:
            v = self.render(vid)
            if asked_by:
                v["asks"] = self.render(asked_by)
            teaching.append(v)

        dialogue = [self.render(vid) for vid in p["follow"]]

        return {
            "question": question,
            "curated": {
                "reason": p["reason"],
                "lead_heading_sa": p["lead_heading_sa"],
                "lead_heading_en": p["lead_heading_en"],
                "follow_heading_sa": p["follow_heading_sa"],
                "follow_heading_en": p["follow_heading_en"],
                "follow_note": p["follow_note"],
            },
            "teaching": teaching,
            "dialogue": dialogue,
            "verses": teaching + dialogue,
            "pool_size": len(teaching) + len(dialogue),
            "top_score": None,
            "score_note": (
                "These verses are chosen, not ranked. For this question the "
                "closest matches by similarity are not the aptest ones."
            ),
        }

    # -- the endpoint's work ---------------------------------------------
    def ask(self, question: str, k: int = 5, context_k: int = 3) -> dict:
        """Retrieve, then split the dialogue.

        Krishna's verses are the answer; Arjuna's are the question restated.
        Embedding similarity cannot tell them apart — a query that describes
        distress is genuinely closest to Arjuna describing distress — so the
        split happens here, on the speaker label, after retrieval.

        A wider pool is retrieved than is shown, because filtering to Krishna
        from a pool of k would often leave one or two verses. Arjuna's matches
        aren't discarded; they move to a secondary block, where they are useful
        as "here is your question, in his words".
        """
        question = (question or "").strip()
        if not question:
            return {
                "question": "", "teaching": [], "dialogue": [], "verses": [],
                "curated": None, "note": "Ask about a situation or feeling.",
            }

        # Route before embedding. One category — despair — gets a hand-chosen
        # set, because cosine similarity ranks 2:20 and 2:22 top for it and
        # those are the wrong verses to hand someone cold. See app/safety.py.
        # Decided here rather than in the HTTP handler so every caller is
        # covered: the API, the demo builder, and anything added later.
        plan = route(question)
        if plan.mode == "curated":
            return self._curated(question, plan)

        # Pool wide enough that the Krishna block fills even when the top of the
        # ranking is dominated by Arjuna.
        pool_size = max(k * 4, 16)
        rows = [r for r in self.retriever.explain(question, k=pool_size)
                if r["id"] in self.verses]

        rendered = [self.render(r["id"], r["score"], r["matched_lang"]) for r in rows]

        teaching = [v for v in rendered if v["speaker"] == "krishna"][:k]
        others = [v for v in rendered if v["speaker"] != "krishna"]

        # Attach the question each teaching verse answers, where the text
        # actually says so. The retriever ranks verses one at a time and has no
        # idea any of them is a reply; this restores that from the speaker
        # labels, so the curated and retrieved routes render the same shape
        # whenever the pairing is true. Most verses get none — 18:58 is 57
        # verses into a monologue, and inventing a question for it would be a
        # confident-looking lie.
        paired: set[str] = set()
        for v in teaching:
            asked_by = self._answers.get(v["id"])
            if asked_by:
                # No score: the pair is a fact about the text, not a hit. A
                # number here would suggest the retriever found it.
                v["asks"] = self.render(asked_by)
                paired.add(asked_by)

        # A verse shown as a question above its answer must not also appear in
        # the Arjuna block below — same verse, twice, one screen apart. Filter
        # before the slice, or promoting one verse silently shortens the block
        # instead of pulling the next match up into it.
        dialogue = [v for v in others if v["id"] not in paired][:context_k]

        return {
            "question": question,
            "model": self.retriever.model_name,
            "curated": None,
            # Krishna first — that is the whole point of the split.
            "teaching": teaching,
            "dialogue": dialogue,
            # Flat list kept for any caller that just wants ranked verses.
            "verses": teaching + dialogue,
            "pool_size": len(rendered),
            # Surfaced rather than hidden: the eval run showed that retrieval
            # score does NOT separate answerable from unanswerable questions
            # (best unanswerable 0.56 vs median answerable 0.52), so this number
            # must not be used as a refusal threshold. It is shown for
            # inspection only.
            "top_score": (teaching or dialogue or [{"score": None}])[0]["score"],
            "score_note": (
                "Similarity is shown for inspection. It is deliberately NOT used "
                "to decide whether the corpus can answer the question — measured "
                "score distributions for answerable and unanswerable questions "
                "overlap. See docs/PRD.md FR-3.3."
            ),
        }
