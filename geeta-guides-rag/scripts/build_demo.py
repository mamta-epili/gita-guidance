"""
scripts/build_demo.py — bake real retrieval results into a standalone demo page.

    python scripts/build_demo.py

Writes: <repo-root>/docs/demo.html   (served by GitHub Pages)

WHAT THIS IS
------------
A single self-contained HTML file with the *actual* answers the pipeline gives,
captured by running each question against all 700 verses. No backend, no API
keys, no network calls at runtime — the data is inlined into the page.

Recorded, not invented. Nothing here is hand-picked: every verse and every score
comes from the same `Guidance.ask()` the live app calls. If retrieval improves,
re-run this and the demo improves with it.

PUBLIC-DOMAIN ONLY
------------------
The page is published, so it is built with `public_only=True`. That drops the
Hindi renderings, which are copyrighted, and keeps the Sanskrit (ancient) and
Purohit Swami's English (d. 1941). The filter runs at the source, in
Guidance.render(), rather than being trimmed here — so there is one place the
rule lives and no way to forget it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)                     # geeta-guides-rag/
REPO_ROOT = os.path.dirname(APP_ROOT)                # gita-guidance/
TEMPLATE = os.path.join(HERE, "demo_template.html")
OUT = os.path.join(REPO_ROOT, "docs", "demo.html")

sys.path.insert(0, APP_ROOT)

# The questions the demo offers. Chosen to show the range rather than to
# flatter: vague guidance questions (the real use case), precise lookups, and
# the one question that bypasses retrieval entirely.
#
# ORDER MATTERS: the page opens on QUESTIONS[0], so the first entry is the
# landing state. The despair question is second — one click away, not the first
# thing a stranger sees on a page linked from a CV. Swap the two lines to lead
# with it.
QUESTIONS = [
    {"q": "I am stressed. Guide me.", "sa": "चिन्ता", "en": "stressed"},
    # Routed to the curated set rather than the retriever — the one place the
    # app overrides cosine similarity, and so the most worth demonstrating.
    # Labelled विषाद, which is the name of Chapter 1 itself: अर्जुनविषादयोग,
    # the Yoga of Arjuna's Despondency. The chapter is this question. The chip
    # stays gentle; the full phrasing appears only once it is clicked.
    {"q": "I don't want to live anymore", "sa": "विषाद", "en": "when nothing seems worth it"},
    {"q": "I am afraid of what is coming.", "sa": "भय", "en": "afraid"},
    {"q": "I am angry and I cannot let it go.", "sa": "क्रोध", "en": "angry"},
    {"q": "Someone I love has died. How do I bear it?", "sa": "शोक", "en": "grieving"},
    {"q": "I am in a situation where I know something wrong is happening "
          "but I cannot say it. What should I do?", "sa": "", "en": "a wrong I can't name"},
    # Aimed at the 2.54 -> 2.55 exchange, so the page carries a second
    # retrieved pairing rather than a third route to 3:37 (which "afraid" and
    # "desire to ruin" already reach). Arjuna: "how can we recognise the saint
    # whose mind is steady? how does he talk, how does he live?" — स्थितप्रज्ञ is
    # his own word for it.
    #
    # THE WORDING HERE IS LOAD-BEARING. Match the ANSWER's language, not the
    # question's. Measured with scripts/try_question.py --want 2:55:
    #
    #   "What does someone who is actually at peace look like?"   0 pairs
    #       -> 2:71, 6:7, 2:64, 14:23 teaching; 2:54 into DIALOGUE at 0.553.
    #          Reads naturally, and is a paraphrase of 2:54 — Arjuna's own
    #          words — so it retrieves the question and loses the answer.
    #   "How do I know when I have really let go of wanting things?"  1 pair
    #       -> 2:54 -> 2:55 at 0.553.  <- chosen
    #   "Is it possible to be content with nothing but yourself?"     1 pair
    #       -> 2:54 -> 2:55 at 0.536, but tops out on 3:17 matched via `hi`,
    #          which prints a "not published here" note on the lead card.
    #
    # Same failure that sank two earlier attempts (3.36->3.37, 5.1->5.2): a
    # question phrased like Arjuna's pulls Arjuna. This one echoes 2:55's
    # "given up the desires of his heart" instead, and pairs.
    {"q": "How do I know when I have really let go of wanting things?",
     "sa": "स्थितप्रज्ञ", "en": "have I really let go"},
    {"q": "How can I do my work well if I am not allowed to care about the result?",
     "sa": "कर्म", "en": "work without attachment"},
    {"q": "What is the chain that starts with wanting something and ends badly?",
     "sa": "मोह", "en": "desire to ruin"},
    {"q": "Is it better to be mediocre at what suits me than excellent at "
          "someone else's role?", "sa": "स्वधर्म", "en": "my own duty"},
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the standalone demo page.")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-index", action="store_true",
                    help="skip the index.html copy that makes the Pages root URL work")
    ap.add_argument("--k", type=int, default=4, help="Krishna verses per question")
    ap.add_argument("--context-k", type=int, default=2, help="Arjuna verses per question")
    args = ap.parse_args()

    from app.guidance import Guidance, NotReady

    try:
        # public_only=True is the whole licence story in one argument.
        g = Guidance(public_only=True)
    except NotReady as e:
        raise SystemExit(f"{e}\n\n  The demo is built from the real index, so it must exist.")

    print(f"corpus : {len(g.verses)} verses")
    print(f"model  : {g.retriever.model_name}\n")

    answers = []
    for item in QUESTIONS:
        res = g.ask(item["q"], k=args.k, context_k=args.context_k)
        answers.append({
            "question": item["q"],
            "chip_sa": item["sa"],
            "chip_en": item["en"],
            "curated": res.get("curated"),
            "teaching": res["teaching"],
            "dialogue": res["dialogue"],
            "pool_size": res["pool_size"],
        })
        top = res["teaching"][0]["id"] if res["teaching"] else "—"
        how = "curated" if res.get("curated") else "retrieved"
        # Which verses render as an exchange. Printed because it is the one
        # thing in the output that no setting guarantees — it depends on what
        # the retriever ranked.
        pairs = [f"{v['asks']['id']}→{v['id']}" for v in res["teaching"] if v.get("asks")]
        print(f"  {item['en'][:28]:28s} → {len(res['teaching'])} teaching, "
              f"{len(res['dialogue'])} dialogue, top {top}  [{how}]"
              + (f"  [pair {', '.join(pairs)}]" if pairs else ""))

    # Guard: a public build must not carry restricted text. Belt and braces on
    # top of public_only — if this ever fires, the filter regressed.
    #
    # Checks the redistributable flag on every rendering of every verse, in
    # every position, rather than looking for Hindi specifically. Hindi is
    # merely what is restricted today; a future corpus could carry a
    # copyrighted English translation as the preferred one, and a
    # Hindi-shaped guard would wave it straight through.
    def every_verse():
        for a in answers:
            for v in a["teaching"] + a["dialogue"]:
                yield v
                if v.get("asks"):          # paired verses render too
                    yield v["asks"]

    leaked = [
        (v["id"], lang, r["translator"])
        for v in every_verse()
        for lang in ("english", "hindi")
        if (r := v.get(lang)) and not r.get("redistributable", False)
    ]
    if leaked:
        listing = "\n    ".join(f"{vid} {lang} — {who}" for vid, lang, who in leaked[:8])
        raise SystemExit(
            f"ABORT: {len(leaked)} renderings are not redistributable but reached "
            f"the page:\n    {listing}\n"
            f"  Guidance(public_only=True) did not filter. Nothing was written."
        )

    data = {
        "generated": dt.date.today().isoformat(),
        "model": g.retriever.model_name,
        "verses_indexed": len(g.verses),
        "answers": answers,
    }

    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()
    if "__DEMO_DATA__" not in html:
        raise SystemExit(f"{TEMPLATE} has no __DEMO_DATA__ placeholder.")

    # ensure_ascii=False keeps the Devanagari readable in the source, and </
    # is escaped so a stray closing tag inside the data cannot break out of
    # the <script> block.
    payload = json.dumps(data, ensure_ascii=False, indent=1).replace("</", "<\\/")
    html = html.replace("__DEMO_DATA__", payload)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    # GitHub Pages serves index.html at the directory root, so without this the
    # bare /gita-guidance/ URL 404s and only /demo.html resolves. Written here
    # rather than by hand: a manual `cp` is correct exactly once and silently
    # stale after the next rebuild, which is how the published root ended up
    # showing links that had already been removed from the demo.
    if not args.no_index:
        index = os.path.join(os.path.dirname(args.out) or ".", "index.html")
        if os.path.abspath(index) != os.path.abspath(args.out):
            shutil.copyfile(args.out, index)

    size = os.path.getsize(args.out) / 1024
    print(f"\nwrote {args.out} ({size:.0f} KB, self-contained)")
    if not args.no_index:
        print(f"  copied to {index}  (Pages root)")
    print(f"  {len(answers)} questions, "
          f"{sum(len(a['teaching']) + len(a['dialogue']) for a in answers)} verses, "
          f"0 restricted")
    print("\n  preview:  open " + args.out)


if __name__ == "__main__":
    main()
