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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)                     # geeta-guides-rag/
REPO_ROOT = os.path.dirname(APP_ROOT)                # gita-guidance/
TEMPLATE = os.path.join(HERE, "demo_template.html")
OUT = os.path.join(REPO_ROOT, "docs", "demo.html")

sys.path.insert(0, APP_ROOT)

# The questions the demo offers. Chosen to show the range rather than to
# flatter: two guidance questions (vague, the real use case), two lookups
# (precise), and one where the honest answer is that the text predates the
# question.
QUESTIONS = [
    {"q": "I am stressed. Guide me.", "sa": "चिन्ता", "en": "stressed"},
    {"q": "I am afraid of what is coming.", "sa": "भय", "en": "afraid"},
    {"q": "I am angry and I cannot let it go.", "sa": "क्रोध", "en": "angry"},
    {"q": "Someone I love has died. How do I bear it?", "sa": "शोक", "en": "grieving"},
    {"q": "I am in a situation where I know something wrong is happening "
          "but I cannot say it. What should I do?", "sa": "", "en": "a wrong I can't name"},
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
            "teaching": res["teaching"],
            "dialogue": res["dialogue"],
            "pool_size": res["pool_size"],
        })
        top = res["teaching"][0]["id"] if res["teaching"] else "—"
        print(f"  {item['en'][:28]:28s} → {len(res['teaching'])} teaching, "
              f"{len(res['dialogue'])} dialogue, top {top}")

    # Guard: a public build must not carry restricted text. Belt and braces on
    # top of public_only — if this ever fires, the filter regressed.
    leaked = [
        v["id"] for a in answers for v in a["teaching"] + a["dialogue"]
        if v.get("hindi") is not None
    ]
    if leaked:
        raise SystemExit(
            f"ABORT: {len(leaked)} verses carry Hindi renderings, which are not "
            f"redistributable: {leaked[:5]}\n  Guidance(public_only=True) did not filter."
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

    size = os.path.getsize(args.out) / 1024
    print(f"\nwrote {args.out} ({size:.0f} KB, self-contained)")
    print(f"  {len(answers)} questions, "
          f"{sum(len(a['teaching']) + len(a['dialogue']) for a in answers)} verses, "
          f"0 restricted")
    print("\n  preview:  open " + args.out)


if __name__ == "__main__":
    main()
