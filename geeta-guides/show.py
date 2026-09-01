"""
show.py — render verses in the canonical display order.

    python show.py 2:47
    python show.py 6:34 6:35 1:29
    python show.py 2:14 --lang sa,en          # Sanskrit + English only
    python show.py 2:47 --plain               # no ANSI, for piping

THE DISPLAY CONTRACT
--------------------
The shloka comes FIRST, in Sanskrit, then its English. Synthesis — anything the
system says in its own words — comes after, never before, and is visually
distinct from quoted scripture.

That ordering is a product decision, not a cosmetic one. A RAG answer that
leads with its own paraphrase invites the reader to trust the paraphrase; one
that leads with the verse invites them to check it. Since the whole premise of
this product is verifiability (BRD BO-1), the source text goes on top.

This module is the single place that ordering is defined, so the CLI, any future
web view, and the generation prompt all quote verses the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
VERSES = os.path.join(HERE, "data", "verses.json")

# Canonical display order. Sanskrit first — it is the text; everything else is
# somebody's rendering of it.
DEFAULT_LANGS = ("sa", "iast", "en", "hi")

WIDTH = 76

BOLD, DIM, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[0m"


def no_colour() -> None:
    global BOLD, DIM, CYAN, YELLOW, RESET
    BOLD = DIM = CYAN = YELLOW = RESET = ""


def load_verses() -> dict[str, dict]:
    if not os.path.exists(VERSES):
        raise SystemExit(f"missing {VERSES}\n  Build it first: make verses")
    with open(VERSES, "r", encoding="utf-8") as f:
        return {v["id"]: v for v in json.load(f)["verses"]}


def strip_leading_ref(text: str) -> str:
    """Drop a leading '2.47' / '||2-47||' style reference from a rendering.

    The source data prefixes many translations with their own verse reference.
    We print the reference ourselves in the header, so repeating it inside the
    quote is noise.
    """
    text = re.sub(r"^\s*\|*\s*\d+[.\-:]\d+\s*\|*\s*", "", text)
    return text.strip()


def wrap(text: str, indent: str = "  ") -> str:
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        out.append(textwrap.fill(para, width=WIDTH,
                                 initial_indent=indent,
                                 subsequent_indent=indent))
    return "\n".join(out)


def pick(renderings: list[dict], prefer: str | None = None) -> dict | None:
    """Choose one rendering: the preferred translator, else the first."""
    if not renderings:
        return None
    if prefer:
        for r in renderings:
            if r["source_key"] == prefer:
                return r
    return renderings[0]


def render(verse: dict, langs=DEFAULT_LANGS,
           prefer_en: str | None = "purohit",
           prefer_hi: str | None = None,
           show_licence: bool = False) -> str:
    """One verse, formatted. Sanskrit first, then renderings."""
    lines: list[str] = []
    vid = verse["id"]
    ch, n = vid.split(":")

    lines.append(f"{CYAN}{'─' * WIDTH}{RESET}")
    lines.append(f"{CYAN}{BOLD}  भगवद्गीता {ch}.{n}{RESET}"
                 f"{DIM}   (chapter {ch}, verse {n}){RESET}")
    lines.append(f"{CYAN}{'─' * WIDTH}{RESET}")

    # --- the shloka itself, first ---
    if "sa" in langs and verse["sanskrit"].get("devanagari"):
        lines.append("")
        lines.append(wrap(verse["sanskrit"]["devanagari"]))

    if "iast" in langs and verse["sanskrit"].get("iast"):
        lines.append("")
        lines.append(f"{DIM}{wrap(verse['sanskrit']['iast'])}{RESET}")

    # --- then the renderings ---
    if "en" in langs:
        r = pick(verse.get("english", []), prefer_en)
        if r:
            lines.append("")
            tag = f"  {DIM}English — {r['translator']}"
            if show_licence and not r["redistributable"]:
                tag += f"  {YELLOW}[not redistributable]{DIM}"
            lines.append(tag + RESET)
            lines.append(wrap(strip_leading_ref(r["text"])))

    if "hi" in langs:
        r = pick(verse.get("hindi", []), prefer_hi)
        if r:
            lines.append("")
            tag = f"  {DIM}हिन्दी — {r['translator']}"
            if show_licence and not r["redistributable"]:
                tag += f"  {YELLOW}[not redistributable]{DIM}"
            lines.append(tag + RESET)
            lines.append(wrap(strip_leading_ref(r["text"])))

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render Gita verses: shloka first, then translations.")
    ap.add_argument("ids", nargs="+", metavar="CHAPTER:VERSE",
                    help="e.g. 2:47  6:34  or a range like 2:47-2:48")
    ap.add_argument("--lang", default=",".join(DEFAULT_LANGS),
                    help=f"comma list from sa,iast,en,hi (default: all)")
    ap.add_argument("--en", default="purohit",
                    help="preferred English translator key (default: purohit, "
                         "the public-domain one)")
    ap.add_argument("--hi", default=None, help="preferred Hindi translator key")
    ap.add_argument("--plain", action="store_true", help="disable colour")
    ap.add_argument("--licence", action="store_true",
                    help="mark renderings that are not redistributable")
    args = ap.parse_args()

    if args.plain or not sys.stdout.isatty():
        no_colour()

    verses = load_verses()
    langs = tuple(s.strip() for s in args.lang.split(",") if s.strip())

    wanted: list[str] = []
    for spec in args.ids:
        m = re.fullmatch(r"(\d+):(\d+)-(?:(\d+):)?(\d+)", spec)
        if m:
            c1, v1, c2, v2 = m.group(1), int(m.group(2)), m.group(3) or m.group(1), int(m.group(4))
            if c1 != c2:
                raise SystemExit(f"ranges must stay within one chapter: {spec}")
            wanted.extend(f"{c1}:{n}" for n in range(v1, v2 + 1))
        else:
            wanted.append(spec)

    missing = [v for v in wanted if v not in verses]
    if missing:
        for vid in missing:
            ch = vid.split(":")[0]
            n = sum(1 for k in verses if k.startswith(f"{ch}:"))
            print(f"no such verse: {vid}"
                  + (f"  (chapter {ch} has {n} verses)" if n else ""),
                  file=sys.stderr)
        raise SystemExit(1)

    for vid in wanted:
        print(render(verses[vid], langs=langs, prefer_en=args.en,
                     prefer_hi=args.hi, show_licence=args.licence))


if __name__ == "__main__":
    main()
