"""
prepare_verses.py — build the verse-addressable corpus for the RAG (Phase 1a).

Output: data/verses.json — 700 verses, each keyed by canonical `chapter:verse`,
carrying Sanskrit (Devanagari + IAST), Hindi, and English, with LICENCE
METADATA PER FIELD.

Why licence-per-field matters
-----------------------------
The Sanskrit shlokas are ancient and public domain. Some English translations
are public domain (Purohit Swami d. 1941). The Hindi translations available in
machine-readable form are NOT — they belong to Gita Press, Chinmaya Mission and
others. That is fine for reading on your own machine and fatal if you publish.

So every text field carries `licence` and `redistributable`. Nothing in this
pipeline decides what you may publish; it records what you'd need to know in
order to decide, at the point where the data enters the system rather than
months later when you've forgotten where it came from.

Serving layer rule: filter on `redistributable` before any text leaves the
machine. `python prepare_verses.py --public-only` writes a second file
containing only redistributable fields, so you can diff what you'd lose.

Source
------
https://github.com/vedicscriptures/bhagavad-gita  (static JSON API)
Verse-aligned by construction, so there is no cross-source alignment risk.

Run:  python prepare_verses.py
      python prepare_verses.py --public-only
      python prepare_verses.py --verify        # counts only, no refetch
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import net  # shared fetching: SSL, retries, atomic cache — see net.py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache", "slok")
OUT_PATH = os.path.join(DATA_DIR, "verses.json")
PUBLIC_PATH = os.path.join(DATA_DIR, "verses.public.json")

API = "https://vedicscriptures.github.io/slok/{chapter}/{verse}"
USER_AGENT = "geeta-guides/0.1 (personal research project; contact: local)"

# ---------------------------------------------------------------------------
# CANONICAL VERSE COUNTS
# ---------------------------------------------------------------------------
# The Gita is conventionally "700 verses" and these counts sum to exactly 700.
#
# Chapter 13 is the known variant: some recensions give 34 verses, others 35
# (counting an opening question from Arjuna as verse 1). Editions therefore
# disagree on the total — 700 vs 701 — and every verse after that point in
# chapter 13 shifts by one. If you ever add a second source, THIS is where
# silent misalignment would come from, which is why the script verifies counts
# rather than trusting them.
CANONICAL_COUNTS = {
    1: 47, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47,
    7: 30, 8: 28, 9: 34, 10: 42, 11: 55, 12: 20,
    13: 34, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78,
}
CANONICAL_TOTAL = sum(CANONICAL_COUNTS.values())  # 700

# ---------------------------------------------------------------------------
# LICENCE TABLE
# ---------------------------------------------------------------------------
# Keyed by the dataset's author codes. Death years drive the assessment:
# most jurisdictions run life + 70 (US/EU) or life + 60 (India).
#
# THIS TABLE IS RESEARCH, NOT LEGAL ADVICE. It reflects what could be
# established from public records; verify anything you intend to publish.
LICENCES = {
    # --- Sanskrit source text -------------------------------------------
    "_slok": dict(translator="(anonymous, ancient)", died=None,
                  licence="public-domain", redistributable=True,
                  note="Sanskrit shloka text; composed c. 2nd century BCE"),
    "_translit": dict(translator="(mechanical transliteration)", died=None,
                      licence="public-domain", redistributable=True,
                      note="IAST romanisation of a public-domain text"),

    # --- English --------------------------------------------------------
    "purohit": dict(translator="Shri Purohit Swami", died=1941,
                    licence="public-domain", redistributable=True,
                    note="d. 1941; PD in US/EU (life+70) and India (life+60)"),
    "siva": dict(translator="Swami Sivananda", died=1963,
                 licence="uncertain", redistributable=False,
                 note="d. 1963; PD in India from 2023 (life+60); "
                      "US/EU term (life+70) runs to 2033"),
    "gambir": dict(translator="Swami Gambirananda", died=1988,
                   licence="copyright", redistributable=False,
                   note="Advaita Ashrama / Ramakrishna Mission"),
    "adi": dict(translator="Swami Adidevananda", died=1983,
                licence="copyright", redistributable=False,
                note="Ramakrishna Mission"),
    "prabhu": dict(translator="A.C. Bhaktivedanta Swami Prabhupada", died=1977,
                   licence="copyright", redistributable=False,
                   note="Bhaktivedanta Book Trust actively enforces this one"),
    "san": dict(translator="Dr. S. Sankaranarayan", died=None,
                licence="copyright", redistributable=False,
                note="modern academic translation"),
    "adidev": dict(translator="Swami Adidevananda", died=1983,
                   licence="copyright", redistributable=False, note=""),

    # --- Hindi ----------------------------------------------------------
    "rams": dict(translator="Swami Ramsukhdas", died=2005,
                 licence="copyright", redistributable=False,
                 note="Gita Press, Gorakhpur"),
    "tej": dict(translator="Swami Tejomayananda", died=None,
                licence="copyright", redistributable=False,
                note="living author; Chinmaya Mission"),
    "chinmay": dict(translator="Swami Chinmayananda", died=1993,
                    licence="copyright", redistributable=False,
                    note="Chinmaya Mission"),
    "sankar": dict(translator="Shankaracharya (modern rendering)", died=None,
                   licence="uncertain", redistributable=False,
                   note="the 8th-c. commentary is PD; THIS Hindi/English "
                        "rendering of it is modern and unattributed"),
}

# Which dataset keys carry which language. Field codes in the source data:
#   et = English translation   ht = Hindi translation
#   ec = English commentary    hc = Hindi commentary   sc = Sanskrit commentary
ENGLISH_KEYS = ["purohit", "siva", "gambir", "adi", "prabhu", "san"]
HINDI_KEYS = ["rams", "tej", "sankar", "chinmay"]


def fetch_verse(chapter: int, verse: int, polite_delay: float = 0.15,
                max_retries: int = 6) -> dict:
    """Fetch one verse, caching to data/cache/slok/ so reruns are offline.

    All the hard-won robustness — certificate handling, backoff on throttling,
    atomic cache writes, corrupt-cache recovery — lives in net.py. This function
    only adds the domain meaning: a 404 here tells you the source's recension
    disagrees with CANONICAL_COUNTS, which is a finding, not a transient error.
    """
    cache_path = os.path.join(CACHE_DIR, f"{chapter}-{verse}.json")
    url = API.format(chapter=chapter, verse=verse)
    try:
        return net.fetch_json(url, cache_path=cache_path,
                              user_agent=USER_AGENT,
                              max_retries=max_retries,
                              polite_delay=polite_delay)
    except net.NotFound:
        raise SystemExit(
            f"HTTP 404 on {url}\n"
            f"  Chapter {chapter} verse {verse} does not exist in this source's "
            f"recension.\n"
            f"  That is a real finding — adjust CANONICAL_COUNTS "
            f"(chapter 13 is the usual culprit)."
        )
    except net.FetchError as e:
        raise SystemExit(f"{e}\n\n  Cached progress is kept — rerun to resume.")


def field(raw: dict, key: str, subkey: str) -> str | None:
    """Pull raw[key][subkey] if present and non-empty."""
    block = raw.get(key)
    if not isinstance(block, dict):
        return None
    text = block.get(subkey)
    if not text or not str(text).strip():
        return None
    return str(text).strip()


def licence_for(key: str) -> dict:
    """Licence record for a dataset author key, with a safe default."""
    lic = LICENCES.get(key)
    if lic is None:
        return dict(translator=key, died=None, licence="unknown",
                    redistributable=False,
                    note="not in the licence table — treat as unclear")
    return lic


def build_verse(chapter: int, verse: int, raw: dict) -> dict:
    """Shape one raw API response into our record."""
    vid = f"{chapter}:{verse}"

    sanskrit = {
        "devanagari": raw.get("slok", "").strip() or None,
        "iast": raw.get("transliteration", "").strip() or None,
        **{k: v for k, v in licence_for("_slok").items() if k != "translator"},
    }

    def renderings(keys, subkey, lang):
        out = []
        for k in keys:
            text = field(raw, k, subkey)
            if text is None:
                continue
            lic = licence_for(k)
            out.append({
                "lang": lang,
                "text": text,
                "translator": lic["translator"],
                "translator_died": lic["died"],
                "licence": lic["licence"],
                "redistributable": lic["redistributable"],
                "licence_note": lic["note"],
                "source_key": k,
            })
        return out

    return {
        "id": vid,
        "chapter": chapter,
        "verse": verse,
        "sanskrit": sanskrit,
        "english": renderings(ENGLISH_KEYS, "et", "en"),
        "hindi": renderings(HINDI_KEYS, "ht", "hi"),
        # Commentary is kept separate from translation: a RAG answer may quote a
        # translation, but a commentary is a third party's interpretation and
        # must never be presented as what the text says.
        "commentary": {
            "hindi": renderings(HINDI_KEYS, "hc", "hi"),
            "english": renderings(ENGLISH_KEYS, "ec", "en"),
        },
    }


def verify(verses: list[dict]) -> int:
    """Check per-chapter counts against CANONICAL_COUNTS. Returns problem count."""
    print("\n" + "=" * 64)
    print("VERSE COUNT VERIFICATION")
    print("=" * 64)
    got: dict[int, int] = {}
    for v in verses:
        got[v["chapter"]] = got.get(v["chapter"], 0) + 1

    problems = 0
    for ch in sorted(CANONICAL_COUNTS):
        want, have = CANONICAL_COUNTS[ch], got.get(ch, 0)
        flag = "ok" if want == have else "MISMATCH"
        if want != have:
            problems += 1
        print(f"  chapter {ch:2d}: expected {want:3d}  got {have:3d}   {flag}")

    total = len(verses)
    print(f"\n  total: expected {CANONICAL_TOTAL}  got {total}")
    if total != CANONICAL_TOTAL:
        problems += 1
        print("  NOTE: 701 usually means chapter 13 was counted with 35 verses.")

    # Contiguity: no gaps, no duplicates.
    ids = {v["id"] for v in verses}
    for ch, want in CANONICAL_COUNTS.items():
        missing = [f"{ch}:{n}" for n in range(1, want + 1) if f"{ch}:{n}" not in ids]
        if missing:
            problems += 1
            print(f"  chapter {ch} missing: {missing[:8]}")
    if len(ids) != len(verses):
        problems += 1
        print(f"  DUPLICATE ids: {len(verses) - len(ids)}")
    return problems


def coverage(verses: list[dict]) -> None:
    """Report how many verses actually have text in each language."""
    print("\n" + "=" * 64)
    print("LANGUAGE COVERAGE")
    print("=" * 64)
    n = len(verses)
    dev = sum(1 for v in verses if v["sanskrit"]["devanagari"])
    iast = sum(1 for v in verses if v["sanskrit"]["iast"])
    print(f"  sanskrit devanagari : {dev:3d}/{n}")
    print(f"  sanskrit IAST       : {iast:3d}/{n}")

    for lang in ("english", "hindi"):
        print(f"\n  {lang}:")
        by_key: dict[str, int] = {}
        for v in verses:
            for r in v[lang]:
                by_key[r["source_key"]] = by_key.get(r["source_key"], 0) + 1
        for k, c in sorted(by_key.items(), key=lambda kv: -kv[1]):
            lic = licence_for(k)
            mark = "PUBLIC " if lic["redistributable"] else "RESTRICT"
            print(f"    {mark} {lic['translator'][:38]:38s} {c:3d}/{n}")

    # The headline number: what survives a --public-only build.
    pub_en = sum(1 for v in verses if any(r["redistributable"] for r in v["english"]))
    pub_hi = sum(1 for v in verses if any(r["redistributable"] for r in v["hindi"]))
    print("\n" + "-" * 64)
    print(f"  verses with a REDISTRIBUTABLE english : {pub_en}/{n}")
    print(f"  verses with a REDISTRIBUTABLE hindi   : {pub_hi}/{n}")
    if pub_hi == 0:
        print("\n  No redistributable Hindi exists in this dataset. For a public")
        print("  build you need a PD Hindi source — Tilak's 'Gita Rahasya'")
        print("  (d. 1920) is on hi.wikisource as scan pages and would need")
        print("  OCR/proofreading. See docs/PRD.md.")


def strip_restricted(verses: list[dict]) -> list[dict]:
    """Return a copy containing only redistributable text fields."""
    out = []
    for v in verses:
        c = json.loads(json.dumps(v))  # deep copy
        c["english"] = [r for r in c["english"] if r["redistributable"]]
        c["hindi"] = [r for r in c["hindi"] if r["redistributable"]]
        c["commentary"] = {
            k: [r for r in rs if r["redistributable"]]
            for k, rs in c["commentary"].items()
        }
        out.append(c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/verses.json")
    parser.add_argument("--public-only", action="store_true",
                        help="also write data/verses.public.json (redistributable only)")
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing verses.json without refetching")
    args = parser.parse_args()

    if args.verify:
        if not os.path.exists(OUT_PATH):
            raise SystemExit(f"{OUT_PATH} not found — run without --verify first.")
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            doc = json.load(f)
        problems = verify(doc["verses"])
        coverage(doc["verses"])
        raise SystemExit(1 if problems else 0)

    os.makedirs(DATA_DIR, exist_ok=True)
    verses = []
    for ch in sorted(CANONICAL_COUNTS):
        want = CANONICAL_COUNTS[ch]
        print(f"chapter {ch:2d}: fetching {want} verses ...", end="", flush=True)
        for n in range(1, want + 1):
            verses.append(build_verse(ch, n, fetch_verse(ch, n)))
        print(" done")

    doc = {
        "meta": {
            "description": "Bhagavad Gita, verse-addressable, Sanskrit/Hindi/English",
            "verse_id_format": "chapter:verse",
            "total_verses": len(verses),
            "canonical_total": CANONICAL_TOTAL,
            "source_dataset": "https://github.com/vedicscriptures/bhagavad-gita",
            "source_api": API,
            "licence_warning": (
                "Text fields carry per-field `licence` and `redistributable`. "
                "The Hindi translations here are NOT redistributable. Filter on "
                "`redistributable` before serving any text off this machine."
            ),
            "recension_note": (
                "Chapter 13 has 34 verses in this build. Editions using 35 will "
                "be offset by one from verse 13:1 onward."
            ),
        },
        "verses": verses,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.2f} MB)")

    problems = verify(verses)
    coverage(verses)

    if args.public_only:
        pub = strip_restricted(verses)
        pubdoc = {**doc, "verses": pub}
        pubdoc["meta"]["licence_warning"] = "Redistributable fields only."
        with open(PUBLIC_PATH, "w", encoding="utf-8") as f:
            json.dump(pubdoc, f, ensure_ascii=False, indent=1)
        print(f"\nwrote {PUBLIC_PATH} ({os.path.getsize(PUBLIC_PATH) / 1e6:.2f} MB)")

    # Show one verse so you can eyeball the shape.
    sample = next(v for v in verses if v["id"] == "2:47")
    print("\n" + "=" * 64)
    print("SAMPLE — 2:47")
    print("=" * 64)
    print(f"  sanskrit : {sample['sanskrit']['devanagari']}")
    print(f"  iast     : {sample['sanskrit']['iast']}")
    for r in sample["hindi"][:1]:
        print(f"  hindi    : {r['text'][:70]}   [{r['translator']}]")
    for r in sample["english"][:1]:
        print(f"  english  : {r['text'][:70]}   [{r['translator']}]")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
