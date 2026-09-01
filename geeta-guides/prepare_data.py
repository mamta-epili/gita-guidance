"""
prepare_data.py — download and clean the training corpus.

Source: Project Gutenberg eBook #2388, "The Song Celestial" (Bhagavad-Gita),
translated by Sir Edwin Arnold. Public domain in the US.

What this does:
  1. Downloads the plain-text UTF-8 edition (cached in data/raw_2388.txt so
     repeated runs don't re-hit Gutenberg's servers).
  2. Strips the Project Gutenberg header and footer boilerplate, so the model
     never learns to produce licence text.
  3. Normalizes line endings to \\n and collapses runs of 3+ blank lines to 2.
  4. Writes data/gita.txt and prints corpus stats (chars, vocab size, vocab).

Run:  python prepare_data.py
"""

import os
import re
import sys

import net  # shared fetching: SSL, retries — see net.py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RAW_PATH = os.path.join(DATA_DIR, "raw_2388.txt")
OUT_PATH = os.path.join(DATA_DIR, "gita.txt")

# Gutenberg serves several mirrors of the same book; try in order.
URLS = [
    "https://www.gutenberg.org/cache/epub/2388/pg2388.txt",
    "https://www.gutenberg.org/files/2388/2388-0.txt",
    "https://www.gutenberg.org/ebooks/2388.txt.utf-8",
]

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)


def download() -> str:
    """Return the raw ebook text, downloading it once and caching to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_PATH):
        print(f"using cached raw text: {RAW_PATH}")
        with open(RAW_PATH, "r", encoding="utf-8") as f:
            return f.read()

    last_err = None
    for url in URLS:
        try:
            print(f"downloading {url} ...")
            raw = net.fetch_text(url, timeout=60)
            with open(RAW_PATH, "w", encoding="utf-8") as f:
                f.write(raw)
            print(f"saved raw text: {RAW_PATH} ({len(raw):,} chars)")
            return raw
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last_err = e
            print(f"  failed: {e}")
    print(f"\nAll mirrors failed. Last error: {last_err}", file=sys.stderr)
    print(
        "You can download it manually in a browser and save it to "
        f"{RAW_PATH}, then re-run this script.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def clean(raw: str) -> str:
    """Strip Gutenberg boilerplate and normalize whitespace."""
    # 1. Normalize line endings first, so offsets are consistent.
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Strip the UTF-8 BOM if the mirror included one.
    text = text.lstrip("﻿")

    # 2. Cut everything before the START marker and after the END marker.
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    else:
        print("WARNING: START marker not found — header may still be present.")

    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    else:
        print("WARNING: END marker not found — footer may still be present.")

    # 3. Gutenberg often leaves a transcriber's note / produced-by line right
    #    after the START marker. Drop leading blank-ish lines.
    text = text.lstrip("\n")

    # 4. Collapse 3+ consecutive newlines (i.e. 2+ blank lines) down to 2
    #    newlines (1 blank line)... actually keep at most 2 blank lines is the
    #    spec: 3+ blank lines -> 2 blank lines, i.e. 4+ \n -> 3 \n.
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # 5. Trim trailing whitespace on each line, and normalize the file ending.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip("\n") + "\n"
    return text


def stats(text: str) -> None:
    """Print the corpus statistics you'll need for the char-level model."""
    vocab = sorted(set(text))
    print()
    print("=" * 60)
    print("CORPUS STATS")
    print("=" * 60)
    print(f"total characters : {len(text):,}")
    print(f"total lines      : {text.count(chr(10)):,}")
    print(f"vocab size       : {len(vocab)}")
    print()
    # repr() so newline/space are visible rather than silently printed
    printable = "".join(vocab)
    print("vocab (as one string):")
    print(repr(printable))
    print()
    print("vocab (indexed):")
    for i, ch in enumerate(vocab):
        name = repr(ch)
        print(f"  {i:3d}: {name}")
    print()
    print("first 400 characters:")
    print("-" * 60)
    print(text[:400])
    print("-" * 60)


def main() -> None:
    raw = download()
    text = clean(raw)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nwrote {OUT_PATH} ({len(text):,} chars)")
    stats(text)


if __name__ == "__main__":
    main()
