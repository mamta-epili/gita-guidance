# Gita guidance

Ask a question in your own words — *"I am stressed, guide me"* — and get the
shlokas that speak to it, in Sanskrit, Hindi and English, quoted verbatim with
chapter and verse.

Two halves, built in that order:

| | |
|---|---|
| **[`geeta-guides/`](geeta-guides/)** | A character-level GPT written from scratch in PyTorch, the 700-verse corpus, and the retrieval evaluation harness. |
| **[`geeta-guides-rag/`](geeta-guides-rag/)** | The app — FastAPI backend, Angular frontend, and a `/lab` page that shows the transformer working. |

---

## Why there is a transformer in a retrieval project

The transformer came first, as a way to understand attention from the inside
rather than from a diagram. It works: 3.26M parameters, trained from scratch on
124,135 characters of the Bhagavad Gita, validation loss **1.6119**.

It is also **not used by the app**, and that was the finding worth having.
Its context window is 256 *characters* — one verse plus a short question fills it
completely, leaving no room to answer in. Fed a real retrieved verse and a real
question, it replies:

> *sto hovole its still pervana. Do exemle, lord, unglike these foeth…*

So the app is **retrieval-only**. The shloka, its translations and its citation
all come verbatim from the corpus. Nothing on that path writes text, which means
nothing on it can invent scripture — the worst failure a product like this could
have.

What did transfer is the understanding. `softmax(QKᵀ/√d)·V` *is* retrieval:
query against keys, weights, weighted sum of values. The RAG is that same
operation with an external corpus and a top-k instead of a softmax.

---

## What it does

- **700 verses**, Sanskrit (Devanagari + IAST), Hindi and English, addressed by
  canonical `chapter:verse`, with licence metadata on every field.
- **Dense retrieval** with BAAI/bge-m3, one vector per language per verse,
  max-pooled — so a Hindi question matches the Hindi rendering without having to
  say which language it is in.
- **Speaker-aware ranking.** The Gita is a dialogue, and embedding similarity
  matches a question to *text that resembles a question* — which is Arjuna, not
  Krishna. Every verse is labelled by speaker, so Krishna's answer leads and
  Arjuna's lament is shown separately.
- **`/lab`** — the character model, live: next-character probabilities, attention
  per head, and the causal mask drawn as a picture.

### Measured

| | BM25 baseline | bge-m3 |
|---|---|---|
| hit@10, all questions | 55.6% | **88.9%** |
| MRR, lookup questions | 0.406 | **0.730** |
| *"i am stressed guide me?"* | 0% | finds it |

The last row is the whole argument for embeddings: the word "stressed" appears
nowhere in 700 verses, so lexical search cannot answer the question the product
exists to answer.

---

## Running it

```bash
cd geeta-guides
./setup.sh          # venv, torch, sentence-transformers
make verses         # build the corpus (700 API calls, cached)
make embed          # embed it — downloads bge-m3, ~2.2 GB, once
```

```bash
cd ../geeta-guides-rag
./setup.sh
make backend        # terminal 1 → :8000
make install
make frontend       # terminal 2 → :4200
```

`http://localhost:4200` is the app, `/lab` is the transformer.
`make help` in either folder lists everything.

---

## Status: working prototype, not yet an MVP

Honest about the gap:

- **No crisis guardrail.** A wellbeing-adjacent app needs to recognise distress
  and return helplines instead of verses. Not built yet, and it is the first
  thing to build.
- **No refusal path.** Ask about cryptocurrency and it will confidently return
  five verses.
- **Evaluation set is 9 questions** against a target of 60, plus 15 refusal
  cases. Every number above is indicative, not yet trustworthy.
- **Not deployed.** Localhost, two terminals, a 2.2 GB model download.

---

## Sources and licences

| Content | Source | Licence |
|---|---|---|
| Sanskrit shlokas | [vedicscriptures](https://github.com/vedicscriptures/bhagavad-gita) | Ancient text — public domain |
| English translation | Shri Purohit Swami (d. 1941) | Public domain |
| Char-GPT training corpus | Edwin Arnold, *The Song Celestial*, 1885 — [Gutenberg #2388](https://www.gutenberg.org/ebooks/2388) | Public domain |
| Hindi translations | Gita Press, Chinmaya Mission | **Copyrighted — not in this repo** |

`data/verses.json` holds all three languages and is deliberately gitignored,
because 0 of 700 Hindi renderings are redistributable. What is committed is
`data/verses.public.json` — the same 700 verses with only the public-domain
fields, which is enough for the app to run. Build the full local corpus with
`make verses`.

The character model was built following Andrej Karpathy's
[*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY).

**Licence:** the code is [MIT](LICENSE). The scripture is not mine to license —
the Sanskrit and the English translations here are public domain, and the Hindi
translations are copyrighted and therefore not in this repository. The
[LICENSE](LICENSE) file spells out which is which.
