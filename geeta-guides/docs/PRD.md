# Product Requirements Document — Gita RAG

| | |
|---|---|
| **Document** | PRD |
| **Product** | Gita RAG |
| **Owner** | OG Techie |
| **Status** | Draft v0.1 |
| **Last updated** | 2026-08-30 |
| **Related** | [BRD.md](BRD.md), [../README.md](../README.md) |

Assumptions A1–A5 in [BRD §0](BRD.md#0-assumptions-this-document-rests-on) apply
to this document too. This PRD specifies **how**; the BRD covers **why**.

---

## 1. Users and their jobs

| User | Job to be done | Success looks like |
|---|---|---|
| **Learner** | "What does the Gita actually say about X?" | Two or three verses, quoted, cited, with a short synthesis |
| **Practitioner** | "Find me the passage about Y so I can quote it" | Exact text, exact citation, no paraphrase |
| **Comparative reader** | "How do translations differ on this verse?" | Same verse, side by side, differences visible |
| **Skeptic** | "Is that really in there?" | A citation they can check, and a system that says no when the answer is no |

---

## 2. Product principles

1. **Retrieval is the product; generation is presentation.** If retrieval
   returns the wrong verses, no prompt engineering saves the answer. Effort goes
   to retrieval in that proportion.
1a. **The verse leads; the explanation follows.** An answer that opens with its
   own paraphrase asks the reader to trust the paraphrase. One that opens with
   the shloka asks them to check it. Since verifiability is the entire premise
   (BO-1), the source text goes on top — Sanskrit, then English. See FR-3.5.
2. **Refusal is a feature.** "The text does not address this" is a correct
   answer and must be reachable.
3. **Every claim traces to a chunk.** If a sentence in the answer can't be
   traced to retrieved text, it should not be in the answer.
4. **Measure before tuning.** No retrieval change ships without a before/after
   number on the eval set.
5. **Small corpus, so favour precision.** 124k characters is small enough that
   exhaustive approaches are affordable. Use that.

---

## 3. Decision record: the role of `gita_gpt.py`

> **This section exists because it's the question the workspace raises, and the
> answer is counterintuitive.**

**Decision: the character-level GPT built in this repo will not appear in the
production RAG pipeline, in any role.**

### Why not

| Property | `gita_gpt.py` | What a RAG generator needs |
|---|---|---|
| Parameters | 3.26 M | 10⁹ – 10¹² |
| Training data | 124 k characters | 10¹² – 10¹³ tokens |
| Vocabulary | 79 characters | 30k–200k subword tokens |
| Context window | 256 **characters** (~45 words) | 8k–200k **tokens** |
| Instruction following | None — no concept of a prompt/response boundary | Required |
| Output at target loss (~1.5 nats/char) | Pronounceable non-words in blank-verse shape | Grounded prose |

The context window alone settles it: at 256 characters, a single retrieved verse
plus a question would not fit, and there would be nothing left over to answer
with. There is no configuration of this model that makes it a generator.

### Alternatives considered and rejected

| Idea | Verdict |
|---|---|
| Use it as an embedding model (take hidden states as vectors) | Rejected. Character-level hidden states encode orthography, not semantics. It would retrieve on spelling similarity |
| Use it as a domain-fit / perplexity scorer to filter chunks | Rejected. Technically possible, but it scores *style*, not relevance, and every chunk in the corpus is in-domain by construction |
| Scale it up until it's usable | Rejected. That is a different project (pretraining), costs six figures, and the result would still be worse than an off-the-shelf open-weights model |

### What actually transfers

The exercise is not wasted; it is Phase 0 for a reason. Five things you build by
hand in `gita_gpt.py` map directly onto decisions in this PRD:

| Built in the exercise | Decides in the RAG |
|---|---|
| **Tokenization** — one integer per character, and watching it fail to capture meaning | Why the RAG uses subword embeddings, and why chunk sizes are measured in tokens, not characters (§5.2) |
| **The position embedding table has exactly `block_size` rows** | Why the context window is a hard architectural cap, not a soft limit — and therefore why retrieval must be selective rather than exhaustive (§5.4) |
| **Attention is O(T²)** — you measured 2.6 GB at `block_size=256, batch=64` | Why stuffing 100 chunks into a prompt costs quadratically, and why reranking to top-5 is cheaper than a bigger context (§5.3) |
| **The token embedding table** — a learned vector per unit of meaning, compared by dot product | The retriever's sentence embeddings are the same idea at a different granularity (§5.2) |
| **`softmax(QKᵀ/√d)·V`** | **Attention *is* retrieval.** Query dotted against keys → relevance weights → weighted sum of values. RAG is the same operation with an external corpus, a top-k instead of a softmax, and a document store instead of a value projection. Once you've written the four lines in `Head.forward`, the whole retrieval architecture below reads as one familiar shape |

That last row is the actual reason to do Phase 0 before Phase 1.

---

## 4. Functional requirements

### FR-1 Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | Ingest plain-text scripture, preserving chapter and verse structure | Must |
| FR-1.2 | Record per-document provenance: source, translator, year, licence | Must |
| FR-1.3 | Strip apparatus (Gutenberg boilerplate, footnote markers) into metadata rather than discarding — footnotes are content | Must |
| FR-1.4 | Re-ingestion is idempotent and versioned; re-indexing does not require rebuilding from source | Should |
| FR-1.5 | Support multiple translations of the same verse, linked by canonical `chapter:verse` ID | Should |

### FR-2 Retrieval

| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | Hybrid retrieval: dense (semantic) and lexical (BM25) in parallel, fused with Reciprocal Rank Fusion | Must |
| FR-2.2 | Cross-encoder reranking of the fused candidate pool down to the final top-k | Must |
| FR-2.3 | Query rewriting/expansion before retrieval, to bridge modern phrasing to archaic text (BRD R-1) | Must |
| FR-2.4 | Every retrieval returns chunk IDs, scores, and stage-by-stage ranks — inspectable, not a black box | Must |
| FR-2.5 | Metadata filtering (by chapter, by translation) | Should |

### FR-3 Generation

| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | Answers cite `chapter:verse` for every substantive claim | Must |
| FR-3.2 | The generator is instructed to answer **only** from retrieved context | Must |
| FR-3.3 | Explicit refusal path when retrieved context does not support an answer | Must |
| FR-3.4 | Post-generation citation verification: every cited verse ID must exist and must have been in the retrieved set | Must |
| FR-3.5 | **The shloka comes first.** Every answer leads with the verse — Devanagari Sanskrit, then IAST, then English, then Hindi — before any synthesis. Quoted scripture is verbatim and visually distinct from anything the system says in its own words | Must |
| FR-3.5a | The display order is defined in exactly one place (`show.py:render`) and shared by the CLI, any web view, and the generation prompt, so a verse is never quoted two different ways | Must |
| FR-3.6 | Where translations disagree materially, surface the disagreement rather than picking one | Should |

### FR-4 Evaluation

| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | A versioned eval set of ≥ 60 question → expected-verse pairs, committed to the repo | Must |
| FR-4.2 | Include ≥ 15 out-of-corpus questions that *must* trigger refusal | Must |
| FR-4.3 | A single command scores retrieval (recall@k, MRR) and generation (faithfulness, citation accuracy) | Must |
| FR-4.4 | Scores are recorded per run so regressions are visible over time | Should |

---

## 5. Architecture

```
question
   │
   ├──> query rewrite / expansion ──┐
   │                                │
   │        ┌───────────────────────┴───────────────────────┐
   │        │                                               │
   │   dense retrieval                              lexical retrieval
   │   (embeddings, top-50)                         (BM25, top-50)
   │        │                                               │
   │        └──────────────> RRF fusion <───────────────────┘
   │                             │
   │                        top ~100
   │                             │
   │                    cross-encoder rerank
   │                             │
   │                        top 5–10
   │                             │
   └────────────────> grounded generation
                                 │
                        citation verification
                                 │
                        answer + citations  ·  or refusal
```

### 5.1 Component choices

Current as of August 2026; revisit before Phase 1 rather than trusting this
table indefinitely.

| Component | Recommendation | Why |
|---|---|---|
| **Embeddings** | Qwen3-Embedding-0.6B locally, or the 4B if memory allows | Top of the MTEB open-source leaderboard through 2026, open weights, runs on MPS, and the 0.6B is small enough for fast iteration. Benchmark 2–3 candidates on *your* eval set before committing — MTEB rank is a shortlist, not an answer |
| **Lexical** | BM25 (rank-bm25, or the vector store's built-in) | Catches proper nouns — Arjuna, Krishna, Kurukshetra — that embeddings blur together |
| **Vector store** | Start with a flat in-memory index or SQLite + numpy | 124k characters is a few thousand chunks. Exact search is instant. A vector database is premature until the corpus is 100× larger |
| **Reranker** | BGE-reranker (open) or Cohere Rerank (hosted) | Cross-encoders reliably outperform bi-encoder scores; reranking a top-100 pool is the stable configuration |
| **Generator** | Hosted instruction-tuned model for v1 | Grounding adherence and refusal behaviour are materially better than small local models, and this is the requirement that matters most (FR-3.2, FR-3.3). Revisit locally once refusal behaviour is measurable |
| **Eval** | Ragas for faithfulness / context precision / answer relevancy, plus your own recall@k | Ragas defined the standard metrics and is reference-free, which suits a corpus with no ground-truth answers |

### 5.2 Chunking — small-to-big

The central tension (BRD R-3): a verse is the right unit to *cite* and the wrong
unit to *understand*, because verses reference their neighbours constantly.

**Resolution: embed small, return big.**

- Index one embedding **per verse** — precise retrieval target, precise citation
- On a hit, return the verse **plus a window of surrounding verses** as the context passed to the generator
- Keep the chunk metadata: `{chapter, verse, translation, licence, neighbours}`
- Additionally index a **modern-language paraphrase** of each verse as a second
  embedding pointing at the same verse ID. This directly attacks the semantic
  gap in BRD R-1: the reader's question matches the paraphrase, the citation
  points at the verse

Do not chunk by character count. The verse structure is free structure — use it.

### 5.3 Why rerank instead of retrieving more

From Phase 0: attention cost is quadratic in sequence length. Passing 50 chunks
to the generator costs roughly 100× the attention of passing 5, degrades
accuracy through the lost-in-the-middle effect, and costs proportionally more.
A cross-encoder over 100 candidates is cheap by comparison and picks better.
This is the same trade-off you'll have felt in `Head.forward`.

### 5.4 Context budget

The generator's context window is a hard cap — the same fact as
`position_embedding_table` having exactly `block_size` rows, at a different
scale. Budget it explicitly: system prompt + question + k chunks × chunk size
must fit with room for the answer. Track it; do not discover it in production.

---

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | p95 end-to-end latency < 4 s; retrieval alone < 500 ms |
| NFR-2 | Cost per query < $0.02 |
| NFR-3 | Runs on a single Apple Silicon machine; embedding and reranking local, generation may be remote |
| NFR-4 | Full pipeline reproducible from `data/` with one command |
| NFR-5 | Every stage logs its inputs, outputs and scores — debuggable without a rerun |
| NFR-6 | No user query text is persisted without explicit opt-in |
| NFR-7 | Licence and provenance metadata travel with every chunk to the answer |

---

## 7. Acceptance criteria

Ship v1 when all of these hold on the committed eval set:

- [ ] Recall@10 ≥ 0.90 on in-corpus questions
- [ ] Faithfulness (Ragas) ≥ 0.95
- [ ] Citation accuracy ≥ 0.95 — every cited verse exists and supports its claim
- [ ] Correct refusal on ≥ 0.90 of out-of-corpus questions
- [ ] p95 latency < 4 s
- [ ] `make eval` produces the full scorecard in one command
- [ ] Ten questions you actually care about return answers you'd quote

---

## 8. Milestones

| Phase | Deliverable | Acceptance |
|---|---|---|
| **0** | `gita_gpt.py` complete | `checks.py --milestone 4` passes |
| **1a** | Verse-aware ingestion + metadata | Every verse addressable by `chapter:verse` |
| **1b** | Eval set, ≥ 60 pairs + ≥ 15 refusal cases | Committed, versioned, reviewed |
| **1c** | Dense index + baseline recall@10 | A number exists, however bad |
| **2a** | Hybrid retrieval + RRF | Recall@10 improves over 1c baseline |
| **2b** | Reranking + query expansion | Recall@10 ≥ 0.90 |
| **3a** | Grounded generation + citations | Citation accuracy ≥ 0.95 |
| **3b** | Refusal path + faithfulness scoring | Faithfulness ≥ 0.95, refusal ≥ 0.90 |
| **4** | Interface + second translation | Daily use |

**Build the eval set at 1b, before any tuning.** Every project that defers it
tunes on vibes and cannot tell improvement from noise.

---

## 9. Out of scope for v1

Carried from [BRD §3](BRD.md#3-scope): multi-user auth, billing, Devanagari
input, conversational memory, doctrinal arbitration, mobile apps. Plus: agentic
multi-hop retrieval, and any fine-tuning of the generator.

---

## 10. Open questions

Carried from [BRD §9](BRD.md#9-open-questions), plus:

5. Verse-level or passage-level as the retrieval unit — decide at 1a, it is
   expensive to change at 2b
6. Who writes the modern-language paraphrases (§5.2)? Generated once by an LLM
   and human-reviewed is the cheap path, but it introduces a generated artifact
   into the retrieval index — is that acceptable given BO-1?
7. How is the eval set built without circularity? Questions written by reading
   the verses will match those verses too easily

---

*Phase 0 is the current work. See [../README.md](../README.md).*
