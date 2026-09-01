# Business Requirements Document — Gita RAG

| | |
|---|---|
| **Document** | BRD |
| **Product** | Gita RAG — a retrieval-augmented question-answering system over scripture |
| **Owner** | OG Techie |
| **Status** | Draft v0.1 |
| **Last updated** | 2026-08-30 |
| **Related** | [PRD.md](PRD.md), [../README.md](../README.md) |

---

## 0. Assumptions this document rests on

> **Flagged for confirmation.** These were inferred from the workspace, not
> stated. Each is cheap to change now and expensive to change after Phase 2.

| # | Assumption | Status | If wrong, change |
|---|---|---|---|
| A1 | The corpus is the Bhagavad Gita: **700 verses in Sanskrit (Devanagari + IAST), Hindi, and English**, sourced from the [vedicscriptures dataset](https://github.com/vedicscriptures/bhagavad-gita) | **Resolved** — built and verified, `data/verses.json` | §2, §3, and the entire PRD retrieval design |
| A2 | Primary users are learners and practitioners asking interpretive questions, not developers querying an API | Open | §2, §4 |
| A3 | Answers must cite chapter and verse; an uncited answer is a defect, not a degraded result | **Satisfied** — canonical `chapter:verse` IDs verified against per-chapter counts summing to 700 | §4, §6 |
| A4 | This is a personal / small-team build, not a funded product with a revenue target | Open | §5 success metrics |
| A5 | All three languages are required — *"geeta means shlokas and guidance from that"* — not English-first | **Confirmed by owner** | §3 scope |
| A6 | **The Hindi translations are not redistributable.** Gita Press, Chinmaya Mission and others hold them. Sanskrit (ancient) and Purohit Swami's English (d. 1941) are public domain | **Open risk — see R-7** | §6 legal constraint; blocks any public deployment |

**Superseded, recorded so the reasoning isn't lost:** Arnold's *Song Celestial*
(the corpus the char-GPT trained on) has **no verse numbers** — 18 chapters of
continuous blank verse. Telang's 1882 SBE translation has none either. Both were
checked and rejected as citation spines before the current source was found.

---

## 1. Executive summary

Scripture is not hard to obtain — the full text of the Gita is free, and this
repo already contains one translation. It is hard to *query*. A reader with a
real question ("what does the text actually say about acting without attachment
to outcome?") has three bad options: read all eighteen chapters, trust a
search engine's summary with no provenance, or ask a general-purpose chatbot
that will answer fluently and cite nothing verifiable.

Gita RAG closes that gap: natural-language questions in, answers out that are
**grounded in retrieved passages and cite chapter and verse**, so the reader can
check the claim against the source. The system's job is not to be wise. It is to
find the right verses and refuse to invent.

---

## 2. Problem statement

| Stakeholder | Current pain | What they need |
|---|---|---|
| **Learner / reader** | Keyword search fails on conceptual questions; the vocabulary of the question rarely matches the vocabulary of a 19th-century verse translation | Semantic retrieval that bridges "detachment" to "renounce the fruit of action" |
| **Practitioner / teacher** | Needs to quote accurately and attribute correctly; a paraphrase is not usable | Verbatim passages with exact citations |
| **Comparative reader** | Translations differ substantially in meaning, and no single one is authoritative | The ability to see the same verse across translations |
| **You (the builder)** | Want a real, non-toy system to learn production LLM engineering on | A project with genuine retrieval difficulty and a hard grounding constraint |

**The core difficulty, stated plainly:** the semantic gap between how people ask
and how scripture is written is unusually wide. Verse translations are archaic,
metaphorical, and densely metaphor-laden. This is a *harder* retrieval problem
than the FAQ-and-support-docs case most RAG tutorials use, which is what makes
it worth building.

---

## 3. Scope

### In scope

- Ingestion of public-domain scripture, starting with the Arnold translation already in `data/gita.txt`
- Chunking that respects verse boundaries rather than character counts
- Hybrid retrieval (semantic + lexical) with reranking
- Grounded answer generation with mandatory citations
- An evaluation set and measured retrieval/faithfulness scores
- A single-user interface (CLI or local web) sufficient to use it daily

### Out of scope for v1

- Multi-user accounts, auth, billing
- Sanskrit-language querying (transliteration in source text is retained and searchable, but Devanagari input is not supported)
- Conversational memory across sessions
- Doctrinal arbitration between commentarial traditions — the system presents what sources say, it does not adjudicate
- Mobile applications

### Explicitly rejected

- **Fine-tuning a model on scripture to answer from its weights.** This is the
  approach that sounds appealing and fails the hardest requirement: a
  fine-tuned model cannot cite, because it has no retrieved passage to point
  at. Retrieval is not an optimization here, it is the product.

---

## 4. Business objectives

| # | Objective | Rationale |
|---|---|---|
| BO-1 | Every substantive answer carries a verifiable chapter:verse citation | Without this the product is a chatbot with extra steps |
| BO-2 | The system says "the text does not address this" rather than improvising | Confabulation on scripture is worse than no answer — it misattributes belief to a source |
| BO-3 | Retrieval quality is measured, not asserted | You cannot improve what you don't score; see PRD §7 |
| BO-4 | Answers are traceable end to end — question → chunks retrieved → chunks actually used | Required to debug, and required to trust |
| BO-5 | The build teaches production LLM engineering as a side effect | An explicit secondary objective, and the reason the workspace exists |

---

## 5. Success criteria

Framed as thresholds, not aspirations. Measured against the evaluation set
defined in the PRD.

| Metric | Target | Floor (ship-blocking below this) |
|---|---|---|
| Retrieval recall@10 on the eval set | ≥ 0.90 | 0.80 |
| Faithfulness / groundedness (claims supported by retrieved context) | ≥ 0.95 | 0.90 |
| Citation accuracy (cited verse actually contains the claim) | ≥ 0.95 | 0.90 |
| Correct abstention on out-of-corpus questions | ≥ 0.90 | 0.75 |
| End-to-end latency, p95 | < 4 s | < 8 s |
| Cost per query | < $0.02 | < $0.05 |

**Non-metric success:** you can ask it a question you actually have and get an
answer you're willing to quote.

---

## 6. Constraints

| Type | Constraint |
|---|---|
| **Legal** | Public-domain sources only (Arnold 1885 is clear). Modern translations and commentaries are under copyright — do not ingest without a licence, and the ingestion pipeline must record provenance and licence per document |
| **Technical** | Development on Apple Silicon with MPS. Local embedding and reranking must fit in unified memory; generation may be a hosted API |
| **Ethical** | The system must not present itself as a spiritual authority. It reports what texts say and attributes clearly. Interpretive disagreement between sources is surfaced, not resolved |
| **Resource** | One part-time developer. Every requirement must survive the question "can one person maintain this?" |
| **Data** | ~124k characters for a single translation — small. This is a blessing for cost and a curse for evaluation-set size |

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-1 | Semantic gap defeats embedding retrieval — modern question vocabulary never matches archaic verse vocabulary | High | High | Hybrid retrieval; query rewriting/expansion; index a modern-language summary alongside each verse and retrieve on both |
| R-2 | Generator confabulates plausible scripture | Medium | Critical | Strict grounding prompt, faithfulness scoring in CI, refusal path, citation verification as a post-generation check |
| R-3 | Verse-level chunks are too small to carry context; passage-level chunks blur the citation | High | Medium | Small-to-big retrieval: embed the verse, return the surrounding passage. See PRD §5.2 |
| R-4 | Evaluation set is too small to detect regressions | Medium | High | Build the eval set *before* tuning; 60+ question/verse pairs minimum; treat it as a first-class artifact |
| R-5 | Scope creep into commentaries, multiple traditions, and multilingual before v1 works on one text | High | Medium | §3 out-of-scope list is binding until success criteria are met on one translation |
| R-6 | The char-level GPT in this repo turns out not to be usable in the product | **Certain** | Low | Already accounted for — see PRD §3, "Decision record: the role of `gita_gpt.py`". The exercise's value is knowledge transfer, not code reuse |
| R-7 | **No redistributable Hindi translation exists in machine-readable form.** Verified: 0 of 700 verses have a Hindi rendering that could legally be published. All three available Hindi translations are in copyright | **Confirmed** | High — blocks public launch, not local use | Licence-per-field is already in the data model, so `--public-only` produces a clean build today (Sanskrit + Purohit English, 700/700). For Hindi: **Tilak's *Gita Rahasya*** (d. 1920, unambiguously PD) is on [hi.wikisource](https://hi.wikisource.org) as un-transcluded scan pages and would need OCR/proofreading. Second option: Jayadayal Goyandka (d. 1965) entered the Indian public domain in January 2026, though Gita Press may contest and the US term differs |

---

## 8. Phases

| Phase | Outcome | Gate to next phase |
|---|---|---|
| **0 — Foundations** *(in progress)* | Char-level GPT built by hand; attention, embeddings, context windows understood from the inside | `checks.py --milestone 4` passes |
| **1 — Ingest & index** | Verse-aware chunking, embeddings, hybrid index over one translation | Eval set of 60+ Q→verse pairs exists and recall@10 is measured |
| **2 — Retrieve & rerank** | Hybrid retrieval + cross-encoder reranking tuned against the eval set | recall@10 ≥ 0.90 |
| **3 — Ground & generate** | Cited answers, refusal path, faithfulness scoring | Faithfulness ≥ 0.95, citation accuracy ≥ 0.95 |
| **4 — Use it** | Daily-usable interface; second translation added | You use it without wanting to fix it first |

---

## 9. Open questions

1. **Which second translation?** Comparative reading is the strongest
   differentiator, but licence status varies. Which public-domain translations
   beyond Arnold are worth the ingestion work?
2. **Verse-level or passage-level as the unit of citation?** Affects chunking,
   eval-set construction, and how answers read.
3. **Is the Gita alone enough, or does the corpus need the Mahabharata framing
   to answer contextual questions?**
4. **Hosted or local generation?** Local keeps cost at zero and data private;
   hosted gives materially better grounding adherence. This is the single
   biggest cost/quality lever in the project.

---

*Next: [PRD.md](PRD.md) specifies how this gets built.*
