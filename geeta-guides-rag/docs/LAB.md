# Demoing the Inspector

What `http://localhost:4200/lab` is, what to check on it, and what to say while
someone is watching over your shoulder.

---

## The one-sentence version

It runs the character-level transformer you wrote from scratch, live on your own
trained weights, and shows what the model is actually doing — the probability it
assigns to every next character, and what each attention head is looking at.

### The distinction that matters

This is **not** the model answering questions on the main page.

- The **guidance app** (`/`) is retrieval. It finds real verses and quotes them
  verbatim. Nothing on that path writes text, so nothing on it can invent
  scripture.
- The **lab** (`/lab`) is the foundations. A 3.26M-parameter model with a
  **256-character** context window that produces convincing gibberish.

Both are true at once, and saying so plainly is more impressive than blurring
it. You built the thing on this page by hand; you were then honest enough to
keep it out of the product.

---

## Before you open it

| Needs | Command | Where |
|---|---|---|
| Backend running | `make backend` | `geeta-guides-rag` |
| Frontend running | `make frontend` | `geeta-guides-rag` |
| A trained checkpoint | `make train` | `geeta-guides` |

No checkpoint, no page — it reads `../geeta-guides/checkpoints/gita_gpt.pt`
directly. If node isn't running, `http://127.0.0.1:8000` serves the same five
sections as a single HTML file with no build step.

---

## The five sections

Each one shows something, has something you can verify on the spot, and has a
line worth saying out loud.

### 1 · One forward pass

Type anything; every keystroke runs the whole model and shows its prediction for
the *next character*. This distribution is the model's entire output —
generation is only sampling from it, appending, and running again.

**Check**

- Type `Arjuna said: my mind is` — top prediction is a space at about **94.5%**,
  entropy **0.52** of **6.304** bits.
- Delete the last letter so it reads `…my mind i` — entropy jumps and `s` takes
  over. Mid-word, fewer things are plausible.
- Forward pass time sits around **12–45 ms**.

**Say**

> "6.30 bits is log₂ of 79 — total uncertainty across the vocabulary. At 0.52
> the model is almost sure. That number is the model's confidence, made
> visible."

### 2 · What the attention heads are looking at

Shading over the text shows how much the final position attended to each earlier
character when making this prediction. Four layers, four heads each — sixteen
views of the same sentence.

**Check**

- Flip through the layer and head dropdowns — the patterns visibly differ. Some
  heads spread attention; others lock onto the previous character or the last
  space.
- Nothing is ever shaded to the *right* of the cursor.
- Tick "average all heads" to see the layer as a whole.

**Say**

> "Nobody assigned those roles. They fell out of gradient descent on 124,135
> characters. That's what multi-head buys you — several kinds of relationship
> tracked at once instead of averaged into one."

### 3 · Generate

Streams 240 characters, one per forward pass, at the speed they're produced.

**Check**

- Output has the *shape* of Arnold's blank verse — capitalised line starts,
  plausible line lengths, pronounceable non-words.
- It means nothing. That is the correct result, not a bug.
- Temperature slider: low makes it repetitive, high makes it noise.

**Say**

> "What you're watching is 240 complete forward passes through a four-layer
> network. One pass over the whole context produces exactly one character."

### 4 · Causality, as a picture

Layer 0, head 0, drawn as a heatmap. Row *t* is what position *t* attended to.
This is the section to linger on.

**Check**

- The upper triangle is pure black. Verified numerically: maximum attention
  weight to a future position is exactly `0.000000`.
- Row sums measured at `0.9997`–`1.0001` — every row is a probability
  distribution.
- Hover any cell for its exact weight.

**Say**

> "That black triangle is one line of code — `masked_fill(tril == 0, -inf)` —
> applied *before* the softmax, not after. Do it after and the rows stop summing
> to one."

### 5 · The whole vocabulary

All 79 symbols the model can represent. Not a word among them.

**Check**

- Count them: 79 — letters, digits, punctuation, space, newline.
- Type `ॐ` or `café` in section 1 — a warning appears saying the
  out-of-vocabulary characters were dropped.

**Say**

> "It learned 'Arjuna' as seven statistical events. A character it never saw has
> no embedding row — it isn't a lookup miss, it's outside the model's universe."

---

## Ninety-second demo

In this order. Each step sets up the next, and the last one is the point.

1. **Read the chips across the top.** 3.26M parameters, vocabulary of 79, 4
   layers, 4 heads, 256-character context, running on `mps` — your Mac's GPU.
2. **Type `Arjuna said: my mind is`.** Point at entropy: 0.52 bits of a possible
   6.30. Say the model is 94.5% sure the next character is a space.
3. **Delete one letter.** Entropy jumps. This is the moment people understand
   what "predicting the next token" actually means.
4. **Scroll to attention, change the head dropdown twice.** Same sentence,
   visibly different patterns.
5. **Scroll to the matrix.** Point at the black upper triangle. One line of
   code, made visible.
6. **Hit Generate** and let it stream while you talk.
7. **Land it:** "On the main page, keyword search scores 62.5% on factual
   questions and **0%** on 'I am stressed, guide me' — because the word
   'stressed' appears nowhere in 700 verses. Understanding what this model does
   and doesn't represent is what makes that number obvious instead of
   surprising."

---

## What they'll ask

**Is this what answers my question on the main page?**

No. Its context window is 256 *characters* — one verse plus a short question
fills it entirely, leaving no room for an answer. The main page is retrieval
over 700 real verses, quoted verbatim.

**Did you actually train this, or is it downloaded?**

Trained from scratch, on 124,135 characters of the Bhagavad Gita, to a
validation loss of **1.6119**. Roughly two minutes on the laptop. Every line of
the attention mechanism was written by hand.

**Why is the output nonsense?**

3.26M parameters trained on 124k characters. It learned the *shape* of the text
— blank-verse line lengths, capitalised openings, English phonotactics — and
that's the correct ceiling for a model this size. Producing meaning would take
roughly a thousand times more of both.

**Why not just call an LLM API?**

The product doesn't need one — it's retrieval-only, so the shloka is always
verbatim and nothing can invent scripture. This page is the foundations that
make the retrieval design legible.

**What is entropy doing there?**

It's the model's uncertainty in bits. log₂(79) ≈ 6.30 means no idea at all; near
zero means near-certainty. It's the single number that makes the prediction
concrete rather than abstract.

---

## If it breaks

| Symptom | Cause | Fix |
|---|---|---|
| Page loads, no numbers anywhere | Backend isn't running | `make backend` |
| "Could not load the model" | No checkpoint | `make train` in `geeta-guides` |
| Attention section empty | Fewer than ~2 characters typed | Type a few more |
| Generate does nothing | Backend restarted mid-stream | Reload the page |
| Devanagari shows as boxes | Missing system font | Cosmetic; the model is unaffected |

---

*Numbers here were measured on the actual checkpoint, not estimated: 3,262,543
parameters, vocabulary 79, validation loss 1.6119, maximum attention weight to a
future position 0.000000, row sums 0.9997–1.0001.*
