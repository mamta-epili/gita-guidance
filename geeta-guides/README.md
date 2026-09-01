# geeta-guides — build a character-level GPT, yourself

A guided workspace for implementing a decoder-only transformer from scratch in
PyTorch, following Andrej Karpathy's
[**Let's build GPT: from scratch, in code, spelled out**](https://www.youtube.com/watch?v=kCc8FmEb1nY).

The corpus is Sir Edwin Arnold's *The Song Celestial* — his 1885 blank-verse
translation of the Bhagavad Gita ([Project Gutenberg #2388](https://www.gutenberg.org/ebooks/2388),
public domain). **124,135 characters, vocabulary of 79.**

**You write the model.** `gita_gpt.py` is a skeleton: signatures, docstrings and
`raise NotImplementedError`. Nothing else in this repo will write it for you.

---

## Setup

**Run this first — there is no `.venv` in the repo yet.** A virtualenv contains
compiled binaries for the machine that built it, so it has to be created on
yours:

```bash
./setup.sh                      # creates .venv, installs torch + numpy,
                                # downloads the corpus, checks your hardware
source .venv/bin/activate
```

`data/gita.txt` *is* already here — it's plain text, so it travels fine.

Then confirm the data is good:

```bash
python checks.py --milestone 0
```

That should print six `PASS` lines before you write a single line of model code.

**Hardware.** `setup.sh` runs `device_check.py` and prints the accelerator it
found. Run it again any time:

```bash
python device_check.py
```

Record the result here so you don't have to keep re-checking:

```
detected device : mps    (Apple Silicon, Darwin arm64, torch 2.13.0, python 3.12.5)
```

You have `mps`, so **keep the default hyperparameters** at the top of
`gita_gpt.py` — the CPU-fallback block stays commented out. (Still worth trying
the fallback config later: it's smaller, and on a corpus this size less capacity
often means a better val loss.)

`device_check.py` doesn't just call `is_available()` — it runs a real matmul on
the device it picked, because those two things disagree more often than you'd
like.

---

## Milestone roadmap

Work top to bottom. After each milestone, run its check. Video timestamps are
approximate — they point at the section, not the exact second.

| # | Build | Video | Check | Expected val loss |
|---|-------|-------|-------|-------------------|
| 0 | *(done for you)* corpus downloaded and cleaned | ~7:52 – 9:28 | `--milestone 0` | — |
| 1 | `load_text`, `build_vocab`, `make_encoder_decoder`, `train_val_split`, `get_batch`, `BigramLanguageModel` | ~9:28 – 38:00 | `--milestone 1` | **~2.45** |
| 2 | `Head` — one masked self-attention head | ~42:13 – 1:18:00 | `--milestone 2` | ~2.3 (if you train it) |
| 3 | `MultiHeadAttention`, `FeedForward`, `Block`, `GPTLanguageModel` | ~1:19:11 – 1:42:00 | `--milestone 3` | — (forward pass only) |
| 4 | `estimate_loss`, `train`, `generate`, then train it | ~1:37:49 onwards | `--milestone 4` | **~1.5** (see caveat) |

Milestone landmarks in the video, in more detail:

- **9:28** tokenization and the train/val split
- **14:27** the data loader — batches of chunks, and the one-position shift
- **22:11** the bigram baseline, loss, and generation ← *milestone 1 lands here*
- **42:13 – 54:42** the three warm-up versions of averaging past context
  (for-loops → matrix multiply → softmax). Worth watching even though the
  skeleton jumps straight to the real thing.
- **1:02:00** "the crux of the video" — version 4, real self-attention ← *milestone 2*
- **1:16:56** why you divide by `sqrt(head_size)` — `checks.py` tests for this
- **1:21:59** multi-headed attention
- **1:24:25** the feed-forward sublayer
- **1:26:48** residual connections
- **1:32:51** LayerNorm (and pre-norm vs. post-norm)
- **1:37:49** scaling up, dropout, the final hyperparameters ← *milestone 3 & 4*

The checks treat "not written yet" as `SKIP`, not `FAIL`, so you can run a
milestone early to see what's still ahead of you.

```bash
python checks.py --milestone 2
python checks.py --milestone all
```

---

## Running it

```bash
make check M=1        # run one milestone
make check-all        # run all of them
make train            # train, then print a 1000-char sample
make sample           # load checkpoints/gita_gpt.pt and sample
make device           # re-report cuda / mps / cpu
make help             # everything else
```

Every `make` target calls `.venv/bin/python` by absolute path, so nothing your
shell does to the word `python` — aliases, pyenv or conda shims, an IDE
activating a different interpreter — can reach these commands. The equivalent
direct calls (`python checks.py --milestone 1`, `python gita_gpt.py`) work too,
whenever your `python` really is the venv's.

Training writes `checkpoints/gita_gpt.pt`, which is gitignored.

---

## Expected results, honestly

The target is a **validation loss of roughly 1.4–1.6**, with `checks.py
--milestone 4` enforcing a loose sanity threshold of **2.0**.

**Reference points, actually measured** on the sealed solution (4 CPU cores,
no GPU) so you have something real to compare against:

| Config | Params | Iterations | Train | Val | Wall clock |
|---|---|---|---|---|---|
| CPU fallback (`n_embd=128`, `block=64`, `batch=12`) | 0.82M | 100 | 2.59 | 2.67 | 7 s |
| CPU fallback | 0.82M | 1500 | 1.79 | **1.86** | 81 s |
| Full (`n_embd=256`, `block=256`, `batch=64`) | 3.26M | — | — | — | ~3 s/iter, ~2.6 GB RAM |

So on CPU the fallback config reaches the milestone-4 threshold in under two
minutes, and was **still improving at 1500 iterations** — run the full 3000 and
you should land near 1.7. The full config on CPU would take roughly 2.5 hours
for 3000 iterations; on `mps` or `cuda` it's minutes, and that's where the
1.4–1.6 range lives.

One caveat worth knowing before it surprises you. Karpathy's tinyshakespeare is
~1.1M characters; this corpus is **~124k, about nine times smaller**, while the
full config is 3.26M parameters. That model has more than enough capacity to
memorize this text, so expect the train and val curves to separate and the val
loss to bottom out and then start **climbing** while train loss keeps falling.
That's overfitting, and on this dataset it's expected behaviour, not a bug in
your code.

Watch the printed val loss and treat its **minimum** as your result. If you land
at 1.6–1.8 rather than 1.4, that's the corpus size, not you. What actually helps
here, in rough order of payoff:

- stop at the val-loss minimum rather than running all 3000 iterations
- keep `dropout=0.2`, or raise it to 0.3
- shrink the model (`n_embd=192`, `n_layer=3`) — less capacity to memorize with
- the CPU-fallback config generalizes better on this corpus, and is worth trying
  even when you do have a GPU

Samples at ~1.8 look like this (real output from the reference at 1500 iters):

```
  Life fixi!
  In not and worshipe prover sto who wre, and steed,
  Brut toou mover not will world arth all cace of path teepamost to Unto fighter
  I soul evere Ener the upon trearth com thesef'self
```

Blank-verse line lengths, capitalized line starts, pronounceable
pseudo-Sanskrit, and no meaning whatsoever. That is the correct outcome for a
character-level model this size — if your samples look like that, you built it
right.

---

## Files

```
prepare_data.py           downloads eBook #2388, strips boilerplate, writes data/gita.txt
device_check.py           reports cuda / mps / cpu and runs a real matmul to prove it
setup.sh                  venv + dependencies + data + hardware check, in one shot
Makefile                  every command, pinned to .venv/bin/python — `make help`
gita_gpt.py               >>> YOUR WORK GOES HERE <<<
checks.py                 milestone tests, runnable one at a time
data/gita.txt             the cleaned corpus (124,135 chars, vocab 79)
reference/solution.py     SEALED — see below

docs/BRD.md               business case for the RAG this is Phase 0 of
docs/PRD.md               how that RAG gets built (§3 is worth reading now)
.claude/agents/           gita-tutor, gita-reviewer
.claude/skills/           milestone-runner, brd-prd
```

---

## Helpers in this repo

Two agents and two skills, all scoped to this project.

| | What it does | When |
|---|---|---|
| **`gita-tutor`** agent | Diagnoses where you're stuck and gives escalating hints — concept, then shape, then the specific line. Never writes your implementation, never opens the reference | While building |
| **`milestone-runner`** skill | Picks the right milestone, runs it, translates each failure into the concept behind it plus a video timestamp | After editing a section |
| **`gita-reviewer`** agent | Reviews your finished code: correctness past what the checks cover, divergences from the reference and whether each is a bug or a defensible choice, then questions to probe whether you actually understand it | After milestone 4 passes |
| **`brd-prd`** skill | Writes BRDs and PRDs in the format used in `docs/` | On the RAG project |

Ask for them by name, or just describe the situation — "run the checks",
"I'm stuck on the mask", "review my implementation" — and the right one loads.

The tutor's refusal to read `reference/solution.py` is enforced by instruction,
not by a sandbox. It's the same honour system as the sealed file itself.

---

## The bigger project

This workspace is **Phase 0** of a retrieval-augmented Q&A system over
scripture. [`docs/BRD.md`](docs/BRD.md) is the business case,
[`docs/PRD.md`](docs/PRD.md) is the build spec.

Worth reading [PRD §3](docs/PRD.md#3-decision-record-the-role-of-gita_gptpy)
before you get attached to the idea of using this model in that system — it
can't be, and the reasons are more interesting than the conclusion. What
transfers is the understanding: `softmax(QKᵀ/√d)·V` is retrieval, and once
you've written it by hand, the whole RAG architecture reads as one familiar
shape.

---

## 🔒 Do not open `reference/solution.py` yet

`reference/solution.py` is a complete working implementation. It exists so you
can **diff against it after you finish**, not so you can consult it while you're
stuck.

Being stuck on why your attention weights don't sum to 1, and working it out
from the error message `checks.py` gives you, *is* the exercise. Reading the
answer costs you the thing you came for. When `python checks.py --milestone 4`
passes on your own code, then:

```bash
diff -u gita_gpt.py reference/solution.py | less
```

If you want to confirm the reference itself works without reading it:

```bash
python checks.py --milestone all --module reference/solution.py   # seconds
python reference/solution.py --max-iters 100                      # trains a bit
```

Both run the file without showing you its contents. The second one uses the full
config, so on CPU give it fifteen minutes or edit the fallback block in
`reference/solution.py` first. It writes to `reference/checkpoints/`, never to
your `checkpoints/`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'` while the prompt says `(.venv)`** —
your shell's `python` isn't the venv's `python`. Usually a pyenv or conda shim,
or an IDE auto-activating its own interpreter in new terminals *after* you
sourced the venv. `checks.py` now detects this and prints both paths. Unblock
immediately with `.venv/bin/python checks.py --milestone N`; find the culprit
with `which -a python python3`.

**`RuntimeError: shape '[...]' is invalid for input of size ...`** in the loss —
`F.cross_entropy` wants `(N, C)` logits and `(N,)` targets, but yours are
`(B, T, V)` and `(B, T)`.

**`IndexError` in `Head.forward` during generation** — you sliced the causal
mask at a fixed `block_size` instead of the actual `T`.

**`IndexError` in the position embedding after ~256 generated characters** —
`generate` isn't cropping the context to the last `block_size` tokens.

**Initial loss is ~4.4 and never moves** — 4.37 is `ln(79)`, i.e. a uniform
distribution. Check that the optimizer step is actually running and that you
called `optimizer.zero_grad(set_to_none=True)` *before* `loss.backward()`.

**Milestone 2 fails "no backward leakage"** — the mask is being applied after
the softmax instead of before it. `masked_fill(..., float('-inf'))` first, then
`softmax`.

**MPS is much slower than you expected** — that's normal for small models; the
per-kernel dispatch overhead dominates. Try the CPU-fallback config and compare;
on this corpus CPU sometimes wins.
