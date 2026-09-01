---
name: milestone-runner
description: Run the right checks.py milestone for the geeta-guides char-level GPT exercise, translate failures into plain language, and point at the matching video timestamp. Use when the user says "run the checks", "check my work", "did milestone 2 pass", "am I done", or asks what to do next in gita_gpt.py. Also use after they finish editing a section of gita_gpt.py.
---

# Milestone runner

Runs and interprets `checks.py` for the character-level GPT exercise in this
repo. The point is not to run a command — it's to turn a failing assertion into
a next action.

## Which milestone

If the user named one, use it. Otherwise infer from what's implemented in
`gita_gpt.py` — grep for `NotImplementedError` and run the lowest milestone
whose functions are written.

```bash
grep -n "NotImplementedError" gita_gpt.py
```

| Milestone | Covers | Run when they've written |
|---|---|---|
| 0 | dataset integrity — no model code involved | anything, or nothing (it passes from the start) |
| 1 | `load_text`, `build_vocab`, `make_encoder_decoder`, `train_val_split`, `get_batch`, `BigramLanguageModel` | the data functions and the bigram |
| 2 | `Head` | one attention head |
| 3 | `MultiHeadAttention`, `FeedForward`, `Block`, `GPTLanguageModel` | the full model |
| 4 | trained quality — needs `checkpoints/gita_gpt.pt` | after `python gita_gpt.py` has trained |

```bash
source .venv/bin/activate && python checks.py --milestone N
```

Use `--milestone all` when they ask "where am I". Add
`--module reference/solution.py` only if they explicitly want to verify the
reference — never as a way to show them the answer.

## Reading the result

Three outcomes, three responses:

- **SKIP** — not written yet. Not a failure. Say what's next, don't debug it.
- **PASS** — say so in one line and name the next milestone. Don't celebrate at length.
- **FAIL** — read the assertion message first. They're written to explain the
  concept, not just report a mismatch. Translate, then point at the video.

## Failure → concept → timestamp

| Failing check | What's actually wrong | Video |
|---|---|---|
| `get_batch targets are inputs shifted by one` | `y` isn't `x` shifted right by one; the offset window is wrong | ~14:27 |
| `get_batch shapes` | Sampling offsets that run off the end, or stacking the wrong dimension | ~14:27 |
| `BigramLanguageModel forward` (loss far from ln(V)) | The reshape for `cross_entropy` is scrambling the logit↔target correspondence | ~22:11 |
| `attention is causal` / `no backward leakage` | Mask applied after softmax instead of before, or transposed | ~1:02:00 |
| `rows sum to 1` | `softmax` over the wrong dimension — must be `dim=-1`, across keys | ~54:42 |
| `affinities are scaled by 1/sqrt(head_size)` | Missing the `k.shape[-1] ** -0.5` factor | ~1:16:56 |
| `Head handles T < block_size` | Mask sliced at a fixed `block_size` instead of the actual `T` | ~1:02:00 |
| `k/q/v projections are bias-free` | `nn.Linear` needs `bias=False` | ~1:02:00 |
| `causal mask is a buffer` | Use `self.register_buffer('tril', ...)`, not a Parameter | ~1:02:00 |
| `MultiHeadAttention shape` | `num_heads * head_size` must equal `n_embd`, then project back | ~1:21:59 |
| `FeedForward 4x expansion` | Hidden layer should be `4 * n_embd` | ~1:24:25 |
| `Block shape, norms, residual` | Missing a residual, or reusing one LayerNorm across both sublayers | ~1:26:48, ~1:32:51 |
| `positional embeddings actually used` | Position table built but never added to token embeddings | ~1:00:18 |
| `gradients reach every parameter` | Something detached or unused in the forward path | — |
| `generate crops context to block_size` | `generate` isn't slicing `idx[:, -block_size:]` before each forward | ~1:37:49 |
| `val loss below 2.0` | Trained too briefly, or the loss never actually fell — check the printed curve | ~1:37:49 |

## Rules

**Report and explain. Do not fix.** The user is writing this implementation
themselves on purpose. Name what's wrong and why it matters; let them write the
correction. If they want a hint ladder rather than a diagnosis, hand off to the
`gita-tutor` agent.

**Never open `reference/solution.py`** to explain a failure. It's sealed until
milestone 4 passes.

**Don't run training to make milestone 4 pass.** Training is theirs to start —
it takes minutes on `mps` and hours on CPU, and it should be their decision.
Tell them the command.

## Reporting format

Keep it short:

```
Milestone 2: 6 passed, 2 failed

✗ attention is causal — the mask is being applied after the softmax, so future
  positions get small nonzero weight instead of exactly zero. masked_fill with
  -inf before softmax. Video ~1:02:00.

✗ affinities are scaled — missing the 1/sqrt(head_size) factor.  Video ~1:16:56.

Fix those two and re-run: python checks.py --milestone 2
```

One line per failure, the concept, the timestamp, the re-run command. No preamble.
