# char-GPT inspector

A local app that makes the character-level transformer from
[`../geeta-guides`](../geeta-guides) **visible** — its next-character
distribution, its attention heads, and its causal mask, running live on your own
trained weights.

FastAPI backend (Python, because torch is Python) + Angular frontend.
No network calls, no paid APIs, no CDN.

---

## Run it

Two terminals.

```bash
# terminal 1 — API
./setup.sh          # once: .venv + torch, fastapi, uvicorn
make backend        # → http://127.0.0.1:8000

# terminal 2 — UI
make install        # once: pnpm install
make frontend       # → http://127.0.0.1:4200
```

Angular's dev server proxies `/api` to `:8000` (`proxy.conf.json`), so there is
no CORS to think about in development.

**Or skip node entirely.** `make backend` alone serves a zero-dependency
fallback page at `http://127.0.0.1:8000` — one HTML file, no build step. It has
the same five sections. Useful on its own, and the fastest way to tell whether a
problem is in the model or the frontend.

Production bundle served by FastAPI:

```bash
make build && make serve
```

`make check` loads the checkpoint and prints what it found, without a server.

### Checkpoint

Read from `../geeta-guides/checkpoints/gita_gpt.pt` by default. Override:

```bash
CHARGPT_CKPT=/path/to/model.pt make backend
```

Missing? Train one: `cd ../geeta-guides && make train`.

---

## What the five sections show

**1 · One forward pass.** Type, and every keystroke runs the model and shows its
prediction for the *next* character. This distribution **is** the model's entire
output — generation is only sampling from it, appending, and running again.

The entropy readout is the most instructive number here. log₂(79) ≈ 6.30 bits
means "no idea at all"; near zero means near-certainty. Type
`Arjuna said: my mind is` and it reads **0.52 bits** — the model is 94.5% sure
the next character is a space. Type a letter and watch it climb: mid-word many
continuations are plausible, after a space far fewer.

**2 · Attention.** How much the *final* position attended to each earlier
character when making this prediction. Four layers × four heads = sixteen views
of the same text; switching between them shows visibly different patterns, which
is what "multi-head" buys. Nothing ever attends to its right.

**3 · Generate.** One character per forward pass, streamed at the speed it is
produced. 240 characters is 240 full passes through the network.

**4 · Causality as a picture.** Layer 0 head 0 as a heatmap. Row *t* is what
position *t* attended to; the upper triangle is black because
`masked_fill(tril == 0, -inf)` zeroed it *before* the softmax. Verified
numerically: max weight to a future position is exactly `0.000000`, row sums
0.9997–1.0001.

**5 · The vocabulary.** All 79 symbols the model can represent — not a word
among them. Type `ॐ` or `café` and the out-of-vocabulary characters are dropped
with a count. Not a bug: a character the model never saw has no embedding row.

---

## Layout

```
app/                     FastAPI backend
  model.py               your architecture, re-declared with attention returns
  main.py                /api/info, /api/step, /api/stream (SSE)
  static/index.html      no-build fallback UI, single file
frontend/                Angular 22, standalone components, signals
  proxy.conf.json        /api → 127.0.0.1:8000
  src/app/core/api.ts    typed client + shared helpers
  src/app/lab/
    next-char.ts         probability bars
    attention-view.ts    per-layer/head shading over the text
    causal-matrix.ts     the triangular heatmap
    generate-stream.ts   SSE streaming
    lab-page.ts          composes the five sections
```

### Why the components are separate

The Gita app you're building is Angular. `<attention-view>`,
`<next-char>` and `<causal-matrix>` are standalone, take their data through
`input()` signals, and are lazy-loaded — so they drop into a `/lab` route there
without dragging this project's plumbing along. Built as one HTML file they
would have to be rewritten.

### Why `app/model.py` duplicates your architecture

Parameter names match exactly, so `load_state_dict` takes your checkpoint with
no shim — but this `Head.forward` can return its attention matrix. Yours doesn't,
because nothing needed it, and adding a return value to a hot training loop to
support a demo would be the wrong trade.

It also reads the architecture *from the weights* (`n_layer` from the count of
`sa.proj.weight` tensors, `block_size` from the position table's rows), so a
retrained model with different hyperparameters loads without editing anything.

### Notes on the frontend

- **pnpm blocks dependency postinstall scripts by default**, and Angular's build
  needs esbuild's (it links the platform binary). `pnpm-workspace.yaml`
  allow-lists them, so a fresh clone installs without an interactive
  `pnpm approve-builds` prompt. Note the schema changed between versions: pnpm 11
  wants `allowBuilds` (a map of name → boolean), pnpm 10 wanted
  `onlyBuiltDependencies` (a list). Both forms are documented in that file. If an
  install reports `ERR_PNPM_IGNORED_BUILDS`, pnpm has usually rewritten the file
  with a `set this to true or false` template — fill in the booleans.
- **EventSource callbacks run outside Angular's zone.** Signals notify their
  consumers directly rather than relying on zone.js, so streaming updates the
  view with no `NgZone.run` — the one place that difference is load-bearing here.
- Every component is `OnPush`; state is signals throughout.

---

## What this is, and isn't

**Is:** a demo of the foundations. The most interesting thing here to anyone who
can read code, because the attention mechanism was written by hand and this
shows it working.

**Isn't:** a question-answering system, and not usable as one. The context window
is 256 **characters**. One verse plus a short question fills it entirely, leaving
no room for an answer — and past 256 the crop silently eats the question, so the
model completes text it can no longer see.

Fed a real retrieved verse and *"I am stressed, guide me"*, it replies:

> *sto hovole its still pervana. Do exemle, lord, unglike these foeth, To The
> verve I slaying the be Thee, O Prince!*

That is the honest ceiling. No prompting or fine-tuning moves it.

**What this page does for the RAG project:** it explains visually why retrieval
needs *semantic* embeddings. In `../geeta-guides`, BM25 scores **62.5%** hit@10
on factual lookup questions and **0%** on *"i am stressed guide me?"* — because
"stressed" appears nowhere in 700 verses. Understanding what a character model
does and does not represent is what makes that number obvious rather than
surprising.
