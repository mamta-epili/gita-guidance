"""
gita_gpt.py — YOUR implementation. Character-level GPT on the Bhagavad Gita.

This file is a SKELETON. Every body is `raise NotImplementedError`. You fill
them in, in the order they appear, which is the order Karpathy builds them in
"Let's build GPT: from scratch, in code, spelled out."

Work milestone by milestone and run the checks after each one:

    python checks.py --milestone 0    # data file integrity (passes already)
    python checks.py --milestone 1    # encode/decode, split, get_batch, bigram
    python checks.py --milestone 2    # single attention head: causal, rows sum to 1
    python checks.py --milestone 3    # multi-head, block, full GPT forward + generate
    python checks.py --milestone 4    # trained val loss below the sanity threshold
    python checks.py --milestone all  # everything

Train / sample:

    python gita_gpt.py                # trains, then prints a sample
    python gita_gpt.py --sample-only  # loads checkpoints/gita_gpt.pt and samples

Rules of the road for yourself:
  - Do not open reference/solution.py until you finish milestone 4.
  - Shapes in docstrings use B = batch, T = time (context positions),
    C = channels (embedding dim), hs = head size, V = vocab size.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------------------
# Default ("full") config. Comfortable on a CUDA GPU; slow but survivable on
# Apple MPS; painful on CPU.

batch_size = 64          # independent sequences processed in parallel
block_size = 256         # maximum context length for predictions
max_iters = 1500         # total training steps (val bottoms near ~750 here)
eval_interval = 250      # how often to estimate train/val loss
eval_iters = 200         # number of batches averaged per loss estimate
learning_rate = 1e-3
n_embd = 256             # embedding dimension (C)
n_head = 4               # number of attention heads (head_size = n_embd // n_head)
n_layer = 4              # number of transformer blocks
dropout = 0.2

# --- CPU FALLBACK -----------------------------------------------------------
# If device_check.py said 'cpu', comment out the block above and uncomment
# this one. Same architecture, small enough to finish in minutes rather than
# hours. Expect a slightly worse final val loss (~1.6-1.8 rather than ~1.5).
#
# batch_size = 12
# block_size = 64
# max_iters = 3000
# eval_interval = 250
# eval_iters = 100
# learning_rate = 1e-3
# n_embd = 128
# n_head = 4
# n_layer = 4
# dropout = 0.2
# ---------------------------------------------------------------------------

# Set this from what device_check.py told you: 'cuda', 'mps', or 'cpu'.
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(1337)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "gita.txt")
CKPT_DIR = os.path.join(HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "gita_gpt.pt")


# ===========================================================================
# MILESTONE 1 (part a) — DATA: load, build the vocabulary, encode, split
# ===========================================================================

def load_text(path: str = DATA_PATH) -> str:
    """Read the whole corpus into a single string.

    Returns
    -------
    str
        The full contents of data/gita.txt (~124k characters). Raise a clear
        error if the file is missing, pointing the user at prepare_data.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run `python prepare_data.py` first to "
            "download and clean the Bhagavad Gita text."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> tuple[list[str], int]:
    """Derive the character-level vocabulary from the corpus.

    The vocabulary is every distinct character that occurs in `text`, in a
    deterministic (sorted) order so that index 7 means the same character on
    every run and across saved checkpoints.

    Returns
    -------
    (chars, vocab_size)
        chars      : list[str], length V, sorted, one entry per distinct char
        vocab_size : int, == len(chars). Should be 79 for this corpus.
    """
    chars = sorted(set(text))
    return chars, len(chars)


def make_encoder_decoder(chars: list[str]):
    """Build the two lookup functions that convert between text and integers.

    This is the world's simplest tokenizer: one token per character. Nothing
    is learned; it is a pair of dictionaries.

    Returns
    -------
    (encode, decode)
        encode : Callable[[str], list[int]]
            'hi' -> [60, 61]. Every character in the input must be in `chars`.
        decode : Callable[[list[int]], str]
            [60, 61] -> 'hi'. Must accept a plain list of ints, and also
            anything list()-able such as a 1-D tensor converted with .tolist().

    Invariant the checks will assert:
        decode(encode(s)) == s   for any s drawn from the corpus.
    """
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[ch] for ch in s]

    def decode(idxs) -> str:
        return "".join(itos[int(i)] for i in idxs)

    return encode, decode


def train_val_split(data: torch.Tensor, frac: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the encoded corpus into a training and a validation tensor.

    This is a *contiguous* split, not a shuffle: the first `frac` of the text
    is train, the remainder is val. Shuffling characters would destroy the
    sequences we are trying to model.

    Parameters
    ----------
    data : torch.Tensor
        1-D tensor of dtype torch.long, shape (N,), the whole encoded corpus.
    frac : float
        Fraction of the data used for training.

    Returns
    -------
    (train_data, val_data)
        Both 1-D torch.long tensors. Their lengths sum to N.
    """
    n = int(frac * len(data))
    return data[:n], data[n:]


# ===========================================================================
# MILESTONE 1 (part b) — BATCHING and the simplest possible baseline
# ===========================================================================

def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor):
    """Sample one random minibatch of (inputs, targets) from the chosen split.

    Pick `batch_size` random starting offsets into the chosen 1-D data tensor.
    From each offset take `block_size` characters as the input, and the SAME
    window shifted one character to the right as the target. That shift is the
    whole supervision signal: at every position t, the target is the character
    that actually followed.

    Parameters
    ----------
    split : str
        'train' or 'val'. Anything else should be an error.
    train_data, val_data : torch.Tensor
        The 1-D encoded tensors from train_val_split.

    Returns
    -------
    (x, y)
        x : torch.long tensor of shape (B, T) == (batch_size, block_size)
        y : torch.long tensor of shape (B, T), where y[b, t] == x[b, t+1]
            for every t < T-1. Both moved to `device`.

    Careful with the highest legal offset: you need block_size+1 characters
    available after it, or the last target runs off the end of the tensor.
    """
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    """Milestone 1 baseline: predict the next character from the current one only.

    No attention, no positions, no context beyond a single character. It is a
    lookup table of shape (V, V): row i holds the logits over the next
    character given that the current character is i. Everything after this
    milestone exists to beat its loss.

    Expected val loss after training: roughly 2.4-2.5. Note that -ln(1/79) is
    about 4.37, which is what a totally uniform model would score, so ~2.45 is
    real learning — just not much of it.
    """

    def __init__(self, vocab_size: int):
        """Create the single (V, V) embedding table. No other parameters."""
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Score the next character at every position.

        Parameters
        ----------
        idx : torch.long tensor, shape (B, T)
            The context characters.
        targets : torch.long tensor of shape (B, T), or None
            When None, skip the loss and return None for it (this is the path
            generate() uses).

        Returns
        -------
        (logits, loss)
            logits : shape (B, T, V) — one score per vocabulary entry, per
                     position, per sequence.
            loss   : scalar tensor (cross entropy) or None.

        The catch: F.cross_entropy wants (N, C) logits and (N,) targets, but
        you have 3-D logits. You will need to reshape both, and then the
        returned logits should still be the (B, T, V) shape described above —
        decide deliberately which shape you hand back and keep it consistent
        with what generate() expects.
        """
        logits = self.token_embedding_table(idx)  # (B, T, V)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively extend `idx` by `max_new_tokens` characters.

        Loop `max_new_tokens` times. Each pass: run the model forward, keep
        only the logits for the LAST time step (that is the prediction for the
        next character), turn them into a probability distribution, sample one
        index from it, and append it to the running sequence.

        Sample rather than taking the argmax — greedy decoding on a character
        model collapses into repeated loops almost immediately.

        Parameters
        ----------
        idx : torch.long tensor, shape (B, T)
            The seed context. A single zero, shape (1, 1), is the usual start.
        max_new_tokens : int

        Returns
        -------
        torch.long tensor of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                        # (B, V)
            probs = F.softmax(logits, dim=-1)                # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)          # (B, T+1)
        return idx


# ===========================================================================
# MILESTONE 2 — SELF-ATTENTION: one head
# ===========================================================================

class Head(nn.Module):
    """One head of masked (causal) self-attention.

    The idea: every position emits a query ("what am I looking for?") and a
    key ("what do I contain?"). The affinity between position t and position s
    is the dot product of t's query with s's key. Softmax those affinities
    into weights, then use them to take a weighted average of the values at
    each position. That average is what the position learns from its past.

    Two details that matter and that the checks will verify:

    1. CAUSALITY. Position t may only attend to positions <= t. Enforce this
       by setting the affinities to future positions to negative infinity
       BEFORE the softmax, so they come out as exactly zero weight. Register
       the lower-triangular mask with self.register_buffer('tril', ...) so it
       moves with .to(device) but is not a learned parameter.

    2. SCALING. Divide the raw affinities by sqrt(head_size). Without it the
       variance of the dot products grows with head_size, the softmax
       saturates toward one-hot, and each position ends up reading from a
       single other position instead of blending.

    The three projections (key, query, value) are nn.Linear with bias=False.
    Dropout is applied to the attention weights after the softmax.
    """

    def __init__(self, head_size: int):
        """Create key/query/value projections, the causal mask buffer, dropout.

        Each projection maps n_embd -> head_size.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, head_size).

        Note that T here is whatever the caller passed, which during
        generation can be SHORTER than block_size. Slice the mask to the
        actual T rather than assuming the full block_size, or generation from
        a short prompt will crash.
        """
        B, T, C = x.shape
        k = self.key(x)      # (B, T, hs)
        q = self.query(x)    # (B, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)    # (B, T, hs)
        return wei @ v       # (B, T, hs)


# ===========================================================================
# MILESTONE 3 — MULTI-HEAD, FEED-FORWARD, and the BLOCK
# ===========================================================================

class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected.

    Multiple heads let the model attend to several different kinds of
    relationship at once (say, one head tracking the previous vowel and
    another tracking line breaks) instead of averaging them all into one
    channel. Run `num_heads` independent Heads over the same input, glue their
    outputs together along the channel dimension, then apply a linear
    projection back into the residual stream, then dropout.

    Store the heads in an nn.ModuleList so their parameters are registered.
    With num_heads * head_size == n_embd, the concatenation lands back at C.
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C), assuming num_heads * head_size == C."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP: the 'think about what you just gathered' step.

    Attention moves information between positions; this moves it between
    channels, independently at each position. Two linear layers with a ReLU
    between them, expanding to 4 * n_embd in the middle and projecting back
    down to n_embd, followed by dropout. The 4x expansion factor is the
    convention from the original Transformer paper.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)."""
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communication, then computation.

    Structure: multi-head self-attention, then feed-forward, each wrapped in a
    residual connection. Residuals give gradients a clean path from the loss
    all the way back to the embeddings, which is what makes stacking several
    blocks trainable at all.

    Use the PRE-norm arrangement: LayerNorm is applied to the input of each
    sublayer, and the residual addition happens on the un-normalized stream.
    (The 2017 paper put the norm after; pre-norm is what everyone actually
    trains now, and it is what Karpathy switches to in the video.) So each
    sublayer contributes: x = x + sublayer(norm(x)).

    You need two separate nn.LayerNorm(n_embd) instances — one per sublayer.
    Reusing one would tie their learned scale and shift together.
    """

    def __init__(self, n_embd: int, n_head: int):
        """head_size is n_embd // n_head, so the heads concatenate back to C."""
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C)."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ===========================================================================
# MILESTONE 3 (cont.) — THE FULL MODEL
# ===========================================================================

class GPTLanguageModel(nn.Module):
    """The whole thing: token + position embeddings, stacked blocks, head.

    Components, in the order the data flows through them:

      token_embedding_table    : nn.Embedding(vocab_size, n_embd)
          What is this character?
      position_embedding_table : nn.Embedding(block_size, n_embd)
          Where in the window is it? Attention is permutation-invariant on its
          own, so without this the model literally cannot tell order. These
          are learned, not the sinusoids from the original paper.
      blocks                   : n_layer Blocks in an nn.Sequential
      ln_f                     : a final nn.LayerNorm(n_embd)
      lm_head                  : nn.Linear(n_embd, vocab_size)
          Projects the residual stream out to one score per vocab entry.

    Also apply the weight initialization Karpathy adds near the end: normal
    with mean 0 and std 0.02 for Linear and Embedding weights, zeros for
    Linear biases. Do it with self.apply(self._init_weights).
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize one submodule; called for every submodule by self.apply.

        nn.Linear  : weight ~ N(0, 0.02); bias zeroed if it exists.
        nn.Embedding: weight ~ N(0, 0.02).
        Anything else: leave alone.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """(B, T) -> logits (B, T, vocab_size), plus loss or None.

        Steps: look up token embeddings (B,T,C); look up position embeddings
        for positions 0..T-1, which is (T,C) and broadcasts across the batch;
        add them; run the blocks; final layer norm; lm_head.

        Build the position indices on the same device as `idx`, not on the
        default device — otherwise this works on CPU and fails on GPU.

        As in the bigram model, compute cross-entropy only when targets is not
        None, reshaping to satisfy F.cross_entropy.
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                          # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )                                                                   # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                            # (B,T,V)

        if targets is None:
            return logits, None

        V = logits.shape[-1]
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """(B, T) -> (B, T + max_new_tokens).

        Same loop as the bigram version, with ONE addition that the bigram
        model did not need: the position embedding table only has block_size
        rows, so before each forward pass you must crop the context to its
        last block_size characters. Forget this and generation crashes the
        moment the sequence grows past block_size.

        Put the model in eval() mode before generating and back in train()
        afterwards if you are sampling mid-training, so dropout is off.
        """
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                           # (B, V)
            probs = F.softmax(logits, dim=-1)                    # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()
        return idx


# ===========================================================================
# MILESTONE 4 — TRAINING
# ===========================================================================

@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor) -> dict:
    """Average the loss over `eval_iters` batches of each split.

    Why not just print the training-batch loss? Because a single minibatch is
    extremely noisy, and you cannot tell a real improvement from luck. This
    averages many batches so the numbers you watch actually mean something.

    Put the model in eval() mode first and back into train() before returning,
    so dropout does not perturb the measurement. The @torch.no_grad() above
    already keeps this from building a graph.

    Returns
    -------
    dict with keys 'train' and 'val', each a float.
    """
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, train_data, val_data)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor,
          chars: list[str]) -> nn.Module:
    """Run the optimization loop.

    Use torch.optim.AdamW with `learning_rate`. For `max_iters` steps:
    fetch a training batch, forward it to get the loss, zero the gradients
    (set_to_none=True), backward, step. Every `eval_interval` steps — and on
    the final step — call estimate_loss and print both numbers.

    Save to CKPT_PATH whenever val loss reaches a new best, not at the end:
    this corpus is small enough that val turns upward well before max_iters,
    so the final model is the most overfit one. Save `chars` alongside the
    weights, so --sample-only can rebuild the identical encode/decode mapping
    from the checkpoint alone, without re-reading the corpus.

    Returns the trained model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    os.makedirs(CKPT_DIR, exist_ok=True)

    best_val = float("inf")

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it:5d}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}", flush=True)

            # Keep the best-generalizing model, not the last one. On a corpus
            # this small val turns upward long before max_iters, so an
            # unconditional save at the end would store the most overfit model.
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model_state_dict": model.state_dict(), "chars": chars},
                    CKPT_PATH,
                )
                print(f"  ↳ new best, saved (val {best_val:.4f})", flush=True)

        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"best val loss {best_val:.4f} -> {CKPT_PATH}", flush=True)
    return model


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    """Wire it together: load data, build the model, train, print a sample.

    Suggested flow:
      - parse --sample-only and --tokens
      - load_text -> build_vocab -> make_encoder_decoder -> encode the whole
        corpus into a torch.long tensor -> train_val_split
      - build GPTLanguageModel(vocab_size).to(device)
      - print the parameter count in millions (a nice sanity signal: the full
        config lands around 3.2M parameters)
      - train, then generate ~1000 characters from a (1,1) zero context and
        print the decoded text
    """
    parser = argparse.ArgumentParser(description="Train a char-level GPT on the Gita.")
    parser.add_argument("--sample-only", action="store_true",
                        help="skip training, load checkpoints/gita_gpt.pt and sample")
    parser.add_argument("--tokens", type=int, default=1000,
                        help="how many characters to generate")
    args = parser.parse_args()

    if args.sample_only:
        # The checkpoint is self-contained: its `chars` is the vocabulary the
        # weights were trained against, so the corpus is never read here.
        # Rebuilding the vocab from data/gita.txt instead would silently
        # mis-decode if that file ever changed.
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(
                f"No checkpoint at {CKPT_PATH}. Run `python gita_gpt.py` to train first."
            )
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        chars = list(ckpt["chars"])
        _, decode = make_encoder_decoder(chars)

        model = GPTLanguageModel(len(chars)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
              f"on {device}", flush=True)
    else:
        text = load_text()
        chars, vocab_size = build_vocab(text)
        encode, decode = make_encoder_decoder(chars)
        data = torch.tensor(encode(text), dtype=torch.long)
        train_data, val_data = train_val_split(data)

        model = GPTLanguageModel(vocab_size).to(device)
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
              f"on {device}", flush=True)
        model = train(model, train_data, val_data, chars)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=args.tokens)[0].tolist()))


if __name__ == "__main__":
    main()
