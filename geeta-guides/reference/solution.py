"""
=============================================================================
  SEALED REFERENCE SOLUTION — DO NOT READ UNTIL YOU HAVE FINISHED MILESTONE 4
=============================================================================

If you are reading this before `python checks.py --milestone 4` passes on your
own gita_gpt.py, close the file. The whole value of the exercise is the twenty
minutes you spend confused about why your attention weights don't sum to one.

Afterwards, diff it:

    diff -u gita_gpt.py reference/solution.py | less

This is a complete, working, self-contained implementation. It mirrors the
structure of gita_gpt.py exactly, so the diff is readable.

Run it directly (it writes to reference/checkpoints/, not yours):

    python reference/solution.py --max-iters 100
=============================================================================
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
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 250
eval_iters = 200
learning_rate = 1e-3
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2

# --- CPU FALLBACK -----------------------------------------------------------
# batch_size = 12
# block_size = 64
# n_embd = 128
# ---------------------------------------------------------------------------

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) is not None
    and torch.backends.mps.is_available()
    else "cpu"
)

torch.manual_seed(1337)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "data", "gita.txt")
CKPT_DIR = os.path.join(HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "solution.pt")


# ===========================================================================
# MILESTONE 0 — DATA
# ===========================================================================

def load_text(path: str = DATA_PATH) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python prepare_data.py` from the project root first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> tuple[list[str], int]:
    chars = sorted(set(text))
    return chars, len(chars)


def make_encoder_decoder(chars: list[str]):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    def decode(idxs) -> str:
        return "".join(itos[int(i)] for i in idxs)

    return encode, decode


def train_val_split(data: torch.Tensor, frac: float = 0.9):
    n = int(frac * len(data))
    return data[:n], data[n:]


# ===========================================================================
# MILESTONE 1 — BATCHING + BIGRAM BASELINE
# ===========================================================================

def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor):
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    # len(data) - block_size is the exclusive upper bound: the last legal
    # start index needs block_size+1 characters available after it.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, V)
        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]                      # (B, V)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ===========================================================================
# MILESTONE 2 — SINGLE ATTENTION HEAD
# ===========================================================================

class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, hs)
        q = self.query(x)   # (B, T, hs)

        # scaled dot-product affinities: (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5

        # causal mask, sliced to the ACTUAL T (may be < block_size when generating)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)   # (B, T, hs)
        return wei @ v      # (B, T, hs)


# ===========================================================================
# MILESTONE 3 — MULTI-HEAD, FEED-FORWARD, BLOCK, FULL MODEL
# ===========================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # pre-norm residuals
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb                                                      # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                                   # (B,T,V)

        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]   # crop: pos table only has block_size rows
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx


# ===========================================================================
# MILESTONE 4 — TRAINING
# ===========================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, iters: int | None = None):
    iters = eval_iters if iters is None else iters
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for k in range(iters):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, iters: int | None = None, chars=None):
    iters = max_iters if iters is None else iters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for it in range(iters):
        if it % eval_interval == 0 or it == iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "chars": chars,
            "config": {
                "block_size": block_size, "n_embd": n_embd,
                "n_head": n_head, "n_layer": n_layer, "dropout": dropout,
            },
        },
        CKPT_PATH,
    )
    print(f"saved checkpoint -> {CKPT_PATH}")
    return model


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Reference char-level GPT on the Gita.")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--max-iters", type=int, default=None,
                        help="override max_iters (useful for a smoke test)")
    args = parser.parse_args()

    text = load_text()
    chars, vocab_size = build_vocab(text)
    encode, decode = make_encoder_decoder(chars)
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(data)

    print(f"device={device}  chars={len(text):,}  vocab={vocab_size}")

    model = GPTLanguageModel(vocab_size).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    if args.sample_only:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        train(model, train_data, val_data, iters=args.max_iters, chars=chars)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n" + "=" * 60)
    print(decode(model.generate(context, max_new_tokens=args.tokens)[0].tolist()))
    print("=" * 60)


if __name__ == "__main__":
    main()
