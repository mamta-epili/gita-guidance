"""
model.py — your trained char-GPT, instrumented so its internals can be watched.

This is architecturally IDENTICAL to gita_gpt.py in ../geeta-guides — same
parameter names, so `load_state_dict` takes your checkpoint with no shim. The
only difference: `Head.forward` optionally hands back its attention matrix, and
`GPTLanguageModel.forward` can return per-layer, per-head attention.

Your own model doesn't return attention because nothing needed it. A visualiser
does, and adding a return value to a hot training loop just to support a demo
would be the wrong trade. So the demo carries its own copy.

Nothing here is trained. It only loads and observes.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# Architecture — must match the checkpoint. Read from it where possible so a
# retrained model with different hyperparameters still loads.
DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "geeta-guides", "checkpoints", "gita_gpt.pt",
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Head(nn.Module):
    """One causal self-attention head. Same as yours, plus an attention return."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool = False):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        attn = wei.detach() if return_attn else None
        wei = self.dropout(wei)
        out = wei @ self.value(x)
        return (out, attn) if return_attn else out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool = False):
        if return_attn:
            pairs = [h(x, return_attn=True) for h in self.heads]
            out = torch.cat([p[0] for p in pairs], dim=-1)
            attns = [p[1] for p in pairs]
            return self.dropout(self.proj(out)), attns
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
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
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, return_attn: bool = False):
        if return_attn:
            sa_out, attns = self.sa(self.ln1(x), return_attn=True)
            x = x + sa_out
            x = x + self.ffwd(self.ln2(x))
            return x, attns
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, return_attn: bool = False):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok + pos

        all_attn = []
        for block in self.blocks:
            if return_attn:
                x, attns = block(x, return_attn=True)
                all_attn.append(attns)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return (logits, all_attn) if return_attn else logits


class CharGPT:
    """Loads your checkpoint and exposes what a visualiser needs."""

    def __init__(self, ckpt_path: str = DEFAULT_CKPT, device: str | None = None):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}.\n"
                f"Train one in ../geeta-guides (`make train`), or set "
                f"CHARGPT_CKPT to point at a .pt file."
            )
        self.device = device or pick_device()
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.chars: list[str] = ck["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

        sd = ck["model_state_dict"]
        # Derive the architecture from the weights rather than hardcoding it, so
        # a retrained model with different hyperparameters still loads.
        self.vocab_size, self.n_embd = sd["token_embedding_table.weight"].shape
        self.block_size = sd["position_embedding_table.weight"].shape[0]
        self.n_layer = sum(1 for k in sd if k.endswith(".sa.proj.weight"))
        self.n_head = sum(
            1 for k in sd if k.startswith("blocks.0.sa.heads.") and k.endswith(".key.weight")
        )
        self.head_size = self.n_embd // self.n_head

        self.model = GPTLanguageModel(
            self.vocab_size, self.n_embd, self.n_head, self.n_layer, self.block_size
        ).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.ckpt_path = ckpt_path

    # -- text <-> ids ----------------------------------------------------
    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def clean(self, s: str) -> str:
        """Drop characters outside the 79-character vocabulary.

        Worth surfacing in the UI: the model literally cannot represent a
        character it never saw. Typing 'ॐ' or 'é' is not a lookup miss, it is
        outside the model's universe.
        """
        return "".join(c for c in s if c in self.stoi)

    def info(self) -> dict:
        return {
            "params": self.n_params,
            "params_m": round(self.n_params / 1e6, 2),
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "head_size": self.head_size,
            "block_size": self.block_size,
            "device": self.device,
            "vocab": self.chars,
            "checkpoint": os.path.basename(self.ckpt_path),
        }

    # -- one step, fully inspectable -------------------------------------
    @torch.no_grad()
    def step(self, text: str, temperature: float = 1.0, top_probs: int = 12,
             want_attn: bool = True) -> dict:
        """Run one forward pass over `text` and report everything observable.

        Returns the next-character distribution and, optionally, the attention
        matrix of every head in every layer for the final position — i.e. what
        each layer was looking at when it made this prediction.
        """
        ids = self.encode(text) or [0]
        cropped = ids[-self.block_size:]
        idx = torch.tensor([cropped], dtype=torch.long, device=self.device)

        if want_attn:
            logits, all_attn = self.model(idx, return_attn=True)
        else:
            logits, all_attn = self.model(idx), []

        last = logits[0, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(last, dim=-1)
        top = torch.topk(probs, min(top_probs, self.vocab_size))

        # Entropy in bits: how undecided the model is at this position.
        p = probs.clamp_min(1e-12)
        entropy_bits = float(-(p * p.log2()).sum())

        out = {
            "context_used": len(cropped),
            "context_dropped": max(0, len(ids) - self.block_size),
            "entropy_bits": round(entropy_bits, 3),
            "max_entropy_bits": round(float(torch.log2(torch.tensor(float(self.vocab_size)))), 3),
            "top": [
                {"char": self.itos[int(i)], "prob": round(float(v), 5)}
                for v, i in zip(top.values, top.indices)
            ],
        }

        if want_attn and all_attn:
            # Attention paid BY the last position TO every earlier position.
            # That is the row of the matrix that actually produced this token.
            out["attention"] = [
                [[round(float(x), 5) for x in head[0, -1, :].tolist()] for head in layer]
                for layer in all_attn
            ]
            # Full lower-triangular matrix for layer 0 head 0, capped for size —
            # this is the picture that shows causality at a glance.
            m = all_attn[0][0][0]
            n = min(m.shape[0], 48)
            out["attention_matrix_l0h0"] = [
                [round(float(x), 4) for x in row[:n].tolist()] for row in m[:n]
            ]
        return out

    @torch.no_grad()
    def sample_one(self, text: str, temperature: float = 1.0) -> str:
        """Sample a single next character."""
        ids = self.encode(text) or [0]
        idx = torch.tensor([ids[-self.block_size:]], dtype=torch.long, device=self.device)
        logits = self.model(idx)
        last = logits[0, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(last, dim=-1)
        return self.itos[int(torch.multinomial(probs, 1))]
