"""
checks.py — milestone tests for your char-level GPT.

Run one milestone at a time, as you finish it:

    python checks.py --milestone 0     # data file integrity (passes already)
    python checks.py --milestone 1     # encode/decode, split, get_batch, bigram
    python checks.py --milestone 2     # single attention head: causal, rows sum to 1
    python checks.py --milestone 3     # multi-head, block, full GPT forward + generate
    python checks.py --milestone 4     # trained val loss below threshold
    python checks.py --milestone all

By default the checks import your gita_gpt.py. To sanity-check the sealed
reference instead (don't do this until you're done):

    python checks.py --milestone all --module reference/solution.py

Each milestone is independent: milestone 2 does not require milestone 1 to
pass, so a stuck test never blocks the rest.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "gita.txt")

VAL_LOSS_THRESHOLD = 2.0

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""


# ---------------------------------------------------------------------------
# tiny test harness
# ---------------------------------------------------------------------------

class Pending(Exception):
    """Raised by a check whose prerequisite hasn't happened yet (e.g. no
    checkpoint because you haven't trained). Reported as SKIP, not FAIL."""


class Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def ok(self, name: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  {GREEN}PASS{RESET} {name}" + (f" {DIM}({detail}){RESET}" if detail else ""))

    def fail(self, name: str, msg: str) -> None:
        self.failed += 1
        print(f"  {RED}FAIL{RESET} {name}")
        for line in str(msg).rstrip().split("\n"):
            print(f"       {line}")

    def skip(self, name: str, msg: str) -> None:
        self.skipped += 1
        print(f"  {YELLOW}SKIP{RESET} {name} {DIM}({msg}){RESET}")


def check(res: Results, name: str, fn) -> None:
    """Run fn(); it returns a detail string, raises AssertionError, or raises
    NotImplementedError (which means 'not written yet' -> SKIP, not FAIL)."""
    try:
        detail = fn()
        res.ok(name, detail or "")
    except NotImplementedError:
        res.skip(name, "not implemented yet")
    except Pending as e:
        res.skip(name, str(e) or "prerequisite not met")
    except AssertionError as e:
        res.fail(name, e)
    except Exception:
        res.fail(name, traceback.format_exc())


def preflight() -> None:
    """Fail early and legibly if torch isn't importable by THIS interpreter.

    The common cause is an activated .venv whose `python` has been shadowed on
    PATH — by a pyenv/conda shim, or by an IDE auto-activating a different
    interpreter in the terminal after you sourced the venv. The prompt still
    says (.venv) while `python` is somebody else's.
    """
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    venv_py = os.path.join(HERE, ".venv", "bin", "python")
    print(f"{RED}torch is not importable by this interpreter.{RESET}\n")
    print(f"  running as   : {sys.executable}")
    print(f"  VIRTUAL_ENV  : {os.environ.get('VIRTUAL_ENV', '(unset)')}")
    print(f"  repo .venv   : {venv_py}"
          f"{'' if os.path.exists(venv_py) else '   <-- MISSING, run ./setup.sh'}")

    if os.path.exists(venv_py):
        if os.path.realpath(sys.executable) != os.path.realpath(venv_py):
            print(
                f"\n{YELLOW}Those first two lines disagree.{RESET} Your shell says the venv is\n"
                "active, but `python` is resolving somewhere else — usually a pyenv or\n"
                "conda shim, or an IDE that auto-activates its own interpreter in new\n"
                "terminals.\n\n"
                "Run it explicitly instead:\n"
                f"  {venv_py} checks.py --milestone <N>\n\n"
                "To fix it properly, find the shadow:\n"
                "  which -a python python3\n"
            )
        else:
            print(
                "\nThis IS the venv interpreter, so torch was never installed into it:\n"
                "  ./setup.sh\n"
            )
    raise SystemExit(2)


def run_sequence(res: "Results", steps, state) -> None:
    """Run checks in order; once one fails to produce what the rest need, list
    the remainder as blocked instead of dropping them silently.

    steps: list of (name, fn, state_key_it_must_produce_or_None)
    """
    blocked_by = None
    for name, fn, produces in steps:
        if blocked_by is not None:
            res.skip(name, f"blocked by {blocked_by}")
            continue
        check(res, name, fn)
        if produces is not None and produces not in state:
            blocked_by = name.split()[0]


def load_module(path: str):
    """Import the student's (or reference's) module from a file path."""
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    if not os.path.exists(path):
        raise SystemExit(f"module not found: {path}")
    name = os.path.splitext(os.path.basename(path))[0] + "_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# MILESTONE 0 — the data file itself. No model code involved.
# ---------------------------------------------------------------------------

def milestone_0(res: Results, mod=None) -> None:
    print(f"\n{'-' * 68}\nMILESTONE 0 — dataset\n{'-' * 68}")

    def exists():
        assert os.path.exists(DATA_PATH), (
            f"{DATA_PATH} is missing. Run: python prepare_data.py"
        )
        return DATA_PATH

    check(res, "data/gita.txt exists", exists)

    if not os.path.exists(DATA_PATH):
        return
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    def size():
        assert len(text) > 50_000, f"only {len(text)} chars; expected ~124k"
        return f"{len(text):,} characters"

    def boilerplate():
        low = text.lower()
        for marker in ("project gutenberg", "gutenberg-tm", "www.gutenberg.org"):
            assert marker not in low, (
                f"found {marker!r} in the corpus — header/footer stripping failed"
            )
        return "no Gutenberg boilerplate"

    def line_endings():
        assert "\r" not in text, "found a carriage return; line endings not normalized"
        return "\\n only"

    def blank_lines():
        runs = re.findall(r"\n+", text)
        longest = max(len(r) for r in runs)
        assert longest <= 3, (
            f"found a run of {longest} newlines "
            f"({longest - 1} blank lines); 3+ blank lines should collapse to 2"
        )
        return f"max {longest - 1} consecutive blank lines"

    def vocab():
        chars = sorted(set(text))
        assert 50 <= len(chars) <= 120, f"vocab size {len(chars)} looks wrong"
        assert "\n" in chars, "newline missing from vocab"
        return f"vocab size {len(chars)}"

    check(res, "corpus is a reasonable size", size)
    check(res, "Gutenberg header/footer stripped", boilerplate)
    check(res, "line endings normalized", line_endings)
    check(res, "3+ blank lines collapsed to 2", blank_lines)
    check(res, "character vocabulary looks sane", vocab)


# ---------------------------------------------------------------------------
# MILESTONE 1 — encode/decode, split, get_batch, bigram
# ---------------------------------------------------------------------------

def milestone_1(res: Results, mod) -> None:
    import torch

    print(f"\n{'-' * 68}\nMILESTONE 1 — tokenizer, split, get_batch, bigram\n{'-' * 68}")

    state = {}

    def load():
        text = mod.load_text()
        assert isinstance(text, str) and len(text) > 50_000, "load_text returned too little"
        state["text"] = text
        return f"{len(text):,} chars"

    def vocab():
        chars, vocab_size = mod.build_vocab(state["text"])
        assert isinstance(chars, (list, tuple)), "build_vocab must return a list of chars first"
        assert vocab_size == len(chars), "vocab_size must equal len(chars)"
        assert list(chars) == sorted(chars), "vocab must be sorted (determinism across runs)"
        assert set(chars) == set(state["text"]), "vocab must be exactly the distinct chars"
        state["chars"] = list(chars)
        state["vocab_size"] = vocab_size
        return f"V = {vocab_size}"

    def roundtrip():
        encode, decode = mod.make_encoder_decoder(state["chars"])
        text = state["text"]
        for sample in ("Arjuna", "\n", text[:500], text[-500:], text[10_000:10_137]):
            ids = encode(sample)
            assert all(isinstance(i, int) for i in ids), "encode must return plain ints"
            assert len(ids) == len(sample), (
                f"encode changed length: {len(sample)} chars -> {len(ids)} ids"
            )
            back = decode(ids)
            assert back == sample, (
                "decode(encode(s)) != s\n"
                f"  in : {sample[:60]!r}\n  out: {back[:60]!r}"
            )
        # must also survive a tensor round trip, which is how generate() uses it
        ids = encode(text[:200])
        assert decode(torch.tensor(ids).tolist()) == text[:200], (
            "decode failed on a list produced by tensor.tolist()"
        )
        state["encode"], state["decode"] = encode, decode
        return "decode(encode(s)) == s"

    def split():
        data = torch.tensor(state["encode"](state["text"]), dtype=torch.long)
        tr, va = mod.train_val_split(data)
        assert tr.dtype == torch.long and va.dtype == torch.long, "splits must be torch.long"
        assert len(tr) + len(va) == len(data), "split lengths must sum to the whole corpus"
        assert abs(len(tr) / len(data) - 0.9) < 0.01, "train split should be ~90%"
        assert torch.equal(tr, data[: len(tr)]), (
            "split must be contiguous (train = the first 90%), not shuffled"
        )
        state["train"], state["val"] = tr, va
        return f"train {len(tr):,} / val {len(va):,}"

    def batch_shapes():
        for split_name in ("train", "val"):
            x, y = mod.get_batch(split_name, state["train"], state["val"])
            assert x.shape == (mod.batch_size, mod.block_size), (
                f"x shape {tuple(x.shape)}, expected {(mod.batch_size, mod.block_size)}"
            )
            assert y.shape == x.shape, f"y shape {tuple(y.shape)} != x shape {tuple(x.shape)}"
            assert x.dtype == torch.long and y.dtype == torch.long, "batches must be torch.long"
            assert x.max().item() < state["vocab_size"], "batch contains an out-of-vocab index"
        return f"(B, T) = ({mod.batch_size}, {mod.block_size})"

    def batch_shift():
        x, y = mod.get_batch("train", state["train"], state["val"])
        assert torch.equal(y[:, :-1], x[:, 1:]), (
            "targets are not the inputs shifted by one:\n"
            "  y[b, t] must equal x[b, t+1] for every t < T-1"
        )
        # and y's final column must be a real continuation, not a repeat/pad
        assert not torch.equal(y[:, -1], x[:, -1]), (
            "y[:, -1] equals x[:, -1]; the last target looks like a copy, not the next char"
        )
        return "y[b,t] == x[b,t+1]"

    def batch_random():
        x1, _ = mod.get_batch("train", state["train"], state["val"])
        x2, _ = mod.get_batch("train", state["train"], state["val"])
        assert not torch.equal(x1, x2), "two consecutive batches are identical; offsets not random"
        return "offsets are random"

    def bigram_forward():
        V = state["vocab_size"]
        m = mod.BigramLanguageModel(V)
        x, y = mod.get_batch("train", state["train"], state["val"])
        m = m.to(x.device)
        logits, loss = m(x, y)
        assert logits.shape[-1] == V, f"last logits dim {logits.shape[-1]}, expected V={V}"
        assert logits.numel() == x.numel() * V, (
            f"logits have {logits.numel()} elements, expected B*T*V = {x.numel() * V}"
        )
        assert loss is not None and torch.isfinite(loss), f"loss is not finite: {loss}"
        # an untrained bigram should sit near ln(V)
        expected = float(torch.log(torch.tensor(float(V))))
        assert abs(loss.item() - expected) < 1.0, (
            f"initial loss {loss.item():.3f} is far from ln(V) = {expected:.3f}; "
            "the (V,V) table is probably not initialized as expected"
        )
        _, none_loss = m(x)
        assert none_loss is None, "forward(idx) with targets=None must return loss None"
        return f"loss {loss.item():.3f} ~ ln(V) {expected:.3f}"

    def bigram_generate():
        m = mod.BigramLanguageModel(state["vocab_size"]).to(mod.device)
        ctx = torch.zeros((1, 1), dtype=torch.long, device=mod.device)
        out = m.generate(ctx, max_new_tokens=50)
        assert out.shape == (1, 51), f"generate returned {tuple(out.shape)}, expected (1, 51)"
        assert torch.equal(out[:, :1], ctx), "generate must keep the seed context as a prefix"
        text = state["decode"](out[0].tolist())
        assert len(text) == 51, "decode of the generated sequence has the wrong length"
        return "(1,1) -> (1,51)"

    # Each entry: (name, fn, state key it must produce for the rest to run).
    # A missing key blocks everything downstream, so the remaining checks are
    # listed as blocked rather than silently omitted — you can see the whole
    # milestone from the first run.
    run_sequence(res, [
        ("load_text reads the corpus", load, "text"),
        ("build_vocab is sorted and complete", vocab, "chars"),
        ("encode/decode round-trips", roundtrip, "encode"),
        ("train_val_split is contiguous 90/10", split, "train"),
        ("get_batch shapes", batch_shapes, None),
        ("get_batch targets are inputs shifted by one", batch_shift, None),
        ("get_batch samples random offsets", batch_random, None),
        ("BigramLanguageModel forward", bigram_forward, None),
        ("BigramLanguageModel generate", bigram_generate, None),
    ], state)


# ---------------------------------------------------------------------------
# MILESTONE 2 — one attention head
# ---------------------------------------------------------------------------

def _capture_attention_weights(mod, head, x):
    """Run `head(x)` while intercepting softmax, to grab the (B,T,T) weights.

    We patch the three ways people spell softmax so this works whether you
    wrote F.softmax(...), torch.softmax(...) or wei.softmax(...).
    """
    import torch
    import torch.nn.functional as F

    captured = []
    orig_f = F.softmax
    orig_t = torch.softmax
    orig_m = torch.Tensor.softmax

    def wrap(orig):
        def inner(*a, **kw):
            out = orig(*a, **kw)
            if out.dim() == 3 and out.shape[-1] == out.shape[-2]:
                captured.append(out.detach().clone())
            return out
        return inner

    F.softmax = wrap(orig_f)
    torch.softmax = wrap(orig_t)
    torch.Tensor.softmax = wrap(orig_m)
    try:
        head(x)
    finally:
        F.softmax = orig_f
        torch.softmax = orig_t
        torch.Tensor.softmax = orig_m
    return captured[0] if captured else None


def milestone_2(res: Results, mod) -> None:
    import torch

    print(f"\n{'-' * 68}\nMILESTONE 2 — single self-attention head\n{'-' * 68}")

    B, T, C = 4, 16, mod.n_embd
    head_size = C // mod.n_head
    state = {}

    def build():
        h = mod.Head(head_size).to(mod.device)
        h.eval()  # dropout off — otherwise nothing below is deterministic
        state["head"] = h
        state["x"] = torch.randn(B, T, C, device=mod.device)
        return f"Head({head_size})"

    def shape():
        out = state["head"](state["x"])
        assert out.shape == (B, T, head_size), (
            f"Head output {tuple(out.shape)}, expected {(B, T, head_size)}"
        )
        return f"(B,T,C) -> (B,T,{head_size})"

    def short_context():
        # during generation T can be much smaller than block_size
        x_short = torch.randn(2, 3, C, device=mod.device)
        out = state["head"](x_short)
        assert out.shape == (2, 3, head_size), f"short-context output {tuple(out.shape)}"
        return "T=3 works (mask sliced, not fixed at block_size)"

    def causal_by_perturbation():
        """Black-box causality: changing the input at position t must not change
        the output at any position < t."""
        h, x = state["head"], state["x"]
        base = h(x)
        t = T // 2
        x2 = x.clone()
        x2[:, t, :] += 10.0
        pert = h(x2)
        before = (base[:, :t, :] - pert[:, :t, :]).abs().max().item()
        after = (base[:, t:, :] - pert[:, t:, :]).abs().max().item()
        assert before < 1e-5, (
            f"positions before t={t} changed by {before:.3e} when the input at t changed.\n"
            "  Information is leaking backwards from the future — the causal mask is\n"
            "  missing, applied after the softmax instead of before, or transposed."
        )
        assert after > 1e-5, (
            "positions from t onward did NOT change; the head appears to ignore its input"
        )
        return f"no leakage (max delta {before:.2e})"

    def weights_are_causal_and_normalized():
        wei = _capture_attention_weights(mod, state["head"], state["x"])
        if wei is None:
            raise AssertionError(
                "could not intercept the (B,T,T) softmax output.\n"
                "  The perturbation test above already covers causality; this check\n"
                "  additionally wants the attention matrix itself. If you computed it\n"
                "  in an unusual way this may be a false alarm."
            )
        assert wei.shape == (B, T, T), f"attention weights {tuple(wei.shape)}, expected {(B,T,T)}"

        # strictly upper triangle must be exactly zero
        upper = torch.triu(torch.ones(T, T, device=wei.device), diagonal=1).bool()
        leak = wei.masked_select(upper.expand(B, T, T)).abs().max().item()
        assert leak < 1e-6, (
            f"attention weight to a FUTURE position is {leak:.3e}, should be 0.\n"
            "  Mask with -inf BEFORE the softmax, not by zeroing after it."
        )

        # every row must sum to 1
        sums = wei.sum(dim=-1)
        err = (sums - 1.0).abs().max().item()
        assert err < 1e-5, (
            f"attention rows sum to as far as {err:.3e} from 1.0.\n"
            "  softmax must be over the LAST dim (dim=-1), i.e. across keys."
        )
        # row 0 attends only to itself
        assert abs(wei[0, 0, 0].item() - 1.0) < 1e-5, (
            "position 0 must attend entirely to itself (it has no past)"
        )
        state["wei"] = wei
        return f"upper-triangle max {leak:.1e}, row-sum error {err:.1e}"

    def scaled():
        """Reconstruct the attention matrix from k/q and compare exactly.

        Rather than eyeballing how peaked the softmax looks, recompute what the
        weights *should* be from the head's own key/query projections, both with
        and without the 1/sqrt(head_size) factor, and see which one you produced.
        """
        import torch.nn.functional as Fn

        wei = state.get("wei")
        if wei is None:
            raise Pending("needs the attention-weights check above to pass first")
        h, x = state["head"], state["x"]
        with torch.no_grad():
            k, q = h.key(x), h.query(x)
        hs = k.shape[-1]
        raw = q @ k.transpose(-2, -1)
        tril = torch.tril(torch.ones(T, T, device=x.device))

        def masked_softmax(w):
            return Fn.softmax(w.masked_fill(tril == 0, float("-inf")), dim=-1)

        want = masked_softmax(raw * hs ** -0.5)
        unscaled = masked_softmax(raw)

        if torch.allclose(wei, want, atol=1e-5):
            return f"matches softmax(QK^T / sqrt({hs}))"
        if torch.allclose(wei, unscaled, atol=1e-5):
            raise AssertionError(
                "your attention weights match softmax(QK^T) exactly — the "
                f"1/sqrt(head_size) = 1/sqrt({hs}) factor is missing.\n"
                "  Without it the dot products grow with head_size, the softmax\n"
                "  saturates toward one-hot, and each position reads from a single\n"
                "  other position instead of blending. Video ~1:16:56."
            )
        diff = (wei - want).abs().max().item()
        raise AssertionError(
            f"attention weights differ from softmax(QK^T/sqrt({hs})) by up to {diff:.3e},\n"
            "  and they don't match the unscaled version either. Check the order of\n"
            "  operations: scale, then mask with -inf, then softmax over dim=-1."
        )

    def no_bias():
        h = state["head"]
        for attr in ("key", "query", "value"):
            lin = getattr(h, attr, None)
            if lin is None:
                raise AssertionError(f"Head has no attribute {attr!r} (expected an nn.Linear)")
            assert lin.bias is None, f"Head.{attr} should be nn.Linear(..., bias=False)"
        return "key/query/value have bias=False"

    def buffer_not_param():
        h = state["head"]
        names = [n for n, _ in h.named_parameters()]
        assert not any("tril" in n for n in names), (
            "the causal mask is registered as a Parameter; it should be a buffer\n"
            "  (self.register_buffer('tril', ...)) so it is not trained"
        )
        assert any("tril" in n for n, _ in h.named_buffers()), (
            "no 'tril' buffer found; register the mask with self.register_buffer"
        )
        return "tril is a buffer"

    run_sequence(res, [
        ("Head constructs", build, "head"),
        ("Head output shape", shape, None),
        ("Head handles T < block_size", short_context, None),
        ("attention is causal (no backward leakage)", causal_by_perturbation, None),
        ("attention weights: future = 0, rows sum to 1",
         weights_are_causal_and_normalized, None),
        ("affinities are scaled by 1/sqrt(head_size)", scaled, None),
        ("k/q/v projections are bias-free", no_bias, None),
        ("causal mask is a buffer, not a parameter", buffer_not_param, None),
    ], state)


# ---------------------------------------------------------------------------
# MILESTONE 3 — multi-head, block, full model
# ---------------------------------------------------------------------------

def milestone_3(res: Results, mod) -> None:
    import torch

    print(f"\n{'-' * 68}\nMILESTONE 3 — multi-head, block, full GPT\n{'-' * 68}")

    B, T, C = 4, 16, mod.n_embd
    head_size = C // mod.n_head
    x = torch.randn(B, T, C, device=mod.device)
    state = {}

    # Deliberately small: a full (batch_size, block_size) batch with gradients
    # enabled needs several GB, and these checks are about correctness, not
    # throughput. Training uses the real sizes.
    SB, ST = 4, min(32, mod.block_size)

    def small_batch(data, b=SB, t=ST):
        ix = torch.randint(len(data) - t - 1, (b,))
        xb = torch.stack([data[i:i + t] for i in ix]).to(mod.device)
        yb = torch.stack([data[i + 1:i + t + 1] for i in ix]).to(mod.device)
        return xb.contiguous(), yb.contiguous()

    def mha():
        m = mod.MultiHeadAttention(mod.n_head, head_size).to(mod.device).eval()
        out = m(x)
        assert out.shape == (B, T, C), (
            f"MultiHeadAttention output {tuple(out.shape)}, expected {(B, T, C)}\n"
            "  n_head * head_size must equal n_embd, then project back to n_embd"
        )
        assert len(list(m.parameters())) > mod.n_head, "heads may not be in an nn.ModuleList"
        return f"{mod.n_head} heads x {head_size}"

    def ffwd():
        m = mod.FeedForward(C).to(mod.device).eval()
        out = m(x)
        assert out.shape == (B, T, C), f"FeedForward output {tuple(out.shape)}, expected {(B,T,C)}"
        widths = [p.shape[0] for p in m.parameters() if p.dim() == 2]
        assert 4 * C in widths, (
            f"expected a hidden width of 4*n_embd = {4 * C}; found linear output dims {widths}"
        )
        return f"{C} -> {4 * C} -> {C}"

    def block():
        m = mod.Block(C, mod.n_head).to(mod.device).eval()
        out = m(x)
        assert out.shape == (B, T, C), f"Block output {tuple(out.shape)}, expected {(B,T,C)}"
        lns = [mm for mm in m.modules() if isinstance(mm, torch.nn.LayerNorm)]
        assert len(lns) >= 2, (
            f"found {len(lns)} LayerNorm(s) in Block; you need two, one per sublayer"
        )
        # residual: with tiny weights the block should still roughly pass x through
        assert (out - x).abs().mean() < x.abs().mean(), (
            "the output looks unrelated to the input; is the residual connection missing?"
        )
        return "2 sublayers, 2 LayerNorms, residual present"

    def gpt_forward():
        text = mod.load_text()
        chars, V = mod.build_vocab(text)
        encode, decode = mod.make_encoder_decoder(chars)
        data = torch.tensor(encode(text), dtype=torch.long)
        tr, va = mod.train_val_split(data)
        state.update(V=V, train=tr, val=va, decode=decode)

        m = mod.GPTLanguageModel(V).to(mod.device)
        m.eval()
        state["model"] = m
        xb, yb = small_batch(tr)
        with torch.no_grad():
            logits, loss = m(xb, yb)
        assert logits.shape[-1] == V, f"logits last dim {logits.shape[-1]}, expected V={V}"
        assert logits.numel() == xb.numel() * V, (
            f"logits have {logits.numel()} elements, expected B*T*V = {xb.numel() * V}"
        )
        assert loss is not None and torch.isfinite(loss), f"loss is not finite: {loss}"
        expected = float(torch.log(torch.tensor(float(V))))
        assert abs(loss.item() - expected) < 1.5, (
            f"initial loss {loss.item():.3f} is far from ln(V) = {expected:.3f}; "
            "check the 0.02-std weight init"
        )
        nparams = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            _, none_loss = m(xb)
        assert none_loss is None, "forward with targets=None must return loss None"
        return f"{nparams / 1e6:.2f}M params, init loss {loss.item():.3f} ~ ln(V) {expected:.3f}"

    def gradients_flow():
        """One real backward pass, tiny, to confirm every parameter is reachable."""
        m = state["model"]
        m.train()
        xb, yb = small_batch(state["train"], b=2, t=16)
        _, loss = m(xb, yb)
        loss.backward()
        dead = [n for n, p in m.named_parameters()
                if p.grad is None or not torch.isfinite(p.grad).all()]
        m.zero_grad(set_to_none=True)
        m.eval()
        assert not dead, (
            "these parameters got no gradient (or a non-finite one), so they are "
            f"disconnected from the loss:\n  {dead[:8]}"
        )
        return f"all {sum(1 for _ in m.parameters())} parameter tensors receive gradients"

    def positions_matter():
        m = state["model"]
        assert hasattr(m, "position_embedding_table"), "no position_embedding_table"
        pe = m.position_embedding_table
        assert pe.num_embeddings == mod.block_size, (
            f"position table has {pe.num_embeddings} rows, expected block_size={mod.block_size}"
        )
        # Direct test: perturb the position table. If its values genuinely feed
        # the forward pass, the logits must move. If the table was built but
        # never added to the token embeddings, nothing changes.
        m.eval()
        idx = torch.randint(0, state["V"], (1, 8), device=mod.device)
        with torch.no_grad():
            a, _ = m(idx)
            saved = pe.weight.detach().clone()
            pe.weight.add_(torch.randn_like(pe.weight) * 2.0)
            b, _ = m(idx)
            pe.weight.copy_(saved)
        delta = (a - b).abs().max().item()
        assert delta > 1e-4, (
            "perturbing position_embedding_table changed the logits by "
            f"{delta:.2e} — i.e. not at all.\n"
            "  The table exists but is never added to the token embeddings, so the\n"
            "  model cannot tell position 0 from position 7. Video ~1:00:18."
        )
        return f"table {pe.num_embeddings} x {pe.embedding_dim}, perturbing it moves logits"

    def gpt_generate():
        m = state["model"]
        ctx = torch.zeros((1, 1), dtype=torch.long, device=mod.device)
        # deliberately longer than block_size, to catch a missing context crop
        n = mod.block_size + 20
        out = m.generate(ctx, max_new_tokens=n)
        assert out.shape == (1, 1 + n), f"generate returned {tuple(out.shape)}, expected {(1, 1+n)}"
        assert out.max().item() < state["V"], "generate produced an out-of-vocab index"
        text = state["decode"](out[0].tolist())
        assert isinstance(text, str) and len(text) == 1 + n
        return f"generated {n} chars past block_size={mod.block_size} without crashing"

    run_sequence(res, [
        ("MultiHeadAttention shape", mha, None),
        ("FeedForward shape and 4x expansion", ffwd, None),
        ("Block shape, norms, residual", block, None),
        ("GPTLanguageModel forward: logits shape + finite loss", gpt_forward, "model"),
        ("positional embeddings actually used", positions_matter, None),
        ("gradients reach every parameter", gradients_flow, None),
        ("generate crops context to block_size", gpt_generate, None),
    ], state)


# ---------------------------------------------------------------------------
# MILESTONE 4 — did it actually learn?
# ---------------------------------------------------------------------------

def milestone_4(res: Results, mod) -> None:
    import torch

    print(f"\n{'-' * 68}\nMILESTONE 4 — trained model quality\n{'-' * 68}")

    ckpt_path = getattr(mod, "CKPT_PATH", None)
    state = {}

    def have_ckpt():
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise Pending("no checkpoint yet — train first: python gita_gpt.py")
        ck = torch.load(ckpt_path, map_location=mod.device, weights_only=False)
        assert isinstance(ck, dict) and "model_state_dict" in ck, (
            "checkpoint should be a dict with a 'model_state_dict' key "
            "(and ideally the vocab alongside it)"
        )
        state["ckpt"] = ck
        return os.path.relpath(ckpt_path, HERE)

    def val_loss():
        text = mod.load_text()
        chars, V = mod.build_vocab(text)
        encode, decode = mod.make_encoder_decoder(chars)
        data = torch.tensor(encode(text), dtype=torch.long)
        tr, va = mod.train_val_split(data)

        m = mod.GPTLanguageModel(V).to(mod.device)
        m.load_state_dict(state["ckpt"]["model_state_dict"])
        m.eval()

        losses = []
        with torch.no_grad():
            for _ in range(20):
                xb, yb = mod.get_batch("val", tr, va)
                _, loss = m(xb, yb)
                losses.append(loss.item())
        val = sum(losses) / len(losses)
        state["val"] = val
        state["model"], state["decode"] = m, decode
        assert val < VAL_LOSS_THRESHOLD, (
            f"val loss {val:.4f} is not below the {VAL_LOSS_THRESHOLD} sanity threshold.\n"
            "  Train longer, or check that the loss printed during training was falling."
        )
        return f"val loss {val:.4f} < {VAL_LOSS_THRESHOLD}"

    def sample_looks_like_english():
        m, decode = state["model"], state["decode"]
        ctx = torch.zeros((1, 1), dtype=torch.long, device=mod.device)
        out = decode(m.generate(ctx, max_new_tokens=600)[0].tolist())
        words = [w for w in re.split(r"[^A-Za-z]+", out) if w]
        assert len(words) > 40, "the sample has almost no word-like runs of letters"
        avg = sum(len(w) for w in words) / len(words)
        assert 2.0 < avg < 9.0, f"average 'word' length {avg:.1f} looks degenerate"
        print(f"{DIM}       --- sample ---{RESET}")
        for line in out.strip().split("\n")[:8]:
            print(f"{DIM}       {line[:70]}{RESET}")
        return f"{len(words)} words, avg length {avg:.1f}"

    run_sequence(res, [
        ("checkpoint exists and is well-formed", have_ckpt, "ckpt"),
        (f"val loss below {VAL_LOSS_THRESHOLD}", val_loss, "model"),
        ("samples look like text", sample_looks_like_english, None),
    ], state)


# ---------------------------------------------------------------------------

MILESTONES = {0: milestone_0, 1: milestone_1, 2: milestone_2, 3: milestone_3, 4: milestone_4}


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone checks for gita_gpt.py")
    parser.add_argument("--milestone", "-m", default="all",
                        help="0, 1, 2, 3, 4, or 'all' (default)")
    parser.add_argument("--module", default="gita_gpt.py",
                        help="path to the module under test (default: gita_gpt.py)")
    args = parser.parse_args()

    if args.milestone == "all":
        which = [0, 1, 2, 3, 4]
    else:
        try:
            which = [int(args.milestone)]
        except ValueError:
            raise SystemExit(f"--milestone must be 0-4 or 'all', got {args.milestone!r}")
        if which[0] not in MILESTONES:
            raise SystemExit(f"no milestone {which[0]}; valid: 0-4")

    mod = None
    if which != [0]:
        preflight()
        mod = load_module(args.module)
        print(f"module under test : {args.module}")
        print(f"device            : {getattr(mod, 'device', '?')}")

    res = Results()
    for n in which:
        MILESTONES[n](res, mod)

    print(f"\n{'=' * 68}")
    parts = [f"{GREEN}{res.passed} passed{RESET}"]
    if res.failed:
        parts.append(f"{RED}{res.failed} failed{RESET}")
    if res.skipped:
        parts.append(f"{YELLOW}{res.skipped} not implemented yet{RESET}")
    print("  " + ", ".join(parts))
    print("=" * 68)
    sys.exit(1 if res.failed else 0)


if __name__ == "__main__":
    main()
