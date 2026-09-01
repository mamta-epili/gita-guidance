"""
main.py — a local server that makes the char-GPT visible.

    make run     →  http://127.0.0.1:8000

Endpoints
---------
GET  /                  the single-page UI
GET  /api/info          architecture, device, vocabulary
POST /api/step          one forward pass: next-char distribution + attention
GET  /api/stream        server-sent events: generate character by character

No paid APIs, no external calls, no CDN. The model runs on this machine.
"""

from __future__ import annotations

import json
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .guidance import Guidance, NotReady
from .model import CharGPT, DEFAULT_CKPT

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
ROOT = os.path.dirname(HERE)
NG_DIST = os.path.join(ROOT, "frontend", "dist", "frontend", "browser")

app = FastAPI(title="char-GPT inspector", docs_url="/api/docs")

# The Angular dev server runs on :4200 and proxies /api here, so same-origin
# requests are the normal path. CORS is only needed if you point a differently
# hosted frontend at this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_gpt: CharGPT | None = None
_guide: Guidance | None = None


def gpt() -> CharGPT:
    """Load the char-GPT once, lazily, so import doesn't block startup."""
    global _gpt
    if _gpt is None:
        _gpt = CharGPT(os.environ.get("CHARGPT_CKPT", DEFAULT_CKPT))
    return _gpt


def guide() -> Guidance:
    """Load the corpus + embedding index once, lazily.

    Lazy matters more here: the embedding model is ~2 GB, and the char-GPT
    inspector must still work on a machine where `make embed` was never run.
    """
    global _guide
    if _guide is None:
        _guide = Guidance(public_only=bool(os.environ.get("PUBLIC_BUILD")))
    return _guide


@app.get("/")
def index():
    """Serve the built Angular app if present, else the no-build fallback page.

    The fallback exists so the backend is useful on its own — no node, no
    install step, nothing to build. It is also the quickest way to tell whether
    a problem is in the model or in the frontend.
    """
    if os.environ.get("SERVE_ANGULAR") and os.path.exists(os.path.join(NG_DIST, "index.html")):
        return FileResponse(os.path.join(NG_DIST, "index.html"))
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/info")
def info():
    g = gpt()
    d = g.info()
    d["ln_vocab"] = round(float(__import__("math").log(d["vocab_size"])), 3)
    return d


class GuidanceIn(BaseModel):
    question: str = Field(default="", max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


@app.post("/api/guidance")
def guidance(body: GuidanceIn):
    """Question in, shlokas out. Retrieval only — nothing here writes text."""
    try:
        g = guide()
    except NotReady as e:
        # 503 rather than 500: the service is fine, the index just isn't built.
        # The message carries the exact command to fix it.
        raise HTTPException(status_code=503, detail=str(e))
    t0 = time.perf_counter()
    out = g.ask(body.question, k=body.k)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


@app.get("/api/guidance/ready")
def guidance_ready():
    """Whether retrieval is available, so the UI can say why it isn't."""
    try:
        g = guide()
        return {
            "ready": True,
            "verses": len(g.verses),
            "model": g.retriever.model_name,
            "dim": g.retriever.dim,
            "rows": int(len(g.retriever.ids)),
        }
    except NotReady as e:
        return {"ready": False, "reason": str(e)}


class StepIn(BaseModel):
    text: str = Field(default="", max_length=4000)
    temperature: float = Field(default=1.0, ge=0.05, le=3.0)
    attention: bool = True


@app.post("/api/step")
def step(body: StepIn):
    g = gpt()
    cleaned = g.clean(body.text)
    t0 = time.perf_counter()
    out = g.step(cleaned, temperature=body.temperature, want_attn=body.attention)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["cleaned"] = cleaned
    out["dropped_chars"] = len(body.text) - len(cleaned)
    # The characters the model can actually see, for the context strip.
    out["context_chars"] = list(cleaned[-g.block_size:])
    return out


@app.get("/api/stream")
def stream(prompt: str = "", n: int = 240, temperature: float = 1.0, delay: float = 0.012):
    """Generate `n` characters, streaming each one as it is sampled.

    Streaming is not decoration — it is what the model actually does. One
    forward pass over the whole context produces exactly one character, then
    the context grows by one and the whole thing runs again. Watching it arrive
    at that speed is watching 700-odd forward passes.
    """
    g = gpt()
    n = max(1, min(n, 2000))

    def events():
        text = g.clean(prompt)
        yield f"data: {json.dumps({'type': 'start', 'prompt': text})}\n\n"
        for i in range(n):
            ch = g.sample_one(text, temperature=temperature)
            text += ch
            payload = {
                "type": "char",
                "char": ch,
                "i": i,
                "cropped": max(0, len(g.encode(text)) - g.block_size),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            if delay:
                time.sleep(delay)
        yield f"data: {json.dumps({'type': 'done', 'text': text})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Angular's hashed bundles, mounted last so /api/* and / win over the catch-all.
if os.path.isdir(NG_DIST):
    app.mount("/", StaticFiles(directory=NG_DIST, html=True), name="angular")
