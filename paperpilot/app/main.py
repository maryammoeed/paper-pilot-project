"""FastAPI app. Stateless by design so it runs on serverless hosts (Vercel) as well as anywhere else.

POST /api/start  -> starts a review and advances it to the first pause / model call
POST /api/step   -> continues a review (optionally with the human's answer)
Every response returns the run's signed state; the browser sends it back with the next step.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import settings
from .engine import Engine, new_state, sign, verify
from .ingest import pdf_to_text
from .llm import MockLLM, build_llm
from .models import Autonomy

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("paperpilot")
HERE = Path(__file__).parent
ROOT = HERE.parent
SAMPLES = {p.stem: p for p in sorted((HERE / "samples").glob("*.md"))}
MAX_BYTES = 4 * 1024 * 1024          # Vercel caps request bodies at 4.5 MB
MAX_WORDS = 20000

app = FastAPI(title="PaperPilot", version="2.0.0")
demo_engine = Engine(MockLLM())
# "hosted" = the host's own key (e.g. a Groq key set as an environment variable on Vercel)
hosted_llm = None if settings.provider == "mock" else build_llm()
hosted_engine = Engine(hosted_llm) if hosted_llm else None


class FairUse:
    """Protects the host's key: a few hosted reviews per visitor per hour, and a daily cap overall.
    Counters live in memory, so on serverless hosts they are best-effort per instance."""

    def __init__(self, per_hour: int, per_day: int):
        self.per_hour, self.per_day = per_hour, per_day
        self.by_ip: dict[str, deque] = defaultdict(deque)
        self.day, self.today = date.today(), 0

    def left_today(self) -> int:
        if date.today() != self.day:
            self.day, self.today = date.today(), 0
        return max(0, self.per_day - self.today)

    def take(self, ip: str) -> str | None:
        if self.left_today() == 0:
            return "Today's free AI reviews are used up. Try the demo, or come back tomorrow."
        q, now = self.by_ip[ip], time.time()
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= self.per_hour:
            mins = int((3600 - (now - q[0])) // 60) + 1
            return (f"You've used your {self.per_hour} free AI reviews for this hour "
                    f"(next one in about {mins} min). Try the demo meanwhile.")
        q.append(now)
        self.today += 1
        return None


fair_use = FairUse(settings.hosted_per_hour, settings.hosted_per_day)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def _respond(st: dict, events: list[dict]) -> dict:
    wait = next((e["seconds"] for e in events if e["type"] == "wait"), None)
    for e in events:
        if e["type"] not in ("step", "wait"):
            log.info({"type": e["type"], "t": e["t"], "stage": e.get("stage"), "checkpoint": e.get("checkpoint")})
    return {"status": "rate_limited" if wait else st["status"], "retry_after": wait,
            "events": [e for e in events if e["type"] != "wait"], "pending": st["pending"],
            "report": st["report"] if st["status"] == "done" else None, "state": st, "sig": sign(st)}


@app.get("/")
def index():
    return FileResponse(ROOT / "public" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "hosted": bool(hosted_llm), "hosted_model": hosted_llm.name if hosted_llm else None,
            "hosted_left_today": fair_use.left_today() if hosted_llm else 0}


@app.get("/api/samples")
def samples():
    return [{"id": k, "title": p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()}
            for k, p in SAMPLES.items()]


@app.post("/api/start")
async def start(
    request: Request,
    autonomy: Autonomy = Form(Autonomy.BALANCED),
    mode: str = Form("auto"),          # demo | hosted | auto (hosted when configured)
    sample: str | None = Form(None),
    text: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    if file and file.filename:
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File is larger than 4 MB.")
        content = pdf_to_text(data) if file.filename.lower().endswith(".pdf") else data.decode("utf-8", "ignore")
    elif sample:
        if sample not in SAMPLES:
            raise HTTPException(404, "Unknown sample.")
        content = SAMPLES[sample].read_text(encoding="utf-8")
    elif text and text.strip():
        content = text
    else:
        raise HTTPException(400, "Provide a sample, a file or some text.")
    words = content.split()
    if len(words) < 80:
        raise HTTPException(400, "That paper is too short to review (need at least ~80 words).")
    if len(words) > MAX_WORDS:
        raise HTTPException(413, f"That paper is very long; please upload one under {MAX_WORDS:,} words.")

    if mode == "auto":
        mode = "hosted" if hosted_llm else "demo"
    if mode == "hosted":
        if not hosted_engine:
            raise HTTPException(400, "Real AI isn't configured on this server. Choose the demo.")
        limit = fair_use.take(_client_ip(request))
        if limit:
            raise HTTPException(429, limit)
        engine = hosted_engine
    else:
        mode, engine = "demo", demo_engine

    st = new_state(content, autonomy, mode, engine.llm.name)
    st, events = await engine.advance(st)
    return _respond(st, events)


class StepIn(BaseModel):
    state: dict
    sig: str
    answer: str | None = None


@app.post("/api/step")
async def step(body: StepIn):
    st = body.state
    if not verify(st, body.sig):
        raise HTTPException(400, "This review's progress could not be verified. Please start again.")
    if st.get("status") not in ("running", "waiting"):
        raise HTTPException(409, "This review has already finished.")
    engine = hosted_engine if st.get("mode") == "hosted" else demo_engine
    if engine is None:
        raise HTTPException(400, "Real AI is no longer configured on this server. Choose the demo.")
    try:
        st, events = await engine.advance(st, body.answer)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _respond(st, events)
