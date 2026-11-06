"""Resumable step engine.

The whole review is a state machine whose state is plain JSON. Each call to `advance()` moves the
run forward until it has made at most ONE model call, reached a human checkpoint, or finished.
That makes every request short (it fits serverless platforms such as Vercel) and makes runs
resumable: the state is handed back to the caller, signed so it cannot be tampered with, and sent
back with the next request, much like a message on a queue between a platform and a worker.

Stages: input guardrails -> [input checkpoint] -> extractor -> critic -> [claims checkpoint]
        -> writer -> output guardrails -> [verdict checkpoint] -> done
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

import httpx

from . import autonomy
from .agents import build_prompt
from .config import settings
from .evaluator import evaluate
from .guardrails import check_input, check_output
from .ingest import parse
from .llm import LLM, RateLimited, extract_json
from .models import Autonomy
from .retrieval import BM25

AGENT_NEXT = {"extractor": "critic", "critic": "claims_checkpoint", "writer": "output_guardrails"}


# ------------------------------------------------------------------------------------------------
# state signing
# ------------------------------------------------------------------------------------------------
def _secret() -> bytes:
    explicit = os.getenv("STATE_SECRET")
    if explicit:
        return explicit.encode()
    # derived from the host's key so it is stable across serverless instances but never exposed
    return hashlib.sha256(("paperpilot|" + settings.api_key).encode()).digest()


def sign(state: dict) -> str:
    body = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_secret(), body, hashlib.sha256).hexdigest()


def verify(state: dict, sig: str) -> bool:
    return hmac.compare_digest(sign(state), sig or "")


# ------------------------------------------------------------------------------------------------
def new_state(text: str, level: Autonomy, mode: str, model_name: str) -> dict:
    return {"v": 1, "t0": time.time(), "mode": mode, "model": model_name, "autonomy": level.value,
            "text": text, "stage": "start", "status": "running", "findings": [], "answers": {},
            "attempt": 1, "feedback": [], "retries_used": 0, "write_round": 0,
            "claims": None, "critique": None, "report": None, "pending": None}


class Engine:
    def __init__(self, llm: LLM, max_retries: int | None = None):
        self.llm = llm
        self.max_retries = max_retries or settings.max_retries

    async def advance(self, st: dict, answer: str | None = None) -> tuple[dict, list[dict]]:
        events: list[dict] = []

        def pub(type_: str, **data: Any) -> None:
            events.append({"type": type_, "t": round(time.time() - st["t0"], 2), **data})

        level = Autonomy(st["autonomy"])
        try:
            if st["status"] == "waiting":
                if answer not in (st["pending"] or {}).get("options", []):
                    raise ValueError("answer must be one of the offered options")
                self._apply_answer(st, answer, pub)
            elif answer is not None:
                raise ValueError("this run is not waiting for an answer")

            called_model = False
            while st["status"] == "running":
                stage = st["stage"]
                if stage == "start":
                    pub("start", model=st["model"], autonomy=level.value, max_retries=self.max_retries)
                    pub("step", stage="input_guardrails", status="start")
                    guard = check_input(st["text"])
                    st["text"], st["findings"] = guard.text, guard.findings
                    pub("step", stage="input_guardrails", status="done", findings=guard.findings)
                    self._checkpoint(st, autonomy.input_checkpoint(level, guard.findings), pub)

                elif stage in ("extractor", "critic", "writer"):
                    if called_model:
                        break                               # one model call per request
                    paper = parse(st["text"])
                    if not paper.chunks:
                        self._stop(st, pub, "No readable text found in this paper.")
                        break
                    index = BM25(paper.chunks)
                    if stage == "extractor" and st["attempt"] == 1:
                        pub("step", stage="knowledge_base", status="done", title=paper.title,
                            chunks=len(paper.chunks), sections=sorted({c.section for c in paper.chunks}))
                    called_model = True
                    await self._agent_attempt(st, stage, paper, index, pub)

                elif stage == "claims_checkpoint":
                    claims = st["claims"]["claims"]
                    self._checkpoint(st, autonomy.claims_checkpoint(level, claims, st["retries_used"]), pub)

                elif stage == "output_guardrails":
                    pub("step", stage="output_guardrails", status="start")
                    guard = check_output(json.dumps(st["report"]))
                    pub("step", stage="output_guardrails", status="done", findings=guard.findings)
                    if guard.findings and any(f["type"] == "toxicity" for f in guard.findings):
                        if st["write_round"] >= 1:
                            self._stop(st, pub, "The review still contained unprofessional language "
                                                "after a rewrite, so it was not released.")
                            break
                        st.update(write_round=1, stage="writer", attempt=1, report=None,
                                  feedback=["Rewrite without unprofessional language: "
                                            + ", ".join(f["detail"] for f in guard.findings)])
                        continue
                    if guard.findings:                       # secret leak: release the redacted text
                        st["report"] = json.loads(guard.text)
                    r = st["report"]
                    majors = sum(w["severity"] == "major" for w in r["weaknesses"])
                    self._checkpoint(st, autonomy.verdict_checkpoint(level, r["recommendation"], r["confidence"], majors), pub)

                elif stage == "finalize":
                    r = st["report"]
                    if st["findings"] and st["answers"].get("input") == "Remove it and mention it in the review":
                        r["integrity_note"] = ("The submission contained hidden instructions addressed to automated "
                                               "reviewers. They were removed and did not influence this review.")
                    r["title"] = parse(st["text"]).title
                    st["status"], st["stage"] = "done", "done"
                    pub("done", report=r)
        except RateLimited as e:
            pub("wait", seconds=round(e.wait, 1))            # the client waits, then resumes this state
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            hint = {401: "the API key was rejected.", 403: "the API key is not allowed to use this model.",
                    404: "the model was not found. It may have been retired.",
                    429: "the provider's daily limit was reached. Try the demo, or come back later."}
            st["status"] = "error"
            pub("error", message=f"Model provider error {code}: {hint.get(code, 'the request failed.')}")
        except httpx.TransportError:
            st["status"] = "error"
            pub("error", message="Could not reach the model provider. Check the connection and try again.")
        return st, events

    # --------------------------------------------------------------------------------------------
    async def _agent_attempt(self, st: dict, agent: str, paper, index, pub) -> None:
        ctx = {"claims": st["claims"], "critique": st["critique"], "answers": st["answers"], "feedback": st["feedback"]}
        pub("step", stage=agent, status="start", attempt=st["attempt"])
        t0 = time.time()
        system, user = build_prompt(agent, paper, index, ctx)
        try:
            raw = await self.llm.generate(agent, system, user, {**ctx, "paper": paper, "index": index})
            obj, problems = evaluate(agent, extract_json(raw), paper)
        except (httpx.HTTPStatusError, httpx.TransportError, RateLimited):
            raise
        except Exception as e:
            obj, problems = None, [f"could not parse model output ({type(e).__name__})"]
        ms = int((time.time() - t0) * 1000)
        if not problems:
            pub("eval", stage=agent, attempt=st["attempt"], passed=True, ms=ms)
            key = {"extractor": "claims", "critic": "critique", "writer": "report"}[agent]
            st[key] = obj.model_dump()
            if agent != "writer":
                st["retries_used"] += st["attempt"] - 1
            st.update(stage=AGENT_NEXT[agent], attempt=1, feedback=[])
            return
        pub("eval", stage=agent, attempt=st["attempt"], passed=False, problems=problems, ms=ms)
        if st["attempt"] >= self.max_retries:
            self._stop(st, pub, f"The {agent} could not produce output that passes the evaluator after "
                                f"{self.max_retries} attempts, so I stopped instead of guessing.")
            return
        st["attempt"] += 1
        st["feedback"] = problems

    def _checkpoint(self, st: dict, decision, pub) -> None:
        d = decision.model_dump()
        if decision.action == "stop":
            pub("decision", **d)
            self._stop(st, pub, "; ".join(decision.reasons))
        elif decision.action == "proceed":
            pub("decision", **d)
            self._apply(st, decision.checkpoint, autonomy.DEFAULTS[decision.checkpoint], pub)
        else:
            st["pending"], st["status"] = d, "waiting"
            pub("ask", **d)

    def _apply_answer(self, st: dict, answer: str, pub) -> None:
        checkpoint = st["pending"]["checkpoint"]
        st["pending"], st["status"] = None, "running"
        pub("answer", checkpoint=checkpoint, answer=answer)
        self._apply(st, checkpoint, answer, pub)

    def _apply(self, st: dict, checkpoint: str, answer: str, pub) -> None:
        if checkpoint == "input":
            st["answers"]["input"] = answer
            if answer == "Stop the review":
                self._stop(st, pub, "Stopped by the reviewer at the input checkpoint.")
            else:
                st["stage"] = "extractor"
        elif checkpoint == "claims":
            st["answers"]["unsupported_claims"] = "flag" if answer.startswith("Yes") else "ignore"
            st["stage"] = "writer"
        elif checkpoint == "verdict":
            if answer.startswith("Change to "):
                st["report"]["recommendation"] = answer.removeprefix("Change to ")
                st["report"]["human_override"] = True
            st["stage"] = "finalize"

    @staticmethod
    def _stop(st: dict, pub, reason: str) -> None:
        st["status"], st["stage"] = "stopped", "stopped"
        pub("stopped", reason=reason)
