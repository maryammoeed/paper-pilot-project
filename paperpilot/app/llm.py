"""LLM clients.

`OpenAICompatibleLLM` talks to any OpenAI-compatible chat endpoint: Groq (free tier), Ollama
(local), or anything else exposing /chat/completions. `MockLLM` is a deterministic, offline
stand-in so the whole pipeline (including retries and pauses) can be demoed without a key.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol

import httpx

from .config import settings
from .ingest import Paper
from .retrieval import BM25


class RateLimited(Exception):
    """The provider asked us to wait longer than is sensible inside one request."""

    def __init__(self, wait: float):
        super().__init__(f"rate limited, retry in {wait:.0f}s")
        self.wait = wait


class LLM(Protocol):
    name: str

    async def generate(self, agent: str, system: str, user: str, ctx: dict[str, Any]) -> str: ...


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.2,
                 transport: httpx.AsyncBaseTransport | None = None, max_wait: float = 8):
        if not base_url or not model:
            raise ValueError("LLM_BASE_URL and LLM_MODEL are required for this provider")
        self.base_url, self.model, self.api_key, self.temperature = base_url.rstrip("/"), model, api_key, temperature
        self.name = f"{model}"
        self.transport, self.max_wait = transport, max_wait

    async def generate(self, agent: str, system: str, user: str, ctx: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": settings.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"     # keep reasoning tokens within free-tier budgets
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=120, transport=self.transport) as client:
            for _ in range(4):
                r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                if r.status_code == 400 and "response_format" in r.text and "response_format" in payload:
                    payload.pop("response_format")       # some models don't support JSON mode
                    continue
                if r.status_code == 400 and "reasoning_effort" in r.text and "reasoning_effort" in payload:
                    payload.pop("reasoning_effort")
                    continue
                if r.status_code == 429:                 # per-minute limit
                    wait = _retry_after(r)
                    if wait is not None and wait <= self.max_wait:
                        await asyncio.sleep(wait if self.transport is None else 0)
                        continue
                    if wait is not None and wait <= 120:
                        raise RateLimited(wait)          # let the caller come back later
                break
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


def _retry_after(r: httpx.Response) -> float | None:
    try:
        return float(r.headers.get("retry-after", "")) + 0.5
    except ValueError:
        m = re.search(r"try again in ([\d.]+)s", r.text)
        return float(m.group(1)) + 0.5 if m else None


# --------------------------------------------------------------------------------------------
# Offline demo model
# --------------------------------------------------------------------------------------------
CLAIM_CUES = re.compile(
    r"\b((we|i|our team) (propose|introduce|present|show|demonstrate|find|built|build|developed|designed|"
    r"implemented|fixed|created|deployed|reduced|improved|achieved|automated|led|added|traced|found|saw|measured)|"
    r"this (paper|work|proposal|project|study) (proposes|presents|shows|introduces|describes)|"
    r"outperforms?|achieves?|improves?|reduces?|state[- ]of[- ]the[- ]art|first to|novel|generali[sz]es?)\b", re.I)
EMPIRICAL = re.compile(r"\b(we (show|demonstrate|find)|outperforms?|achieves?|achieved|improves?|improved|"
                       r"reduces?|reduced|increased|decreased|state[- ]of[- ]the[- ]art|first to|matches|"
                       r"generali[sz]es?)\b|\d+(\.\d+)?\s*%", re.I)
SENT = re.compile(r"(?<=[.!?])\s+")
RESULT_SECTIONS = re.compile(r"experiment|result|evaluation", re.I)

ASPECTS = [
    # aspect, retrieval query, "is it present?" regex, severity if missing, comment, strength text, question
    ("baselines", "baseline compared comparison prior methods results",
     r"\b(baselines?|compar(ed|ing|ison)\b)", "major",
     "Results are not contrasted against strong, recent baselines, so the size of the improvement is hard to judge.",
     "Compares the method against reference systems.",
     "How does the method perform against strong, recent baselines evaluated under the same setup?"),
    ("statistics", "results accuracy score improvement runs seeds",
     r"\b(standard deviation|std|confidence interval|significan\w*|random seeds|p\s*<)\b", "major",
     "Gains are reported as single numbers with no variance, seeds or significance testing, so it is unclear if they are reliable.",
     "Reports variance or significance for its results.",
     "Can you report variance across random seeds and a significance test for the main results?"),
    ("ablation", "ablation component contribution remove",
     r"\bablat\w*", "minor",
     "There is no ablation study isolating which component is responsible for the improvement.",
     "Includes ablations that isolate each component.",
     "Which component drives the improvement? An ablation would make this clear."),
    ("reproducibility", "code data hyperparameters implementation details",
     r"\b(code (is|will be)( publicly)? (available|released)|github|hyperparameters?|learning rate)\b", "minor",
     "Implementation details (hyperparameters, code or data availability) are too sparse to reproduce the results.",
     "Gives enough implementation detail to support reproduction.",
     "Will you release code and the exact hyperparameters used?"),
    ("limitations", "limitations future work weaknesses",
     r"\blimitations?\b", "minor",
     "The paper does not discuss its limitations or failure cases.",
     "Discusses its limitations openly.",
     "What are the main limitations and failure cases of the approach?"),
]
QUESTIONS = {a[0]: a[6] for a in ASPECTS}


def _sentences(chunk_text: str) -> list[str]:
    return [s.strip() for s in SENT.split(chunk_text) if len(s.split()) >= 5]


def _common_tokens(paper: Paper) -> set[str]:
    """Tokens that appear in most chunks (e.g. the method's name) say nothing about support."""
    from collections import Counter
    from .retrieval import tokenize
    df = Counter(t for c in paper.chunks for t in set(tokenize(c.text)))
    return {t for t, f in df.items() if f >= max(2, len(paper.chunks) // 2)}


def _overlap(claim: str, text: str, common: set[str]) -> int:
    from .retrieval import tokenize
    return len((set(tokenize(claim)) - common) & set(tokenize(text)))


class MockLLM:
    """Heuristic stand-in for an LLM. Produces schema-valid JSON from the paper's own text.

    It deliberately makes one grounding mistake on the critic's first attempt (a paraphrased
    'quote'), so the evaluator's retry loop is visible in the demo.
    """

    name = "demo-mode (offline)"

    def __init__(self, delay: float | None = None):
        self.delay = settings.mock_delay if delay is None else delay

    async def generate(self, agent: str, system: str, user: str, ctx: dict[str, Any]) -> str:
        if self.delay:
            await asyncio.sleep(self.delay)   # so the live timeline is watchable
        return json.dumps(getattr(self, f"_{agent}")(ctx))

    # -- agents ---------------------------------------------------------------------------
    def _extractor(self, ctx):
        paper: Paper = ctx["paper"]
        index: BM25 = ctx["index"]
        claims, seen = [], set()
        common = _common_tokens(paper)
        has_results = any(RESULT_SECTIONS.search(c.section) for c in paper.chunks)
        for ch in paper.chunks:
            if has_results and RESULT_SECTIONS.search(ch.section):
                continue
            for s in _sentences(ch.text):
                if CLAIM_CUES.search(s) and s not in seen and len(claims) < 5:
                    seen.add(s)
                    support = [c for c, _ in index.search(s, k=6)
                               if (RESULT_SECTIONS.search(c.section) if has_results else s not in c.text)
                               and _overlap(s, c.text, common) >= 3]
                    ev = [{"chunk_id": ch.id, "quote": s}]
                    if support:
                        q = _sentences(support[0].text)
                        if q:
                            ev.append({"chunk_id": support[0].id, "quote": q[0]})
                    if not EMPIRICAL.search(s):      # a contribution statement, not a result
                        conf = 0.9
                    else:
                        conf = 0.86 if support else 0.38
                    claims.append({"text": s, "evidence": ev, "confidence": conf})
        if not claims:
            # no paper-style phrasing: fall back to the most informative sentences, as plain contributions
            sents = [(c.id, x) for c in paper.chunks for x in _sentences(c.text) if len(x.split()) >= 8]
            sents.sort(key=lambda p: (not re.search(r"\d", p[1]), -len(p[1].split())))
            for cid, x in sents[:3]:
                claims.append({"text": x, "evidence": [{"chunk_id": cid, "quote": x}], "confidence": 0.6})
        abstract = next((c.text for c in paper.chunks if c.section.lower().startswith("abstract")), paper.chunks[0].text)
        return {"title": paper.title, "summary": " ".join(_sentences(abstract)[:2]) or abstract[:300], "claims": claims}

    def _critic(self, ctx):
        paper: Paper = ctx["paper"]
        index: BM25 = ctx["index"]
        full = " ".join(f"{c.section} {c.text}" for c in paper.chunks)
        strengths, issues = [], []
        for aspect, query, present, severity, comment, strength, _q in ASPECTS:
            if re.search(present, full, re.I):
                strengths.append(strength)
                continue
            hits = [c for c, _ in index.search(query, k=6) if RESULT_SECTIONS.search(c.section)] or \
                   [c for c, _ in index.search(query, k=1)]
            ch = hits[0] if hits else paper.chunks[-1]
            quote = _sentences(ch.text)[0] if _sentences(ch.text) else ch.text[:160]
            issues.append({"aspect": aspect, "severity": severity, "comment": comment,
                           "evidence": [{"chunk_id": ch.id, "quote": quote}], "confidence": 0.8})
        long_sents = [s for c in paper.chunks for s in _sentences(c.text) if len(s.split()) > 38]
        if long_sents:
            issues.append({"aspect": "clarity", "severity": "minor",
                           "comment": "Several sentences are very long and hard to follow; splitting them would help readers.",
                           "evidence": [{"chunk_id": next(c.id for c in paper.chunks if long_sents[0] in c.text), "quote": long_sents[0]}],
                           "confidence": 0.7})
        if re.search(r"\d+(\.\d+)?\s*(%|percent|times\b|x\b|GB\b)", full):
            strengths.append("Quantifies its gains with concrete numbers.")
        strengths.insert(0, "Tackles a relevant problem with a clearly stated goal.")
        # one deliberate grounding slip on the first attempt, fixed after evaluator feedback
        if not ctx.get("feedback") and issues:
            issues[0]["evidence"][0]["quote"] = "The authors only report a single headline number for their method."
        return {"strengths": strengths, "issues": issues}

    def _writer(self, ctx):
        claims, critique, answers = ctx["claims"], ctx["critique"], ctx.get("answers", {})
        issues = [i for i in critique["issues"]]
        for c in claims["claims"]:
            if c["confidence"] < 0.5 and answers.get("unsupported_claims", "flag") == "flag":
                issues.append({"aspect": "claims", "severity": "major",
                               "comment": f"This claim is not backed by the reported experiments: \"{c['text'][:140]}\"",
                               "evidence": c["evidence"][:1], "confidence": 0.75})
        majors = sum(i["severity"] == "major" for i in issues)
        minors = len(issues) - majors
        rec = ("accept" if majors == 0 and minors <= 1 else "minor revision" if majors == 0
               else "major revision" if majors <= 4 else "reject")
        soundness = max(1, 4 - majors // 2 - (minors > 2) + (majors == 0))
        clarity = 4 if not any(i["aspect"] == "clarity" for i in issues) else 3
        contribution = 4 if claims["claims"] else 3
        qs = []
        for i in issues:
            if i["aspect"] == "claims":
                qs.append("What evidence supports the claim: " + i["comment"].split(": ", 1)[-1].strip('"').rstrip(".") + "?")
            elif i["aspect"] in QUESTIONS:
                qs.append(QUESTIONS[i["aspect"]])
        qs = qs[:5]
        return {"summary": claims["summary"], "strengths": critique["strengths"], "weaknesses": issues,
                "questions_for_authors": qs or ["Could you share code and data to support reproduction?"],
                "scores": {"soundness": soundness, "clarity": clarity, "contribution": contribution},
                "recommendation": rec, "confidence": 0.72}


def build_llm() -> LLM:
    if settings.provider == "mock":
        return MockLLM()
    base, model, key = settings.resolved()
    return OpenAICompatibleLLM(base, model, key, settings.temperature)
