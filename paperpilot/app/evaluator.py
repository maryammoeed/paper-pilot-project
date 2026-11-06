"""The evaluator agent: checks every agent output against explicit rules.

Rules are deterministic so failures are explainable. Anything that fails goes back to the agent
with the list of problems as feedback, up to `settings.max_retries` attempts.
"""
from __future__ import annotations

import re
import unicodedata

from pydantic import ValidationError

from .ingest import Paper
from .models import ClaimsOutput, CritiqueOutput, ReviewReport

SCHEMAS = {"extractor": ClaimsOutput, "critic": CritiqueOutput, "writer": ReviewReport}
_ws = re.compile(r"\s+")
_TYPO = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
                      "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
                      "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-", "\u00ad": "",
                      "\u00a0": " ", "\u2009": " ", "\u202f": " "})


def _norm(s: str) -> str:
    """Compare text the way a human would: ignore case, spacing and typographic quote/dash styles."""
    s = unicodedata.normalize("NFKC", s).translate(_TYPO).replace("\u2026", "...")
    s = _ws.sub(" ", s.lower()).strip()
    return s.strip("\"'").rstrip(".…").strip()


def _grounded(paper: Paper, evidence) -> list[str]:
    """Every quote must really appear in the paper.

    A quote that sits in a different chunk than the one cited (or spans two neighbouring chunks)
    is still grounded, so the citation is corrected instead of failing the whole output."""
    problems = []
    texts = [_norm(c.text) for c in paper.chunks]
    for ev in evidence:
        q = _norm(ev.quote)
        if not q:
            problems.append("evidence has an empty quote")
            continue
        ch = paper.chunk(ev.chunk_id)
        if ch is not None and q in _norm(ch.text):
            continue
        hit = next((i for i, t in enumerate(texts) if q in t), None)
        if hit is None:   # sentence split across two neighbouring chunks
            hit = next((i for i in range(len(texts) - 1) if q in texts[i] + " " + texts[i + 1]), None)
        if hit is not None:
            ev.chunk_id = paper.chunks[hit].id
        else:
            problems.append(f"quote does not appear in the paper: \"{ev.quote[:80]}\"")
    return problems


def evaluate(agent: str, data: dict, paper: Paper) -> tuple[object | None, list[str]]:
    """Return (validated_model, problems). Empty problems == pass."""
    try:
        obj = SCHEMAS[agent].model_validate(data)
    except ValidationError as e:
        return None, [f"schema: {err['loc']} {err['msg']}" for err in e.errors()[:6]]

    problems: list[str] = []
    if agent == "extractor":
        if not obj.claims:
            problems.append("no claims extracted")
        for c in obj.claims:
            if not c.evidence:
                problems.append(f"claim has no evidence: \"{c.text[:60]}\"")
            problems += _grounded(paper, c.evidence)

    elif agent == "critic":
        aspects = [i.aspect.lower() for i in obj.issues]
        if len(aspects) != len(set(aspects)):
            problems.append("duplicate aspects in issues; merge them")
        for i in obj.issues:
            if i.severity == "major" and not i.evidence:
                problems.append(f"major issue '{i.aspect}' has no evidence")
            problems += _grounded(paper, i.evidence)

    elif agent == "writer":
        majors = sum(w.severity == "major" for w in obj.weaknesses)
        if obj.recommendation == "accept" and majors:
            problems.append("recommends 'accept' despite major weaknesses")
        if obj.recommendation == "reject" and majors == 0:
            problems.append("recommends 'reject' without any major weakness")
        if majors >= 2 and obj.scores.soundness >= 4:
            problems.append("soundness score is inconsistent with the number of major weaknesses")
        if not obj.questions_for_authors:
            problems.append("no questions for the authors")
        for w in obj.weaknesses:
            problems += _grounded(paper, w.evidence)

    return obj, problems
