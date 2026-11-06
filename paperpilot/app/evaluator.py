"""The evaluator agent: checks every agent output against explicit rules.

Rules are deterministic so failures are explainable. Anything that fails goes back to the agent
with the list of problems as feedback, up to `settings.max_retries` attempts.
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from .ingest import Paper
from .models import ClaimsOutput, CritiqueOutput, ReviewReport

SCHEMAS = {"extractor": ClaimsOutput, "critic": CritiqueOutput, "writer": ReviewReport}
_ws = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _ws.sub(" ", s.lower()).strip().strip('"').strip()


def _grounded(paper: Paper, evidence) -> list[str]:
    problems = []
    for ev in evidence:
        ch = paper.chunk(ev.chunk_id)
        if ch is None:
            problems.append(f"evidence cites chunk '{ev.chunk_id}', which does not exist")
        elif _norm(ev.quote) not in _norm(ch.text):
            problems.append(f"quote is not verbatim from {ev.chunk_id}: \"{ev.quote[:80]}\"")
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
