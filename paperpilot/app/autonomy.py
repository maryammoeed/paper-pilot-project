"""Adaptive autonomy: decide at each checkpoint whether to proceed, ask the human, or stop.

Instead of a fixed "always automate" or "always ask", each checkpoint computes a risk score from
observable signals (guardrail hits, unsupported claims, evaluator retries, how consequential the
next step is, model confidence). The user's autonomy setting sets the threshold above which the
agent pauses. Some conditions stop the run no matter the setting.

Every decision records its reasons, so the user can see *why* the system paused or proceeded.
"""
from __future__ import annotations

from .models import Autonomy, Decision

THRESHOLD = {Autonomy.HANDS_ON: 0.15, Autonomy.BALANCED: 0.45, Autonomy.HANDS_OFF: 0.8}

# what the system does on its own when it does not need to ask
DEFAULTS = {
    "input": "Remove it and mention it in the review",
    "claims": "Yes, flag them",
    "verdict": "Finalize",
}


def _decide(level: Autonomy, checkpoint: str, risk: float, reasons: list[str],
            question: str, options: list[str]) -> Decision:
    risk = round(min(risk, 1.0), 2)
    if risk >= THRESHOLD[level]:
        return Decision(action="ask", risk=risk, reasons=reasons, question=question,
                        options=options, checkpoint=checkpoint)
    reasons = reasons + [f"risk {risk} is below the {level.value.replace('_', '-')} threshold "
                         f"{THRESHOLD[level]}, so I proceeded with: {DEFAULTS[checkpoint]}"]
    return Decision(action="proceed", risk=risk, reasons=reasons, checkpoint=checkpoint)


def input_checkpoint(level: Autonomy, findings: list[dict]) -> Decision:
    inj = [f for f in findings if f["type"] == "prompt_injection"]
    sec = [f for f in findings if f["type"] == "secret"]
    risk, reasons = 0.0, []
    if inj:
        risk += 0.7
        reasons.append(f"{len(inj)} hidden instruction(s) aimed at the reviewer were found and removed")
    if sec:
        risk += 0.3
        reasons.append(f"{len(sec)} secret(s) were redacted before any model saw the text")
    if not reasons:
        return Decision(action="proceed", risk=0.0, checkpoint="input",
                        reasons=["input guardrails found no secrets or hidden instructions"])
    snippet = inj[0]["detail"][:160].rstrip(".!?") if inj else ""
    q = (f"The paper contains text that tries to instruct the reviewer: \"{snippet}\". I removed it before "
         f"the agents ran. How should the review treat this?") if inj else \
        "Secrets were redacted from the input. Continue the review?"
    opts = ["Remove it and mention it in the review", "Remove it silently", "Stop the review"] if inj else \
           ["Remove it and mention it in the review", "Stop the review"]
    return _decide(level, "input", risk, reasons, q, opts)


def claims_checkpoint(level: Autonomy, claims: list[dict], retries_used: int) -> Decision:
    weak = [c for c in claims if c["confidence"] < 0.5]
    risk = 0.1 + 0.18 * len(weak) + 0.1 * retries_used
    reasons = [f"{len(weak)} of {len(claims)} claims have no supporting result in the experiments"]
    if retries_used:
        reasons.append(f"the evaluator sent agent work back {retries_used} time(s)")
    preview = "; ".join(f"\"{c['text'][:90]}\"" for c in weak[:2])
    q = (f"{len(weak)} claim(s) are not backed by the experiments I could find ({preview}). "
         f"Should the review count them as weaknesses?")
    return _decide(level, "claims", risk, reasons, q, ["Yes, flag them", "No, they may be supported elsewhere"])


def verdict_checkpoint(level: Autonomy, recommendation: str, confidence: float, majors: int) -> Decision:
    impact = {"accept": 0.25, "minor revision": 0.3, "major revision": 0.5, "reject": 0.75}[recommendation]
    risk = impact + 0.3 * (1 - confidence)
    reasons = [f"recommendation '{recommendation}' with {majors} major weakness(es)",
               f"writer confidence {confidence:.2f}"]
    others = [r for r in ["accept", "minor revision", "major revision", "reject"] if r != recommendation]
    q = f"I'm about to recommend **{recommendation}**. This is the most consequential output. Finalize it?"
    return _decide(level, "verdict", risk, reasons, q, ["Finalize"] + [f"Change to {o}" for o in others])
