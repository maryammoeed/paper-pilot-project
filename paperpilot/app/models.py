"""Typed contracts passed between agents. Every LLM output is validated against one of these."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    chunk_id: str = Field(description="ID of the paper chunk the quote comes from, e.g. 'c4'")
    quote: str = Field(description="Verbatim text copied from that chunk")


class Claim(BaseModel):
    text: str
    evidence: list[Evidence] = []
    confidence: float = Field(ge=0, le=1)


class ClaimsOutput(BaseModel):
    title: str
    summary: str
    claims: list[Claim]


class Issue(BaseModel):
    aspect: str = Field(description="e.g. baselines, ablation, statistics, reproducibility, clarity")
    severity: Literal["minor", "major"]
    comment: str
    evidence: list[Evidence] = []
    confidence: float = Field(ge=0, le=1)


class CritiqueOutput(BaseModel):
    strengths: list[str]
    issues: list[Issue]


class Scores(BaseModel):
    soundness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    contribution: int = Field(ge=1, le=5)


Recommendation = Literal["accept", "minor revision", "major revision", "reject"]


class ReviewReport(BaseModel):
    summary: str
    strengths: list[str]
    weaknesses: list[Issue]
    questions_for_authors: list[str]
    scores: Scores
    recommendation: Recommendation
    confidence: float = Field(ge=0, le=1)


class Autonomy(str, Enum):
    HANDS_ON = "hands_on"
    BALANCED = "balanced"
    HANDS_OFF = "hands_off"


class Decision(BaseModel):
    action: Literal["proceed", "ask", "stop"]
    risk: float
    reasons: list[str]
    question: str | None = None
    options: list[str] = []
    checkpoint: str | None = None
