"""
Pydantic schemas for API request and response bodies.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    case_description: str = Field(
        ...,
        min_length=50,
        max_length=10_000,
        description="Free-text description of the legal case / facts.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of similar cases to return.",
    )


# ── Sub-models ───────────────────────────────────────────────────────────────

class LegalIssue(BaseModel):
    label: str
    score: float
    text: str


class NamedEntity(BaseModel):
    text: str
    label: str          # PER | ORG | LOC
    start: int
    end: int
    score: float


class SimilarCase(BaseModel):
    case_id: str
    title: str
    court: str | None = None
    date: str | None = None
    snippet: str
    score: float           # hybrid re-rank score (higher = more similar)
    source: str            # "faiss" | "bm25" | "both"
    summary: str           # T5-generated summary


# ── Response ─────────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    legal_issues: list[LegalIssue]
    entities: list[NamedEntity]
    similar_cases: list[SimilarCase]
    query_summary: str    # T5 summary of the *input* description
    processing_meta: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
