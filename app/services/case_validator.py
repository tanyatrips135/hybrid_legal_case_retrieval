"""
Post-retrieval relevance validation for retrieved legal cases.

This module adds a robust validation layer after hybrid retrieval. It prefers an
LLM-backed scoring flow (Groq) and automatically falls back to a deterministic
rule-based validator if API credentials or client setup are unavailable.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from app.config import settings
from app.models.schemas import SimilarCase

logger = logging.getLogger(__name__)

_LABELS = (
    "HIGHLY_RELEVANT",
    "MODERATELY_RELEVANT",
    "WEAKLY_RELEVANT",
    "NOT_RELEVANT",
)


class CaseValidator:
    """Validates retrieved cases against the query summary."""

    def __init__(self) -> None:
        self._client: Any = None
        self.loaded = False

    def load(self) -> None:
        """Initialize Groq client when validation is enabled."""
        if not settings.VALIDATION_ENABLED:
            logger.info("Case validation disabled by configuration.")
            self.loaded = False
            return

        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set. Validation will run in fallback mode.")
            self.loaded = False
            return

        try:
            from groq import Groq

            self._client = Groq(api_key=settings.GROQ_API_KEY)
            self.loaded = True
            logger.info("Case validator initialized with Groq model: %s", settings.VALIDATION_MODEL)
        except Exception as exc:
            logger.warning("Could not initialize Groq validator (%s). Fallback mode enabled.", exc)
            self.loaded = False

    def validate_cases(self, query_summary: str, cases: list[SimilarCase]) -> dict[str, Any]:
        """Validate cases and return normalized scoring output."""
        if not cases:
            return {
                "cases": [],
                "summary": {
                    "relevant_count": 0,
                    "highly_relevant_count": 0,
                    "decision": "REJECT_RESULTS",
                },
                "mode": "empty",
            }

        if not self.loaded or self._client is None:
            return self._fallback_validate(query_summary, cases)

        prompt = self._build_prompt(query_summary, cases)
        try:
            response = self._client.chat.completions.create(
                model=settings.VALIDATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict legal evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.VALIDATION_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            payload = json.loads(content)
            normalized = self._normalize_output(payload, len(cases))
            normalized["mode"] = "llm"
            return normalized
        except Exception as exc:
            logger.warning("LLM validation failed (%s). Falling back to heuristic mode.", exc)
            return self._fallback_validate(query_summary, cases)

    @staticmethod
    def _build_prompt(query_summary: str, cases: list[SimilarCase]) -> str:
        cases_text = "\n\n".join(
            [
                f"Case {i + 1}: {(c.summary or c.snippet)[:2500]}"
                for i, c in enumerate(cases)
            ]
        )

        return f"""
You are a LEGAL RELEVANCE VALIDATION SYSTEM.

TASK:
Determine whether each retrieved case is relevant to the given query case.

QUERY CASE SUMMARY:
{query_summary}

RETRIEVED CASE SUMMARIES:
{cases_text}

---------------------------------------
SCORING METRICS (0 to 5):
1. LEGAL ISSUE MATCH
2. FACTUAL SIMILARITY
3. ENTITY OVERLAP (people, acts, sections)
4. OUTCOME RELEVANCE
5. CONTEXTUAL ALIGNMENT

---------------------------------------
INSTRUCTIONS:
- Score each metric (0-5)
- Compute FINAL_SCORE = average of all 5 metrics
- Classify:
    - FINAL_SCORE >= 4 -> \"HIGHLY_RELEVANT\"
    - FINAL_SCORE >= 3 -> \"MODERATELY_RELEVANT\"
    - FINAL_SCORE >= 2 -> \"WEAKLY_RELEVANT\"
    - else -> \"NOT_RELEVANT\"
- Provide short reasoning (1-2 lines)

---------------------------------------
OUTPUT STRICT JSON FORMAT:

{{
  \"cases\": [
    {{
      \"case_id\": 1,
      \"scores\": {{
        \"legal_issue\": int,
        \"factual_similarity\": int,
        \"entity_overlap\": int,
        \"outcome_relevance\": int,
        \"context_alignment\": int
      }},
      \"final_score\": float,
      \"label\": \"HIGHLY_RELEVANT | MODERATELY_RELEVANT | WEAKLY_RELEVANT | NOT_RELEVANT\",
      \"reason\": \"short explanation\"
    }}
  ],
  \"summary\": {{
    \"relevant_count\": int,
    \"highly_relevant_count\": int,
    \"decision\": \"SHOW_RESULTS | REJECT_RESULTS\"
  }}
}}

---------------------------------------
FINAL DECISION RULE:
- If >= 2 cases are HIGHLY_RELEVANT -> SHOW_RESULTS
- If >= 50% cases are at least MODERATELY_RELEVANT -> SHOW_RESULTS
- Else -> REJECT_RESULTS

RETURN ONLY JSON. NO EXTRA TEXT.
"""

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    @staticmethod
    def _score_to_label(final_score: float) -> str:
        if final_score >= 4.0:
            return "HIGHLY_RELEVANT"
        if final_score >= 3.0:
            return "MODERATELY_RELEVANT"
        if final_score >= 2.0:
            return "WEAKLY_RELEVANT"
        return "NOT_RELEVANT"

    def _fallback_validate(self, query_summary: str, cases: list[SimilarCase]) -> dict[str, Any]:
        q_tokens = set(self._tokenize(query_summary))

        scored_cases: list[dict[str, Any]] = []
        for idx, case in enumerate(cases, start=1):
            c_tokens = set(self._tokenize(case.summary or case.snippet))
            if not q_tokens or not c_tokens:
                overlap = 0.0
            else:
                overlap = len(q_tokens & c_tokens) / math.sqrt(len(q_tokens) * len(c_tokens))

            # Map lexical overlap to 0..5 scale and apply tiny metric variation.
            base = max(0.0, min(5.0, round(overlap * 6.0, 2)))
            metrics = {
                "legal_issue": int(round(base)),
                "factual_similarity": int(round(max(0.0, base - 0.3))),
                "entity_overlap": int(round(max(0.0, base - 0.6))),
                "outcome_relevance": int(round(max(0.0, base - 0.4))),
                "context_alignment": int(round(base)),
            }
            metrics = {k: max(0, min(5, v)) for k, v in metrics.items()}

            final_score = round(sum(metrics.values()) / 5.0, 2)
            label = self._score_to_label(final_score)
            scored_cases.append(
                {
                    "case_id": idx,
                    "scores": metrics,
                    "final_score": final_score,
                    "label": label,
                    "reason": "Heuristic fallback validation (LLM unavailable).",
                }
            )

        return self._normalize_output({"cases": scored_cases, "summary": {}}, len(cases), mode="fallback")

    def _normalize_output(
        self,
        payload: dict[str, Any],
        expected_count: int,
        mode: str | None = None,
    ) -> dict[str, Any]:
        raw_cases = payload.get("cases", []) if isinstance(payload, dict) else []

        normalized_cases: list[dict[str, Any]] = []
        for i in range(expected_count):
            raw = raw_cases[i] if i < len(raw_cases) and isinstance(raw_cases[i], dict) else {}
            scores_raw = raw.get("scores", {}) if isinstance(raw.get("scores", {}), dict) else {}
            scores = {
                "legal_issue": int(max(0, min(5, scores_raw.get("legal_issue", 0)))),
                "factual_similarity": int(max(0, min(5, scores_raw.get("factual_similarity", 0)))),
                "entity_overlap": int(max(0, min(5, scores_raw.get("entity_overlap", 0)))),
                "outcome_relevance": int(max(0, min(5, scores_raw.get("outcome_relevance", 0)))),
                "context_alignment": int(max(0, min(5, scores_raw.get("context_alignment", 0)))),
            }

            parsed_final = raw.get("final_score", None)
            if parsed_final is None:
                final_score = round(sum(scores.values()) / 5.0, 2)
            else:
                final_score = round(float(parsed_final), 2)

            label = str(raw.get("label", "")).strip().upper()
            if label not in _LABELS:
                label = self._score_to_label(final_score)

            reason = str(raw.get("reason", "")).strip() or "No reason provided."
            normalized_cases.append(
                {
                    "case_id": i + 1,
                    "scores": scores,
                    "final_score": final_score,
                    "label": label,
                    "reason": reason,
                }
            )

        highly_relevant_count = sum(1 for c in normalized_cases if c["label"] == "HIGHLY_RELEVANT")
        relevant_count = sum(
            1
            for c in normalized_cases
            if c["label"] in {"HIGHLY_RELEVANT", "MODERATELY_RELEVANT", "WEAKLY_RELEVANT"}
        )
        moderately_or_better = sum(
            1
            for c in normalized_cases
            if c["label"] in {"HIGHLY_RELEVANT", "MODERATELY_RELEVANT"}
        )

        show_results = (
            highly_relevant_count >= 2
            or (expected_count > 0 and (moderately_or_better / expected_count) >= 0.5)
        )
        decision = "SHOW_RESULTS" if show_results else "REJECT_RESULTS"

        normalized = {
            "cases": normalized_cases,
            "summary": {
                "relevant_count": relevant_count,
                "highly_relevant_count": highly_relevant_count,
                "decision": decision,
            },
        }
        if mode:
            normalized["mode"] = mode
        return normalized
