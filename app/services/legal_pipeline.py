"""
Legal RAG Pipeline
==================
Orchestrates the end-to-end flow:

    Input text
        → IssueExtractor  (LegalBERT)
        → NERExtractor     (RoBERTa)
        → HybridRetriever  (FAISS + BM25)
        → Summarizer       (T5)
        → AnalyzeResponse
"""

from __future__ import annotations
import logging
import time
from typing import Any

from app.models.schemas import (
    AnalyzeResponse,
    LegalIssue,
    NamedEntity,
    SimilarCase,
)
from app.services.issue_extractor import IssueExtractor
from app.services.ner_extractor import NERExtractor
from app.services.retriever import HybridRetriever
from app.services.summarizer import Summarizer

logger = logging.getLogger(__name__)


class LegalPipeline:
    """Singleton pipeline — instantiated once at app startup."""

    def __init__(self) -> None:
        self.issue_extractor = IssueExtractor()
        self.ner_extractor = NERExtractor()
        self.retriever = HybridRetriever()
        self.summarizer = Summarizer()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all models sequentially."""
        self.issue_extractor.load()
        self.ner_extractor.load()
        self.retriever.load()
        self.summarizer.load()

    @property
    def models_loaded(self) -> dict[str, bool]:
        return {
            "legal_bert_issue_extraction": self.issue_extractor.loaded,
            "roberta_ner": self.ner_extractor.loaded,
            "faiss_bm25_retriever": self.retriever.loaded,
            "t5_summarizer": self.summarizer.loaded,
        }

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, case_description: str, top_k: int = 5) -> AnalyzeResponse:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        # ── Step 1: Legal Issue Extraction ───────────────────────────────────
        t = time.perf_counter()
        issues: list[LegalIssue] = self.issue_extractor.extract(case_description)
        timings["issue_extraction_ms"] = round((time.perf_counter() - t) * 1000, 1)
        logger.info("Issues: %s", [i.label for i in issues])

        # ── Step 2: Named Entity Recognition ────────────────────────────────
        t = time.perf_counter()
        entities: list[NamedEntity] = self.ner_extractor.extract(case_description)
        timings["ner_ms"] = round((time.perf_counter() - t) * 1000, 1)
        logger.info("Entities: %d found", len(entities))

        # ── Step 3: Query enrichment for retrieval ───────────────────────────
        # Append top issue labels + entity texts to improve retrieval recall
        enriched_query = self._build_retrieval_query(
            case_description, issues, entities
        )

        # ── Step 4: Hybrid Retrieval (FAISS + BM25) ──────────────────────────
        t = time.perf_counter()
        raw_hits = self.retriever.retrieve(enriched_query, top_k=top_k)
        timings["retrieval_ms"] = round((time.perf_counter() - t) * 1000, 1)
        logger.info("Retrieved %d candidates", len(raw_hits))

        # ── Step 5: Summarize query description ──────────────────────────────
        t = time.perf_counter()
        query_summary = self.summarizer.summarize(case_description)
        timings["query_summary_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # ── Step 6: Summarize each retrieved case + build SimilarCase ────────
        t = time.perf_counter()
        similar_cases = self._build_similar_cases(raw_hits)
        timings["case_summaries_ms"] = round((time.perf_counter() - t) * 1000, 1)

        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        logger.info("Pipeline complete in %.0f ms", timings["total_ms"])

        return AnalyzeResponse(
            legal_issues=issues,
            entities=entities,
            similar_cases=similar_cases,
            query_summary=query_summary,
            processing_meta={
                "timings": timings,
                "models_loaded": self.models_loaded,
                "enriched_query_length": len(enriched_query),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_retrieval_query(
        description: str,
        issues: list[LegalIssue],
        entities: list[NamedEntity],
    ) -> str:
        """
        Combine the case description with high-confidence issue labels and
        entity texts to produce a richer retrieval query.
        """
        extras: list[str] = []

        for iss in issues[:3]:
            if iss.score >= 0.60:
                extras.append(iss.label)

        for ent in entities[:5]:
            if ent.score >= 0.80:
                extras.append(ent.text)

        if extras:
            return description + " " + " ".join(extras)
        return description


    def _build_similar_cases(self, raw_hits: list[dict[str, Any]]) -> list[SimilarCase]:
        cases: list[SimilarCase] = []
        texts_to_summarize: list[str] = []
        needs_summary_idx: list[int] = []
        metas: list[dict[str, Any]] = []

        for hit in raw_hits:
            if hit.get("_mock"):
                metas.append(hit)
            else:
                meta = self.retriever.get_meta(hit["meta_idx"])
                metas.append({**meta, **hit})

        for i, (meta, hit) in enumerate(zip(metas, raw_hits)):
            existing_summary = meta.get("summary", "").strip()  # "" for CJPE records
            case_id = str(meta.get("case_id", ""))
            full_text = self.retriever.get_full_text(case_id)
            display_snippet = full_text or meta.get("text_snippet", "")

            cases.append(
                SimilarCase(
                    case_id=case_id or "UNKNOWN",
                    title=case_id or "UNKNOWN",
                    court=None,
                    date=None,
                    snippet=display_snippet[:12000],
                    score=round(float(hit.get("score", 0.0)), 6),
                    source=str(meta.get("source", hit.get("source", "faiss"))),
                    summary=existing_summary or "",
                )
            )

            if not existing_summary:
                # CJPE record — empty summary, run T5
                texts_to_summarize.append(full_text or display_snippet)
                needs_summary_idx.append(i)

        if texts_to_summarize:
            t5_summaries = self.summarizer.summarize_batch(texts_to_summarize)
            for idx, t5_summary in zip(needs_summary_idx, t5_summaries):
                cases[idx] = cases[idx].model_copy(update={"summary": t5_summary})

        return cases
