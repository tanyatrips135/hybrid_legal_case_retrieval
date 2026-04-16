"""
tests/test_pipeline.py
----------------------
Run with:  pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch


# ── IssueExtractor ────────────────────────────────────────────────────────────

class TestIssueExtractor:
    def test_mock_extract_returns_issues(self):
        from app.services.issue_extractor import IssueExtractor
        ex = IssueExtractor()
        # mock path (no model loaded)
        results = ex._mock_extract("The petitioner seeks bail under Section 302 IPC.")
        assert len(results) >= 1
        assert all(r.score > 0 for r in results)
        assert all(r.label for r in results)

    def test_mock_extract_bail(self):
        from app.services.issue_extractor import IssueExtractor
        results = IssueExtractor._mock_extract("Application for bail on medical grounds.")
        labels = [r.label for r in results]
        assert "Bail / Anticipatory Bail" in labels

    def test_mock_extract_property(self):
        from app.services.issue_extractor import IssueExtractor
        results = IssueExtractor._mock_extract("Land acquisition and property dispute.")
        labels = [r.label for r in results]
        assert "Property / Land" in labels


# ── NERExtractor ──────────────────────────────────────────────────────────────

class TestNERExtractor:
    def test_mock_returns_entities(self):
        from app.services.ner_extractor import NERExtractor
        ex = NERExtractor()
        results = ex._mock_extract("Any text")
        assert len(results) >= 1
        assert all(e.label in {"PER", "ORG", "LOC"} for e in results)
        assert all(e.score > 0 for e in results)


# ── BM25 ──────────────────────────────────────────────────────────────────────

class TestBM25:
    def test_basic_ranking(self):
        from app.services.retriever import BM25
        corpus = [
            ["bail", "murder", "ipc", "section"],
            ["property", "land", "acquisition"],
            ["bail", "anticipatory", "section", "438"],
        ]
        bm25 = BM25(corpus)
        hits = bm25.top_k(["bail", "section"], k=3)
        # docs 0 and 2 both contain "bail" and "section" → should rank above doc 1
        top_indices = [h[0] for h in hits]
        assert 0 in top_indices
        assert 2 in top_indices
        assert top_indices[0] != 1  # doc 1 should not be top

    def test_zero_score_on_empty_query(self):
        from app.services.retriever import BM25
        corpus = [["hello", "world"], ["foo", "bar"]]
        bm25 = BM25(corpus)
        hits = bm25.top_k([], k=2)
        assert all(score == 0.0 for _, score in hits)


# ── HybridRetriever RRF ───────────────────────────────────────────────────────

class TestRRFFusion:
    def test_both_sources_boosted(self):
        from app.services.retriever import HybridRetriever
        dense  = [(0, 0.9), (1, 0.8), (2, 0.7)]
        sparse = [(2, 15.0), (0, 12.0), (3, 10.0)]
        fused = HybridRetriever._rrf_fuse(dense, sparse)
        # doc 0 appears in both → should be near top
        top_ids = [f["meta_idx"] for f in fused[:2]]
        assert 0 in top_ids

    def test_unique_docs_included(self):
        from app.services.retriever import HybridRetriever
        dense  = [(0, 0.9)]
        sparse = [(1, 10.0)]
        fused = HybridRetriever._rrf_fuse(dense, sparse)
        ids = {f["meta_idx"] for f in fused}
        assert ids == {0, 1}

    def test_source_labels(self):
        from app.services.retriever import HybridRetriever
        dense  = [(0, 0.9)]
        sparse = [(0, 5.0)]
        fused = HybridRetriever._rrf_fuse(dense, sparse)
        assert fused[0]["source"] == "bm25+faiss"


# ── Summarizer ────────────────────────────────────────────────────────────────

class TestSummarizer:
    def test_mock_returns_string(self):
        from app.services.summarizer import Summarizer
        s = Summarizer()
        result = s._mock_summary("This case involves a property dispute.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_batch(self):
        from app.services.summarizer import Summarizer
        s = Summarizer()
        texts = ["Case one about bail.", "Case two about property."]
        results = s.summarize_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)


# ── CaseValidator ─────────────────────────────────────────────────────────────

class TestCaseValidator:
    def test_fallback_validate_shape(self):
        from app.services.case_validator import CaseValidator
        from app.models.schemas import SimilarCase

        validator = CaseValidator()
        cases = [
            SimilarCase(
                case_id="C1",
                title="Case 1",
                snippet="Contract breach and delayed delivery dispute.",
                score=0.91,
                source="faiss+bm25",
                summary="Dispute over delayed delivery and financial damages.",
            ),
            SimilarCase(
                case_id="C2",
                title="Case 2",
                snippet="Criminal theft and assault matter.",
                score=0.71,
                source="faiss+bm25",
                summary="A criminal case unrelated to contract law.",
            ),
        ]

        out = validator._fallback_validate(
            "Breach of contract involving delayed delivery and damages.",
            cases,
        )
        assert "cases" in out
        assert "summary" in out
        assert len(out["cases"]) == 2
        assert out["summary"]["decision"] in {"SHOW_RESULTS", "REJECT_RESULTS"}


# ── Pipeline integration (mocked models) ─────────────────────────────────────

class TestPipeline:
    def test_run_returns_response(self):
        from app.services.legal_pipeline import LegalPipeline
        from app.models.schemas import AnalyzeResponse

        pipeline = LegalPipeline()
        # All models unloaded → mock mode
        result = pipeline.run(
            "The petitioner seeks bail under Section 302 IPC for alleged murder "
            "during a property dispute in Delhi High Court.",
            top_k=3,
        )
        assert isinstance(result, AnalyzeResponse)
        assert len(result.legal_issues) >= 1
        assert isinstance(result.query_summary, str)
        assert len(result.similar_cases) <= 3

    def test_models_loaded_dict(self):
        from app.services.legal_pipeline import LegalPipeline
        p = LegalPipeline()
        ml = p.models_loaded
        assert set(ml.keys()) == {
            "legal_bert_issue_extraction",
            "roberta_ner",
            "faiss_bm25_retriever",
            "t5_summarizer",
            "retrieval_validator",
        }

    def test_enriched_query_appends_labels(self):
        from app.services.legal_pipeline import LegalPipeline
        from app.models.schemas import LegalIssue, NamedEntity
        issues = [LegalIssue(label="Bail / Anticipatory Bail", score=0.9, text="bail")]
        entities = [NamedEntity(text="Delhi", label="LOC", start=0, end=5, score=0.95)]
        q = LegalPipeline._build_retrieval_query("Base query.", issues, entities)
        assert "Bail / Anticipatory Bail" in q
        assert "Delhi" in q

    def test_pipeline_adds_validation_annotations(self):
        from app.services.legal_pipeline import LegalPipeline

        pipeline = LegalPipeline()
        result = pipeline.run(
            "Commercial dispute concerning delayed delivery and contract breach "
            "causing significant monetary losses to the buyer.",
            top_k=2,
        )

        assert "validation" in result.processing_meta
        assert len(result.similar_cases) <= 2
        if result.similar_cases:
            assert result.similar_cases[0].validation_label is not None
