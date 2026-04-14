"""
tests/test_api.py
-----------------
Integration tests for the FastAPI endpoints.
Uses TestClient so no real server is needed.

Run with:  pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# We need to patch the pipeline load so models don't download during CI
import app.services.legal_pipeline as pipeline_module


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a mock pipeline injected into app state."""
    from main import app
    from app.services.legal_pipeline import LegalPipeline
    from app.models.schemas import (
        AnalyzeResponse, LegalIssue, NamedEntity, SimilarCase
    )

    mock_pipeline = MagicMock(spec=LegalPipeline)
    mock_pipeline.models_loaded = {
        "legal_bert_issue_extraction": False,
        "roberta_ner": False,
        "faiss_bm25_retriever": False,
        "t5_summarizer": False,
    }
    mock_pipeline.run.return_value = AnalyzeResponse(
        legal_issues=[
            LegalIssue(label="Bail / Anticipatory Bail", score=0.88, text="bail")
        ],
        entities=[
            NamedEntity(text="Supreme Court", label="ORG", start=0, end=13, score=0.97)
        ],
        similar_cases=[
            SimilarCase(
                case_id="ILDC_001",
                title="State vs. Sharma",
                court="Supreme Court of India",
                date="2021-03-15",
                snippet="The petitioner applied for bail citing medical grounds.",
                score=0.812,
                source="faiss+bm25",
                summary="Mock summary of State vs. Sharma.",
            )
        ],
        query_summary="Mock T5 summary of the input case.",
        processing_meta={
            "timings": {"total_ms": 123},
            "models_loaded": mock_pipeline.models_loaded,
        },
    )

    with TestClient(app, raise_server_exceptions=True) as c:
        app.state.pipeline = mock_pipeline
        yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_health_model_keys(self, client):
        resp = client.get("/health")
        ml = resp.json()["models_loaded"]
        assert "legal_bert_issue_extraction" in ml
        assert "t5_summarizer" in ml


# ── Frontend ──────────────────────────────────────────────────────────────────

class TestFrontend:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Indian Legal RAG" in resp.text


# ── Analyze endpoint ──────────────────────────────────────────────────────────

class TestAnalyze:
    VALID_DESCRIPTION = (
        "The petitioner seeks anticipatory bail under Section 438 CrPC. "
        "He was named as accused in an FIR registered under Sections 420 and 120B IPC "
        "for alleged cheating in a financial transaction worth Rs 50 lakh in Delhi."
    )

    def test_analyze_success(self, client):
        resp = client.post(
            "/api/analyze",
            json={"case_description": self.VALID_DESCRIPTION, "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "legal_issues" in data
        assert "entities" in data
        assert "similar_cases" in data
        assert "query_summary" in data
        assert "processing_meta" in data

    def test_analyze_response_shape(self, client):
        resp = client.post(
            "/api/analyze",
            json={"case_description": self.VALID_DESCRIPTION, "top_k": 1},
        )
        data = resp.json()
        issue = data["legal_issues"][0]
        assert "label" in issue
        assert "score" in issue
        assert "text" in issue

        case = data["similar_cases"][0]
        for field in ("case_id", "title", "score", "source", "summary", "snippet"):
            assert field in case, f"Missing field: {field}"

    def test_analyze_short_text_rejected(self, client):
        resp = client.post(
            "/api/analyze",
            json={"case_description": "Too short.", "top_k": 5},
        )
        assert resp.status_code == 422  # Pydantic validation error

    def test_analyze_top_k_bounds(self, client):
        # top_k > 20 should fail validation
        resp = client.post(
            "/api/analyze",
            json={"case_description": self.VALID_DESCRIPTION, "top_k": 99},
        )
        assert resp.status_code == 422

    def test_analyze_missing_body(self, client):
        resp = client.post("/api/analyze", json={})
        assert resp.status_code == 422
