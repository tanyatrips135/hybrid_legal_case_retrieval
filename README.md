# Indian Legal RAG System

End-to-end retrieval-augmented generation pipeline for Indian legal case analysis.

```
User Input (case description)
        │
        ▼
┌──────────────────────────────┐
│  Legal Issue Extraction      │  LegalBERT fine-tuned on IndicLegalQA
│  (legal_bert_issue_extract.) │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  Named Entity Recognition    │  RoBERTa base — PER, ORG, LOC
│  (roberta-base, zero-shot)   │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  Query Enrichment            │  Append top issue labels + high-conf entities
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  Hybrid Retrieval            │  FAISS dense  +  BM25 sparse
│  (FAISS + BM25 → RRF)        │  fused via Reciprocal Rank Fusion
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│  Judgment Summarization      │  T5 fine-tuned on IN-Abs
│  (t5-legal-explainer)        │  → query summary + per-case summaries
└────────────┬─────────────────┘
             │
             ▼
         Final Output
   (similar cases, summaries,
    legal issues, named entities)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` and set `DEVICE=cuda` in `.env`.

### 2. Configure paths

```bash
cp .env.example .env
# Edit .env if your model directories differ from the defaults
```

### 3. Place your models

```
models/
├── legal_bert_issue_extraction/    ← LegalBERT fine-tuned on IndicLegalQA
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── t5-legal-explainer/             ← T5 fine-tuned on IN-Abs
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   └── tokenizer.json
│
└── all-mpnet-base-v2/              ← SBERT + FAISS artefacts
    ├── legal_cases.faiss
    ├── legal_cases_meta.jsonl      ← one JSON per line (see schema below)
    └── legal_embeddings.npy
```

#### Metadata JSONL schema (one record per line)

```json
{
  "case_id":  "ILDC_00001",
  "title":    "Maneka Gandhi vs Union of India",
  "court":    "Supreme Court of India",
  "date":     "1978-01-25",
  "text":     "<full judgment text used for BM25 and T5 input>",
  "snippet":  "<short excerpt shown in the UI>"
}
```

### 4. Verify artefacts

```bash
python scripts/verify_models.py
```

### 5. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## API Reference

### `GET /health`

Returns pipeline status and per-model load state.

```json
{
  "status": "ok",
  "models_loaded": {
    "legal_bert_issue_extraction": true,
    "roberta_ner": false,
    "faiss_bm25_retriever": true,
    "t5_summarizer": true
  }
}
```

### `POST /api/analyze`

**Request body:**
```json
{
  "case_description": "string (50–10 000 chars)",
  "top_k": 5
}
```

**Response:**
```json
{
  "legal_issues": [
    { "label": "Bail / Anticipatory Bail", "score": 0.88, "text": "bail" }
  ],
  "entities": [
    { "text": "Supreme Court of India", "label": "ORG", "start": 0, "end": 22, "score": 0.97 }
  ],
  "similar_cases": [
    {
      "case_id": "ILDC_001",
      "title": "State vs. Sharma",
      "court": "Supreme Court of India",
      "date": "2021-03-15",
      "snippet": "...",
      "score": 0.812,
      "source": "faiss+bm25",
      "summary": "T5-generated summary..."
    }
  ],
  "query_summary": "T5 summary of the input case description.",
  "processing_meta": {
    "timings": { "issue_extraction_ms": 45, "total_ms": 820 },
    "models_loaded": { ... }
  }
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run entirely in **mock mode** — no model weights needed.

---

## Placeholder Guide

Every model integration has a clearly marked `# ── PLACEHOLDER ──` block.
When the real model directory is present, the `try` block loads it automatically.
If loading fails (missing directory, missing packages), the system falls back to
mock/rule-based output and logs a `WARNING` — the API remains fully functional.

| Service | File | Placeholder notes |
|---|---|---|
| LegalBERT | `app/services/issue_extractor.py` | Loads from `LEGAL_BERT_DIR`; falls back to keyword rules |
| RoBERTa NER | `app/services/ner_extractor.py` | Downloads from HuggingFace Hub; set `ROBERTA_NER_MODEL` to a local path for offline use |
| FAISS | `app/services/retriever.py` | Requires `faiss-cpu` installed + index file present |
| BM25 | `app/services/retriever.py` | Pure Python, no external dep; built automatically from metadata JSONL |
| T5 | `app/services/summarizer.py` | Loads from `T5_DIR`; falls back to truncation-based mock |

---

## Project Structure

```
indian_legal_rag/
├── main.py                          # FastAPI app + lifespan loader
├── requirements.txt
├── .env.example
│
├── app/
│   ├── config.py                    # All settings (override via .env)
│   ├── routers/
│   │   ├── analyze.py               # POST /api/analyze + GET /
│   │   └── health.py                # GET /health
│   ├── services/
│   │   ├── pipeline.py              # Master orchestrator
│   │   ├── issue_extractor.py       # LegalBERT
│   │   ├── ner_extractor.py         # RoBERTa NER
│   │   ├── retriever.py             # FAISS + BM25 + RRF
│   │   └── summarizer.py            # T5
│   └── models/
│       └── schemas.py               # Pydantic request/response models
│
├── templates/
│   └── index.html                   # Single-page frontend
│
├── scripts/
│   └── verify_models.py             # Pre-flight artefact check
│
└── tests/
    ├── conftest.py
    ├── test_pipeline.py             # Unit tests (pure mock, no models needed)
    └── test_api.py                  # API integration tests (mocked pipeline)
```

---

## Configuration Reference

All settings live in `app/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cpu` | `cpu` or `cuda` |
| `LEGAL_BERT_DIR` | `models/legal_bert_issue_extraction` | LegalBERT directory |
| `T5_DIR` | `models/t5-legal-explainer` | T5 directory |
| `SBERT_DIR` | `models/all-mpnet-base-v2` | SentenceTransformer directory |
| `ROBERTA_NER_MODEL` | `Jean-Baptiste/roberta-large-ner-english` | Hub ID or local path |
| `FAISS_INDEX` | `models/all-mpnet-base-v2/legal_cases.faiss` | FAISS index file |
| `META_JSONL` | `models/all-mpnet-base-v2/legal_cases_meta.jsonl` | Case metadata |
| `TOP_K_FAISS` | `20` | FAISS candidates |
| `TOP_K_BM25` | `20` | BM25 candidates |
| `TOP_K_FINAL` | `5` | Results after RRF |
| `T5_MAX_INPUT_TOKENS` | `512` | T5 input truncation |
| `T5_MAX_NEW_TOKENS` | `256` | T5 generation length |
| `T5_NUM_BEAMS` | `4` | Beam search width |
