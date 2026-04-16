"""
Configuration — all paths and hyper-parameters in one place.
Override via environment variables or a .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Model directories ────────────────────────────────────────────────────
    MODELS_DIR: Path = Path("models")

    # LegalBERT (fine-tuned on IndicLegalQA for issue extraction)
    LEGAL_BERT_DIR: Path = Path("models/legal_bert_issue_extraction")

    # RoBERTa base for NER (person, org, location)
    ROBERTA_NER_MODEL: str = "Jean-Baptiste/roberta-large-ner-english"
    # Set to a local path if you want fully offline NER, e.g.:
    # ROBERTA_NER_MODEL: Path = Path("models/roberta_ner")

    # Sentence-BERT for embeddings (FAISS index was built with this)
    SBERT_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    SBERT_DIR: Path = Path("models/all-mpnet-base-v2")

    # T5 summarizer (fine-tuned on IN-Abs)
    T5_DIR: Path = Path("models/t5-legal-explainer")

    # ── FAISS / metadata ─────────────────────────────────────────────────────
    FAISS_INDEX: Path = Path("models/all-mpnet-base-v2/legal_cases.faiss")
    META_JSONL: Path = Path("models/all-mpnet-base-v2/legal_cases_meta.jsonl")
    FULL_TEXT_JSONL: Path = Path("app/models/all-mpnet-base-v2/merged_corpus.jsonl")
    EMBEDDINGS_NPY: Path = Path("models/all-mpnet-base-v2/legal_embeddings.npy")

    # ── Retrieval ────────────────────────────────────────────────────────────
    TOP_K_FAISS: int = 20          # candidates fetched from FAISS
    TOP_K_BM25: int = 20           # candidates fetched from BM25
    TOP_K_FINAL: int = 5           # results returned after re-ranking
    BM25_CACHE_PATH: Path = Path("app/models/all-mpnet-base-v2/bm25_cache.pkl")

    # ── Post-retrieval validation ────────────────────────────────────────────
    VALIDATION_ENABLED: bool = True
    VALIDATION_MODEL: str = "llama-3.3-70b-versatile"
    VALIDATION_TEMPERATURE: float = 0.1
    VALIDATION_ENFORCE_DECISION: bool = False
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # ── NER labels to keep ───────────────────────────────────────────────────
    NER_KEEP_LABELS: list[str] = ["PER", "ORG", "LOC",
                                   "PERSON", "ORGANIZATION", "LOCATION"]

    # ── T5 generation ────────────────────────────────────────────────────────
    T5_MAX_INPUT_TOKENS: int = 512
    T5_MAX_NEW_TOKENS: int = 256
    T5_MIN_NEW_TOKENS: int = 48
    T5_NUM_BEAMS: int = 4
    T5_SUMMARY_SENTENCES: int = 3

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE: str = "cpu"            # "cuda" if GPU available

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
