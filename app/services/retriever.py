"""
Hybrid Retrieval — FAISS + BM25
================================
1. Dense retrieval   : FAISS ANN search over all-mpnet-base-v2 embeddings.
2. Sparse retrieval  : BM25 over case text snippets from the metadata JSONL.
3. Hybrid re-ranking : Reciprocal Rank Fusion (RRF) to merge both ranked lists.

The metadata JSONL is expected to have one JSON object per line with at
minimum these fields:
    {
        "case_id":  "<string>",
        "title":    "<string>",
        "court":    "<string | null>",
        "date":     "<string | null>",
        "text":     "<string>",          # full or snippet text for BM25
        "snippet":  "<string>"           # short display snippet (optional)
    }
"""

from __future__ import annotations
import hashlib
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# RRF constant — controls how much low-ranked hits are penalised
RRF_K = 60
BM25_CACHE_VERSION = "v1"


# ══════════════════════════════════════════════════════════════════════════════
# BM25 (pure-Python, no external dependency)
# ══════════════════════════════════════════════════════════════════════════════

class BM25:
    """Minimal BM25 implementation — no external library required."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._build_df()

    def _build_df(self) -> None:
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        dl = len(doc)
        tf_map: dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1

        total = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            num = tf * (self.k1 + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self.idf[term] * num / denom
        return total

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scores = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# ══════════════════════════════════════════════════════════════════════════════
# Retriever
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """FAISS dense + BM25 sparse, fused via RRF."""

    def __init__(self) -> None:
        self._faiss_index: Any = None
        self._embedder: Any = None
        self._meta: list[dict[str, Any]] = []
        self._full_text: dict[str, str] = {}
        self._bm25: BM25 | None = None
        self.loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        self._load_metadata()
        self._load_full_text()
        self._load_faiss()
        self._load_embedder()
        self._build_bm25()

    def _load_metadata(self) -> None:
        path: Path = settings.META_JSONL
        if not path.exists():
            logger.warning(
                "Metadata JSONL not found at %s — retrieval will use mocks.", path
            )
            return
        logger.info("Loading case metadata from %s", path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self._meta.append(json.loads(line))
        logger.info("Loaded %d case records.", len(self._meta))

    def _load_full_text(self) -> None:
        path: Path = settings.FULL_TEXT_JSONL
        if not path.exists():
            logger.warning(
                "Full text JSONL not found at %s — summaries will use snippet.",
                path,
            )
            return

        logger.info("Loading full text from %s", path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)
                case_id = str(rec.get("case_id", "")).strip()
                text = str(rec.get("text", rec.get("text_snippet", ""))).strip()
                if case_id and text:
                    self._full_text[case_id] = text

        logger.info("Full text loaded for %d cases.", len(self._full_text))

    def _load_faiss(self) -> None:
        index_path: Path = settings.FAISS_INDEX
        if not index_path.exists():
            logger.warning("FAISS index not found at %s.", index_path)
            return

        # ── PLACEHOLDER ── faiss-cpu must be installed ────────────────────
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.read_index(str(index_path))
            logger.info(
                "FAISS index loaded — %d vectors, dim=%d.",
                self._faiss_index.ntotal,
                self._faiss_index.d,
            )
        except ImportError:
            logger.warning("faiss-cpu not installed. Dense retrieval disabled.")
        except Exception as exc:
            logger.warning("FAISS load error: %s", exc)
        # ── END PLACEHOLDER ───────────────────────────────────────────────

    def _load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            # Try local model FIRST
            if settings.SBERT_DIR.exists():
                try:
                    self._embedder = SentenceTransformer(str(settings.SBERT_DIR))
                    logger.info("SentenceTransformer loaded from local dir.")
                    return
                except Exception as e:
                    logger.warning("Local SBERT load failed: %s", e)

            # Fallback to HuggingFace model
            self._embedder = SentenceTransformer(settings.SBERT_MODEL)
            logger.info("SentenceTransformer loaded from HF: %s", settings.SBERT_MODEL)

        except ImportError:
            logger.warning("sentence-transformers not installed.")
        
    def _build_bm25(self) -> None:
        if not self._meta:
            return

        cache_path = settings.BM25_CACHE_PATH
        signature = self._bm25_signature()

        cached = self._load_bm25_cache(cache_path, signature)
        if cached is not None:
            self._bm25 = cached
            logger.info("BM25 index loaded from cache: %s", cache_path)
            return

        logger.info("Building BM25 index over %d documents…", len(self._meta))
        corpus_tokens = [
            self._tokenize(
                self._full_text.get(
                    str(rec.get("case_id", "")),
                    rec.get("text_snippet", ""),
                )
            )
            for rec in self._meta
        ]
        self._bm25 = BM25(corpus_tokens)
        logger.info("BM25 index built.")

        self._save_bm25_cache(cache_path, signature, self._bm25)

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._meta:
            return self._mock_results(top_k)

        faiss_hits = self._dense_search(query, settings.TOP_K_FAISS)
        bm25_hits = self._bm25_search(query, settings.TOP_K_BM25)
        fused = self._rrf_fuse(faiss_hits, bm25_hits)
        return fused[:top_k]

    def embed_query(self, query: str) -> np.ndarray | None:
        """Return the query embedding (used by the pipeline for other tasks)."""
        if self._embedder is None:
            return None
        return self._embedder.encode(query, convert_to_numpy=True)

    # ── Dense search ──────────────────────────────────────────────────────────

    def _dense_search(
        self, query: str, k: int
    ) -> list[tuple[int, float]]:
        """Returns list of (meta_index, similarity_score)."""
        if self._faiss_index is None or self._embedder is None:
            return []

        vec = self._embedder.encode(query, convert_to_numpy=True).reshape(1, -1)
        vec = vec.astype(np.float32)

        # Normalise for cosine
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        distances, indices = self._faiss_index.search(vec, k)
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            hits.append((int(idx), float(dist)))
        return hits

    # ── Sparse (BM25) search ──────────────────────────────────────────────────

    def _bm25_search(
        self, query: str, k: int
    ) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = self._tokenize(query)
        return self._bm25.top_k(tokens, k)

    # ── RRF fusion ────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_fuse(
        dense: list[tuple[int, float]],
        sparse: list[tuple[int, float]],
    ) -> list[dict[str, Any]]:
        scores: dict[int, float] = {}
        sources: dict[int, set[str]] = {}

        for rank, (idx, _) in enumerate(dense):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(idx, set()).add("faiss")

        for rank, (idx, _) in enumerate(sparse):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(idx, set()).add("bm25")

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {"meta_idx": idx, "score": sc, "source": "+".join(sorted(sources[idx]))}
            for idx, sc in ranked
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

    def get_meta(self, idx: int) -> dict[str, Any]:
        if 0 <= idx < len(self._meta):
            return self._meta[idx]
        return {}

    def get_full_text(self, case_id: str) -> str:
        return self._full_text.get(case_id, "")

    def _bm25_signature(self) -> str:
        """Signature derived from corpus inputs so cache invalidates on data changes."""
        meta_stat = settings.META_JSONL.stat() if settings.META_JSONL.exists() else None
        full_stat = (
            settings.FULL_TEXT_JSONL.stat() if settings.FULL_TEXT_JSONL.exists() else None
        )

        payload = {
            "version": BM25_CACHE_VERSION,
            "meta_path": str(settings.META_JSONL),
            "meta_mtime_ns": getattr(meta_stat, "st_mtime_ns", None),
            "meta_size": getattr(meta_stat, "st_size", None),
            "full_path": str(settings.FULL_TEXT_JSONL),
            "full_mtime_ns": getattr(full_stat, "st_mtime_ns", None),
            "full_size": getattr(full_stat, "st_size", None),
            "meta_len": len(self._meta),
            "full_len": len(self._full_text),
        }
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _load_bm25_cache(cache_path: Path, signature: str) -> BM25 | None:
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)

            if not isinstance(payload, dict):
                return None
            if payload.get("signature") != signature:
                return None

            bm25 = payload.get("bm25")
            if isinstance(bm25, BM25):
                return bm25
        except Exception as exc:
            logger.warning("Could not load BM25 cache at %s (%s).", cache_path, exc)

        return None

    @staticmethod
    def _save_bm25_cache(cache_path: Path, signature: str, bm25: BM25) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(
                    {"signature": signature, "version": BM25_CACHE_VERSION, "bm25": bm25},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info("BM25 cache written to %s", cache_path)
        except Exception as exc:
            logger.warning("Could not write BM25 cache at %s (%s).", cache_path, exc)

    # ── Mock fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def _mock_results(top_k: int) -> list[dict[str, Any]]:
        return [
            {
                "meta_idx": i,
                "score": round(0.95 - i * 0.08, 4),
                "source": "faiss+bm25",
                "_mock": True,
                "case_id": f"ILDC_{1000 + i}",
                "title": f"[Mock] Example Legal Case {i + 1}",
                "court": "Supreme Court of India",
                "date": "2022-01-01",
                "snippet": "This is a placeholder case. Add the FAISS index...",
                "summary": None,
                "label": None,
                "source": "faiss+bm25",
                "text_snippet": "This is a placeholder case. Add the FAISS index and metadata JSONL to enable real retrieval.",
            }
            for i in range(top_k)
        ]
