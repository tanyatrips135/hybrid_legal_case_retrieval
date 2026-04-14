"""
Judgment Summarization
======================
Uses a T5 model fine-tuned on the IN-Abs Indian legal summarization dataset.

For each retrieved case text, the T5 generates a plain-language summary.
The same model also summarises the user's *query* case description.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import settings

logger = logging.getLogger(__name__)

# Prompt prefix for summarization input
_PREFIX = ""

_SEQ2SEQ_MODEL_TYPES = {
    "bart",
    "t5",
    "mt5",
    "pegasus",
    "mbart",
    "led",
    "longt5",
    "m2m_100",
    "marian",
    "encoder-decoder",
}


def _preview(text: str, limit: int = 280) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


class Summarizer:
    """Wraps the seq2seq legal summarizer (BART/T5 compatible)."""

    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._model: Any = None
        self.loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_seq2seq_model(model_ref: str, local_only: bool) -> bool:
        try:
            cfg = AutoConfig.from_pretrained(model_ref, local_files_only=local_only)
        except Exception:
            return False
        model_type = str(getattr(cfg, "model_type", "")).lower()
        return model_type in _SEQ2SEQ_MODEL_TYPES

    def load(self) -> None:
        preferred_local = Path(settings.T5_DIR)
        fallback_local = Path("app/models")
        hub_fallback = "facebook/bart-large-cnn"

        candidates: list[tuple[str, bool]] = []
        if preferred_local.exists():
            candidates.append((str(preferred_local), True))
        if fallback_local.exists() and fallback_local != preferred_local:
            candidates.append((str(fallback_local), True))
        candidates.append((hub_fallback, False))

        logger.info("Loading seq2seq summarizer (BART/T5 compatible)")

        # ── PLACEHOLDER ─────────────────────────────────────────────────────
        for model_ref, local_only in candidates:
            if not self._is_seq2seq_model(model_ref, local_only=local_only):
                logger.warning(
                    "Skipping non-seq2seq model config at %s", model_ref
                )
                continue

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_ref,
                    local_files_only=local_only,
                )
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_ref,
                    local_files_only=local_only,
                )
                self._model.eval()
                if settings.DEVICE != "cpu" and torch.cuda.is_available():
                    self._model = self._model.to(settings.DEVICE)

                self.loaded = True
                logger.info("Summarizer model loaded from %s", model_ref)
                return
            except Exception as exc:
                logger.warning("Failed to load summarizer model at %s (%s)", model_ref, exc)

        logger.warning("No valid seq2seq summarizer model loaded. Running in MOCK mode.")
        self._model = None
        self._tokenizer = None
        self.loaded = False
        # ── END PLACEHOLDER ──────────────────────────────────────────────────

    # ── Inference ─────────────────────────────────────────────────────────────

    def summarize(self, text: str) -> str:
        """Return a plain-language summary of *text*."""
        if not self.loaded:
            mock = self._mock_summary(text)
            logger.debug(
                "Summarizer mock output | input_preview=%s | output_preview=%s",
                _preview(text),
                _preview(mock),
            )
            return mock

        input_text = _PREFIX + text.strip()
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=settings.T5_MAX_INPUT_TOKENS,
            truncation=True,
        )
        if settings.DEVICE != "cpu" and torch.cuda.is_available():
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=settings.T5_MAX_NEW_TOKENS,
                min_new_tokens=settings.T5_MIN_NEW_TOKENS,
                num_beams=settings.T5_NUM_BEAMS,
                early_stopping=True,
            )

        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        finalized = self._finalize_summary(decoded)

        logger.debug(
            (
                "Summarizer single output | max_input=%d | max_new=%d | min_new=%d "
                "| input_preview=%s | raw_preview=%s | finalized_preview=%s"
            ),
            settings.T5_MAX_INPUT_TOKENS,
            settings.T5_MAX_NEW_TOKENS,
            settings.T5_MIN_NEW_TOKENS,
            _preview(text),
            _preview(decoded),
            _preview(finalized),
        )
        return finalized

    def summarize_batch(self, texts: list[str]) -> list[str]:
        """Batch summarize for efficiency when processing multiple cases."""
        if not self.loaded:
            mock_outputs = [self._mock_summary(t) for t in texts]
            for idx, (src, out) in enumerate(zip(texts, mock_outputs)):
                logger.debug(
                    "Summarizer mock batch output | idx=%d | input_preview=%s | output_preview=%s",
                    idx,
                    _preview(src),
                    _preview(out),
                )
            return mock_outputs

        prefixed = [_PREFIX + t.strip() for t in texts]
        inputs = self._tokenizer(
            prefixed,
            return_tensors="pt",
            max_length=settings.T5_MAX_INPUT_TOKENS,
            truncation=True,
            padding=True,
        )
        if settings.DEVICE != "cpu" and torch.cuda.is_available():
            inputs = {k: v.to(settings.DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=settings.T5_MAX_NEW_TOKENS,
                min_new_tokens=settings.T5_MIN_NEW_TOKENS,
                num_beams=settings.T5_NUM_BEAMS,
                early_stopping=True,
            )

        raw_outputs = [self._tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        finalized_outputs = [self._finalize_summary(raw) for raw in raw_outputs]

        for idx, (src, raw, final) in enumerate(zip(texts, raw_outputs, finalized_outputs)):
            logger.debug(
                (
                    "Summarizer batch output | idx=%d | max_input=%d | max_new=%d | min_new=%d "
                    "| input_preview=%s | raw_preview=%s | finalized_preview=%s"
                ),
                idx,
                settings.T5_MAX_INPUT_TOKENS,
                settings.T5_MAX_NEW_TOKENS,
                settings.T5_MIN_NEW_TOKENS,
                _preview(src),
                _preview(raw),
                _preview(final),
            )

        return finalized_outputs

    def _finalize_summary(self, text: str) -> str:
        # Keep raw generation length, but drop a trailing sentence fragment.
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""

        if normalized[-1] in ".!?":
            return normalized

        complete = [
            m.group(0).strip()
            for m in re.finditer(r"[^.!?]+[.!?](?=\s|$)", normalized)
            if m.group(0).strip()
        ]
        if complete:
            return " ".join(complete)

        # If there are no complete sentences at all, return as-is.
        return normalized

    # ── Mock fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def _mock_summary(text: str) -> str:
        words = text.split()
        preview = " ".join(words[:25]) + ("…" if len(words) > 25 else "")
        return (
            f"[Mock summary — seq2seq model not loaded] "
            f"The document discusses: {preview}"
        )
