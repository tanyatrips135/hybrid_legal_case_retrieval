"""
Legal Issue Extraction
======================
Uses the LegalBERT model fine-tuned on the IndicLegalQA dataset.

The model is a token-classifier / sequence-classifier that tags spans of
text with legal-issue labels (e.g. "BAIL", "CONTRACT", "PROPERTY", etc.).
We wrap it with a transformers pipeline and post-process the results.
"""

from __future__ import annotations
import logging
from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
import transformers

from app.config import settings
from app.models.schemas import LegalIssue

logger = logging.getLogger(__name__)

# ── Label normalisation map ───────────────────────────────────────────────────
# Adapt this to match the actual label set in config.json produced during
# fine-tuning on IndicLegalQA.  Keys are whatever the model outputs.
LABEL_DISPLAY = {
    "LABEL_0": "No Issue",
    "LABEL_1": "Constitutional Matter",
    "LABEL_2": "Criminal Offence",
    "LABEL_3": "Civil Dispute",
    "LABEL_4": "Property / Land",
    "LABEL_5": "Family / Matrimonial",
    "LABEL_6": "Commercial / Contract",
    "LABEL_7": "Service / Employment",
    "LABEL_8": "Taxation",
    "LABEL_9": "Bail / Anticipatory Bail",
    "LABEL_10": "Writ Petition",
    "LABEL_11": "Intellectual Property",
    "LABEL_12": "Environmental",
    "LABEL_13": "Banking / Finance",
    "LABEL_14": "Arbitration",
    # extend as needed based on actual fine-tuning labels
}


class IssueExtractor:
    """Wraps the fine-tuned LegalBERT for legal-issue span extraction."""

    def __init__(self) -> None:
        self._pipe: Any = None
        self.loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        model_dir = str(settings.LEGAL_BERT_DIR)
        logger.info("Loading LegalBERT from %s", model_dir)

        # ── PLACEHOLDER ── replace this block when the real model dir is
        # present on the inference machine. ──────────────────────────────────
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_dir, local_files_only=True
            )
            model.eval()
            if settings.DEVICE != "cpu" and torch.cuda.is_available():
                model = model.to(settings.DEVICE)

            self._pipe = transformers.pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if settings.DEVICE != "cpu" else -1,
            )
            self.loaded = True
            logger.info("LegalBERT loaded successfully.")
        except Exception as exc:
            logger.warning(
                "LegalBERT could not be loaded (%s). "
                "Running in MOCK mode — replace model dir to enable real inference.",
                exc,
            )
            self._pipe = None
            self.loaded = False
        # ── END PLACEHOLDER ──────────────────────────────────────────────────

    # ── Inference ─────────────────────────────────────────────────────────────

    def extract(self, text: str) -> list[LegalIssue]:
        if not self.loaded or self._pipe is None:
            return self._mock_extract(text)

        # Truncate to model max tokens
        results = self._pipe(text[:2000])

        issues: list[LegalIssue] = []
        seen: set[str] = set()

        for span in results:
            raw_label: str = span.get("entity_group", span.get("entity", ""))
            display = LABEL_DISPLAY.get(raw_label, raw_label)

            if display == "No Issue":
                continue

            word: str = span.get("word", "").strip()
            score: float = float(span.get("score", 0.0))

            if score < 0.50:
                continue

            key = f"{display}:{word[:30]}"
            if key not in seen:
                seen.add(key)
                issues.append(
                    LegalIssue(label=display, score=round(score, 4), text=word)
                )

        # de-dup by label, keep highest score per label
        best: dict[str, LegalIssue] = {}
        for iss in issues:
            if iss.label not in best or iss.score > best[iss.label].score:
                best[iss.label] = iss

        return sorted(best.values(), key=lambda x: x.score, reverse=True)

    # ── Mock fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def _mock_extract(text: str) -> list[LegalIssue]:
        """Rule-based fallback used when the model directory is absent."""
        keywords = {
            "bail": "Bail / Anticipatory Bail",
            "murder": "Criminal Offence",
            "assault": "Criminal Offence",
            "property": "Property / Land",
            "contract": "Commercial / Contract",
            "divorce": "Family / Matrimonial",
            "matrimon": "Family / Matrimonial",
            "writ": "Writ Petition",
            "constitution": "Constitutional Matter",
            "tax": "Taxation",
            "employ": "Service / Employment",
            "arbitrat": "Arbitration",
            "patent": "Intellectual Property",
            "trademark": "Intellectual Property",
            "environ": "Environmental",
        }
        lower = text.lower()
        found: dict[str, LegalIssue] = {}
        for kw, label in keywords.items():
            if kw in lower and label not in found:
                found[label] = LegalIssue(label=label, score=0.75, text=kw)
        return list(found.values()) or [
            LegalIssue(label="Civil Dispute", score=0.60, text="general")
        ]
