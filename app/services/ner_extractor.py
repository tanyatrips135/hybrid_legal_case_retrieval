"""
Named Entity Recognition
========================
Uses RoBERTa base (Jean-Baptiste/roberta-large-ner-english by default) for
extracting PERSON, ORGANIZATION, and LOCATION entities from the case text.

The model is NOT fine-tuned; it runs zero-shot NER using the standard
CoNLL label set.  Swap ROBERTA_NER_MODEL in config.py for a custom checkpoint.
"""

from __future__ import annotations
import logging
import re
from typing import Any

from transformers import TokenClassificationPipeline
import transformers

from app.config import settings
from app.models.schemas import NamedEntity

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    # huggingface roberta-large-ner-english uses these group names
    "PER": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "PERSON": "PER",
    "ORGANIZATION": "ORG",
    "LOCATION": "LOC",
    "MISC": None,  # ignored
}

# Noise tokens produced by subword tokenization
_NOISE = re.compile(r"^(##|\u2581|Ġ)", re.UNICODE)


class NERExtractor:
    """Wraps RoBERTa NER pipeline."""

    def __init__(self) -> None:
        self._pipe: TokenClassificationPipeline | None = None
        self.loaded = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        model_id = str(settings.ROBERTA_NER_MODEL)
        logger.info("Loading RoBERTa NER from %s", model_id)

        # ── PLACEHOLDER ── The model is downloaded from HuggingFace Hub on
        # first run.  To run fully offline, download it once and point
        # ROBERTA_NER_MODEL in config.py to a local directory. ───────────────
        try:
            self._pipe = transformers.pipeline(
                "ner",
                model=model_id,
                aggregation_strategy="simple",
                device=0 if settings.DEVICE != "cpu" else -1,
            )
            self.loaded = True
            logger.info("RoBERTa NER loaded.")
        except Exception as exc:
            logger.warning(
                "RoBERTa NER could not be loaded (%s). "
                "Running in MOCK mode.",
                exc,
            )
            self._pipe = None
            self.loaded = False
        # ── END PLACEHOLDER ──────────────────────────────────────────────────

    # ── Inference ─────────────────────────────────────────────────────────────

    def extract(self, text: str) -> list[NamedEntity]:
        if not self.loaded or self._pipe is None:
            return self._mock_extract(text)

        keep_labels = set(settings.NER_KEEP_LABELS)
        raw: list[dict[str, Any]] = self._pipe(text[:2000])  # type: ignore[arg-type]

        entities: list[NamedEntity] = []
        for span in raw:
            raw_label: str = span.get("entity_group", span.get("entity", ""))
            mapped = _LABEL_MAP.get(raw_label.upper())

            if mapped is None or mapped not in keep_labels:
                continue

            word: str = _NOISE.sub("", span.get("word", "")).strip()
            if not word or len(word) < 2:
                continue

            score: float = float(span.get("score", 0.0))
            if score < 0.70:
                continue

            entities.append(
                NamedEntity(
                    text=word,
                    label=mapped,
                    start=int(span.get("start", 0)),
                    end=int(span.get("end", 0)),
                    score=round(score, 4),
                )
            )

        # deduplicate by (text, label)
        seen: set[tuple[str, str]] = set()
        unique: list[NamedEntity] = []
        for ent in entities:
            key = (ent.text.lower(), ent.label)
            if key not in seen:
                seen.add(key)
                unique.append(ent)

        return unique

    # ── Mock fallback ─────────────────────────────────────────────────────────

    @staticmethod
    def _mock_extract(text: str) -> list[NamedEntity]:
        """Returns a small set of placeholder entities."""
        return [
            NamedEntity(
                text="Supreme Court of India",
                label="ORG",
                start=0,
                end=22,
                score=0.99,
            ),
            NamedEntity(
                text="Delhi",
                label="LOC",
                start=0,
                end=5,
                score=0.95,
            ),
        ]
