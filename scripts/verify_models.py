#!/usr/bin/env python3
"""
scripts/verify_models.py
------------------------
Run this before starting the server to confirm all model artefacts are in place.

    python scripts/verify_models.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

CHECKS = [
    # (path, description, required)
    (ROOT / "app/models/legal_bert_issue_extraction/config.json",       "LegalBERT config",           True),
    (ROOT / "app/models/legal_bert_issue_extraction/model.safetensors", "LegalBERT weights",           True),
    (ROOT / "app/models/legal_bert_issue_extraction/tokenizer.json",    "LegalBERT tokenizer",         True),
    (ROOT / "app/models/t5-legal-explainer/config.json",                "T5 config",                   True),
    (ROOT / "app/models/t5-legal-explainer/model.safetensors",          "T5 weights",                  True),
    (ROOT / "app/models/t5-legal-explainer/tokenizer.json",             "T5 tokenizer",                True),
    (ROOT / "app/models/all-mpnet-base-v2/legal_cases.faiss",           "FAISS index",                 True),
    (ROOT / "app/models/all-mpnet-base-v2/legal_cases_meta.jsonl",      "Case metadata JSONL",         True),
    (ROOT / "app/models/all-mpnet-base-v2/legal_embeddings.npy",        "Pre-computed embeddings",     False),
]

OK, WARN, FAIL = "✅", "⚠️ ", "❌"

errors = 0
for path, desc, required in CHECKS:
    if path.exists():
        size_mb = path.stat().st_size / 1_048_576
        print(f"  {OK}  {desc:45s} ({size_mb:.1f} MB)")
    elif required:
        print(f"  {FAIL}  {desc:45s} — NOT FOUND: {path.relative_to(ROOT)}")
        errors += 1
    else:
        print(f"  {WARN}  {desc:45s} — not found (optional): {path.relative_to(ROOT)}")

# Spot-check metadata JSONL
meta_path = ROOT / "app/models/all-mpnet-base-v2/legal_cases_meta.jsonl"
if meta_path.exists():
    with meta_path.open() as f:
        first_line = f.readline().strip()
    if first_line:
        try:
            rec = json.loads(first_line)
            # required_fields = {"case_id", "title", "text"}
            required_fields = {"case_id", "text_snippet", "source"}
            missing = required_fields - rec.keys()
            if missing:
                print(f"\n  {WARN}  metadata.jsonl first record missing fields: {missing}")
            else:
                print(f"\n  {OK}  metadata.jsonl schema OK (sample case_id: {rec['case_id']})")
        except json.JSONDecodeError as e:
            print(f"\n  {FAIL}  metadata.jsonl first line is not valid JSON: {e}")
            errors += 1

print()
if errors:
    print(f"  {errors} required file(s) missing. Fix before starting the server.")
    sys.exit(1)
else:
    print("  All required artefacts present. You can now run:  uvicorn main:app --reload")
