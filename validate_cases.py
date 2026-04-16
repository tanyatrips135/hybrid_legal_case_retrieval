import os
import json
from typing import List, Dict
from groq import Groq

# ----------------------------
# CONFIG
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"  # strong reasoning

client = Groq(api_key=GROQ_API_KEY)

# ----------------------------
# PROMPT TEMPLATE
# ----------------------------
def build_validation_prompt(query_summary: str, case_summaries: List[str]) -> str:
    cases_text = "\n\n".join([f"Case {i+1}: {c}" for i, c in enumerate(case_summaries)])

    prompt = f"""
You are a LEGAL RELEVANCE VALIDATION SYSTEM.

TASK:
Determine whether each retrieved case is relevant to the given query case.

QUERY CASE SUMMARY:
{query_summary}

RETRIEVED CASE SUMMARIES:
{cases_text}

---------------------------------------
SCORING METRICS (0 to 5):
1. LEGAL ISSUE MATCH
2. FACTUAL SIMILARITY
3. ENTITY OVERLAP (people, acts, sections)
4. OUTCOME RELEVANCE
5. CONTEXTUAL ALIGNMENT

---------------------------------------
INSTRUCTIONS:
- Score each metric (0-5)
- Compute FINAL_SCORE = average of all 5 metrics
- Classify:
    - FINAL_SCORE >= 4 → "HIGHLY_RELEVANT"
    - FINAL_SCORE >= 3 → "MODERATELY_RELEVANT"
    - FINAL_SCORE >= 2 → "WEAKLY_RELEVANT"
    - else → "NOT_RELEVANT"
- Provide short reasoning (1-2 lines)

---------------------------------------
OUTPUT STRICT JSON FORMAT:

{{
  "cases": [
    {{
      "case_id": 1,
      "scores": {{
        "legal_issue": int,
        "factual_similarity": int,
        "entity_overlap": int,
        "outcome_relevance": int,
        "context_alignment": int
      }},
      "final_score": float,
      "label": "HIGHLY_RELEVANT | MODERATELY_RELEVANT | WEAKLY_RELEVANT | NOT_RELEVANT",
      "reason": "short explanation"
    }}
  ],
  "summary": {{
    "relevant_count": int,
    "highly_relevant_count": int,
    "decision": "SHOW_RESULTS | REJECT_RESULTS"
  }}
}}

---------------------------------------
FINAL DECISION RULE:
- If >= 2 cases are HIGHLY_RELEVANT → SHOW_RESULTS
- If >= 50% cases are at least MODERATELY_RELEVANT → SHOW_RESULTS
- Else → REJECT_RESULTS

RETURN ONLY JSON. NO EXTRA TEXT.
"""
    return prompt


# ----------------------------
# GROQ CALL
# ----------------------------
def validate_cases(query_summary: str, case_summaries: List[str]) -> Dict:
    prompt = build_validation_prompt(query_summary, case_summaries)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a strict legal evaluator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # deterministic
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except Exception:
        # fallback if LLM breaks format
        return {"error": "Invalid JSON from LLM", "raw": response.choices[0].message.content}

    return result


# ----------------------------
# OPTIONAL: POST-PROCESS METRICS
# ----------------------------
def compute_aggregate_metrics(result: Dict) -> Dict:
    cases = result.get("cases", [])

    avg_score = sum(c["final_score"] for c in cases) / len(cases) if cases else 0

    distribution = {
        "HIGHLY_RELEVANT": 0,
        "MODERATELY_RELEVANT": 0,
        "WEAKLY_RELEVANT": 0,
        "NOT_RELEVANT": 0
    }

    for c in cases:
        distribution[c["label"]] += 1

    return {
        "average_score": avg_score,
        "distribution": distribution
    }


# ----------------------------
# USAGE EXAMPLE
# ----------------------------
if __name__ == "__main__":
    query = "Dispute regarding breach of contract involving delayed delivery and financial damages."

    cases = [
        "Case involving breach of contract due to delayed shipment and compensation claim.",
        "Criminal case related to theft and assault.",
        "Commercial dispute regarding contract termination and damages."
    ]

    result = validate_cases(query, cases)

    print("LLM Validation Output:\n", json.dumps(result, indent=2))

    metrics = compute_aggregate_metrics(result)
    print("\nAggregate Metrics:\n", json.dumps(metrics, indent=2))