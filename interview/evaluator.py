"""
Evaluator — Call 1 of the three-call-per-turn architecture.
Assesses response depth with integrated drift detection.
"""

import json
from config import llm_call_json
from prompts.system_evaluator import get_evaluator_prompt


def evaluate_response(
    question_text: str,
    response_text: str,
    competency_name: str,
    cognitive_task: str,
    depth_spec: str,
    experience_level: str,
) -> dict:
    """
    Evaluate a candidate's response against a specific competency.

    Integrates drift detection (the drift detector runs as part of evaluation).

    Returns structured evaluation:
    {
        "drift_check": "on-scope" | "off-scope",
        "drift_description": str | None,
        "accuracy": "correct" | "partially_correct" | "incorrect",
        "accuracy_note": str | None,
        "causal_reasoning": "absent" | "partial" | "sufficient",
        "causal_evidence": str,
        "evidence_grounding": "absent" | "partial" | "sufficient",
        "grounding_evidence": str,
        "gap_diagnosis": str,
        "what_is_missing": str,
    }
    """
    system_prompt, user_prompt = get_evaluator_prompt(
        competency_name=competency_name,
        depth_spec=depth_spec,
        cognitive_task=cognitive_task,
        experience_level=experience_level,
        question_text=question_text,
        response_text=response_text,
    )

    try:
        result = llm_call_json(system_prompt, user_prompt)
    except Exception as e:
        # Fallback: return a conservative evaluation
        result = {
            "drift_check": "on-scope",
            "drift_description": None,
            "accuracy": "partially_correct",
            "accuracy_note": f"Evaluation failed: {str(e)}",
            "causal_reasoning": "partial",
            "causal_evidence": "Unable to evaluate — API error",
            "evidence_grounding": "partial",
            "grounding_evidence": "Unable to evaluate — API error",
            "gap_diagnosis": "evaluation_failed",
            "what_is_missing": "Evaluation could not be completed due to API error",
            "novelty": "rephrased",
            "dimension_probed": "unknown",
            "adaptability": "not_applicable",
            "struggle_detected": False,
        }

    # Ensure gap_diagnosis overrides to scope_drift if drift detected
    if result.get("drift_check") == "off-scope":
        result["gap_diagnosis"] = "scope_drift"

    # Ensure accuracy gate: if incorrect, override gap to accuracy_low
    if result.get("accuracy") == "incorrect":
        result["gap_diagnosis"] = "accuracy_low"

    # Validate all required fields
    required_fields = [
        "drift_check", "accuracy", "causal_reasoning",
        "evidence_grounding", "gap_diagnosis", "what_is_missing",
        "novelty", "dimension_probed", "adaptability"
    ]
    for field in required_fields:
        if field not in result:
            result[field] = "N/A"
            
    if "struggle_detected" not in result:
        result["struggle_detected"] = False

    return result
