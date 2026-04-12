"""
Report Generator — Post-interview structured evaluation report.
"""

import json
from config import llm_call_json


def generate_report(candidate_state: dict, interview_log: str) -> dict:
    """
    Generate the post-interview evaluation report.

    Args:
        candidate_state: The full in-memory state dict
        interview_log: The raw transcript text

    Returns:
    {
        "recruiter_report": {
            "candidate_name": str,
            "role": str,
            "experience_level": str,
            "competency_scores": [
                {
                    "competency": str,
                    "depth": "surface" | "partial" | "deep",
                    "confidence": "low" | "medium" | "high",
                    "exit_condition": str,
                    "key_evidence": str,
                    "key_gap": str,
                    "tradeoff_awareness": str,
                }
            ],
            "ranked_summary": str,
            "difficulty_trajectory": str,
            "flags": [str],
        },
        "candidate_feedback": {
            "strengths": [str],
            "development_areas": [str],
            "overall_impression": str,
        }
    }
    """
    system_prompt = """You are an interview evaluation report generator. Produce a comprehensive, fair, and evidence-based evaluation report.

Given the candidate state and interview log, generate a structured report.

CRITICAL RULES:
1. Every score must be backed by specific evidence from the interview.
2. Tradeoff awareness is ONLY a negative signal if the candidate was directly asked about tradeoffs and couldn't answer. If tradeoffs were "not_probed", that is the SYSTEM's gap, not the candidate's — do not penalize.
3. Frame the report relative to the candidate's declared experience level.
4. Candidate feedback must be constructive — acknowledge strengths before gaps. No harsh criticism.
5. Cite specific quotes or paraphrases from the interview log.

Return ONLY a JSON object with this schema:
{
  "recruiter_report": {
    "candidate_name": "name",
    "experience_level": "level",
    "competency_scores": [
      {
        "competency": "name",
        "depth": "surface" or "partial" or "deep",
        "confidence": "low" or "medium" or "high",
        "exit_condition": "depth_reached" or "budget_exhausted" or "plateau_detected",
        "key_evidence": "strongest signal with quote",
        "key_gap": "most significant gap",
        "tradeoff_awareness": "not_probed" or "probed_and_absent" or "probed_and_present"
      }
    ],
    "ranked_summary": "Strongest signal: [competency]. Weakest signal: [competency].",
    "difficulty_trajectory": "description of how difficulty changed over the interview and why",
    "flags": ["any concerns — drift patterns, accuracy issues, plateau patterns"]
  },
  "candidate_feedback": {
    "strengths": ["specific things the candidate did well"],
    "development_areas": ["specific areas to develop, phrased constructively"],
    "overall_impression": "1-2 sentence overall assessment"
  }
}

IMPORTANT: Return ONLY the JSON object. No explanations, no markdown fences."""

    # Build a condensed state summary for the prompt
    state_summary = _build_state_summary(candidate_state)

    # Truncate log if too long (to fit in context window)
    log_text = interview_log
    if len(log_text) > 6000:
        log_text = log_text[:6000] + "\n\n[... log truncated for brevity ...]"

    user_prompt = f"""CANDIDATE STATE SUMMARY:
{state_summary}

INTERVIEW LOG:
{log_text}"""

    try:
        report = llm_call_json(system_prompt, user_prompt)
    except Exception as e:
        # Fallback: build basic report from state
        report = _build_fallback_report(candidate_state)

    return report


def _build_state_summary(state: dict) -> str:
    """Build a condensed text summary of candidate state for the report prompt."""
    lines = []
    p = state["profile"]
    lines.append(f"Candidate: {p['name']}")
    lines.append(f"Experience Level: {p['experience_level']}")
    lines.append(f"Background: {p['background_summary']}")
    lines.append(f"Difficulty Calibration: {p['difficulty_calibration']}")
    lines.append(f"Total Exchanges: {state['progress']['total_exchanges']}")
    lines.append("")

    for comp_name, comp in state["competency_signals"].items():
        lines.append(f"### {comp_name}")
        lines.append(f"Status: {comp['status']}")
        lines.append(f"Probes: {comp['probes_completed']}")
        lines.append(f"Exit: {comp.get('exit_condition', 'N/A')}")
        lines.append(f"Tradeoffs: {comp.get('tradeoff_awareness', 'not_probed')}")
        lines.append(f"Contradiction Tested: {comp.get('contradiction_tested', False)}")
        lines.append(f"Understanding Score: {comp.get('understanding_score', 0.0):.2f}")
        lines.append(f"Probe Intensity Level: {comp.get('probe_intensity_level', 1)}")

        if comp.get("verdict"):
            v = comp["verdict"]
            lines.append(f"Depth: {v.get('depth_assessment', 'N/A')}")
            lines.append(f"Confidence: {v.get('confidence', 'N/A')}")
            lines.append(f"Evidence: {v.get('key_evidence', 'N/A')}")
            lines.append(f"Gap: {v.get('key_gap', 'N/A')}")

        for i, probe in enumerate(comp.get("probe_history", []), 1):
            lines.append(f"  Probe {i}:")
            lines.append(f"    Q: {probe.get('question', '?')[:150]}")
            lines.append(f"    gap={probe.get('gap_diagnosis', '?')}, "
                        f"reasoning={probe.get('causal_reasoning', '?')}, "
                        f"grounding={probe.get('evidence_grounding', '?')}, "
                        f"accuracy={probe.get('accuracy', '?')}")
            lines.append(f"    novelty={probe.get('novelty', '?')}, "
                        f"dimension={probe.get('dimension_probed', '?')}, "
                        f"adaptability={probe.get('adaptability', '?')}, "
                        f"struggle={probe.get('struggle_detected', False)}")
            missing = probe.get("what_is_missing", "")
            if missing and missing != "None":
                lines.append(f"    missing: {missing[:120]}")
        lines.append("")

    if state.get("cross_competency_signals"):
        lines.append("Cross-Competency Signals:")
        for sig in state["cross_competency_signals"]:
            lines.append(f"  - {sig}")

    if state.get("difficulty_calibration_log"):
        lines.append("Calibration Log:")
        for entry in state["difficulty_calibration_log"]:
            lines.append(f"  - After {entry['after_competency']}: {entry['calibration']} — {entry['reasoning']}")

    return "\n".join(lines)


def _build_fallback_report(state: dict) -> dict:
    """Build a comprehensive report from state without LLM, including full probe history."""
    scores = []
    for comp_name, comp in state["competency_signals"].items():
        v = comp.get("verdict", {})

        # Build per-probe detail array
        probe_details = []
        for i, probe in enumerate(comp.get("probe_history", []), 1):
            probe_details.append({
                "probe_number": i,
                "question": probe.get("question", "N/A"),
                "gap_diagnosis": probe.get("gap_diagnosis", "N/A"),
                "causal_reasoning": probe.get("causal_reasoning", "N/A"),
                "evidence_grounding": probe.get("evidence_grounding", "N/A"),
                "accuracy": probe.get("accuracy", "N/A"),
                "novelty": probe.get("novelty", "N/A"),
                "dimension_probed": probe.get("dimension_probed", "N/A"),
                "adaptability": probe.get("adaptability", "N/A"),
                "struggle_detected": probe.get("struggle_detected", False),
                "what_is_missing": probe.get("what_is_missing", "N/A"),
                "follow_up_strategy": probe.get("follow_up_strategy", "N/A"),
            })

        scores.append({
            "competency": comp_name,
            "depth": v.get("depth_assessment", "surface"),
            "confidence": v.get("confidence", "low"),
            "exit_condition": comp.get("exit_condition", "unknown"),
            "key_evidence": v.get("key_evidence", "N/A"),
            "key_gap": v.get("key_gap", "N/A"),
            "tradeoff_awareness": comp.get("tradeoff_awareness", "not_probed"),
            "contradiction_tested": comp.get("contradiction_tested", False),
            "understanding_score": round(comp.get("understanding_score", 0.0), 3),
            "probe_intensity_level": comp.get("probe_intensity_level", 1),
            "probes_completed": comp.get("probes_completed", 0),
            "probe_details": probe_details,
        })

    # Sort by depth for ranking
    depth_order = {"deep": 2, "partial": 1, "surface": 0}
    sorted_scores = sorted(scores, key=lambda x: depth_order.get(x["depth"], 0), reverse=True)

    return {
        "recruiter_report": {
            "candidate_name": state["profile"]["name"],
            "experience_level": state["profile"]["experience_level"],
            "competency_scores": scores,
            "ranked_summary": f"Strongest: {sorted_scores[0]['competency']}. Weakest: {sorted_scores[-1]['competency']}." if sorted_scores else "No scores.",
            "difficulty_trajectory": state["profile"]["difficulty_calibration"],
            "flags": [],
        },
        "candidate_feedback": {
            "strengths": ["Participated in all competency areas."],
            "development_areas": ["Report generated from fallback — detailed feedback not available."],
            "overall_impression": "The interview covered all planned competencies. See individual scores for details.",
        },
    }

