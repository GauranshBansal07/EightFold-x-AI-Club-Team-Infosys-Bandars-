"""
Question Generator — Call 3 of the three-call-per-turn architecture.
Generates contextual, conversational interview questions.
"""

import json
from config import llm_call_json
from prompts.system_question_gen import get_question_gen_prompt
from memory.candidate_state import CandidateState
from memory.state_renderer import render_state_for_prompt


def generate_question(
    strategist_decision: dict,
    candidate_state: CandidateState,
    competency_specs: dict,
    candidate_background: str,
    difficulty_level: str,
) -> dict:
    """
    Generate the next interview question.

    Returns:
    {
        "question_text": str,
        "cognitive_task": str,
        "target_competency": str,
        "internal_rationale": str,
    }
    """
    state_content = render_state_for_prompt(candidate_state)
    decision_str = json.dumps(strategist_decision, indent=2)

    # Get the spec for the target competency
    target_comp = strategist_decision.get("target_competency", "")
    comp_spec = competency_specs.get(target_comp, {})
    comp_spec_str = json.dumps(comp_spec, indent=2) if comp_spec else "No specific spec available."

    system_prompt, user_prompt = get_question_gen_prompt(
        candidate_state_content=state_content,
        strategist_decision_json=decision_str,
        target_competency_spec=comp_spec_str,
        candidate_background=candidate_background,
        difficulty_level=difficulty_level,
    )

    try:
        result = llm_call_json(system_prompt, user_prompt)
    except Exception as e:
        # Fallback: generate a generic question
        result = {
            "question_text": f"Can you tell me about your experience with {target_comp}? Walk me through a specific example or project where this was important.",
            "cognitive_task": "explanation",
            "target_competency": target_comp,
            "internal_rationale": f"Fallback generic question due to API error: {str(e)}",
        }

    # Validate required fields
    if "question_text" not in result:
        result["question_text"] = f"Tell me about your approach to {target_comp}."
    if "cognitive_task" not in result:
        result["cognitive_task"] = "explanation"
    if "target_competency" not in result:
        result["target_competency"] = target_comp
    if "internal_rationale" not in result:
        result["internal_rationale"] = "Auto-generated question"

    return result
