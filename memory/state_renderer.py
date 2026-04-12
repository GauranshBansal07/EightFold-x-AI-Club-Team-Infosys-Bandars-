"""
State Renderer — Reads candidate state and formats it for system prompt injection.
"""

from memory.candidate_state import CandidateState


def render_state_for_prompt(candidate_state: CandidateState) -> str:
    """
    Render the current candidate state into a string suitable for
    injection into system prompts for the evaluator, strategist,
    and question generator.
    """
    md_content = candidate_state.get_state_md()
    if md_content:
        return md_content

    # Fallback: render from in-memory state
    state = candidate_state.get_state()
    lines = []
    lines.append(f"Candidate: {state['profile']['name']}")
    lines.append(f"Experience: {state['profile']['experience_level']}")
    lines.append(f"Phase: {state['progress']['current_phase']}")
    lines.append(f"Current Competency: {state['progress']['current_competency']}")
    lines.append(f"Remaining: {', '.join(state['progress']['competencies_remaining'])}")
    lines.append(f"Difficulty: {state['profile']['difficulty_calibration']}")
    lines.append(f"Total Exchanges: {state['progress']['total_exchanges']}")
    return "\n".join(lines)
