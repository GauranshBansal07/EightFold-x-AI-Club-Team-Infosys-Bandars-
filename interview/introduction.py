"""
Introduction Phase Handler — Manages the candidate introduction phase.
"""

from config import llm_call, llm_call_json
from prompts.system_interviewer import (
    INTERVIEWER_GREETING_PROMPT,
    get_introduction_analysis_prompt,
    INTRODUCTION_FOLLOWUP_PROMPT,
)


def generate_greeting(role_title: str) -> str:
    """Generate the initial greeting message."""
    prompt = INTERVIEWER_GREETING_PROMPT.format(role_title=role_title)
    try:
        greeting = llm_call(
            system_prompt="You are a friendly, professional AI interviewer called HireMind. Generate only the greeting text, nothing else.",
            user_prompt=prompt,
            temperature=0.7,
        )
        return greeting
    except Exception:
        return (
            f"Hi there! I'm HireMind, and I'll be conducting your interview today "
            f"for the {role_title} position. I'll ask you questions across a few areas "
            f"relevant to this role — take your time, think out loud if you'd like. "
            f"To start, could you tell me a bit about yourself? Your background, what "
            f"you've been working on recently, and any projects you're particularly proud of?"
        )


def process_introduction(
    introduction_text: str,
    jd_text: str,
    competency_list: list[str],
) -> dict:
    """
    Analyze the candidate's introduction.

    Returns:
    {
        "background_summary": str,
        "experience_level_confirmed": str,
        "projects": [{"name": str, "description": str, "relevant_competencies": [str], "potential_signals": str}],
        "competency_mapping_notes": str,
    }
    """
    comp_str = "\n".join(f"- {c}" for c in competency_list)

    system_prompt, user_prompt = get_introduction_analysis_prompt(
        jd_text=jd_text,
        competency_list=comp_str,
        introduction_text=introduction_text,
    )

    try:
        result = llm_call_json(system_prompt, user_prompt)
    except Exception:
        result = {
            "background_summary": introduction_text[:200],
            "experience_level_confirmed": "junior",
            "projects": [],
            "competency_mapping_notes": "Unable to analyze introduction — proceeding with defaults.",
        }

    return result


def generate_followup(introduction_text: str) -> str | None:
    """
    Generate a follow-up question if the introduction was too brief.
    Returns None if introduction seems sufficient.
    """
    # Simple heuristic: if intro is less than 50 words, ask a follow-up
    word_count = len(introduction_text.split())
    if word_count >= 50:
        return None

    prompt = INTRODUCTION_FOLLOWUP_PROMPT.format(introduction_text=introduction_text)
    try:
        return llm_call(
            system_prompt="You are a friendly AI interviewer. Generate only a brief follow-up question.",
            user_prompt=prompt,
            temperature=0.7,
        )
    except Exception:
        return "That's great! Could you tell me a bit more about one of those projects or what you've been working on recently?"
