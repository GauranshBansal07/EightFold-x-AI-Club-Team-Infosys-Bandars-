"""
JD Parser — Extracts structured role profile from raw job description text.
Sends JD to LLM and returns role title, competencies, cognitive tasks, technologies.
"""

from config import llm_call_json


def parse_jd(jd_text: str) -> dict:
    """
    Parse a raw job description into a structured role profile.

    Returns:
        {
            "role_title": str,
            "competencies": [
                {
                    "name": str,
                    "description": str,
                    "cognitive_task": str,  # decision / explanation / comparison / design / debugging
                }
            ],
            "technologies_domains": [str],
        }
    """
    system_prompt = """You are a job description analyzer. Your task is to extract structured information from a job description for use in an adaptive interview system.

Extract the following and return ONLY a JSON object:

{
  "role_title": "the job title",
  "competencies": [
    {
      "name": "competency name (e.g., 'Problem Solving', 'System Design', 'Analytical Reasoning', 'Communication Clarity') — these are COMPETENCIES, not technologies or skills",
      "description": "brief description of what this competency means for this specific role",
      "cognitive_task": "the cognitive task type that best tests this competency — one of: decision, explanation, comparison, design, debugging"
    }
  ],
  "technologies_domains": ["key technologies or domains mentioned — these inform question context, NOT evaluation criteria"]
}

RULES:
1. Extract 3-5 core COMPETENCIES — not skills or technologies. Competencies are things like "problem solving", "system design", "analytical reasoning", "communication clarity", "data modeling", "stakeholder management".
2. Technologies like "Python", "React", "AWS" go in technologies_domains, not competencies.
3. Each competency must have a cognitive_task that best tests it. Choose from: decision, explanation, comparison, design, debugging.
4. Keep competency descriptions specific to THIS role, not generic.

IMPORTANT: Return ONLY the JSON object. No explanations, no markdown fences."""

    user_prompt = f"""JOB DESCRIPTION:
{jd_text}"""

    result = llm_call_json(system_prompt, user_prompt)

    # Validate structure
    assert "role_title" in result, "Missing role_title in parsed JD"
    assert "competencies" in result, "Missing competencies in parsed JD"
    assert len(result["competencies"]) >= 3, "Need at least 3 competencies"

    # Cap at 5 competencies
    if len(result["competencies"]) > 5:
        result["competencies"] = result["competencies"][:5]

    return result
