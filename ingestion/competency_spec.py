"""
Competency Spec Generator — Generates per-competency depth specifications.
Cached after generation; not regenerated per turn.
"""

from config import llm_call_json


def generate_competency_specs(parsed_jd: dict, experience_level: str) -> dict:
    """
    Generate detailed depth specifications for each competency.

    Returns a dict keyed by competency name:
    {
        "Competency Name": {
            "depth_spec": str,
            "common_drift_patterns": [str],
            "probe_templates": {
                "reasoning_absent": str,
                "evidence_absent": str,
                "textbook_pattern": str,
                "scope_drift": str,
                "accuracy_low": str,
            },
            "tradeoff_probes": [str],
        }
    }
    """
    role_title = parsed_jd["role_title"]
    technologies = ", ".join(parsed_jd.get("technologies_domains", []))

    system_prompt = f"""You are an interview design specialist. Generate detailed depth specifications for interview competencies.

ROLE: {role_title}
CANDIDATE EXPERIENCE LEVEL: {experience_level}
KEY TECHNOLOGIES/DOMAINS: {technologies}

For each competency provided, generate a detailed specification and return ONLY a JSON object where each key is the competency name.

Each competency spec should have this structure:
{{
  "CompetencyName": {{
    "depth_spec": "What does 'sufficient depth' look like for this competency in THIS role? Be specific. For a backend engineering role, 'problem solving' depth means identifying constraints, considering algorithmic alternatives, reasoning about complexity tradeoffs. For a product management role, it means decomposing user problems, identifying competing metrics, reasoning about prioritization.",
    "common_drift_patterns": ["Where do candidates typically escape to when weak on this? Be specific to this competency."],
    "probe_templates": {{
      "reasoning_absent": "A template follow-up question for when the candidate stated WHAT but not WHY",
      "evidence_absent": "A template follow-up question for when the candidate gave abstract/theoretical answers",
      "textbook_pattern": "A constraint introduction template that invalidates the standard answer — must be realistic for this role",
      "scope_drift": "A redirect template that acknowledges drift and steers back",
      "accuracy_low": "A reframe template that approaches from a different angle"
    }},
    "tradeoff_probes": ["2-3 specific tradeoff questions for this competency — these are question-generation resources, not evaluation criteria"]
  }}
}}

IMPORTANT: Return ONLY the JSON object. No explanations, no markdown fences."""

    competency_list = "\n".join(
        f"- {c['name']}: {c['description']} (cognitive task: {c['cognitive_task']})"
        for c in parsed_jd["competencies"]
    )

    user_prompt = f"""COMPETENCIES TO SPECIFY:
{competency_list}"""

    result = llm_call_json(system_prompt, user_prompt)
    return result
