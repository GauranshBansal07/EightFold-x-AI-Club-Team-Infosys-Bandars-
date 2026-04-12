"""
Drift Detector — Detects when candidate responses drift from the asked question.
Integrated into the evaluator (Call 1), this provides helper logic and validation.
"""


def detect_drift_signals(
    question_cognitive_task: str,
    response_text: str,
) -> dict:
    """
    Lightweight local drift detection to supplement the LLM evaluator.

    This checks for obvious structural mismatches between the question type
    and the response shape. The LLM evaluator does the detailed drift analysis;
    this catches clear cases for logging and validation.

    Returns:
    {
        "likely_drift": bool,
        "drift_type": str | None,
        "notes": str,
    }
    """
    response_lower = response_text.lower().strip()
    word_count = len(response_text.split())

    # Very short responses are suspicious but not necessarily drift
    if word_count < 10:
        return {
            "likely_drift": False,
            "drift_type": None,
            "notes": "Very brief response — may indicate uncertainty or lack of knowledge.",
        }

    # Check for common drift patterns based on cognitive task
    if question_cognitive_task == "decision":
        # Decision questions expect: choice + justification
        # Drift signal: no choice indicators, just mechanism explanation
        choice_indicators = ["i would", "i'd choose", "i prefer", "the best", "i'd go with",
                             "i think", "my choice", "i'd pick", "i'd use", "i recommend",
                             "the right approach", "better to"]
        has_choice = any(ind in response_lower for ind in choice_indicators)
        if not has_choice:
            return {
                "likely_drift": True,
                "drift_type": "decision_to_explanation",
                "notes": "Decision question answered with explanation/description — no clear choice made.",
            }

    elif question_cognitive_task == "comparison":
        # Comparison questions expect: both options discussed
        # Drift signal: only one option addressed
        comparison_indicators = ["on the other hand", "whereas", "compared to", "unlike",
                                 "in contrast", "however", "but", "while", "alternatively",
                                 "versus", " vs ", "difference"]
        has_comparison = any(ind in response_lower for ind in comparison_indicators)
        if not has_comparison and word_count > 30:
            return {
                "likely_drift": True,
                "drift_type": "comparison_to_single",
                "notes": "Comparison question but response only addresses one option.",
            }

    elif question_cognitive_task == "design":
        # Design questions expect: structured approach with constraints
        # Drift signal: just definitions
        design_indicators = ["i would design", "the architecture", "components",
                            "workflow", "steps", "first", "then", "the system",
                            "we could", "the approach", "implementation",
                            "layer", "module", "service"]
        has_design = any(ind in response_lower for ind in design_indicators)
        if not has_design and word_count > 30:
            return {
                "likely_drift": True,
                "drift_type": "design_to_definition",
                "notes": "Design question answered with definitions rather than system construction.",
            }

    return {
        "likely_drift": False,
        "drift_type": None,
        "notes": "No obvious structural drift detected.",
    }
