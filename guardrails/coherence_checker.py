"""
Coherence Checker — Validates interviewer question coherence.
Runs between Call 2 (Strategist) and Call 3 (Question Generator).
Updated for adversarial strategies and contradiction requirements.
"""

from memory.candidate_state import CandidateState

# All valid strategies (including new adversarial ones)
VALID_STRATEGIES = {
    "redirect", "reframe", "probe_reasoning", "probe_evidence",
    "break_textbook", "forced_tradeoff", "contradiction_test",
    "adversarial_constraint", "extreme_scenario", "probe_tradeoffs",
}


def check_coherence(
    strategist_decision: dict,
    candidate_state: CandidateState,
) -> dict:
    """
    Verify the strategist's decision doesn't produce a redundant question.

    Checks:
    1. Target competency hasn't already been fully scored
    2. Follow-up strategy doesn't re-probe an established signal
    3. Strategy is valid
    4. Contradiction tested before advancing
    5. Cross-competency signal redundancy

    Returns:
    {
        "is_coherent": bool,
        "adjusted_decision": dict | None,
        "adjustment_reason": str | None,
    }
    """
    state = candidate_state.get_state()
    action = strategist_decision.get("action")
    target = strategist_decision.get("target_competency")
    strategy = strategist_decision.get("follow_up_strategy")

    # ─── Check 1: Target competency not already scored ─────────────
    if target and target in state["competency_signals"]:
        comp_data = state["competency_signals"][target]
        if comp_data["status"] == "scored":
            for comp_name, comp_info in state["competency_signals"].items():
                if comp_info["status"] != "scored":
                    return {
                        "is_coherent": False,
                        "adjusted_decision": {
                            **strategist_decision,
                            "action": "advance",
                            "target_competency": comp_name,
                            "follow_up_strategy": None,
                        },
                        "adjustment_reason": f"Competency '{target}' already scored. Redirecting to '{comp_name}'.",
                    }
            # All scored
            return {
                "is_coherent": False,
                "adjusted_decision": {
                    **strategist_decision,
                    "action": "advance",
                    "target_competency": None,
                    "follow_up_strategy": None,
                },
                "adjustment_reason": "All competencies already scored. Interview should complete.",
            }

    # ─── Check 2: Don't re-probe established signals ──────────────
    if action == "follow_up" and target and strategy:
        comp_data = state["competency_signals"].get(target, {})
        probe_history = comp_data.get("probe_history", [])

        if strategy == "probe_reasoning":
            for probe in probe_history:
                if probe.get("causal_reasoning") == "sufficient":
                    # Reasoning established → escalate to adversarial
                    return {
                        "is_coherent": False,
                        "adjusted_decision": {
                            **strategist_decision,
                            "follow_up_strategy": "adversarial_constraint" if not comp_data.get("contradiction_tested") else "probe_evidence",
                        },
                        "adjustment_reason": "Causal reasoning already sufficient. Escalating to adversarial_constraint.",
                    }

        elif strategy == "probe_evidence":
            for probe in probe_history:
                if probe.get("evidence_grounding") == "sufficient":
                    # Evidence established → use contradiction or tradeoff
                    if not comp_data.get("contradiction_tested"):
                        next_strat = "contradiction_test"
                    elif comp_data.get("tradeoff_awareness") == "not_probed":
                        next_strat = "forced_tradeoff"
                    else:
                        next_strat = "break_textbook"
                    return {
                        "is_coherent": False,
                        "adjusted_decision": {
                            **strategist_decision,
                            "follow_up_strategy": next_strat,
                        },
                        "adjustment_reason": f"Evidence grounding already established. Switching to {next_strat}.",
                    }

        elif strategy in ("probe_tradeoffs", "forced_tradeoff"):
            if comp_data.get("tradeoff_awareness") != "not_probed":
                if not comp_data.get("contradiction_tested"):
                    next_strat = "contradiction_test"
                else:
                    next_strat = "break_textbook"
                return {
                    "is_coherent": False,
                    "adjusted_decision": {
                        **strategist_decision,
                        "follow_up_strategy": next_strat,
                    },
                    "adjustment_reason": f"Tradeoffs already probed. Switching to {next_strat}.",
                }

        elif strategy == "contradiction_test":
            if comp_data.get("contradiction_tested"):
                # Already tested contradiction — use another adversarial strategy
                return {
                    "is_coherent": False,
                    "adjusted_decision": {
                        **strategist_decision,
                        "follow_up_strategy": "extreme_scenario",
                    },
                    "adjustment_reason": "Contradiction already tested. Switching to extreme_scenario.",
                }

    # ─── Check 3: Strategy validity ────────────────────────────────
    if strategy and strategy not in VALID_STRATEGIES:
        return {
            "is_coherent": False,
            "adjusted_decision": {
                **strategist_decision,
                "follow_up_strategy": "probe_reasoning",
            },
            "adjustment_reason": f"Invalid strategy '{strategy}'. Falling back to probe_reasoning.",
        }

    # ─── Check 4: Contradiction gate before advancing ──────────────
    if action == "advance" and target:
        current_comp = state["progress"]["current_competency"]
        if current_comp and current_comp in state["competency_signals"]:
            comp_data = state["competency_signals"][current_comp]
            probes_done = comp_data.get("probes_completed", 0)
            if (not comp_data.get("contradiction_tested")
                    and probes_done < 5
                    and strategist_decision.get("exit_condition") != "budget_exhausted"):
                return {
                    "is_coherent": False,
                    "adjusted_decision": {
                        **strategist_decision,
                        "action": "follow_up",
                        "exit_condition": None,
                        "target_competency": current_comp,
                        "follow_up_strategy": "contradiction_test",
                    },
                    "adjustment_reason": "Contradiction test required before advancing. Blocking exit.",
                }

    # ─── Check 5: Cross-competency signal enrichment ───────────────
    if action == "advance" and target:
        cross_signals = state.get("cross_competency_signals", [])
        relevant_signals = [s for s in cross_signals if target.lower() in s.lower()]
        if relevant_signals:
            return {
                "is_coherent": True,
                "adjusted_decision": {
                    **strategist_decision,
                    "cross_competency_note": (
                        strategist_decision.get("cross_competency_note", "") or ""
                    ) + " " + "; ".join(relevant_signals),
                },
                "adjustment_reason": None,
            }

    return {
        "is_coherent": True,
        "adjusted_decision": None,
        "adjustment_reason": None,
    }
