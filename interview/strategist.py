"""
Strategist — Call 2 of the three-call-per-turn architecture.
Decides follow-up vs advance, applies exit conditions, maps gap to strategy.
Handles mathematical novelty tracking, diminishing returns, and orthogonal dimension rotation.
"""

import json
from config import llm_call_json
from prompts.system_strategist import get_strategist_prompt
from memory.candidate_state import CandidateState
from memory.state_renderer import render_state_for_prompt


def decide_next(
    evaluation_result: dict,
    candidate_state: CandidateState,
    competency_specs: dict,
) -> dict:
    """
    Decide what happens next based on evaluation.

    Returns:
    {
        "action": "follow_up" | "advance",
        "exit_condition": str | None,
        "target_competency": str,
        "follow_up_strategy": str | None,
        "cross_competency_note": str | None,
        "coherence_note": str | None,
    }
    """
    state = candidate_state.get_state()
    current_comp = state["progress"]["current_competency"]
    comp_data = state["competency_signals"].get(current_comp, {})
    remaining = state["progress"]["competencies_remaining"]

    # ─── Pre-check mathematical conditions locally before LLM call ─────────
    probes_done = comp_data.get("probes_completed", 0)
    probe_history = comp_data.get("probe_history", [])
    contradiction_tested = comp_data.get("contradiction_tested", False)

    local_exit = None
    local_plateau = False

    # Plateau detection — triggers strategy escalation or ceiling exit
    if len(probe_history) >= 2:
        last_two_gaps = [p.get("gap_diagnosis") for p in probe_history[-2:]]
        if (last_two_gaps[0] == last_two_gaps[1]
                and last_two_gaps[0] != "no_gap"
                and last_two_gaps[0] is not None):
            local_plateau = True

    # Orthogonal Dimension Switch Detection
    force_dimension_switch = False
    if len(probe_history) >= 2:
        last_two_dims = [p.get("dimension_probed") for p in probe_history[-2:]]
        if last_two_dims[0] == last_two_dims[1] and last_two_dims[0] not in ("unknown", None, "N/A"):
            force_dimension_switch = True

    # ─── New Evidence-Based Math Exits ──────────────────────────────────────
    if evaluation_result.get("gap_diagnosis") == "evaluation_failed":
        local_exit = "api_error"
    else:
        score = comp_data.get("understanding_score", 0.0)
        gap = evaluation_result.get("gap_diagnosis")
        
        # Track consecutive novelty and strain
        consecutive_rephrased = 0
        consecutive_worse = 0
        for p in probe_history[-2:]:
            if p.get("novelty") == "rephrased":
                consecutive_rephrased += 1
            if p.get("adaptability") == "worse":
                consecutive_worse += 1

        if probes_done > 0:
            if score >= 0.75 and gap == "no_gap" and contradiction_tested:
                local_exit = "depth_reached"
            elif score < 0.4 and local_plateau:
                local_exit = "plateau_ceiling"
            elif consecutive_rephrased >= 2:
                local_exit = "diminishing_returns"
            elif consecutive_worse >= 2 or evaluation_result.get("struggle_detected", False):
                local_exit = "candidate_strain"
            elif probes_done >= 5 and score >= 0.6:
                local_exit = "soft_cap_reached"
                
            # Overwhelming signal bypass
            if score > 0.85:
                local_exit = "depth_reached"

    # If dimension rotation forced, bypass normal error strategy explicitly locally
    force_strategy_payload = None
    if force_dimension_switch and not local_exit:
        force_strategy_payload = "dimension_rotation"

    # ─── LLM call for nuanced decision ─────────────────────────────
    state_content = render_state_for_prompt(candidate_state)
    specs_str = json.dumps(competency_specs, indent=2)
    eval_str = json.dumps(evaluation_result, indent=2)

    system_prompt, user_prompt = get_strategist_prompt(
        candidate_state_content=state_content,
        evaluation_result_json=eval_str,
        competency_specs=specs_str,
    )

    try:
        result = llm_call_json(system_prompt, user_prompt)
    except Exception as e:
        # Fallback: use local mathematical logic explicitly
        if local_exit:
            filtered = [c for c in remaining if c != current_comp]
            next_comp = filtered[0] if filtered else None
            result = {
                "action": "advance",
                "exit_condition": local_exit,
                "target_competency": next_comp or current_comp,
                "follow_up_strategy": None,
                "cross_competency_note": None,
                "coherence_note": None,
            }
        else:
            gap = evaluation_result.get("gap_diagnosis", "reasoning_absent")
            if force_strategy_payload:
                strategy = force_strategy_payload
            else:
                strategy = _gap_to_strategy(gap, comp_data, local_plateau)
            
            result = {
                "action": "follow_up",
                "exit_condition": None,
                "target_competency": current_comp,
                "follow_up_strategy": strategy,
                "cross_competency_note": None,
                "coherence_note": f"Fallback API error logic override. Strategy requested: {strategy}",
            }

    # ─── Safety net: Enforce contradiction, plateau, & rotation constraints ────────
    
    if result.get("action") == "follow_up" and force_strategy_payload:
        result["follow_up_strategy"] = force_strategy_payload
        result["coherence_note"] = "Local mathematically forced dimension switch triggered due to repetitive sub-dimensions."
    
    # If the LLM tries to advance but we have not met the mathematical score conditions
    if result.get("action") == "advance" and not local_exit:
        if force_strategy_payload:
             result["action"] = "follow_up"
             result["exit_condition"] = None
             result["follow_up_strategy"] = force_strategy_payload
             result["target_competency"] = current_comp
        elif not contradiction_tested:
            result["action"] = "follow_up"
            result["exit_condition"] = None
            result["follow_up_strategy"] = "contradiction_test"
            result["target_competency"] = current_comp
            result["coherence_note"] = "Contradiction test required before early advancing."
        elif local_plateau:
            result["action"] = "follow_up"
            result["exit_condition"] = None
            result["follow_up_strategy"] = _get_plateau_strategy(probe_history)
            result["target_competency"] = current_comp
            result["coherence_note"] = "Plateau escalates strategy rather than advancing below limit thresholds."

    # ─── Override: Force mathematical exit constraint if triggered ───────────
    if local_exit and local_exit != "api_error" and result.get("action") != "advance":
        filtered = [c for c in remaining if c != current_comp]
        next_comp = filtered[0] if filtered else None
        result["action"] = "advance"
        result["exit_condition"] = local_exit
        if next_comp:
            result["target_competency"] = next_comp
        result["follow_up_strategy"] = None
        
    if local_exit == "api_error" and result.get("action") != "advance":
        filtered = [c for c in remaining if c != current_comp]
        next_comp = filtered[0] if filtered else None
        result["action"] = "advance"
        result["exit_condition"] = "api_error"
        if next_comp:
            result["target_competency"] = next_comp
        result["follow_up_strategy"] = None

    # If advancing but no more competencies present mathematically, complete exactly
    if result.get("action") == "advance" and not remaining:
        result["target_competency"] = None

    return result


def _gap_to_strategy(gap: str, comp_data: dict, is_plateau: bool) -> str:
    """Map gap diagnosis strictly to hard adversarial follow-up strategies."""
    if is_plateau:
        return _get_plateau_strategy(comp_data.get("probe_history", []))

    mapping = {
        "scope_drift": "constraint_injection",
        "accuracy_low": "failure_mode_exploration",
        "reasoning_absent": "quantification_attack",
        "evidence_absent": "constraint_injection",
        "textbook_pattern": "tradeoff_forcing",
    }
    if gap == "no_gap":
        if not comp_data.get("contradiction_tested", False):
            return "contradiction_test"
        return "tradeoff_forcing"
    return mapping.get(gap, "constraint_injection")


def _get_plateau_strategy(probe_history: list) -> str:
    """Pick an escalation strategy for plateau breaking aggressively."""
    used = set()
    for p in probe_history:
        s = p.get("follow_up_strategy", "")
        if s in ("contradiction_test", "failure_mode_exploration", "quantification_attack"):
            used.add(s)

    for strategy in ["contradiction_test", "failure_mode_exploration", "quantification_attack"]:
        if strategy not in used:
            return strategy
    return "contradiction_test"
