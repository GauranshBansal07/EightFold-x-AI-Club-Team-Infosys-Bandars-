"""
Loop Controller — Outer/inner loop orchestration + exit conditions.
This is the central orchestrator that wires evaluator → coherence check → strategist → question generator.
"""

import json
from interview.evaluator import evaluate_response
from interview.strategist import decide_next
from interview.question_generator import generate_question
from interview.introduction import generate_greeting, process_introduction, generate_followup
from guardrails.coherence_checker import check_coherence
from memory.candidate_state import CandidateState
from memory.interview_log import InterviewLog
from memory.state_renderer import render_state_for_prompt
from report.report_generator import generate_report


class InterviewController:
    """
    Orchestrates the full interview lifecycle:
    introduction → competency_probing (outer × inner loop) → complete → report
    """

    def __init__(self):
        self.candidate_state = CandidateState()
        self.interview_log = InterviewLog()
        self.phase = "setup"  # setup → introduction → probing → complete
        self.exchange_count = 0
        self.intro_exchange_count = 0

        # Set during ingestion
        self.jd_text = ""
        self.parsed_jd = None
        self.competency_specs = None
        self.experience_level = ""
        self.candidate_background = ""

        # Current question context (for evaluating the next response)
        self.current_question = None  # dict with question_text, cognitive_task, target_competency
        self.competency_order = []    # ordered list of competency names

    def setup(self, jd_text: str, parsed_jd: dict, competency_specs: dict, experience_level: str):
        """Initialize the interview after ingestion is complete."""
        self.jd_text = jd_text
        self.parsed_jd = parsed_jd
        self.competency_specs = competency_specs
        self.experience_level = experience_level

        # Extract competency names in order
        self.competency_order = [c["name"] for c in parsed_jd["competencies"]]

        # Initialize state files
        self.candidate_state.initialize(experience_level, self.competency_order)
        self.phase = "introduction"

    def start_interview(self) -> str:
        """Generate the opening greeting and return it."""
        role_title = self.parsed_jd["role_title"]
        greeting = generate_greeting(role_title)
        self.exchange_count += 1

        # Initialize log (name pending — will update after intro)
        self.interview_log.initialize(
            jd_title=role_title,
            candidate_name="[pending]",
            experience_level=self.experience_level,
        )

        return greeting

    def process_candidate_message(self, candidate_text: str) -> dict:
        """
        Process a candidate message and generate the agent's response.

        Returns:
        {
            "agent_response": str,          # The text to show the candidate
            "phase": str,                   # Current phase
            "is_complete": bool,            # Whether the interview is done
            "current_competency": str,      # Currently probing (or None)
            "competencies_done": list,      # List of scored competencies
            "competencies_remaining": list, # List of remaining competencies
            "report": dict | None,          # Final report if complete
            "internal_notes": str,          # Internal notes for logging
        }
        """
        if self.phase == "introduction":
            return self._handle_introduction(candidate_text)
        elif self.phase == "probing":
            return self._handle_probing(candidate_text)
        else:
            return {
                "agent_response": "The interview has concluded. Thank you for your time!",
                "phase": "complete",
                "is_complete": True,
                "current_competency": None,
                "competencies_done": self._get_scored_competencies(),
                "competencies_remaining": [],
                "report": None,
                "internal_notes": "Interview already complete.",
            }

    def _handle_introduction(self, candidate_text: str) -> dict:
        """Handle the introduction phase."""
        self.intro_exchange_count += 1
        self.exchange_count += 1

        # Process the introduction
        competency_names = [c["name"] for c in self.parsed_jd["competencies"]]
        intro_analysis = process_introduction(
            introduction_text=candidate_text,
            jd_text=self.jd_text,
            competency_list=competency_names,
        )

        # Update candidate state with profile
        projects = intro_analysis.get("projects", [])
        name = self._extract_name(candidate_text, intro_analysis)
        
        # Override the controller's underlying experience property dynamically
        exp = intro_analysis.get("experience_level_confirmed", "")
        if exp:
            self.experience_level = exp

        self.candidate_state.update_profile(
            name=name,
            background_summary=intro_analysis.get("background_summary", ""),
            projects=projects,
            experience_confirmed=exp,
        )

        # Build background string for question generation
        self.candidate_background = intro_analysis.get("background_summary", "")
        if projects:
            proj_strs = [f"- {p.get('name', 'unnamed')}: {p.get('description', '')}" for p in projects]
            self.candidate_background += "\nProjects:\n" + "\n".join(proj_strs)

        # Log the introduction exchange
        internal_note = (
            f"Background: {intro_analysis.get('background_summary', 'N/A')}. "
            f"Projects: {', '.join(p.get('name', '') for p in projects)}. "
            f"Competency mapping: {intro_analysis.get('competency_mapping_notes', 'N/A')}"
        )
        self.interview_log.append_introduction(
            exchange_num=self.exchange_count,
            interviewer_text="[Introduction greeting]",
            candidate_text=candidate_text,
            internal_note=internal_note,
        )

        # Check if we need a follow-up (brief intro)
        if self.intro_exchange_count < 2:
            followup = generate_followup(candidate_text)
            if followup:
                return {
                    "agent_response": followup,
                    "phase": "introduction",
                    "is_complete": False,
                    "current_competency": None,
                    "competencies_done": [],
                    "competencies_remaining": self.competency_order,
                    "report": None,
                    "internal_notes": internal_note,
                }

        # Transition to competency probing
        self.phase = "probing"
        first_comp = self.competency_order[0]
        self.candidate_state.start_competency(first_comp)

        # Generate first question
        first_question = self._generate_initial_question(first_comp)
        self.current_question = first_question

        # Build transition response
        transition = (
            f"Thanks for sharing that, {name}! "
            f"Let's dive into some questions. "
            f"{first_question['question_text']}"
        )

        return {
            "agent_response": transition,
            "phase": "probing",
            "is_complete": False,
            "current_competency": first_comp,
            "competencies_done": [],
            "competencies_remaining": self.competency_order,
            "report": None,
            "internal_notes": f"Transitioning to competency probing. First competency: {first_comp}. Rationale: {first_question.get('internal_rationale', '')}",
        }

    def _handle_probing(self, candidate_text: str) -> dict:
        """Handle the competency probing phase — the core three-call loop."""
        self.exchange_count += 1

        if not self.current_question:
            return self._error_response("No current question context. Generating a new question.")

        current_comp = self.current_question["target_competency"]
        comp_data = self.candidate_state.get_competency_data(current_comp)
        comp_spec = self.competency_specs.get(current_comp, {})
        depth_spec = comp_spec.get("depth_spec", "Assess depth of understanding.")

        # ─── CALL 1: Evaluate ──────────────────────────────────────
        evaluation = evaluate_response(
            question_text=self.current_question["question_text"],
            response_text=candidate_text,
            competency_name=current_comp,
            cognitive_task=self.current_question.get("cognitive_task", "explanation"),
            depth_spec=depth_spec,
            experience_level=self.experience_level,
        )

        # Record the probe in state
        probe_data = {
            "question": self.current_question["question_text"],
            "cognitive_task": self.current_question.get("cognitive_task", "explanation"),
            "response": candidate_text,
            "follow_up_strategy": self.current_question.get("_follow_up_strategy"),  # strategy that produced this question
            "candidate_claims": evaluation.get("candidate_claims", []),
            **{k: v for k, v in evaluation.items() if k != "candidate_claims"},
        }
        self.candidate_state.add_probe(current_comp, probe_data)

        # ─── CALL 2: Strategize ───────────────────────────────────
        strategy = decide_next(
            evaluation_result=evaluation,
            candidate_state=self.candidate_state,
            competency_specs=self.competency_specs,
        )

        # ─── GUARDRAIL: Coherence Check ───────────────────────────
        coherence = check_coherence(strategy, self.candidate_state)
        if not coherence["is_coherent"] and coherence["adjusted_decision"]:
            strategy = coherence["adjusted_decision"]

        # ─── Handle strategy result ───────────────────────────────
        action = strategy.get("action", "follow_up")
        follow_up_strategy = strategy.get("follow_up_strategy")
        exit_condition = strategy.get("exit_condition")

        # Decision text for logging
        decision = "follow-up on same competency" if action == "follow_up" else "advance to next competency"
        reason = exit_condition or f"Strategy: {follow_up_strategy}"

        # Log the exchange
        eval_for_log = {**evaluation, "follow_up_strategy": follow_up_strategy or "N/A"}
        self.interview_log.append_exchange(
            exchange_num=self.exchange_count,
            competency_name=current_comp,
            question_text=self.current_question["question_text"],
            cognitive_task=self.current_question.get("cognitive_task", "explanation"),
            target_competency=current_comp,
            candidate_text=candidate_text,
            evaluation_result=eval_for_log,
            decision=decision,
            reason=reason,
        )

        # ─── If advancing, finalize current competency ────────────
        if action == "advance":
            # Determine depth from probe history
            depth, confidence, key_evidence, key_gap = self._compute_verdict(current_comp)
            self.candidate_state.finalize_competency(
                competency_name=current_comp,
                exit_condition=exit_condition or "depth_reached",
                depth=depth,
                confidence=confidence,
                key_evidence=key_evidence,
                key_gap=key_gap,
            )

            # Update difficulty calibration (4-tier)
            current_state = self.candidate_state.get_state()
            cal = strategy.get("difficulty_calibration", current_state["profile"]["difficulty_calibration"])
            cal_reason = strategy.get("difficulty_reasoning", "")
            current_cal = current_state["profile"]["difficulty_calibration"]
            if cal != current_cal and cal in ("baseline", "constraint_heavy", "ambiguous", "adversarial"):
                self.candidate_state.update_calibration(current_comp, cal, cal_reason)

            # Add cross-competency signal if present
            cross_note = strategy.get("cross_competency_note")
            if cross_note:
                self.candidate_state.add_cross_competency_signal(cross_note)

            # Check if all competencies done
            target_comp = strategy.get("target_competency")
            if not target_comp or target_comp not in self.candidate_state.get_state()["competency_signals"]:
                # Interview complete
                return self._complete_interview()

            # Start next competency
            self.candidate_state.start_competency(target_comp)

        target_comp = strategy.get("target_competency", current_comp)

        # ─── CALL 3: Generate Question ────────────────────────────
        comp_data = self.candidate_state.get_competency_data(target_comp)
        pil_difficulty = comp_data.get("probe_intensity_level", 1)
        next_question = generate_question(
            strategist_decision=strategy,
            candidate_state=self.candidate_state,
            competency_specs=self.competency_specs,
            candidate_background=self.candidate_background,
            difficulty_level=pil_difficulty,
        )
        # Prevent 100% exact text repetition due to LLM generator API failures
        if getattr(self, 'current_question', None) and next_question["question_text"] == self.current_question.get("question_text"):
            # If we exactly repeat the question, force advance to break loop
            competencies_rem = self.candidate_state.get_state()["progress"]["competencies_remaining"]
            filtered = [c for c in competencies_rem if c != target_comp]
            next_comp = filtered[0] if filtered else None
            
            if next_comp:
                next_question = self._generate_initial_question(next_comp)
                target_comp = next_comp
            else:
                return self._complete_interview()

        # Tag the question with the strategy that produced it (for tracking)
        next_question["_follow_up_strategy"] = strategy.get("follow_up_strategy")
        self.current_question = next_question

        return {
            "agent_response": next_question["question_text"],
            "phase": "probing",
            "is_complete": False,
            "current_competency": target_comp,
            "competencies_done": self._get_scored_competencies(),
            "competencies_remaining": self.candidate_state.get_state()["progress"]["competencies_remaining"],
            "report": None,
            "internal_notes": f"Eval: {evaluation.get('gap_diagnosis')} | Strategy: {action}/{follow_up_strategy} | Rationale: {next_question.get('internal_rationale', '')}",
        }

    def _complete_interview(self) -> dict:
        """Finalize the interview and generate report."""
        self.phase = "complete"
        self.candidate_state.get_state()["progress"]["current_phase"] = "complete"

        # Generate report
        state = self.candidate_state.get_state()
        log = self.interview_log.get_full_log()

        try:
            report = generate_report(state, log)
        except Exception as e:
            report = {"error": f"Report generation failed: {str(e)}"}

        # Inject raw candidate state into report for transparency
        report["raw_candidate_state"] = state

        try:
            from config import llm_call
            name = self.candidate_state.get_state().get('profile', {}).get('name', 'candidate')
            role = self.parsed_jd.get('role_title', 'this role')
            sys_prompt = "You are HireMind, an AI interviewer. The interview is now complete. Generate a warm, concise (2-3 sentences max) concluding message thanking the candidate and mentioning they'll receive a report."
            user_prompt = f"Candidate name: {name}\nRole: {role}"
            thank_you = llm_call(sys_prompt, user_prompt, temperature=0.7)
        except Exception:
            thank_you = (
                f"That wraps up our interview! Thank you so much for your time and thoughtful responses. "
                f"You'll receive a detailed evaluation report shortly. Best of luck!"
            )

        return {
            "agent_response": thank_you,
            "phase": "complete",
            "is_complete": True,
            "current_competency": None,
            "competencies_done": self._get_scored_competencies(),
            "competencies_remaining": [],
            "report": report,
            "internal_notes": "Interview complete. Report generated.",
        }

    def _generate_initial_question(self, competency_name: str) -> dict:
        """Generate the first question for a competency."""
        initial_decision = {
            "action": "advance",
            "exit_condition": None,
            "target_competency": competency_name,
            "follow_up_strategy": None,
            "difficulty_calibration": self.candidate_state.get_state()["profile"]["difficulty_calibration"],
            "difficulty_reasoning": "Initial question for competency.",
            "cross_competency_note": None,
            "coherence_note": None,
        }

        comp_data = self.candidate_state.get_competency_data(competency_name)
        pil_difficulty = comp_data.get("probe_intensity_level", 1)
        return generate_question(
            strategist_decision=initial_decision,
            candidate_state=self.candidate_state,
            competency_specs=self.competency_specs,
            candidate_background=self.candidate_background,
            difficulty_level=pil_difficulty,
        )

    def _compute_verdict(self, competency_name: str) -> tuple[str, str, str, str]:
        """Compute overall verdict for a competency from probe history.
        
        Uses mathematical signals: reasoning, evidence, novelty, adaptability, 
        and understanding score. Penalizes stagnation and rephrasing aggressively.
        """
        comp_data = self.candidate_state.get_competency_data(competency_name)
        probes = comp_data.get("probe_history", [])

        if not probes:
            return "surface", "low", "No probes completed.", "No signal gathered."

        # ─── Aggregate core signals ───────────────────────────────────
        reasoning_scores = []
        grounding_scores = []
        gaps = []
        evidence_fragments = []

        score_map = {"absent": 0, "partial": 1, "sufficient": 2}

        novelty_rephrased_count = 0
        adaptability_stagnant_count = 0
        adaptability_worse_count = 0
        total_probes = len(probes)

        for p in probes:
            r = score_map.get(p.get("causal_reasoning", "absent"), 0)
            g = score_map.get(p.get("evidence_grounding", "absent"), 0)
            reasoning_scores.append(r)
            grounding_scores.append(g)
            gaps.append(p.get("gap_diagnosis", ""))
            if p.get("causal_evidence") and p.get("causal_evidence") != "N/A":
                evidence_fragments.append(p.get("causal_evidence", ""))

            # Track negative signals
            if p.get("novelty") == "rephrased":
                novelty_rephrased_count += 1
            if p.get("adaptability") == "same":
                adaptability_stagnant_count += 1
            elif p.get("adaptability") == "worse":
                adaptability_worse_count += 1

        avg_reasoning = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0
        avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0
        avg_total = (avg_reasoning + avg_grounding) / 2

        # ─── Penalty: Novelty stagnation ──────────────────────────────
        # If most probes produced rephrased content, candidate didn't expand
        rephrased_ratio = novelty_rephrased_count / total_probes if total_probes > 0 else 0
        stagnant_ratio = adaptability_stagnant_count / total_probes if total_probes > 0 else 0

        # Apply mathematical penalty to the avg_total
        # Heavy rephrasing = candidate just restating, penalize depth
        if rephrased_ratio >= 0.5:
            avg_total -= 0.4 * rephrased_ratio  # Up to -0.4 penalty

        # Stagnation under pressure = candidate couldn't deepen
        if stagnant_ratio >= 0.5:
            avg_total -= 0.3 * stagnant_ratio  # Up to -0.3 penalty

        # Worsening under pressure = strong negative signal
        if adaptability_worse_count >= 1:
            avg_total -= 0.2 * adaptability_worse_count

        # ─── Understanding Score cross-check ──────────────────────────
        understanding_score = comp_data.get("understanding_score", 0.0)

        # ─── Depth assessment (tightened thresholds + score gate) ─────
        if avg_total >= 1.6 and understanding_score >= 0.7:
            depth = "deep"
        elif avg_total >= 0.9 and understanding_score >= 0.4:
            depth = "partial"
        else:
            depth = "surface"

        # ─── Confidence (strict multi-signal requirement) ─────────────
        no_gap_count = sum(1 for g in gaps if g == "no_gap")
        has_strong_evidence = avg_grounding >= 1.5  # At least partial+ across probes
        has_adaptability = adaptability_stagnant_count < (total_probes * 0.5)  # Less than half stagnant
        has_novelty = rephrased_ratio < 0.5  # Majority of probes showed new info

        if no_gap_count >= 2 and has_strong_evidence and has_adaptability and has_novelty:
            confidence = "high"
        elif no_gap_count >= 1 and (has_strong_evidence or has_adaptability):
            confidence = "medium"
        else:
            confidence = "low"

        key_evidence = evidence_fragments[0] if evidence_fragments else "No strong signal captured."
        key_gap = probes[-1].get("what_is_missing", "No specific gap identified.") if probes else "No probes."

        return depth, confidence, key_evidence, key_gap

    def _get_scored_competencies(self) -> list:
        """Get list of scored competency names."""
        state = self.candidate_state.get_state()
        return [
            name for name, data in state["competency_signals"].items()
            if data["status"] == "scored"
        ]

    def _extract_name(self, text: str, analysis: dict) -> str:
        """Extract candidate name from text or analysis."""
        # Try to find a name-like pattern in the first sentence
        bg = analysis.get("background_summary", "")
        words = text.split()
        # Common patterns: "I am [name]", "My name is [name]", "Hi, I'm [name]"
        for i, w in enumerate(words):
            wl = w.lower().rstrip(".,!?")
            if wl in ("am", "i'm", "im") and i + 1 < len(words):
                name_candidate = words[i + 1].strip(".,!?")
                if name_candidate[0].isupper() and len(name_candidate) > 1:
                    return name_candidate
            if wl == "name" and i + 2 < len(words) and words[i + 1].lower() == "is":
                name_candidate = words[i + 2].strip(".,!?")
                if name_candidate[0].isupper() and len(name_candidate) > 1:
                    return name_candidate
        return "Candidate"

    def _error_response(self, note: str) -> dict:
        """Generate an error/fallback response."""
        return {
            "agent_response": "Let me ask you another question.",
            "phase": self.phase,
            "is_complete": False,
            "current_competency": self.candidate_state.get_state()["progress"].get("current_competency"),
            "competencies_done": self._get_scored_competencies(),
            "competencies_remaining": self.candidate_state.get_state()["progress"]["competencies_remaining"],
            "report": None,
            "internal_notes": f"Error: {note}",
        }
