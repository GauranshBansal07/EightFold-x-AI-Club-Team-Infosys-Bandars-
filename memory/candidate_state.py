"""
Candidate State — Structured state management (candidate_state.md).
Read before every API call, updated after every evaluation.
"""

import os
import json
from config import CANDIDATE_STATE_FILE, STATE_FILES_DIR


class CandidateState:
    """Manages the structured candidate state as both a .md file and in-memory dict."""

    def __init__(self):
        self.file_path = CANDIDATE_STATE_FILE
        self._ensure_dir()
        # In-memory representation
        self.state = {
            "profile": {
                "name": "",
                "experience_level": "",
                "background_summary": "",
                "relevant_projects": [],
                "difficulty_calibration": "baseline",
            },
            "progress": {
                "current_phase": "introduction",
                "current_competency": None,
                "competencies_remaining": [],
                "total_exchanges": 0,
            },
            "competency_signals": {},  # competency_name -> signal data
            "cross_competency_signals": [],
            "difficulty_calibration_log": [],
        }

    def _ensure_dir(self):
        os.makedirs(STATE_FILES_DIR, exist_ok=True)

    def initialize(self, experience_level: str, competency_names: list[str]):
        """Initialize state for a new interview."""
        self.state["profile"]["experience_level"] = experience_level
        self.state["progress"]["competencies_remaining"] = list(competency_names)

        for comp in competency_names:
            self.state["competency_signals"][comp] = {
                "status": "not_started",
                "probes_completed": 0,
                "exit_condition": None,
                "tradeoff_awareness": "not_probed",
                "contradiction_tested": False,
                "candidate_claims": [],  # testable claims extracted from responses
                "probe_history": [],
                "understanding_score": 0.0,
                "probe_intensity_level": 1,
                "verdict": None,
            }
        self._write_file()

    def update_profile(self, name: str, background_summary: str,
                       projects: list, experience_confirmed: str):
        """Update candidate profile from introduction analysis."""
        self.state["profile"]["name"] = name
        self.state["profile"]["background_summary"] = background_summary
        self.state["profile"]["relevant_projects"] = projects
        if experience_confirmed:
            self.state["profile"]["experience_level"] = experience_confirmed
        self._write_file()

    def start_competency(self, competency_name: str):
        """Begin probing a competency."""
        self.state["progress"]["current_phase"] = "competency_probing"
        self.state["progress"]["current_competency"] = competency_name
        if competency_name in self.state["competency_signals"]:
            self.state["competency_signals"][competency_name]["status"] = "in_progress"
        self._write_file()

    def add_probe(self, competency_name: str, probe_data: dict):
        """Add a probe result to a competency's history."""
        comp = self.state["competency_signals"].get(competency_name)
        if comp is None:
            return
        comp["probes_completed"] += 1
        comp["probe_history"].append(probe_data)
        self.state["progress"]["total_exchanges"] += 1

        # Track tradeoff probing
        if probe_data.get("follow_up_strategy") == "forced_tradeoff":
            comp["tradeoff_awareness"] = "probed_and_present" if probe_data.get("gap_diagnosis") == "no_gap" else "probed_and_absent"
        elif probe_data.get("follow_up_strategy") == "probe_tradeoffs":
            comp["tradeoff_awareness"] = "probed_and_present" if probe_data.get("gap_diagnosis") == "no_gap" else "probed_and_absent"

        # Track contradiction testing
        if probe_data.get("follow_up_strategy") in ("contradiction_test", "adversarial_constraint"):
            comp["contradiction_tested"] = True

        # Track candidate claims
        claims = probe_data.get("candidate_claims", [])
        if claims:
            comp["candidate_claims"].extend(claims)

        # ─── Calculate Mathematical Understanding Score ─────────────
        r_map = {"absent": 0.0, "partial": 0.5, "sufficient": 1.0}
        e_map = {"absent": 0.0, "partial": 0.5, "sufficient": 1.0}
        a_map = {"worse": 0.0, "same": 0.5, "improved": 1.0, "not_applicable": 0.5}

        reasoning = r_map.get(probe_data.get("causal_reasoning", "absent"), 0.0)
        evidence = e_map.get(probe_data.get("evidence_grounding", "absent"), 0.0)
        adaptability = a_map.get(probe_data.get("adaptability", "not_applicable"), 0.5)
        
        tradeoffs = 0.5  # Neutral default prior to probing
        if comp["tradeoff_awareness"] == "probed_and_present":
            tradeoffs = 1.0
        elif comp["tradeoff_awareness"] == "probed_and_absent":
            tradeoffs = 0.0

        comp["understanding_score"] = (0.35 * reasoning) + (0.35 * evidence) + (0.2 * adaptability) + (0.1 * tradeoffs)

        # ─── Probe Intensity Level (PIL) Escalation Loop ────────────
        a_status = probe_data.get("adaptability", "not_applicable")
        novelty = probe_data.get("novelty", "new_info")
        
        if a_status == "same" or novelty == "rephrased":
            comp["probe_intensity_level"] = min(4, comp["probe_intensity_level"] + 1)
            
        if probe_data.get("struggle_detected", False) or a_status == "worse":
            comp["probe_intensity_level"] = max(2, comp["probe_intensity_level"] - 1)

        # Maximum constraint guardrail
        if comp["probe_intensity_level"] >= 4:
            comp["consecutive_high_pil"] = comp.get("consecutive_high_pil", 0) + 1
            if comp["consecutive_high_pil"] > 2:
                comp["probe_intensity_level"] = 3
                comp["consecutive_high_pil"] = 0
        else:
            comp["consecutive_high_pil"] = 0

        self._write_file()

    def finalize_competency(self, competency_name: str, exit_condition: str,
                            depth: str, confidence: str,
                            key_evidence: str, key_gap: str):
        """Score and finalize a competency."""
        comp = self.state["competency_signals"].get(competency_name)
        if comp is None:
            return
        comp["status"] = "scored"
        comp["exit_condition"] = exit_condition
        comp["verdict"] = {
            "depth_assessment": depth,
            "confidence": confidence,
            "key_evidence": key_evidence,
            "key_gap": key_gap,
        }

        # Move from remaining to done
        remaining = self.state["progress"]["competencies_remaining"]
        if competency_name in remaining:
            remaining.remove(competency_name)

        # Check if all done
        if not remaining:
            self.state["progress"]["current_phase"] = "complete"
            self.state["progress"]["current_competency"] = None
        self._write_file()

    def update_calibration(self, competency_name: str, calibration: str, reasoning: str):
        """Update difficulty calibration after a competency."""
        self.state["profile"]["difficulty_calibration"] = calibration
        self.state["difficulty_calibration_log"].append({
            "after_competency": competency_name,
            "calibration": calibration,
            "reasoning": reasoning,
        })
        self._write_file()

    def add_cross_competency_signal(self, signal: str):
        """Add a cross-competency signal."""
        self.state["cross_competency_signals"].append(signal)
        self._write_file()

    def get_state(self) -> dict:
        """Return the in-memory state dict."""
        return self.state

    def get_state_md(self) -> str:
        """Read and return the markdown file contents."""
        if not os.path.exists(self.file_path):
            return ""
        with open(self.file_path, "r") as f:
            return f.read()

    def get_current_competency_probes(self) -> list:
        """Get probe history for the current competency."""
        current = self.state["progress"]["current_competency"]
        if current and current in self.state["competency_signals"]:
            return self.state["competency_signals"][current]["probe_history"]
        return []

    def get_competency_data(self, name: str) -> dict:
        """Get signal data for a specific competency."""
        return self.state["competency_signals"].get(name, {})

    def _write_file(self):
        """Render the in-memory state to the markdown file."""
        s = self.state
        p = s["profile"]
        pr = s["progress"]

        lines = [
            "# Candidate State — Live\n",
            "## Candidate Profile",
            f"- **Name:** {p['name'] or '[pending]'}",
            f"- **Experience Level:** {p['experience_level']}",
            f"- **Background Summary:** {p['background_summary'] or '[pending — introduction phase]'}",
        ]

        # Projects
        if p["relevant_projects"]:
            proj_strs = []
            for proj in p["relevant_projects"]:
                if isinstance(proj, dict):
                    proj_strs.append(f"{proj.get('name', 'unnamed')}: {proj.get('description', '')}")
                else:
                    proj_strs.append(str(proj))
            lines.append(f"- **Relevant Projects:** {'; '.join(proj_strs)}")
        else:
            lines.append("- **Relevant Projects:** [none identified yet]")

        lines.append(f"- **Difficulty Calibration:** {p['difficulty_calibration']}")
        lines.append("")

        # Progress
        lines.append("## Interview Progress")
        lines.append(f"- **Current Phase:** {pr['current_phase']}")
        lines.append(f"- **Current Competency:** {pr['current_competency'] or 'N/A'}")
        lines.append(f"- **Competencies Remaining:** {', '.join(pr['competencies_remaining']) if pr['competencies_remaining'] else 'None'}")
        lines.append(f"- **Total Exchanges:** {pr['total_exchanges']}")
        lines.append("")

        # Competency Signals
        lines.append("## Competency Signals\n")
        for comp_name, comp in s["competency_signals"].items():
            lines.append(f"### {comp_name}")
            lines.append(f"- **Status:** {comp['status']}")
            lines.append(f"- **Probes Completed:** {comp['probes_completed']}")
            lines.append(f"- **Exit Condition:** {comp['exit_condition'] or 'null'}")
            lines.append(f"- **Tradeoff Awareness:** {comp['tradeoff_awareness']}")
            lines.append(f"- **Contradiction Tested:** {'✅ Yes' if comp.get('contradiction_tested', False) else '❌ No'}")

            # Candidate Claims
            claims = comp.get('candidate_claims', [])
            if claims:
                lines.append(f"- **Testable Claims:** {'; '.join(claims[:5])}")

            # Escalation & Exits
            lines.append(f"- **Understanding Score:** {comp.get('understanding_score', 0.0)}")
            lines.append(f"- **Probe Intensity Level:** {comp.get('probe_intensity_level', 1)}")
            lines.append("")

            # Probe History
            if comp["probe_history"]:
                lines.append("#### Probe History")
                for i, probe in enumerate(comp["probe_history"], 1):
                    lines.append(f"**Probe {i}:**")
                    lines.append(f"- Question Asked: \"{probe.get('question', '')}\"")
                    lines.append(f"- Cognitive Task Targeted: {probe.get('cognitive_task', 'N/A')}")
                    lines.append(f"- Drift Check: {probe.get('drift_check', 'N/A')}")
                    lines.append(f"- Causal Reasoning: {probe.get('causal_reasoning', 'N/A')} — Evidence: \"{probe.get('causal_evidence', '')}\"")
                    lines.append(f"- Evidence Grounding: {probe.get('evidence_grounding', 'N/A')} — Evidence: \"{probe.get('grounding_evidence', '')}\"")
                    lines.append(f"- Accuracy Gate: {probe.get('accuracy', 'N/A')} — Note: \"{probe.get('accuracy_note', '')}\"")
                    lines.append(f"- Gap Diagnosis: {probe.get('gap_diagnosis', 'N/A')}")
                    lines.append(f"- Follow-Up Strategy Selected: {probe.get('follow_up_strategy', 'null')}")
                    lines.append(f"- Novelty: {probe.get('novelty', 'new_info')}")
                    lines.append(f"- Dimension Probed: {probe.get('dimension_probed', 'unknown')}")
                    lines.append(f"- Adaptability: {probe.get('adaptability', 'not_applicable')}")
                    lines.append(f"- Struggle Detected: {'✅ Yes' if probe.get('struggle_detected', False) else '❌ No'}")
                    lines.append(f"- What Is Missing: \"{probe.get('what_is_missing', '')}\"")
                    lines.append("")

            # Verdict
            if comp["verdict"]:
                v = comp["verdict"]
                lines.append("#### Competency Verdict")
                lines.append(f"- **Depth Assessment:** {v['depth_assessment']}")
                lines.append(f"- **Confidence:** {v['confidence']}")
                lines.append(f"- **Key Evidence:** \"{v['key_evidence']}\"")
                lines.append(f"- **Key Gap:** \"{v['key_gap']}\"")
                lines.append(f"- **Tradeoff Awareness:** {comp['tradeoff_awareness']}")
                lines.append("")

            lines.append("")

        # Cross-Competency Signals
        if s["cross_competency_signals"]:
            lines.append("## Cross-Competency Signals")
            for signal in s["cross_competency_signals"]:
                lines.append(f"- {signal}")
            lines.append("")

        # Difficulty Calibration Log
        if s["difficulty_calibration_log"]:
            lines.append("## Difficulty Calibration Log")
            for entry in s["difficulty_calibration_log"]:
                lines.append(f"- After {entry['after_competency']}: {entry['calibration']} — {entry['reasoning']}")
            lines.append("")

        with open(self.file_path, "w") as f:
            f.write("\n".join(lines))
