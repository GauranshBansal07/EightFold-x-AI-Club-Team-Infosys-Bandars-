"""
Interview Log — Append-only raw transcript (interview_log.md).
Never edited, only appended to.
"""

import os
from datetime import datetime
from config import INTERVIEW_LOG_FILE, STATE_FILES_DIR


class InterviewLog:
    """Manages the append-only interview transcript."""

    def __init__(self):
        self.file_path = INTERVIEW_LOG_FILE
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(STATE_FILES_DIR, exist_ok=True)

    def initialize(self, jd_title: str, candidate_name: str, experience_level: str):
        """Create the log file with session metadata."""
        content = f"""# Interview Log — Raw Transcript

## Session Metadata
- **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **JD Title:** {jd_title}
- **Candidate:** {candidate_name}
- **Experience Level:** {experience_level}

---
"""
        with open(self.file_path, "w") as f:
            f.write(content)

    def append_introduction(self, exchange_num: int, interviewer_text: str,
                            candidate_text: str, internal_note: str):
        """Append an introduction exchange."""
        entry = f"""
### Exchange {exchange_num} — Phase: Introduction
**Interviewer:** {interviewer_text}
**Candidate:** {candidate_text}
**Internal Note:** {internal_note}

---
"""
        with open(self.file_path, "a") as f:
            f.write(entry)

    def append_exchange(self, exchange_num: int, competency_name: str,
                        question_text: str, cognitive_task: str,
                        target_competency: str, candidate_text: str,
                        evaluation_result: dict, decision: str, reason: str):
        """Append a competency probing exchange with full evaluation details."""
        eval_section = f"""  - Drift Check: {evaluation_result.get('drift_check', 'N/A')}
  - Causal Reasoning: {evaluation_result.get('causal_reasoning', 'N/A')}
  - Evidence Grounding: {evaluation_result.get('evidence_grounding', 'N/A')}
  - Accuracy: {evaluation_result.get('accuracy', 'N/A')}
  - Gap Diagnosis: {evaluation_result.get('gap_diagnosis', 'N/A')}
  - Follow-Up Strategy: {evaluation_result.get('follow_up_strategy', 'N/A')}
  - What Is Missing: {evaluation_result.get('what_is_missing', 'N/A')}"""

        entry = f"""
### Exchange {exchange_num} — Phase: Competency Probing — {competency_name}
**Interviewer:** {question_text}
**Cognitive Task:** {cognitive_task}
**Target Competency:** {target_competency}
**Candidate:** {candidate_text}
**Evaluation Result:**
{eval_section}
**Decision:** {decision}
**Reason:** {reason}

---
"""
        with open(self.file_path, "a") as f:
            f.write(entry)

    def get_full_log(self) -> str:
        """Read and return the full log contents."""
        if not os.path.exists(self.file_path):
            return ""
        with open(self.file_path, "r") as f:
            return f.read()
