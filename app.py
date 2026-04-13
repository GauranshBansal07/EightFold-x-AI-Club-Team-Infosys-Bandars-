"""
HireMind — AI-Powered Adaptive Interview Agent
Streamlit Entry Point — UI Only, No Business Logic
"""

import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import GROQ_API_KEY_1, GROQ_API_KEY_2, STATE_FILES_DIR
from ingestion.jd_parser import parse_jd
from ingestion.competency_spec import generate_competency_specs
from interview.loop_controller import InterviewController

# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="HireMind — Adaptive Interview Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #a18cd1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
        color: #c4c4f5;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .sidebar-header h3 {
        margin: 0;
        font-weight: 600;
    }

    .competency-card {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
    }

    .competency-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
    }

    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-not-started { background: #2a2a4a; color: #8888aa; }
    .status-in-progress { background: #1a3a5c; color: #4dabf7; border: 1px solid #4dabf7; }
    .status-scored { background: #1a3c2a; color: #51cf66; border: 1px solid #51cf66; }

    .depth-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .depth-surface { background: #3d1f1f; color: #ff6b6b; }
    .depth-partial { background: #3d3a1f; color: #ffd43b; }
    .depth-deep { background: #1f3d2e; color: #51cf66; }

    .report-section {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .report-section h3 {
        color: #a18cd1;
        border-bottom: 1px solid #2a2a4a;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .metric-card h2 {
        font-size: 2rem;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .chat-msg-agent {
        background: linear-gradient(135deg, #1e1e3f 0%, #2a1e4a 100%);
        border-left: 3px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
        color: #e4e4f5;
    }

    .chat-msg-candidate {
        background: linear-gradient(135deg, #1e3a2e 0%, #1e2a3f 100%);
        border-left: 3px solid #51cf66;
        padding: 1rem 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
        color: #e4e4f5;
    }

    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #2a2a4a !important;
        background: #1a1a2e !important;
        color: #e4e4f5 !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .phase-indicator {
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .phase-intro { background: #1a3a5c; color: #4dabf7; }
    .phase-probing { background: #3d3a1f; color: #ffd43b; }
    .phase-complete { background: #1f3d2e; color: #51cf66; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────
def init_session_state():
    defaults = {
        "stage": "setup",        # setup / interview / report
        "controller": None,
        "messages": [],          # Chat messages
        "parsed_jd": None,
        "competency_specs": None,
        "report": None,
        "jd_text": "",
        "experience_level": "junior",
        "processing": False,
        "interview_complete": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 HireMind</h1>
    <p>AI-Powered Adaptive Interview Agent — Eightfold.AI × AI Club BITS Pilani</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: SETUP
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.stage == "setup":

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Paste Job Description")
        jd_text = st.text_area(
            "Enter the full job description text:",
            height=300,
            placeholder="Paste the complete job description here...\n\nExample: We are looking for a backend software engineer with experience in Python, distributed systems, database design...",
            key="jd_input",
        )

    with col2:

        st.markdown("""
        **How it works:**
        1. Paste a JD → system extracts competencies
        2. Introduce yourself naturally — the system will dynamically deduce your exact experience tier.
        3. Adaptive interview — questions adapt in real time to your responses.
        4. Structured evaluation report with evidence.
        """)

    st.markdown("---")

    if st.button("🚀 Start Interview", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("Please paste a job description first.")
        else:
            with st.spinner("🔍 Analyzing job description and generating competency specifications..."):
                try:
                    # Phase 0: Ingestion
                    parsed_jd = parse_jd(jd_text)
                    st.session_state.parsed_jd = parsed_jd

                    experience_level = "unknown"
                    competency_specs = generate_competency_specs(parsed_jd, experience_level)
                    st.session_state.competency_specs = competency_specs

                    # Initialize controller
                    controller = InterviewController()
                    controller.setup(jd_text, parsed_jd, competency_specs, experience_level)

                    # Generate greeting
                    greeting = controller.start_interview()

                    st.session_state.controller = controller
                    st.session_state.jd_text = jd_text
                    st.session_state.experience_level = experience_level
                    st.session_state.messages = [{"role": "agent", "content": greeting}]
                    st.session_state.stage = "interview"
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error during setup: {str(e)}")
                    st.exception(e)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: INTERVIEW
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "interview":

    # ─── Sidebar: Interview Progress ──────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>📊 Interview Progress</h3>
        </div>
        """, unsafe_allow_html=True)

        controller = st.session_state.controller
        if controller:
            state = controller.candidate_state.get_state()
            phase = state["progress"]["current_phase"]

            # Phase indicator
            phase_class = {
                "introduction": "phase-intro",
                "competency_probing": "phase-probing",
                "complete": "phase-complete",
            }.get(phase, "phase-intro")

            phase_label = {
                "introduction": "📝 Introduction",
                "competency_probing": "🔬 Competency Probing",
                "complete": "✅ Complete",
            }.get(phase, phase)

            st.markdown(
                f'<div class="phase-indicator {phase_class}">{phase_label}</div>',
                unsafe_allow_html=True
            )

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                total = state["progress"]["total_exchanges"]
                st.metric("Exchanges", total)
            with col2:
                done = len([c for c in state["competency_signals"].values() if c["status"] == "scored"])
                total_comp = len(state["competency_signals"])
                st.metric("Progress", f"{done}/{total_comp}")

            # Difficulty calibration
            cal = state["profile"]["difficulty_calibration"]
            cal_emoji = {"baseline": "📊", "escalated": "⬆️", "eased": "⬇️"}.get(cal, "📊")
            st.markdown(f"**Difficulty:** {cal_emoji} {cal.title()}")

            st.markdown("---")

            # Competency cards
            st.markdown("### Competencies")
            current_comp = state["progress"]["current_competency"]

            for comp_name, comp_data in state["competency_signals"].items():
                status = comp_data["status"]
                probes = comp_data["probes_completed"]
                is_current = comp_name == current_comp

                # Status styling
                status_class = f"status-{status.replace('_', '-')}"
                status_label = status.replace("_", " ").title()

                # Build card content
                card_border = "border-color: #667eea;" if is_current else ""
                current_marker = " 👈" if is_current else ""

                st.markdown(f"""
                <div class="competency-card" style="{card_border}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="color: #e4e4f5;">{comp_name}{current_marker}</strong>
                        <span class="status-badge {status_class}">{status_label}</span>
                    </div>
                    <div style="color: #8888aa; font-size: 0.85rem;">
                        Probes: {probes}/4
                        {'| Tradeoffs: ' + comp_data.get('tradeoff_awareness', 'not_probed') if status == 'scored' else ''}
                    </div>
                    {"<div style='margin-top: 0.5rem;'><span class='depth-badge depth-" + comp_data['verdict']['depth_assessment'] + "'>" + comp_data['verdict']['depth_assessment'].title() + "</span> | Confidence: " + comp_data['verdict']['confidence'].title() + "</div>" if comp_data.get('verdict') else ""}
                </div>
                """, unsafe_allow_html=True)

            # State files viewer
            st.markdown("---")
            with st.expander("📁 Raw State Files"):
                tab1, tab2 = st.tabs(["Candidate State", "Interview Log"])
                with tab1:
                    state_md = controller.candidate_state.get_state_md()
                    st.code(state_md if state_md else "Not yet created.", language="markdown")
                with tab2:
                    log = controller.interview_log.get_full_log()
                    st.code(log if log else "Not yet created.", language="markdown")

    # ─── Main: Chat Interface ─────────────────────────────────────
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "agent":
            st.markdown(
                f'<div class="chat-msg-agent">🧠 <strong>HireMind:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-msg-candidate">👤 <strong>You:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    # Input area
    if not st.session_state.get("interview_complete", False):
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your response:",
                height=120,
                placeholder="Type your answer here... Take your time and think out loud.",
                key="user_input",
            )
            submitted = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
    else:
        submitted = False
        user_input = ""

    if submitted and user_input.strip() and st.session_state.controller:
        # Add user message
        st.session_state.messages.append({"role": "candidate", "content": user_input.strip()})

        with st.spinner("🤔 Processing your response..."):
            try:
                result = st.session_state.controller.process_candidate_message(user_input.strip())

                # Add agent response
                st.session_state.messages.append({"role": "agent", "content": result["agent_response"]})

                # Check if interview is complete
                if result["is_complete"]:
                    st.session_state.report = result.get("report")
                    st.session_state.interview_complete = True

            except Exception as e:
                import traceback
                trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace.txt")
                with open(trace_path, "w") as f:
                    f.write(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "agent",
                    "content": f"I apologize — I encountered a technical issue. Let me continue. Could you repeat your last point?"
                })
                st.error(f"Internal error: {str(e)}")

        st.rerun()

    if st.session_state.get("interview_complete", False):
        st.markdown("---")
        if st.button("📊 View Final Report", type="primary", use_container_width=True):
            st.session_state.stage = "report"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: REPORT
# ═══════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "report":

    report = st.session_state.report

    if not report:
        st.warning("Report not available. The interview may not have completed properly.")
        if st.button("← Back to Setup"):
            st.session_state.stage = "setup"
            st.rerun()
    else:
        recruiter = report.get("recruiter_report", {})
        feedback = report.get("candidate_feedback", {})

        # ─── Report Header ────────────────────────────────────────
        st.markdown(f"""
        <div class="report-section" style="text-align: center; border: 2px solid #667eea;">
            <h2 style="color: #a18cd1; margin: 0;">📋 Interview Evaluation Report</h2>
            <p style="color: #8888aa; margin-top: 0.5rem;">
                Candidate: <strong style="color: #e4e4f5;">{recruiter.get('candidate_name', 'N/A')}</strong> |
                Level: <strong style="color: #e4e4f5;">{recruiter.get('experience_level', 'N/A')}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ─── Summary Metrics ──────────────────────────────────────
        scores = recruiter.get("competency_scores", [])
        if scores:
            cols = st.columns(len(scores))
            for i, score in enumerate(scores):
                with cols[i]:
                    depth_emoji = {"deep": "🟢", "partial": "🟡", "surface": "🔴"}.get(score["depth"], "⚪")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.8rem; color: #8888aa; margin-bottom: 0.5rem;">{score['competency']}</div>
                        <h2>{depth_emoji} {score['depth'].title()}</h2>
                        <div style="font-size: 0.75rem; color: #8888aa;">
                            Confidence: {score.get('confidence', 'N/A').title()}<br>
                            Exit: {score.get('exit_condition', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ─── Ranked Summary ───────────────────────────────────────
        ranked = recruiter.get("ranked_summary", "")
        if ranked:
            st.markdown(f"""
            <div class="report-section">
                <h3>🏆 Ranked Summary</h3>
                <p style="color: #e4e4f5; font-size: 1.1rem;">{ranked}</p>
            </div>
            """, unsafe_allow_html=True)

        # ─── Per-Competency Details ───────────────────────────────
        st.markdown('<div class="report-section"><h3>📊 Competency Details</h3></div>', unsafe_allow_html=True)

        for score in scores:
            with st.expander(f"**{score['competency']}** — {score['depth'].title()} ({score.get('confidence', 'N/A')} confidence)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Key Evidence:**")
                    st.info(score.get("key_evidence", "N/A"))
                with col2:
                    st.markdown("**⚠️ Key Gap:**")
                    st.warning(score.get("key_gap", "N/A"))

                tradeoff = score.get("tradeoff_awareness", "not_probed")
                tradeoff_label = {
                    "not_probed": "⚪ Not probed (system gap, not candidate gap)",
                    "probed_and_present": "🟢 Demonstrated when asked",
                    "probed_and_absent": "🔴 Asked but could not surface tradeoffs",
                }.get(tradeoff, tradeoff)
                st.markdown(f"**Tradeoff Awareness:** {tradeoff_label}")
                st.markdown(f"**Exit Condition:** {score.get('exit_condition', 'N/A')}")

        # ─── Difficulty Trajectory ────────────────────────────────
        trajectory = recruiter.get("difficulty_trajectory", "")
        if trajectory:
            st.markdown(f"""
            <div class="report-section">
                <h3>📈 Difficulty Trajectory</h3>
                <p style="color: #e4e4f5;">{trajectory}</p>
            </div>
            """, unsafe_allow_html=True)

        # ─── Flags ────────────────────────────────────────────────
        flags = recruiter.get("flags", [])
        if flags:
            st.markdown('<div class="report-section"><h3>🚩 Flags</h3></div>', unsafe_allow_html=True)
            for flag in flags:
                st.markdown(f"- ⚠️ {flag}")

        # ─── Candidate Feedback ───────────────────────────────────
        if feedback:
            st.markdown("---")
            st.markdown("## 💬 Candidate Feedback")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 💪 Strengths")
                for s in feedback.get("strengths", []):
                    st.markdown(f"✅ {s}")
            with col2:
                st.markdown("### 🌱 Areas to Develop")
                for d in feedback.get("development_areas", []):
                    st.markdown(f"📌 {d}")

            overall = feedback.get("overall_impression", "")
            if overall:
                st.info(f"**Overall:** {overall}")

        # ─── Raw Data Access ──────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📁 Transparency — Raw Data")

        tab1, tab2, tab3 = st.tabs(["Candidate State", "Interview Log", "Raw Report JSON"])

        with tab1:
            controller = st.session_state.controller
            if controller:
                st.code(controller.candidate_state.get_state_md(), language="markdown")

        with tab2:
            if controller:
                st.code(controller.interview_log.get_full_log(), language="markdown")

        with tab3:
            st.json(report)

        # ─── Actions ──────────────────────────────────────────────
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col2:
            if st.button("📥 Download Report", use_container_width=True):
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=report_json,
                    file_name="hiremind_report.json",
                    mime="application/json",
                )
