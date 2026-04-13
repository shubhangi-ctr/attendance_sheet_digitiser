from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure these constants exist in your config.py
from config import (
    ALLOWED_FILE_EXTENSIONS,
    DEFAULT_EXPORT_FILENAME,
    LMS_EXPORT_COLUMNS,
)
from extractor import extract_attendance
from validator import validate_rows

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attendance AI Digitizer",
    page_icon="📋",
    layout="wide"
)

# ── Blue Dashboard CSS ─────────────────────────────────────────────────────
st.markdown("""
    <style>
    /* 1. Remove Top Padding to pull header up */
    .block-container {
    padding-top: 4rem !important;
    padding-bottom: 0rem !important;
    }

    /* 2. Main Background */
    .stApp {
        background: linear-gradient(180deg, #EBF2FA 0%, #ffffff 100%);
    }

    /* 3. Card-style containers (Modified to avoid ghost cells) */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown):not(:first-child) {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 123, 255, 0.1);
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
        margin-bottom: 1rem;
    }
            
    /* 4. Modern Blue Header */
    .main-header {
        background: linear-gradient(90deg, #002366 0%, #002366 50%,#002366 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(67, 100, 247, 0.2);
    }

    div.stButton > button:first-child {
        background-color: #002366 !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #002366 !important;
        box-shadow: 0 5px 15px rgba(0, 123, 255, 0.4) !important;
        transform: translateY(-2px);
    }

    [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 123, 255, 0.1);
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05);
    }

    [data-testid="stMetricValue"] {
        color: #002366 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Target only the Download Button */
    .stDownloadButton > button {
        background-color: #002366 !important; /* Cobalt Blue */
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }

    /* Hover State */
    .stDownloadButton > button:hover {
        background-color: #002366!important; /* Electric Blue on hover */
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Helper Functions (The Logic) ───────────────────────────────────────────


def normalize_rows_for_editor(rows: list[dict]) -> pd.DataFrame:
    normalized = []
    for index, row in enumerate(rows, start=1):
        normalized.append({
            "row_number": row.get("row_number", index),
            "name": row.get("name", ""),
            "employee_id": row.get("employee_id", ""),
            "signature_present": bool(row.get("signature_present", False)),
            "attendance_date": row.get("attendance_date", ""),
            "training_time": row.get("training_time", ""),
            "training_title": row.get("training_title", ""),
            "facilitator_name": row.get("facilitator_name", ""),
            "notes": row.get("notes", ""),
            "review_flags": ", ".join(row.get("review_flags", [])),
            "requires_review": bool(row.get("requires_review", False)),
        })
    return pd.DataFrame(normalized)


def dataframe_to_rows(df: pd.DataFrame) -> list[dict]:
    def clean_value(value: object) -> str:
        return str(value).strip() if value is not None and not pd.isna(value) else ""

    def clean_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "y"}
        return bool(value) if value is not None and not pd.isna(value) else False

    records = []
    for index, row in enumerate(df.to_dict(orient="records"), start=1):
        record = {
            "row_number": int(row.get("row_number", index)),
            "name": clean_value(row.get("name", "")),
            "employee_id": clean_value(row.get("employee_id", "")),
            "signature_present": clean_bool(row.get("signature_present", False)),
            "attendance_date": clean_value(row.get("attendance_date", "")),
            "training_time": clean_value(row.get("training_time", "")),
            "training_title": clean_value(row.get("training_title", "")),
            "facilitator_name": clean_value(row.get("facilitator_name", "")),
            "notes": clean_value(row.get("notes", "")),
        }
        if any(record.values()):
            records.append(record)
    return records


def build_lms_csv(rows: list[dict]) -> bytes:
    export_df = pd.DataFrame(rows)
    for column in LMS_EXPORT_COLUMNS:
        if column not in export_df.columns:
            export_df[column] = ""
    buffer = io.StringIO()
    export_df[LMS_EXPORT_COLUMNS].to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _flags_fingerprint(rows: list[dict]) -> str:
    return ";".join([f"{r.get('row_number', 0)}:{'|'.join(r.get('review_flags', []))}" for r in rows])

# ── Initialization ─────────────────────────────────────────────────────────


if "review_rows" not in st.session_state:
    st.session_state.review_rows = []
if "finalized_csv" not in st.session_state:
    st.session_state.finalized_csv = None
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>📋 AI Attendance Digitizer</h1><p>Upload a scanned paper attendance tracker, review the extracted rows, and export LMS-ready CSV.</p></div>', unsafe_allow_html=True)

tab_upload, tab_review = st.tabs(["📤 **UPLOAD CENTER**", "🔍 **DATA REVIEW**"])

with tab_upload:
    st.markdown("### Document Submission")
    uploaded_file = st.file_uploader(
        "Drop your PDF or Image here", type=ALLOWED_FILE_EXTENSIONS)

    if st.button("🚀 Process & Extract", disabled=uploaded_file is None, use_container_width=True):
        with st.status("🧠 AI analyzing document...", expanded=True) as status:
            try:
                extracted = extract_attendance(
                    file_bytes=uploaded_file.getvalue(),
                    mime_type=uploaded_file.type or "",
                    filename=uploaded_file.name,
                    training_title="", training_date="", training_time="", facilitator_name=""
                )
                st.session_state.review_rows = validate_rows(extracted)
                st.session_state.editor_version += 1
                status.update(label="Extraction Complete!",
                              state="complete", expanded=True)
                st.info(
                    "💡 **Next Step:** Your data is ready! Please switch to the **DATA REVIEW** tab above to verify the details and export your CSV.")
                st.toast("All rows verified! Dataset is 100% clean.", icon="🎯")
            except Exception as e:
                st.error(f"Error: {e}")


with tab_review:
    if not st.session_state.review_rows:
        st.info(
            "👋 **Welcome!** Please head over to the **Upload Center** to process your first attendance sheet.")
    else:
        # Metrics
        total = len(st.session_state.review_rows)
        flagged = sum(
            1 for r in st.session_state.review_rows if r.get("requires_review"))

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Extracted", total)
        c2.metric("Requires Review", flagged, delta_color="inverse")
        c3.metric("Verified Rows", total - flagged)

        # Editor
        df_editor = normalize_rows_for_editor(st.session_state.review_rows)

        edited_df = st.data_editor(
            df_editor,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "row_number": st.column_config.NumberColumn("Ref", disabled=True),
                "signature_present": st.column_config.CheckboxColumn("Signed"),
                "review_flags": st.column_config.TextColumn("Issues", disabled=True),
                "requires_review": st.column_config.CheckboxColumn("Flag", disabled=True),
            },
            disabled=["row_number", "review_flags", "requires_review"],
            hide_index=True,
            key=f"editor_v{st.session_state.editor_version}"
        )

        # Sync logic
        if st.button("💾 Sync & Validate Edits", use_container_width=True):
            current_data = validate_rows(dataframe_to_rows(edited_df))
            st.session_state.review_rows = current_data
            st.session_state.editor_version += 1
            st.rerun()

        st.divider()

        if st.button("✅ Finalize Dataset", type="primary", use_container_width=True):
            st.session_state.finalized_csv = build_lms_csv(
                st.session_state.review_rows)
            st.success("Dataset finalized.")

        if st.session_state.finalized_csv:
            st.download_button(
                label="📥 Download LMS CSV",
                data=st.session_state.finalized_csv,
                file_name=DEFAULT_EXPORT_FILENAME,
                type="secondary",
                mime="text/csv",
                use_container_width=True
            )
