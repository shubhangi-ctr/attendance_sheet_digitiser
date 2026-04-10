from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    ALLOWED_FILE_EXTENSIONS,
    DEFAULT_EXPORT_FILENAME,
    LMS_EXPORT_COLUMNS,
)
from extractor import extract_attendance
from validator import validate_rows


APP_DIR = Path(__file__).resolve().parent

# Metadata defaults — the AI reads these from the scanned sheet automatically.
# Empty strings force the model to rely on what it sees in the image.
_DEFAULT_TRAINING_TITLE = ""
_DEFAULT_TRAINING_DATE = ""
_DEFAULT_FACILITATOR_NAME = ""


def normalize_rows_for_editor(rows: list[dict]) -> pd.DataFrame:
    normalized = []
    for index, row in enumerate(rows, start=1):
        normalized.append(
            {
                "row_number": row.get("row_number", index),
                "name": row.get("name", ""),
                "employee_id": row.get("employee_id", ""),
                "signature_present": bool(row.get("signature_present", False)),
                "attendance_date": row.get("attendance_date", ""),
                "training_title": row.get("training_title", ""),
                "facilitator_name": row.get("facilitator_name", ""),
                "notes": row.get("notes", ""),
                "review_flags": ", ".join(row.get("review_flags", [])),
                "requires_review": bool(row.get("requires_review", False)),
            }
        )
    return pd.DataFrame(normalized)


def dataframe_to_rows(df: pd.DataFrame) -> list[dict]:
    def clean_value(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value).strip()

    def clean_bool(value: object) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "y"}
        return bool(value)

    def parse_row_number(value: object, fallback: int) -> int:
        if value is None or pd.isna(value):
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    records = []
    for index, row in enumerate(df.to_dict(orient="records"), start=1):
        name = clean_value(row.get("name", ""))
        employee_id = clean_value(row.get("employee_id", ""))
        attendance_date = clean_value(row.get("attendance_date", ""))
        training_title = clean_value(row.get("training_title", ""))
        facilitator_name = clean_value(row.get("facilitator_name", ""))
        notes = clean_value(row.get("notes", ""))
        signature_present = clean_bool(row.get("signature_present", False))

        if not any([name, employee_id, attendance_date, training_title, facilitator_name, notes, signature_present]):
            continue

        record = {
            "row_number": parse_row_number(row.get("row_number"), index),
            "name": name,
            "employee_id": employee_id,
            "signature_present": signature_present,
            "attendance_date": attendance_date,
            "training_title": training_title,
            "facilitator_name": facilitator_name,
            "notes": notes,
        }
        records.append(record)
    return records


def build_lms_csv(rows: list[dict]) -> bytes:
    export_df = pd.DataFrame(rows)
    for column in LMS_EXPORT_COLUMNS:
        if column not in export_df.columns:
            export_df[column] = ""
    export_df = export_df[LMS_EXPORT_COLUMNS]

    buffer = io.StringIO()
    export_df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_validation_summary(rows: list[dict]) -> list[str]:
    issues: list[str] = []
    flagged_rows = [row for row in rows if row.get("requires_review")]
    duplicate_ids = sorted(
        {
            row.get("employee_id", "")
            for row in flagged_rows
            if "Duplicate employee ID" in row.get("review_flags", [])
        }
    )
    invalid_id_count = sum(
        1 for row in flagged_rows if "Employee ID format needs review" in row.get("review_flags", [])
    )
    invalid_name_count = sum(
        1
        for row in flagged_rows
        if any(
            flag in row.get("review_flags", [])
            for flag in (
                "Attendee name looks invalid",
                "Attendee name needs review",
                "Name column contained mixed content",
            )
        )
    )
    repaired_id_count = sum(
        1 for row in flagged_rows if "Employee ID column needed repair" in row.get("review_flags", [])
    )

    if invalid_id_count:
        issues.append(
            f"{invalid_id_count} row(s) have employee IDs that do not match the expected `HI1234` pattern.")
    if repaired_id_count:
        issues.append(
            f"{repaired_id_count} row(s) had employee IDs repaired from mixed or shifted column text.")
    if invalid_name_count:
        issues.append(
            f"{invalid_name_count} row(s) have names that look malformed or need confirmation.")
    if duplicate_ids:
        issues.append(
            f"Duplicate employee IDs found: {', '.join(duplicate_ids)}.")
    if not issues and flagged_rows:
        issues.append(
            "Some rows still need manual confirmation before export.")
    return issues


def initialize_state() -> None:
    st.session_state.setdefault("raw_rows", [])
    st.session_state.setdefault("review_rows", [])
    st.session_state.setdefault("source_filename", "")
    st.session_state.setdefault("source_mime_type", "")
    st.session_state.setdefault("finalized_csv", None)


def clear_finalized_state() -> None:
    st.session_state["finalized_csv"] = None


# ── Page Setup ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Attendance Sheet Digitiser",
                   page_icon="📋", layout="wide")
initialize_state()

st.title("📋Attendance Sheet Digitiser")
st.caption("Upload a scanned paper attendance tracker, review the extracted rows, and export LMS-ready CSV.")

# ── Upload & Extract ────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload scanned attendance sheet",
    type=ALLOWED_FILE_EXTENSIONS,
    help=f"{', '.join(ext.upper() for ext in ALLOWED_FILE_EXTENSIONS)} files are supported.",
)

extract_clicked = st.button(
    "Extract Attendance", type="primary", disabled=uploaded_file is None)

if extract_clicked and uploaded_file is not None:
    clear_finalized_state()
    with st.spinner("Extracting rows from the attendance sheet..."):
        try:
            extracted_rows = extract_attendance(
                file_bytes=uploaded_file.getvalue(),
                mime_type=uploaded_file.type or "",
                filename=uploaded_file.name,
                training_title=_DEFAULT_TRAINING_TITLE,
                training_date=_DEFAULT_TRAINING_DATE,
                facilitator_name=_DEFAULT_FACILITATOR_NAME,
            )
            validated_rows = validate_rows(extracted_rows)
            st.session_state["raw_rows"] = validated_rows
            st.session_state["review_rows"] = validated_rows
            st.session_state["source_filename"] = uploaded_file.name
            st.session_state["source_mime_type"] = uploaded_file.type or ""
            st.success(f"Extracted {len(validated_rows)} attendance row(s).")
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")

# ── Review Table ────────────────────────────────────────────────────────────

if st.session_state["review_rows"]:
    st.subheader("Review Extracted Rows")
    review_df = normalize_rows_for_editor(st.session_state["review_rows"])
    edited_df = st.data_editor(
        review_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "row_number": st.column_config.NumberColumn("Row", disabled=True),
            "signature_present": st.column_config.CheckboxColumn("Signature Present"),
            "review_flags": st.column_config.TextColumn("Validation Flags", disabled=True, width="large"),
            "requires_review": st.column_config.CheckboxColumn("Needs Review", disabled=True),
        },
        disabled=["row_number", "review_flags", "requires_review"],
        hide_index=True,
        key="main_editor",
    )

    reviewed_rows = validate_rows(dataframe_to_rows(edited_df))
    st.session_state["review_rows"] = reviewed_rows

    flagged_rows = [row for row in reviewed_rows if row.get("requires_review")]
    total_rows = len(reviewed_rows)
    flagged_count = len(flagged_rows)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Extracted Rows", total_rows)
    metric_col2.metric("Rows Requiring Review", flagged_count)
    metric_col3.metric("Ready Rows", total_rows - flagged_count)

    # ── Flagged Rows (editable) ─────────────────────────────────────────
    if flagged_rows:
        st.subheader("Flagged Entries")
        st.warning(
            "Some rows need confirmation before export. Edit or delete flagged entries below.")
        for issue in build_validation_summary(reviewed_rows):
            st.caption(issue)

        flagged_df = normalize_rows_for_editor(flagged_rows)
        edited_flagged_df = st.data_editor(
            flagged_df[["row_number", "name", "employee_id",
                        "signature_present", "notes", "review_flags"]],
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "row_number": st.column_config.NumberColumn("Row", disabled=True),
                "signature_present": st.column_config.CheckboxColumn("Signature Present"),
                "review_flags": st.column_config.TextColumn("Flags", disabled=True, width="large"),
            },
            disabled=["row_number", "review_flags"],
            hide_index=True,
            key="flagged_editor",
        )

        if st.button("Sync Flagged Changes"):
            # Build a lookup of edited flagged rows by row_number
            edited_flagged_records = edited_flagged_df.to_dict(
                orient="records")
            edited_by_row_num: dict[int, dict] = {}
            for rec in edited_flagged_records:
                rn = rec.get("row_number")
                if rn is not None and not pd.isna(rn):
                    edited_by_row_num[int(rn)] = rec

            # Deleted row numbers (rows that were in flagged but no longer in the edited df)
            original_flagged_row_nums = {r["row_number"] for r in flagged_rows}
            surviving_row_nums = set(edited_by_row_num.keys())
            deleted_row_nums = original_flagged_row_nums - surviving_row_nums

            # Merge edits back into the full reviewed_rows list
            merged: list[dict] = []
            for row in reviewed_rows:
                rn = row["row_number"]
                if rn in deleted_row_nums:
                    continue  # User deleted this flagged row
                if rn in edited_by_row_num:
                    edits = edited_by_row_num[rn]
                    row = dict(row)  # copy
                    for field in ("name", "employee_id", "notes"):
                        val = edits.get(field)
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            row[field] = str(val).strip()
                    sig = edits.get("signature_present")
                    if sig is not None:
                        row["signature_present"] = bool(sig)
                merged.append(row)

            # Re-validate and update state
            re_validated = validate_rows(dataframe_to_rows(
                normalize_rows_for_editor(merged)
            ))
            st.session_state["review_rows"] = re_validated
            st.rerun()

    # ── Export ──────────────────────────────────────────────────────────
    st.divider()
    finalize_clicked = st.button("Finalize LMS-ready CSV", type="primary")
    if finalize_clicked:
        st.session_state["finalized_csv"] = build_lms_csv(reviewed_rows)
        st.success("Ready for download!")

    if st.session_state["finalized_csv"] is not None:
        st.download_button(
            "📥Download CSV",
            data=st.session_state["finalized_csv"],
            file_name=DEFAULT_EXPORT_FILENAME,
            type="primary",
            mime="text/csv",
        )
