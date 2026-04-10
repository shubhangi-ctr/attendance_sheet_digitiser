"""Append-only audit trail for finalized attendance submissions."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from config import AUDIT_COLUMNS, AUDIT_VISIBLE_COLUMNS, SUBMISSION_ID_LENGTH


def append_audit_event(
    *,
    log_path: str | Path,
    submitted_by: str,
    source_filename: str,
    source_mime_type: str,
    training_title: str,
    training_date: str,
    facilitator_name: str,
    extracted_rows: list[dict],
    reviewed_rows: list[dict],
) -> str:
    """Append a finalized submission record and return the submission ID."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    submission_id = uuid4().hex[:SUBMISSION_ID_LENGTH]
    event = {
        "submission_id": submission_id,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted_by": submitted_by,
        "source_filename": source_filename,
        "source_mime_type": source_mime_type,
        "training_title": training_title,
        "training_date": training_date,
        "facilitator_name": facilitator_name,
        "extracted_row_count": len(extracted_rows),
        "review_required_count": sum(1 for row in reviewed_rows if row.get("requires_review")),
        "extracted_rows_json": json.dumps(extracted_rows, ensure_ascii=True),
        "reviewed_rows_json": json.dumps(reviewed_rows, ensure_ascii=True),
    }

    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(event)

    return submission_id


def read_audit_log(log_path: str | Path) -> pd.DataFrame:
    """Read the audit log CSV and return visible columns."""
    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame()

    audit_df = pd.read_csv(log_path)
    return audit_df[AUDIT_VISIBLE_COLUMNS]
