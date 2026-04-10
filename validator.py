"""Validation and normalization for extracted attendance rows."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from config import (
    EMPLOYEE_ID_PATTERN,
    EMPLOYEE_ID_SEARCH_PATTERN,
    ILLEGIBLE_HINTS,
    NAME_PATTERN,
    SECTION_LABELS,
)


def _string_value(row: Mapping[str, object], key: str) -> str:
    return str(row.get(key, "") or "").strip()


def _bool_value(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


def _normalize_employee_id(employee_id: str) -> str:
    normalized = employee_id.upper().strip()
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    # Fix common handwriting misreads: M->H (MI->HI), H1->HI, HL->HI
    normalized = re.sub(r"^[MH][1IL](?=\d)", "HI", normalized)
    if re.fullmatch(r"[MH]\d{3,4}", normalized):
        normalized = "HI" + normalized[1:]
    normalized = normalized.replace("O", "0")
    return normalized


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip())
    normalized = normalized.strip(".,;:/\\|")
    return normalized


def _extract_employee_id_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in EMPLOYEE_ID_SEARCH_PATTERN.findall(text or ""):
        normalized = _normalize_employee_id(match)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _remove_employee_id_candidates(text: str) -> str:
    cleaned = EMPLOYEE_ID_SEARCH_PATTERN.sub(" ", text or "")
    return _normalize_name(cleaned)


def _looks_like_name_fragment(value: str) -> bool:
    if not value:
        return False
    if _is_placeholder_text(value):
        return False
    # Reject strings that contain digits -- likely employee IDs or ID fragments
    if any(char.isdigit() for char in value):
        return False
    # Reject bare prefixes that look like employee ID stems (HI, MI, H1, etc.)
    if re.fullmatch(r"[HMhm][IiLl1]?", value.strip()):
        return False
    letters_only = re.sub(r"[^A-Za-z]", "", value)
    return len(letters_only) >= 2


def _merge_name_parts(*parts: str) -> str:
    merged: list[str] = []
    for part in parts:
        normalized = _normalize_name(part)
        if not normalized:
            continue
        if not _looks_like_name_fragment(normalized):
            continue
        if normalized not in merged:
            merged.append(normalized)
    return _normalize_name(" ".join(merged))


def _is_placeholder_text(value: str) -> bool:
    return value.strip().lower() in SECTION_LABELS


def _looks_like_invalid_name(name: str) -> bool:
    lowered = name.lower()
    if _is_placeholder_text(lowered):
        return True
    if any(char.isdigit() for char in name):
        return True
    if len(re.sub(r"[^A-Za-z]", "", name)) < 2:
        return True
    return False


def _normalize_row(row: Mapping[str, object], index: int) -> dict:
    raw_employee_id = _string_value(row, "employee_id")
    raw_name = _string_value(row, "name")
    notes = _string_value(row, "notes")

    employee_id_candidates = (
        _extract_employee_id_candidates(raw_employee_id)
        or _extract_employee_id_candidates(raw_name)
        or _extract_employee_id_candidates(notes)
    )
    employee_id = employee_id_candidates[0] if employee_id_candidates else _normalize_employee_id(raw_employee_id)

    employee_id_residual = _remove_employee_id_candidates(raw_employee_id)
    name_without_ids = _remove_employee_id_candidates(raw_name)
    name = _merge_name_parts(employee_id_residual, name_without_ids)

    repair_notes: list[str] = []
    if employee_id and employee_id != _normalize_employee_id(raw_employee_id):
        repair_notes.append("employee ID was repaired from adjacent text")
    if name and name != _normalize_name(raw_name):
        repair_notes.append("name was cleaned to remove non-name content")

    if repair_notes:
        notes = "; ".join([notes, *repair_notes]).strip("; ").strip()

    return {
        "row_number": int(row.get("row_number") or index),
        "name": name,
        "employee_id": employee_id,
        "signature_present": _bool_value(row, "signature_present"),
        "attendance_date": _string_value(row, "attendance_date"),
        "training_title": _string_value(row, "training_title"),
        "facilitator_name": _string_value(row, "facilitator_name"),
        "notes": notes,
    }


def _should_drop_row(row: Mapping[str, object]) -> bool:
    employee_id = _string_value(row, "employee_id")
    name = _string_value(row, "name")
    notes = _string_value(row, "notes")

    if not any([employee_id, name, notes]):
        return True
    if _is_placeholder_text(employee_id.lower()) and not name:
        return True
    if _is_placeholder_text(name.lower()) and not employee_id:
        return True
    if employee_id.lower() == "submitattendancerows" or name.lower() == "submit_attendance_rows":
        return True
    return False


def _flag_row(row: Mapping[str, object], duplicate_employee_ids: set[str]) -> list[str]:
    flags: list[str] = []
    name = _string_value(row, "name")
    employee_id = _string_value(row, "employee_id")
    notes = _string_value(row, "notes").lower()
    attendance_date = _string_value(row, "attendance_date")

    if not name:
        flags.append("Missing attendee name")
    elif any(hint in name.lower() or hint in notes for hint in ILLEGIBLE_HINTS):
        flags.append("Attendee name may be illegible")
    elif _looks_like_invalid_name(name):
        flags.append("Attendee name looks invalid")
    elif not NAME_PATTERN.match(name):
        flags.append("Attendee name needs review")
    if "name was cleaned to remove non-name content" in notes:
        flags.append("Name column contained mixed content")

    if not employee_id:
        flags.append("Missing employee ID")
    elif not EMPLOYEE_ID_PATTERN.match(employee_id):
        flags.append("Employee ID format needs review")
    if "employee id was repaired from adjacent text" in notes:
        flags.append("Employee ID column needed repair")

    if employee_id and employee_id in duplicate_employee_ids:
        flags.append("Duplicate employee ID")

    if not _bool_value(row, "signature_present"):
        flags.append("Missing signature")

    if not attendance_date:
        flags.append("Missing attendance date")

    return flags


def validate_rows(rows: list[dict]) -> list[dict]:
    """Normalize, filter, and flag extracted attendance rows."""
    normalized_rows: list[dict] = []
    for index, row in enumerate(rows, start=1):
        normalized = _normalize_row(row, index)
        if _should_drop_row(normalized):
            continue
        normalized_rows.append(normalized)

    duplicate_employee_ids = {
        employee_id
        for employee_id, count in Counter(
            row["employee_id"] for row in normalized_rows if row.get("employee_id")
        ).items()
        if count > 1
    }

    validated_rows: list[dict] = []
    for row in normalized_rows:
        review_flags = _flag_row(row, duplicate_employee_ids)
        normalized = dict(row)
        normalized["review_flags"] = review_flags
        normalized["requires_review"] = bool(review_flags)
        validated_rows.append(normalized)
    return validated_rows
