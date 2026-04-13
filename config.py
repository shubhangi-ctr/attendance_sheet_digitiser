"""Centralized configuration for the Clinical Attendance Capture application."""

from __future__ import annotations

import re
from dotenv import load_dotenv
import os

load_dotenv()

# ── API Configuration ───────────────────────────────────────────────────────

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash-lite"
API_MAX_RETRIES = 3
API_MAX_OUTPUT_TOKENS = 8000
API_TEMPERATURE = 0


# ── Image Processing ────────────────────────────────────────────────────────

IMAGE_LIMIT_BYTES = 20 * 1024 * 1024  # Gemini supports 20 MB inline
MAX_BASE64_IMAGE_BYTES = IMAGE_LIMIT_BYTES - 32_768
MODEL_IMAGE_MAX_DIMENSION = 1800
VIEW_IMAGE_MEDIA_TYPE = "image/jpeg"
JPEG_QUALITY = 88
PDF_SCALE_FACTOR = 2

SUPPORTED_IMAGE_TYPES: dict[str, str] = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
}

FILENAME_IMAGE_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


# ── Extraction ──────────────────────────────────────────────────────────────

JSON_ARRAY_KEYS = ("rows", "attendance", "attendees",
                   "data", "records", "entries")

ROW_KEYS = {
    "row_number",
    "name",
    "employee_id",
    "signature_present",
    "attendance_date",
    "training_time",
    "training_title",
    "facilitator_name",
    "notes",
}


EXTRACTION_PROMPT_TEMPLATE = """
Extract attendance from this clinical training sign-in sheet.
Read the uploaded image directly and extract the attendance table accurately.

Return a JSON array where each item follows this schema:
{{
  "row_number": 1,
  "name": "",
  "employee_id": "",
  "signature_present": true,
  "attendance_date": "",
  "training_time":"",
  "training_title": "",
  "facilitator_name": "",
  "notes": ""
}}

You are a highly capable intelligent document processor. You will extract data from clinical training sign-in sheets, even when text is messy and handwritten.
Below are FEW-SHOT EXAMPLES demonstrating successful extraction from this complex document:

Example 1: A neat entry.
Input: (Row 1 from image_0.png) HI257 DOMADIA DEEP, scribbled signature.
Output (JSON item):
{{
  "row_number": 1,
  "name": "DOMADIA DEEP",
  "employee_id": "HI257",
  "signature_present": true,
  "attendance_date": "24 July 2024",
  "training_time":"04:00 PM - 05:00 PM",
  "training_title": "Introductory session on Power BI",
  "facilitator_name": "Rushi Shah, Meghanshu Bhatia",
  "notes": ""
}}

Example 2: A messy ID.
Input: (Row 6 from image_0.png) HI, Dhaval Vasavada, signature ink decider decided.
Instruction: Extract 'HI' as the best-read ID and add a note.
Output (JSON item):
{{
  "row_number": 6,
  "name": "Dhaval Vasavada",
  "employee_id": "HI",
  "signature_present": true,
  "attendance_date": "24 July 2024",
  "training_time":"04:00 PM - 05:00 PM",
  "training_title": "Introductory session on Power BI",
  "facilitator_name": "Rushi Shah, Meghanshu Bhatia",
  "notes": "Employee ID HI prefix detected, but digits are missing. Attendee name confirmed."
}}

Example 3: A confused ID and messy name.
Input: (Row 3 from image_0.png) Messey, looks like a mix of M/H, I/1, ... HI pattern, digits are present. Name looks messy.
Output (JSON item):
{{
  "row_number": 3,
  "name": "Pratika Karna",
  "employee_id": "HI202",
  "signature_present": true,
  "attendance_date": "24 July 2024",
  "training_time":"04:00 PM - 05:00 PM",
  "training_title": "Introductory session on Power BI",
  "facilitator_name": "Rushi Shah, Meghanshu Bhatia",
  "notes": ""
}}

Instructions:
- The sheet may have top metadata such as Session, Date, Time, and Trainer/Speaker.
- The sheet may contain a Trainer/Speaker table and a separate Attendees table.
- Extract ONLY rows from the Attendees section for LMS attendance output.
- Ignore trainer/speaker rows.
- The Attendees table columns are Employee ID, Name, and Signature.
- If Time is visible at the top (e.g., "04:00 PM to 5:00 PM"), use it as "training_time".
- Read the table row-by-row in visible order.
- Use one JSON item per attendee row.
- Preserve the visible attendee row order from top to bottom.
- For PDFs, you may receive multiple page images. Preserve reading order across pages.

Column separation rules (CRITICAL):
- Keep each extracted value in its correct column: employee IDs only in `employee_id`, names only in `name`.
- Never combine `employee_id` and `name` into one field.
- Employee IDs follow the pattern "HI" followed by 1-4 digits (e.g. HI257, HI36, HI1234).
- If handwriting makes the prefix ambiguous (could be HI, MI, H1, Hl), always normalize to "HI".
- Names are full human names (first name + last name). They never contain digits.

Signature handling:
- Do not try to read signature handwriting as a person's name.
- Treat `signature_present` as only a yes/no decision based on whether the signature cell contains ink.
- Set "signature_present" to true if the signature cell contains any pen mark, initials, scribble, or signature.
- Set "signature_present" to false only when the signature cell is clearly blank.

Handwriting tips:
- Capture handwritten names and employee IDs as carefully as possible.
- If a handwritten value is unclear, keep the best reading and mention the uncertainty in "notes".
- Common handwriting confusions: M vs H, I vs 1 vs l, O vs 0, Z vs 2.
- When in doubt about the employee ID prefix, prefer "HI" (the standard prefix).
- Count ALL rows including those written in different ink colors (black, blue, pink).

Metadata extraction:
- If Session or training title is visible at the top, use it as "training_title".
- If Date is visible at the top, use it as "attendance_date".
- If Trainer/Speaker names are visible, combine them as "facilitator_name" (comma-separated if multiple).
- If any top metadata is not visible or unclear, use these defaults when appropriate:
  training_title = "{training_title}"
  attendance_date = "{training_date}"
  training_time = "{training_time}"
  facilitator_name = "{facilitator_name}"
- Return ONLY the JSON array. No markdown, no commentary.
""".strip()

REPAIR_PROMPT_TEMPLATE = """
Convert the following model output into a valid JSON array only.

Target schema for each item:
{{
  "row_number": 1,
  "name": "",
  "employee_id": "",
  "signature_present": true,
  "attendance_date": "",
  "training_time":"",
  "training_title": "",
  "facilitator_name": "",
  "notes": ""
}}

Rules:
- Return JSON only.
- If the content is wrapped in an object, extract the attendance rows.
- Do not add commentary, markdown, or explanations.

Model output to repair:
{raw_text}
""".strip()


# ── Validation ──────────────────────────────────────────────────────────────

EMPLOYEE_ID_PATTERN = re.compile(r"^HI\d{1,4}$")
EMPLOYEE_ID_SEARCH_PATTERN = re.compile(
    r"\b[HM][1IL]\s*[-]?\s*\d{1,4}\b", re.IGNORECASE)
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,79}$")

ILLEGIBLE_HINTS = ("illegible", "unclear", "unknown", "hard to read", "?")

SECTION_LABELS = {
    "employee id",
    "name",
    "signature",
    "attendees",
    "trainer",
    "trainer/speaker",
    "speaker",
    "session",
    "date",
    "time",
}


# ── Export ───────────────────────────────────────────────────────────────────

LMS_EXPORT_COLUMNS = [
    "name",
    "employee_id",
    "signature_present",
    "attendance_date",
    "training_time",
    "training_title",
    "facilitator_name",
    "notes",
]

DEFAULT_EXPORT_FILENAME = "attendance_lms_export.csv"


# ── Audit ────────────────────────────────────────────────────────────────────

AUDIT_COLUMNS = [
    "submission_id",
    "submitted_at_utc",
    "submitted_by",
    "source_filename",
    "source_mime_type",
    "training_title",
    "training_date",
    "facilitator_name",
    "extracted_row_count",
    "review_required_count",
    "extracted_rows_json",
    "reviewed_rows_json",
]

AUDIT_VISIBLE_COLUMNS = [
    "submission_id",
    "submitted_at_utc",
    "submitted_by",
    "source_filename",
    "training_title",
    "training_date",
    "facilitator_name",
    "extracted_row_count",
    "review_required_count",
]

SUBMISSION_ID_LENGTH = 12


# ── Supported Upload Types ──────────────────────────────────────────────────

ALLOWED_FILE_EXTENSIONS = ["pdf", "png", "jpg", "jpeg"]
