"""Vision-based attendance extraction using the Google Gemini API."""

from __future__ import annotations

import json
import os
import re
import time
from io import BytesIO
from typing import Any

import fitz
from PIL import Image, ImageOps
from PIL import ImageEnhance

from config import (
    API_MAX_OUTPUT_TOKENS,
    API_MAX_RETRIES,
    API_TEMPERATURE,
    EXTRACTION_PROMPT_TEMPLATE,
    FILENAME_IMAGE_TYPES,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL,
    IMAGE_LIMIT_BYTES,
    JPEG_QUALITY,
    JSON_ARRAY_KEYS,
    MAX_BASE64_IMAGE_BYTES,
    MODEL_IMAGE_MAX_DIMENSION,
    PDF_SCALE_FACTOR,
    REPAIR_PROMPT_TEMPLATE,
    ROW_KEYS,
    SUPPORTED_IMAGE_TYPES,
    VIEW_IMAGE_MEDIA_TYPE,
)


# ── PDF / Image Handling ────────────────────────────────────────────────────


def _pdf_to_png_images(file_bytes: bytes) -> list[bytes]:
    """Convert each page of a PDF to a PNG image at high resolution."""
    document = fitz.open(stream=file_bytes, filetype="pdf")
    images: list[bytes] = []
    try:
        for page in document:
            matrix = fitz.Matrix(PDF_SCALE_FACTOR, PDF_SCALE_FACTOR)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pixmap.tobytes("png"))
    finally:
        document.close()
    return images


def _encode_image(image: Image.Image, media_type: str) -> bytes:
    """Encode a PIL Image to bytes in the specified media type."""
    buffer = BytesIO()
    if media_type == "image/jpeg":
        export_image = image.convert("RGB")
        export_image.save(
            buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True,
        )
    else:
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _resize_for_model(
    image: Image.Image, max_dimension: int = MODEL_IMAGE_MAX_DIMENSION,
) -> Image.Image:
    """Resize an image so its longest side does not exceed *max_dimension*."""
    width, height = image.size
    largest_side = max(width, height)
    if largest_side <= max_dimension:
        return image

    scale = max_dimension / largest_side
    resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(resized_size, Image.Resampling.LANCZOS)


def _split_image(image: Image.Image) -> list[Image.Image]:
    """Split an image in half along its longer axis."""
    width, height = image.size
    if width <= 1 and height <= 1:
        raise ValueError(
            "Unable to split image further while keeping chunk size under the limit.")

    if height >= width and height > 1:
        midpoint = height // 2
        return [
            image.crop((0, 0, width, midpoint)),
            image.crop((0, midpoint, width, height)),
        ]

    midpoint = width // 2
    return [
        image.crop((0, 0, midpoint, height)),
        image.crop((midpoint, 0, width, height)),
    ]


def _load_image(file_bytes: bytes) -> Image.Image:
    """Load an image from bytes and correct EXIF orientation."""
    image = Image.open(BytesIO(file_bytes))
    image.load()
    return ImageOps.exif_transpose(image)


def _prepare_image_bytes(image: Image.Image) -> tuple[bytes, str]:
    """Resize and encode an image; return *(raw_bytes, mime_type)*."""
    gray_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray_image)
    processed_image = enhancer.enhance(2.0)  # Boost contrast significantly

    prepared = _resize_for_model(processed_image)
    encoded = _encode_image(prepared, VIEW_IMAGE_MEDIA_TYPE)

    # Gemini supports 20 MB inline, but split if still too large
    if len(encoded) <= IMAGE_LIMIT_BYTES:
        return encoded, VIEW_IMAGE_MEDIA_TYPE

    # Recursive split for extremely large images
    parts: list[tuple[bytes, str]] = []
    for tile in _split_image(prepared):
        parts.append(_prepare_image_bytes(tile))
    return parts[0]  # Return first tile as fallback


def _collect_image_parts(
    file_bytes: bytes, mime_type: str, filename: str,
) -> list[tuple[bytes, str]]:
    """Return a list of *(image_bytes, mime_type)* tuples for the upload."""
    parts: list[tuple[bytes, str]] = []

    if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
        page_images = _pdf_to_png_images(file_bytes)
        if not page_images:
            raise ValueError(
                "The uploaded PDF did not contain any renderable pages.")
        for page_bytes in page_images:
            parts.append(_prepare_image_bytes(_load_image(page_bytes)))
    else:
        resolved_type = SUPPORTED_IMAGE_TYPES.get(mime_type)
        if not resolved_type:
            for suffix, resolved in FILENAME_IMAGE_TYPES.items():
                if filename.lower().endswith(suffix):
                    resolved_type = resolved
                    break
        if not resolved_type:
            raise ValueError(
                f"Unsupported upload type: {mime_type or filename}")
        parts.append(_prepare_image_bytes(_load_image(file_bytes)))

    return parts


# ── JSON Parsing ────────────────────────────────────────────────────────────


def _coerce_rows_payload(payload: Any) -> list[dict]:
    """Validate and coerce *payload* into a list of attendance row dicts."""
    if isinstance(payload, list):
        if not payload:
            return []

        normalized_rows: list[dict] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(
                    "The model response list did not contain attendance row objects.")
            if not (set(item.keys()) & ROW_KEYS):
                raise ValueError(
                    "The model response list did not contain attendance row fields.")
            normalized_rows.append(item)
        return normalized_rows

    if isinstance(payload, dict):
        for key in JSON_ARRAY_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return _coerce_rows_payload(value)

    raise ValueError(
        "The model response did not contain a valid attendance row array.")


def _extract_json_array(text: str) -> list[dict]:
    """Parse a JSON array of attendance rows from *text*."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return _coerce_rows_payload(parsed)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
            return _coerce_rows_payload(parsed)
        except json.JSONDecodeError:
            continue

    raise ValueError("The model response did not contain a valid JSON array.")


# ── Gemini API Helpers ──────────────────────────────────────────────────────


def _generate_with_retry(client: Any, **kwargs: Any) -> Any:
    """Call ``client.models.generate_content`` with retry on server errors."""
    last_error: Exception | None = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            status_code = getattr(exc, "status_code",
                                  getattr(exc, "code", None))
            is_retryable = status_code is not None and (
                (isinstance(status_code, int) and status_code >= 500)
            )
            exc_name = exc.__class__.__name__
            if "ServerError" in exc_name or "InternalError" in exc_name:
                is_retryable = True

            if not is_retryable or attempt == API_MAX_RETRIES:
                raise

            last_error = exc
            time.sleep(attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini request failed without returning a response.")


def _repair_json_array(client: Any, model: str, raw_text: str) -> list[dict]:
    """Send malformed text back to the model for JSON repair."""
    from google.genai import types  # noqa: WPS433 (local import is intentional)

    repair_prompt = REPAIR_PROMPT_TEMPLATE.format(raw_text=raw_text)

    response = _generate_with_retry(
        client,
        model=model,
        contents=repair_prompt,
        config=types.GenerateContentConfig(
            temperature=API_TEMPERATURE,
            max_output_tokens=API_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
        ),
    )

    response_text = response.text
    if not response_text or not response_text.strip():
        raise ValueError("The repair response did not contain text output.")
    return _extract_json_array(response_text)


# ── Row Normalization ───────────────────────────────────────────────────────


def _normalize_record(
    row: dict, index: int, training_title: str, training_date: str, facilitator_name: str,
) -> dict:
    """Normalize a single extracted row into the canonical schema."""

    def to_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "y"}
        return bool(value)

    def normalize_employee_id(value: object) -> str:
        text = str(value or "").upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)
        # Fix common handwriting misreads: M→H (MI→HI), H1→HI, HL→HI
        text = re.sub(r"^[MH][1IL](?=\d)", "HI", text)
        if re.fullmatch(r"[MH]\d{3,4}", text):
            text = "HI" + text[1:]
        text = text.replace("O", "0")
        return text

    def normalize_name(value: object) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    notes = str(row.get("notes", "")).strip()
    return {
        "row_number": int(row.get("row_number") or index),
        "name": normalize_name(row.get("name", "")),
        "employee_id": normalize_employee_id(row.get("employee_id", "")),
        "signature_present": to_bool(row.get("signature_present", False)),
        "attendance_date": str(row.get("attendance_date") or training_date).strip(),
        "training_title": str(row.get("training_title") or training_title).strip(),
        "facilitator_name": str(row.get("facilitator_name") or facilitator_name).strip(),
        "notes": notes,
    }


# ── Public API ──────────────────────────────────────────────────────────────


def extract_attendance(
    *,
    file_bytes: bytes,
    mime_type: str,
    filename: str,
    training_title: str,
    training_date: str,
    facilitator_name: str,
) -> list[dict]:
    """Extract attendance rows from a scanned sheet using Gemini Vision.

    Returns a list of normalized attendance-row dicts.
    """
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{GEMINI_API_KEY_ENV} is not set.")

    try:
        from google import genai  # noqa: WPS433
        from google.genai import types  # noqa: WPS433
    except ImportError as exc:
        raise RuntimeError(
            "The `google-genai` package is not installed.") from exc

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)

    # ── Build content parts (images + prompt) ───────────────────────────
    image_parts = _collect_image_parts(file_bytes, mime_type, filename)

    parts: list[Any] = []
    for img_bytes, img_mime in image_parts:
        parts.append(
            types.Part.from_bytes(data=img_bytes, mime_type=img_mime),
        )

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        training_title=training_title,
        training_date=training_date,
        facilitator_name=facilitator_name,
    )
    parts.append(types.Part.from_text(text=prompt))

    # ── Structured-output schema for Gemini ─────────────────────────────
    attendance_row_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "row_number": {"type": "INTEGER"},
                "name": {"type": "STRING"},
                "employee_id": {"type": "STRING"},
                "signature_present": {"type": "BOOLEAN"},
                "attendance_date": {"type": "STRING"},
                "training_title": {"type": "STRING"},
                "facilitator_name": {"type": "STRING"},
                "notes": {"type": "STRING"},
            },
            "required": [
                "row_number", "name", "employee_id", "signature_present",
                "attendance_date", "training_title", "facilitator_name", "notes",
            ],
        },
    }

    # ── Call Gemini with structured output ───────────────────────────────
    response = _generate_with_retry(
        client,
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=API_TEMPERATURE,
            max_output_tokens=API_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=attendance_row_schema,
        ),
    )

    # ── Parse response ──────────────────────────────────────────────────
    response_text = response.text
    if not response_text or not response_text.strip():
        raise ValueError("The Gemini response was empty.")

    try:
        parsed_rows = _extract_json_array(response_text)
    except ValueError:
        parsed_rows = _repair_json_array(client, model, response_text)

    return [
        _normalize_record(row, index, training_title,
                          training_date, facilitator_name)
        for index, row in enumerate(parsed_rows, start=1)
    ]
