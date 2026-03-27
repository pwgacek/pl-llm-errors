from __future__ import annotations

import json
import string
from typing import Literal

UPPERCASE_LETTERS = string.ascii_uppercase
ANSWER_FIELD_KEYS = ("odpowiedź", "odpowiedz", "answer")
VerificationResult = Literal["CORRECT", "INCORRECT", "ERROR"]


def verify_response(raw: str, expected: dict) -> VerificationResult:
    kind = expected.get("type")

    if kind == "multiple_choice_index":
        return verify_multiple_choice_index(raw, expected)

    if kind == "multiple_choice_letter":
        return verify_multiple_choice_letter(raw, expected)

    if kind == "open_contained":
        return verify_open_contained(raw, expected)

    if kind == "entailment":
        return verify_entailment(raw, expected)

    return "ERROR"


def verify_multiple_choice_index(raw: str, expected: dict) -> VerificationResult:
    predicted = extract_letter(raw)
    if predicted is None:
        return "ERROR"

    correct_letter = index_to_letter(expected["correct_index"])
    if correct_letter is None:
        return "ERROR"

    return "CORRECT" if predicted == correct_letter else "INCORRECT"


def verify_multiple_choice_letter(raw: str, expected: dict) -> VerificationResult:
    predicted = extract_letter(raw)
    if predicted is None:
        return "ERROR"

    return "CORRECT" if predicted == expected["correct_letter"].upper() else "INCORRECT"


def verify_open_contained(raw: str, expected: dict) -> VerificationResult:
    answer = extract_json_field(raw, ANSWER_FIELD_KEYS)
    if answer is None:
        normalized_answer = raw.strip().lower()
    else:
        normalized_answer = answer.strip().lower()

    if not normalized_answer:
        return "ERROR"

    normalized_expected = {
        accepted.strip().lower()
        for accepted in expected["accepted_answers"]
        if isinstance(accepted, str) and accepted.strip()
    }

    if normalized_answer in normalized_expected:
        return "CORRECT"

    return "INCORRECT"


def verify_entailment(raw: str, expected: dict) -> VerificationResult:
    answer = extract_json_field(raw, ANSWER_FIELD_KEYS)
    if answer is None:
        normalized_answer = raw.strip().upper()
    else:
        normalized_answer = answer.strip().upper()

    if normalized_answer not in {"NEUTRAL", "CONTRADICTION", "ENTAILMENT"}:
        return "ERROR"

    return "CORRECT" if normalized_answer == expected["judgment"].upper() else "INCORRECT"


def extract_letter(text: str) -> str | None:
    payload = parse_json_payload(text)
    if payload is None:
        return None

    raw_answer = extract_json_field_from_dict(
        payload,
        ANSWER_FIELD_KEYS,
    )
    if raw_answer is None:
        return None
    if isinstance(raw_answer, int):
        return index_to_letter(raw_answer)
    if not isinstance(raw_answer, str):
        return None

    normalized = raw_answer.strip().upper()
    if not normalized:
        return None
    if normalized.isdigit():
        letter = index_to_letter(int(normalized))
        if letter is not None:
            return letter
    if len(normalized) == 1 and normalized in UPPERCASE_LETTERS:
        return normalized
    return None


def index_to_letter(index: int) -> str | None:
    if 0 <= index < len(UPPERCASE_LETTERS):
        return UPPERCASE_LETTERS[index]
    return None


def extract_json_field(text: str, keys: tuple[str, ...]) -> str | None:
    payload = parse_json_payload(text)
    if payload is None:
        return None

    value = extract_json_field_from_dict(payload, keys)
    if isinstance(value, str):
        return value
    return None


def extract_json_field_from_dict(payload: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def parse_json_payload(text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None
