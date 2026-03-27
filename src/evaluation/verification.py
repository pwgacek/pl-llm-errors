from __future__ import annotations

from collections import Counter
import json
import re
import string

UPPERCASE_LETTERS = string.ascii_uppercase
ANSWER_FIELD_KEYS = ("odpowiedź", "odpowiedz", "answer")
VerificationResult = tuple[float, bool]


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

    return 0.0, True


def verify_multiple_choice_index(raw: str, expected: dict) -> VerificationResult:
    predicted = extract_letter(raw)
    if predicted is None:
        return 0.0, True

    correct_letter = index_to_letter(expected["correct_index"])
    if correct_letter is None:
        return 0.0, True

    return (1.0, False) if predicted == correct_letter else (0.0, False)


def verify_multiple_choice_letter(raw: str, expected: dict) -> VerificationResult:
    predicted = extract_letter(raw)
    if predicted is None:
        return 0.0, True

    return (1.0, False) if predicted == expected["correct_letter"].upper() else (0.0, False)


def verify_open_contained(raw: str, expected: dict) -> VerificationResult:
    answer = extract_json_field(raw, ANSWER_FIELD_KEYS)
    if answer is None:
        normalized_answer = normalize_for_f1(raw)
    else:
        normalized_answer = normalize_for_f1(answer)

    if not normalized_answer:
        return 0.0, True

    normalized_expected = [
        normalize_for_f1(accepted)
        for accepted in expected["accepted_answers"]
        if isinstance(accepted, str)
    ]
    normalized_expected = [candidate for candidate in normalized_expected if candidate]

    if not normalized_expected:
        return 0.0, True

    best_f1 = max(compute_f1_score(normalized_answer, candidate) for candidate in normalized_expected)
    return best_f1, False


def verify_entailment(raw: str, expected: dict) -> VerificationResult:
    answer = extract_json_field(raw, ANSWER_FIELD_KEYS)
    if answer is None:
        normalized_answer = raw.strip().upper()
    else:
        normalized_answer = answer.strip().upper()

    if normalized_answer not in {"NEUTRAL", "CONTRADICTION", "ENTAILMENT"}:
        return 0.0, True

    return (1.0, False) if normalized_answer == expected["judgment"].upper() else (0.0, False)


def normalize_for_f1(text: str) -> str:
    lowered = text.strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return " ".join(cleaned.split())


def compute_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2.0 * precision * recall) / (precision + recall)


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
