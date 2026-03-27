from __future__ import annotations

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

    if kind == "open_short_answer":
        return verify_open_short_answer(raw, expected)

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


def verify_open_short_answer(raw: str, expected: dict) -> VerificationResult:
    answer = extract_json_field(raw, ANSWER_FIELD_KEYS)
    if answer is None:
        normalized_answer = normalize(raw)
    else:
        normalized_answer = normalize(answer)

    normalized_expected = [ normalize(accepted) for accepted in expected["accepted_answers"]]

    if not normalized_expected or not normalized_answer:
        return 0.0, True

    best_score = max(
        levenshtein_score(normalized_answer, candidate)
        for candidate in normalized_expected
    )
    return best_score, False


def normalize(text: str) -> str:
    return text.strip().lower()


def levenshtein_distance(s1: str, s2: str) -> int:
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )

    return dist[rows - 1][cols - 1]


def levenshtein_score(pred: str, ref: str) -> float:
    max_len = max(len(pred), len(ref))
    distance = levenshtein_distance(pred, ref)
    return 1.0 - (distance / max_len)




def verify_entailment(raw: str, expected: dict) -> VerificationResult:
    answer = extract_json_field(raw, ANSWER_FIELD_KEYS)
    if answer is None:
        normalized_answer = raw.strip().upper()
    else:
        normalized_answer = answer.strip().upper()

    if normalized_answer not in {"NEUTRAL", "CONTRADICTION", "ENTAILMENT"}:
        return 0.0, True

    return (1.0, False) if normalized_answer == expected["judgment"].upper() else (0.0, False)


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
