from __future__ import annotations

import json
import re
import string

UPPERCASE_LETTERS = string.ascii_uppercase
ANSWER_FIELD_KEYS = ("odpowiedź", "odpowiedz", "answer")
VerificationResult = tuple[float, bool]
BBH_MIN_ENTITY_SCORE = 0.8


def verify_response(raw: str, expected: dict) -> VerificationResult:
    kind = expected.get("type")

    if kind == "open_short_answer":
        return verify_open_short_answer(raw, expected)

    if kind == "bbh_position_match":
        return verify_bbh_position_match(raw, expected)


    return 0.0, True

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


def verify_bbh_position_match(raw: str, expected: dict) -> VerificationResult:
    correct_order_raw = expected.get("correct_order")
    if not isinstance(correct_order_raw, list) or not correct_order_raw:
        return 0.0, True

    normalized_correct_order: list[list[str]] = []
    for slot in correct_order_raw:
        if not isinstance(slot, list) or not slot:
            return 0.0, True
        normalized_slot = [normalize(str(variant)) for variant in slot if normalize(str(variant))]
        if not normalized_slot:
            return 0.0, True
        normalized_correct_order.append(normalized_slot)

    predicted_entities = extract_order_entities(raw)

    target_keys = [slot[0] for slot in normalized_correct_order]
    predicted_keys = [
        map_predicted_entity_to_key(entity, target_keys)
        for entity in predicted_entities[: len(target_keys)]
    ]

    correct_positions = 0
    for idx, target_key in enumerate(target_keys):
        predicted_key = predicted_keys[idx] if idx < len(predicted_keys) else None
        if predicted_key == target_key:
            correct_positions += 1

    return correct_positions / len(target_keys), False


def normalize(text: str) -> str:
    return text.strip().lower()


def normalize_entity_token(text: str) -> str:
    normalized = normalize(text)
    normalized = re.sub(r"^\d+[\).:\-]\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" ,;|>")


def extract_order_entities(raw: str) -> list[str]:
    answer = extract_json_field(raw, ANSWER_FIELD_KEYS)
    source = answer if answer is not None else raw

    numbered_matches = re.findall(r"(?:^|\n)\s*\d+[\).:\-]\s*([^\n]+)", source, flags=re.IGNORECASE)
    if len(numbered_matches) >= 2:
        entities = [normalize_entity_token(match) for match in numbered_matches]
        return [entity for entity in entities if entity]

    normalized = source.replace("\n", ",")
    normalized = re.sub(r"\s*(?:->|>|\|)\s*", ",", normalized)
    normalized = re.sub(r"\s*;\s*", ",", normalized)
    parts = [normalize_entity_token(part) for part in normalized.split(",")]
    return [part for part in parts if part]


def map_predicted_entity_to_key(predicted_entity: str, keys: list[str]) -> str | None:
    normalized_predicted = normalize_entity_token(predicted_entity)
    if not normalized_predicted:
        return None

    best_key: str | None = None
    best_score = 0.0

    for key in keys:
        score = levenshtein_score(normalized_predicted, key)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= BBH_MIN_ENTITY_SCORE:
        return best_key
    return None


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
    if max_len == 0:
        return 1.0
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
