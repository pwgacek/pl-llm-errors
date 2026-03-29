from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TASK_ORDER_PROMPTS: dict[str, str] = {
    "birds": "Wymień ptaki w kolejności od lewej do prawej.",
    "books": "Wymień książki w kolejności od lewej do prawej.",
    "vehicles": "Wymień pojazdy w kolejności od najnowszego do najstarszego.",
    "fruits": "Wymień owoce w kolejności od najtańszego do najdroższego.",
    "golfers": "Wymień golfistów w kolejności od 1. miejsca do 7. miejsca.",
}


def _normalize_english_entity(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned)
    return cleaned


def _normalize_polish_entity_for_mapping(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+ksi[aą]żka$", "", cleaned)
    return cleaned


def _split_entities(raw: str) -> list[str]:
    normalized = re.sub(r"\s+(i|and)\s+", ", ", raw, flags=re.IGNORECASE)
    parts: list[str] = []
    for part in normalized.split(","):
        cleaned = re.sub(r"^(i|and)\s+", "", part.strip(), flags=re.IGNORECASE)
        if cleaned:
            parts.append(cleaned)
    return parts


def _extract_entities_from_prompt(text: str) -> list[str]:
    match = re.search(r":\s*(.*?)\.", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Cannot extract entity list from prompt")
    return _split_entities(match.group(1))


def _extract_polish_stem_without_options(text: str) -> str:
    marker = "\nOpcje:"
    if marker in text:
        return text.split(marker, 1)[0].strip()

    marker_alt = "Opcje:"
    if marker_alt in text:
        return text.split(marker_alt, 1)[0].strip()

    raise ValueError("Polish prompt does not contain 'Opcje:' section")


def _detect_task_type(stem: str) -> str:
    lowered = stem.lower()

    if re.search(r"\bptak", lowered):
        return "birds"
    if re.search(r"\bksi[aą]ż", lowered):
        return "books"
    if re.search(r"\bgolfist", lowered):
        return "golfers"
    if re.search(r"\bowoc", lowered):
        return "fruits"
    if re.search(r"\bpojazd", lowered):
        return "vehicles"

    raise ValueError("Cannot detect task type from prompt stem")


def _build_task_order_question(task_type: str) -> str:
    return TASK_ORDER_PROMPTS[task_type]


def _build_aliases(entities: list[str], task_type: str) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for entity in entities:
        canonical = entity.strip().lower()
        variants = {
            canonical,
            canonical.replace("  ", " "),
        }

        if task_type == "books" and not canonical.endswith(" książka"):
            variants.add(f"{canonical} książka")

        aliases[canonical] = sorted(variants)
    return aliases


def _numbered_line_sequence(entities: list[str]) -> str:
    return "\n".join(f"{idx + 1}. {entity}" for idx, entity in enumerate(entities))


def _numbered_inline_sequence(entities: list[str]) -> str:
    return ", ".join(f"{idx + 1}. {entity}" for idx, entity in enumerate(entities))


def _build_accepted_answers(solution_order: list[str], aliases: dict[str, list[str]]) -> list[str]:
    alias_variants_per_slot: list[list[str]] = []
    for entity in solution_order:
        canonical = entity.strip().lower()
        variants = aliases.get(canonical)
        if not variants:
            variants = [canonical]
        alias_variants_per_slot.append(variants)

    accepted: set[str] = set()

    def add_for_sequence(seq: list[str]) -> None:
        accepted.add(", ".join(seq))
        accepted.add(" ".join(seq))
        accepted.add(" -> ".join(seq))
        accepted.add(" > ".join(seq))
        accepted.add(" | ".join(seq))
        accepted.add(_numbered_inline_sequence(seq))
        accepted.add(_numbered_line_sequence(seq))

    # Canonical sequence
    canonical_seq = [entity.strip().lower() for entity in solution_order]
    add_for_sequence(canonical_seq)

    # Mixed alias variants (first alias from each slot)
    first_alias_seq = [variants[0] for variants in alias_variants_per_slot]
    add_for_sequence(first_alias_seq)

    # Another mixed alias variants (last alias from each slot)
    last_alias_seq = [variants[-1] for variants in alias_variants_per_slot]
    add_for_sequence(last_alias_seq)

    return sorted(answer for answer in accepted if answer.strip())


def _build_static_en_to_pl_dictionary(
    polish_lines: list[str],
    english_solved: list[dict[str, object]],
) -> dict[str, str]:
    en_to_pl: dict[str, str] = {}

    for idx, raw in enumerate(polish_lines):
        polish_record = json.loads(raw)
        english_record = english_solved[idx]

        polish_input = str(polish_record.get("input", ""))
        english_input = str(english_record.get("input", ""))
        if not polish_input or not english_input:
            raise ValueError(f"Missing input fields in record {idx}")

        polish_entities = _extract_entities_from_prompt(polish_input)
        english_entities = _extract_entities_from_prompt(english_input)
        if len(polish_entities) != 7 or len(english_entities) != 7:
            raise ValueError(f"Expected 7 entities in record {idx}")

        for english_entity, polish_entity in zip(english_entities, polish_entities):
            key = _normalize_english_entity(english_entity)
            value = _normalize_polish_entity_for_mapping(polish_entity)

            existing = en_to_pl.get(key)
            if existing is not None and existing != value:
                raise ValueError(
                    "Conflicting EN->PL mapping for "
                    f"'{english_entity}' (normalized: '{key}'): '{existing}' vs '{value}'"
                )
            en_to_pl[key] = value

    return en_to_pl


def _map_solution_order_to_polish(
    english_solution_order: list[str],
    en_to_pl: dict[str, str],
) -> list[str]:
    mapped: list[str] = []
    for english_entity in english_solution_order:
        key = _normalize_english_entity(english_entity)
        if key not in en_to_pl:
            raise ValueError(f"English solution entity '{english_entity}' not found in static EN->PL dictionary")
        mapped.append(en_to_pl[key])
    return mapped


def build_open_bbh_dataset(
    polish_path: Path,
    english_solved_path: Path,
    output_path: Path,
) -> None:
    polish_lines = [line for line in polish_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    english_solved = json.loads(english_solved_path.read_text(encoding="utf-8"))

    if len(polish_lines) != len(english_solved):
        raise ValueError(
            f"Different number of records: polish={len(polish_lines)} english_solved={len(english_solved)}"
        )

    en_to_pl = _build_static_en_to_pl_dictionary(polish_lines=polish_lines, english_solved=english_solved)

    out_lines: list[str] = []

    for idx, raw in enumerate(polish_lines):
        polish_record = json.loads(raw)
        english_record = english_solved[idx]

        polish_input = str(polish_record.get("input", ""))
        english_input = str(english_record.get("input", ""))
        english_solution_order = list(english_record.get("solution_order", []))

        if not polish_input or not english_input or not english_solution_order:
            raise ValueError(f"Missing fields in record {idx}")

        polish_entities = _extract_entities_from_prompt(polish_input)
        if len(polish_entities) != 7:
            raise ValueError(f"Expected 7 entities in record {idx}")

        polish_solution_order = _map_solution_order_to_polish(
            english_solution_order=english_solution_order,
            en_to_pl=en_to_pl,
        )

        stem = _extract_polish_stem_without_options(polish_input)
        task_type = _detect_task_type(stem)
        order_question = _build_task_order_question(task_type)
        open_prompt = f"{stem}\n{order_question}"

        aliases = _build_aliases(polish_entities, task_type=task_type)
        accepted_answers = _build_accepted_answers(polish_solution_order, aliases)

        out_record = {
            "input": open_prompt,
            "target": accepted_answers,
        }
        out_lines.append(json.dumps(out_record, ensure_ascii=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build open BBH-PL dataset without options, with ordering targets."
    )
    parser.add_argument(
        "--polish",
        type=Path,
        default=Path("datasets/bbh-logical-deduction-seven-objects-pl.jsonl"),
        help="Path to Polish BBH JSONL dataset.",
    )
    parser.add_argument(
        "--english-solved",
        type=Path,
        default=Path("src/scripts/translate/bbh-english-solved.json"),
        help="Path to solved English BBH JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/bbh-logical-deduction-seven-objects-pl-open.jsonl"),
        help="Output path for open-form BBH JSONL.",
    )
    args = parser.parse_args()

    build_open_bbh_dataset(
        polish_path=args.polish,
        english_solved_path=args.english_solved,
        output_path=args.output,
    )
    print(f"Saved open dataset to: {args.output}")


if __name__ == "__main__":
    main()
