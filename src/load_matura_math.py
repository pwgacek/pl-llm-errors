from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_TYPE = "Egzaminy Maturalne"
TARGET_NAME = "Matematyka"


def load_matura_math_questions(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load questions from a JSONL file filtered to Matura Math entries."""
    path = Path(jsonl_path)
    questions: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                record.get("type") == TARGET_TYPE
                and record.get("name") == TARGET_NAME
            ):
                questions.append(record)

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load only 'Egzaminy Maturalne' -> 'Matematyka' questions from a JSONL file."
    )
    parser.add_argument(
        "--input",
        default="datasets/llmzszl-test.jsonl",
        help="Path to input JSONL file (default: datasets/llmzszl-test.jsonl)",
    )
    args = parser.parse_args()

    filtered = load_matura_math_questions(args.input)
    print(f"Loaded {len(filtered)} Matematyka questions.")

    if filtered:
        print("First question:")
        print(filtered[0].get("question", ""))


if __name__ == "__main__":
    main()
