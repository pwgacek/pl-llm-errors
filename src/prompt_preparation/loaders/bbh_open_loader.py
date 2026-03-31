from __future__ import annotations

import json
from pathlib import Path

from .base import Loader
from ..questions import BBHOpenQuestion, Question


class BBHOpenLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        questions: list[Question] = []

        lines = self._load_lines(path)
        lines = self._deterministic_sample(lines, num_samples, seed)

        for line in lines:
            record = json.loads(line)
            input_text = str(record.get("input", "")).strip()
            correct_order = record.get("correct_order", [])

            if not input_text:
                raise ValueError(f"Invalid BBH open record, missing input: {line}")

            if not isinstance(correct_order, list) or not correct_order:
                raise ValueError(f"Invalid BBH open record, missing correct_order list: {line}")

            normalized_correct_order: list[list[str]] = []
            for slot in correct_order:
                if not isinstance(slot, list) or not slot:
                    raise ValueError(f"Invalid BBH open record, malformed correct_order slot: {line}")
                normalized_slot = [str(variant).strip() for variant in slot if str(variant).strip()]
                if not normalized_slot:
                    raise ValueError(f"Invalid BBH open record, empty correct_order slot: {line}")
                normalized_correct_order.append(normalized_slot)

            questions.append(
                BBHOpenQuestion(
                    text=input_text,
                    correct_order=normalized_correct_order,
                )
            )

        return questions
