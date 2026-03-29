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
            target = record.get("target", [])

            if not input_text or not isinstance(target, list) or not target:
                raise ValueError(f"Invalid BBH open record, missing input or target list: {line}")

            answers = [str(answer).strip() for answer in target if str(answer).strip()]
            if not answers:
                raise ValueError(f"Invalid BBH open record, empty target list: {line}")

            questions.append(BBHOpenQuestion(text=input_text, answers=answers))

        return questions
