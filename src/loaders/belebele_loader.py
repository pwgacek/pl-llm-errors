from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from questions import MultipleChoiceQuestion, Question

from .base import Loader

JSONL_PATH = Path("datasets/belebele-pol.jsonl")


class BelebeleLoader(Loader):
    def load(self) -> list[Question]:
        """Load questions from Belebele JSONL file."""
        path = JSONL_PATH
        questions: list[Question] = []

        with path.open("r", encoding="utf-8") as file:
            for line_no, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question_text = str(record.get("question", ""))
                answers = [
                    str(record.get("mc_answer1", "")),
                    str(record.get("mc_answer2", "")),
                    str(record.get("mc_answer3", "")),
                    str(record.get("mc_answer4", "")),
                ]
                correct_num = int(record.get("correct_answer_num", 0))
                correct_index = correct_num - 1  # Shift from 1-4 to 0-3
                questions.append(MultipleChoiceQuestion(question_text, answers, correct_index))

        return questions