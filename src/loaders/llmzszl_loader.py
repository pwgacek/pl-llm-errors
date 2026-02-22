from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from questions import LlmzszlQuestion, Question

from .base import Loader


TARGET_TYPE = "Egzaminy Maturalne"
TARGET_NAME = "Matematyka"


class LLMZSZLLoader(Loader):
    def load(self) -> list[Question]:
        """Load questions from a JSONL file filtered to Matura Math entries."""
        questions: list[Question] = []

        with Path("datasets/llmzszl.jsonl").open("r", encoding="utf-8") as file:
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
                    question_text = str(record.get("question", ""))
                    answers = list(record.get("answers", []))
                    correct_index = int(record.get("correct_answer_index", -1))
                    questions.append(LlmzszlQuestion(question_text, answers, correct_index))

        return questions
