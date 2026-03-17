from __future__ import annotations

import json
from pathlib import Path

from questions import LlmzszlQuestion, Question

from .base import Loader


TARGET_TYPE = "Egzaminy Maturalne"


class LLMZSZLLoader(Loader):
    def load(self, num_samples: int | None = None, seed: int = 42) -> list[Question]:
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
                ):
                    question_text = str(record.get("question", ""))
                    answers = list(record.get("answers", []))
                    correct_index = int(record.get("correct_answer_index", -1))
                    questions.append(LlmzszlQuestion(question_text, answers, correct_index))

        return self._deterministic_sample(questions, num_samples, seed)
