from __future__ import annotations

import csv
from pathlib import Path

from questions import PolqaQuestion, Question

from .base import Loader

class PolQALoader(Loader):
    def load(self, num_samples: int | None = None, seed: int = 42) -> list[Question]:
        """Load questions from the PolQA CSV file, keeping only relevant=True rows, optionally sampling deterministically."""
        questions: list[Question] = []
        seen: set[tuple[str, str]] = set()

        with Path("datasets/polqa.csv").open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("relevant", "").strip() != "True":
                    continue

                question_text = row.get("question", "").strip()
                context = row.get("passage_text", "").strip()
                key = (question_text, context)
                if key in seen:
                    continue
                seen.add(key)

                answers = PolqaQuestion.parse_answers(row.get("answers", "[]"))
                questions.append(PolqaQuestion(question_text, context, answers))

        return self._deterministic_sample(questions, num_samples, seed)