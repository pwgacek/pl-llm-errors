from __future__ import annotations

import csv
from pathlib import Path

from questions import OpenQuestion, Question

from .base import Loader

JSONL_PATH = Path("datasets/polqa.csv")

class PolQALoader(Loader):
    def load(self) -> list[Question]:
        """Load questions from the PolQA CSV file, keeping only relevant=True rows."""
        
        questions: list[Question] = []
        seen: set[tuple[str, str]] = set()

        with JSONL_PATH.open("r", encoding="utf-8") as file:
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

                answers = OpenQuestion.parse_answers(row.get("answers", "[]"))
                questions.append(OpenQuestion(question_text, context, answers))

        return questions