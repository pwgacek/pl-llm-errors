from __future__ import annotations
import ast
from ..questions import PolQAQuestion, Question
from .base import Loader
from pathlib import Path

class PolQALoader(Loader):

    @staticmethod
    def _parse_answers(raw: str) -> list[str]:
        parsed = ast.literal_eval(raw)
        return [str(a) for a in parsed]
 
        

    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        questions: list[Question] = []
        seen: set[tuple[str, str]] = set()

        rows = self._load_rows(path)
        for row in rows:
            if row.get("relevant", "").strip() == "True":
                question_text = row.get("question", "").strip()
                context = row.get("passage_text", "").strip()
                answers = self._parse_answers(row.get("answers", "[]"))

                if not question_text or not context or not answers:
                    raise ValueError(f"Invalid POLQA record, missing question text or passage text: {row}")

                # handle duplicates as duplicated fields does not always is set correctly
                key = (question_text, context)
                if key in seen:
                    continue
                seen.add(key)

                questions.append(PolQAQuestion(question_text, context, answers))

        return self._deterministic_sample(questions, num_samples, seed)