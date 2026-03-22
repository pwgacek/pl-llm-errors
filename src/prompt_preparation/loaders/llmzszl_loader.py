import json
from questions import LlmzszlQuestion, Question
from .base import Loader
from pathlib import Path


TARGET_TYPE = "Egzaminy Maturalne"


class LLMZSZLLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        questions: list[Question] = []

        lines = self._load_lines(path)

        for line in lines:
            record = json.loads(line)

            if (
                record.get("type") == TARGET_TYPE
            ):
                question_text = str(record.get("question", ""))
                answers = list(record.get("answers", []))
                correct_index = int(record.get("correct_answer_index", -1))

                if not question_text or not answers or correct_index < 0 or correct_index >= len(answers):
                    raise ValueError(f"Invalid LLMZSZL record, missing question text, answers, or correct index: {line}")

                questions.append(LlmzszlQuestion(question_text, answers, correct_index))

        return self._deterministic_sample(questions, num_samples, seed)
