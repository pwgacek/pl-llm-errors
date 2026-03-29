import json
from ..questions import LlmzszlQuestion, Question
from .base import Loader
from pathlib import Path


TARGET_TYPE = "Egzaminy Maturalne"


class LLMZSZLLoader(Loader):
    def __init__(self, answer_permutation: list[int] | None = None) -> None:
        self.answer_permutation = answer_permutation

    @staticmethod
    def _validate_permutation(permutation: list[int], num_answers: int) -> None:
        expected = list(range(num_answers))
        if sorted(permutation) != expected:
            raise ValueError(
                f"Invalid LLMZSZL answer_permutation={permutation}, expected permutation of {expected}"
            )

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

                if self.answer_permutation is not None:
                    self._validate_permutation(self.answer_permutation, len(answers))
                    answers = [answers[idx] for idx in self.answer_permutation]
                    correct_index = self.answer_permutation.index(correct_index)

                questions.append(LlmzszlQuestion(question_text, answers, correct_index))

        return self._deterministic_sample(questions, num_samples, seed)
