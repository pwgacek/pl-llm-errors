import json
from ..questions import LlmzszlQuestion, Question
from .base import Loader
from pathlib import Path


class LLMZSZLLoader(Loader):
    def load(self, path: Path, num_samples: int, seed: int) -> list[Question]:
        questions: list[Question] = []

        lines = self._load_lines(path)

        for line in lines:
            record = json.loads(line)

            question_text = str(record.get("question", ""))
            answer = str(record.get("answer", ""))

            if not question_text or not answer:
                raise ValueError(f"Invalid LLMZSZL record, missing question or answer: {line}")

            questions.append(LlmzszlQuestion(question_text, answer))

        return self._deterministic_sample(questions, num_samples, seed)
