from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Prompt, render_template


class LlmzszlMCPrompt(Prompt):
    def __init__(self, question: str, answers: list[str], correct_answer_index: int | None = None) -> None:
        self.question = question
        self.answers = answers
        self.correct_answer_index = correct_answer_index

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        question = error_generator.apply(self.question)
        answers = [error_generator.apply(a) for a in self.answers]
        choices = self._format_lettered_choices(answers)

        return render_template(
            "llmzszl_mc",
            {
                "question": question,
                "choices": choices,
            },
        )
