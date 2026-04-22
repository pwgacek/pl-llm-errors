from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Prompt, render_template


class PolQAPrompt(Prompt):
    def __init__(self, question: str, context: str, answers: list[str]) -> None:
        self.question = question
        self.context = context
        self.answers = answers

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        question = error_generator.apply(self.question)
        context = error_generator.apply(self.context)
        return render_template(
            "polqa",
            {
                "context": context,
                "question": question,
            },
        )