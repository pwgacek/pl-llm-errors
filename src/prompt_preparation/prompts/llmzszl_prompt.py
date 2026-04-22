from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Prompt, render_template


class LlmzszlPrompt(Prompt):
    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answers = [answer]

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        question = error_generator.apply(self.question)
        return render_template("llmzszl", {"question": question})