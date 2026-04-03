from __future__ import annotations

from ..errors.base import ErrorGenerator

from .base import Question


class LlmzszlQuestion(Question):
    def __init__(self, question: str, answer: str) -> None:
        self.question = question
        self.answers = [answer]

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        question = error_generator.apply(self.question)

        return (
            "Odpowiedz na poniższe pytanie krótko i zwięźle.\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"WYRAŻENIE\"}\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
        )
