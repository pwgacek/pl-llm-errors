from __future__ import annotations

from ..errors.base import ErrorGenerator

from .base import Question


class PolQAQuestion(Question):
    def __init__(self, question: str, context: str, answers: list[str]) -> None:
        self.question = question
        self.context = context
        self.answers = answers

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        question = error_generator.apply(self.question)
        context = error_generator.apply(self.context)

        return (
            "Odpowiedz na pytanie korzystając wyłącznie z dostarczonego kontekstu.\n"
            "Zwróć jedno słowo lub krótką frazę.\n"
            "Nie odpowiadaj pełnym zdaniem.\n"
            "Nie podawaj wyjaśnień.\n"
            f"<KONTEKST>{context}</KONTEKST>\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
        )