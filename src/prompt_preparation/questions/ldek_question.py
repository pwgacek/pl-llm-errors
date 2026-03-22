from __future__ import annotations

from errors.base import ErrorGenerator

from .base import Question


class LDEKQuestion(Question):
    def __init__(self, question: str, answers: list[str], correct_answer: str) -> None:
        self.question = question
        self.answers = answers
        self.correct_answer = correct_answer

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        answers = [error_generator.apply(answer) for answer in self.answers]
        choices = self._format_lettered_choices(answers)
        question = error_generator.apply(self.question)
        return (
            "Wybierz poprawną odpowiedź spośród możliwych.\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
            f"<ODPOWIEDZI>{choices}</ODPOWIEDZI>\n"
        )
