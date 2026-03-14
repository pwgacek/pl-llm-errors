from __future__ import annotations

from errors.base import ErrorGenerator

from .base import Question


class LDEKQuestion(Question):
    def __init__(self, question: str, answers: list[str], correct_answer: str) -> None:
        self.question = question
        self.answers = answers
        self.correct_answer = correct_answer.upper()

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        letters = ["A", "B", "C", "D", "E"]
        answers = [error_generator.apply(answer) for answer in self.answers]
        choices = "\n".join(f"{letters[i]}. {answer}" for i, answer in enumerate(answers))
        question = error_generator.apply(self.question)
        return (
            "Przemyśl pytanie krok po kroku, a następnie wybierz poprawną odpowiedź spośród możliwych.\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
            f"<ODPOWIEDZI>{choices}</ODPOWIEDZI>\n"
        )
