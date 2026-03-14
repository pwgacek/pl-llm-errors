from __future__ import annotations

from errors.base import ErrorGenerator

from .base import Question


class LlmzszlQuestion(Question):
    def __init__(self, question: str, answers: list[str], correct_answer_index: int) -> None:
        self.question = question
        self.answers = answers
        self.correct_answer_index = correct_answer_index

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        question = error_generator.apply(self.question)
        answers = [error_generator.apply(answer) for answer in self.answers]

        letters = ["A", "B", "C", "D"]
        choices = "\n".join(f"{letters[i]}. {answer}" for i, answer in enumerate(answers))

        return (
            "Przemyśl pytanie krok po kroku, a następnie wybierz poprawną odpowiedź spośród podanych.\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
            f"<ODPOWIEDZI>{choices}</ODPOWIEDZI>\n"
        )
