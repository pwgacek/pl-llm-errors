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
        choices = self._format_lettered_choices(answers)

        return (
            "Odpowiedz na poniższe pytanie, wybierając poprawną odpowiedź spośród podanych.\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
            f"<ODPOWIEDZI>{choices}</ODPOWIEDZI>\n"
        )
