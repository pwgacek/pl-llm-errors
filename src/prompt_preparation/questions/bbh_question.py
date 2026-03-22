from __future__ import annotations

from errors.base import ErrorGenerator

from .base import Question


class BBHQuestion(Question):
    def __init__(self, text: str, options: list[str], answer: str) -> None:
        super().__init__()
        self.text = text
        self.options = options
        self.answer = answer

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        options = [error_generator.apply(option) for option in self.options]
        choices = self._format_lettered_choices(options)
        text = error_generator.apply(self.text)

        return (
            "Wybierz poprawną odpowiedź spośród podanych.\n"
            "Przemyśl swoją odpowiedź krok po kroku.\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{text}</PYTANIE>\n"
            f"<ODPOWIEDZI>\n{choices}\n</ODPOWIEDZI>\n"
        )