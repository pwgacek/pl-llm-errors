from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Question


class BBHOpenQuestion(Question):
    def __init__(self, text: str, answers: list[str]) -> None:
        self.text = text
        self.answers = answers

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        text = error_generator.apply(self.text)
        return (
            "Przemyśl swoją odpowiedź krok po kroku.\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"NAZWY OBIEKTÓW ODDZIELONE PRZECINKAMI\"}\n"
            f"<PYTANIE>{text}</PYTANIE>\n"
        )
