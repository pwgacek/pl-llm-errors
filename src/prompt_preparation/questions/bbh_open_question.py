from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Question


class BBHOpenQuestion(Question):
    def __init__(self, text: str, correct_order: list[list[str]]) -> None:
        self.text = text
        self.correct_order = correct_order

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        text = error_generator.apply(self.text)
        return (
            "Przemyśl swoją odpowiedź krok po kroku.\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"NAZWY OBIEKTÓW ODDZIELONE PRZECINKAMI\"}\n"
            f"<PYTANIE>{text}</PYTANIE>\n"
        )
