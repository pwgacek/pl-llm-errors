from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Question


class BBHQuestion(Question):
    def __init__(self, text: str, correct_order: list[list[str]]) -> None:
        self.text = text
        self.correct_order = correct_order

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        text = error_generator.apply(self.text)
        return (
            "Rozwiąż poniższe zadanie krok po kroku.\n"
            "\n"
            "Zasady:\n"
            "- zapisuj kolejne kroki rozumowania,\n"
            "- wykorzystuj wszystkie podane informacje,\n"
            "- rozumowanie ma być zwięzłe, ale kompletne.\n"
            "\n"
            "Na końcu podaj wynik jako uporządkowaną listę wszystkich obiektów w formacie:\n"
            "<ODPOWIEDŹ>obiekt1, obiekt2, obiekt3, obiekt4, obiekt5, obiekt6, obiekt7</ODPOWIEDŹ>\n"
            "\n"
            f"<ZADANIE>{text}</ZADANIE>\n"
        )
