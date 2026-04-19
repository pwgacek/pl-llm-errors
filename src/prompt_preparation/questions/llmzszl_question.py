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
            "Odpowiedz na poniższe pytanie, podając uzasadnienie krok po kroku.\n"
            "\n"
            "Zasady:\n"
            "- zapisuj kolejne kroki rozumowania,\n"
            "- wykonuj niezbędne obliczenia,\n"
            "- nie pomijaj istotnych przekształceń,\n"
            "- uzasadnienie ma być zwięzłe, ale pełne.\n"
            "\n"
            "Na końcu podaj odpowiedź w osobnej linii w formacie:\n"
            "<ODPOWIEDŹ>...</ODPOWIEDŹ>\n"
            "\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
        )
