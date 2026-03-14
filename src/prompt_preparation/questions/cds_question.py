from __future__ import annotations

from errors.base import ErrorGenerator

from .base import Question


class CdsQuestion(Question):
    def __init__(self, sentence_a: str, sentence_b: str, entailment_judgment: str) -> None:
        self.sentence_a = sentence_a
        self.sentence_b = sentence_b
        self.entailment_judgment = entailment_judgment.upper()

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        sentence_a = error_generator.apply(self.sentence_a)
        sentence_b = error_generator.apply(self.sentence_b)
        return (
            "Przeanalizuj krok po kroku relację między dwoma zdaniami i określ, czy drugie zdanie jest konsekwencją pierwszego (ENTAILMENT), "
            "sprzeciwia się mu (CONTRADICTION), czy też nie ma jasnej relacji (NEUTRAL).\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"RELATION\"} gdzie RELATION to ENTAILMENT, CONTRADICTION lub NEUTRAL.\n\n"
            f"<ZDANIE_A>{sentence_a}</ZDANIE_A>\n"
            f"<ZDANIE_B>{sentence_b}</ZDANIE_B>\n"
        )