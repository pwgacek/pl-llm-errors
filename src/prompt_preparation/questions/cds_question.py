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
            "Sklasyfikuj relację między przesłanką a hipotezą jako ENTAILMENT (wynikanie), CONTRADICTION (sprzeczność) lub NEUTRAL (neutralność).\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"RELATION\"}.\n\n"
            f"<ZDANIE_A>{sentence_a}</ZDANIE_A>\n"
            f"<ZDANIE_B>{sentence_b}</ZDANIE_B>\n"
        )