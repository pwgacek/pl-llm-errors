from __future__ import annotations

import json

from .base import Question, VerificationResult

VALID_LETTERS = frozenset("ABCDE")


class LDEKQuestion(Question):
    def __init__(self, question: str, answers: list[str], correct_answer: str) -> None:
        self.question = question
        self.answers = answers
        self.correct_answer = correct_answer.upper()

    def verify_answer(self, llm_answer: str) -> VerificationResult:
        predicted = self._extract_answer_letter(llm_answer)
        if predicted is None:
            return VerificationResult.ERROR

        if predicted == self.correct_answer:
            return VerificationResult.CORRECT

        return VerificationResult.INCORRECT

    @staticmethod
    def _extract_answer_letter(text: str) -> str | None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None

        answer_raw = None
        for key in ("odpowiedź", "odpowiedz", "answer"):
            value = payload.get(key)
            if value is not None:
                answer_raw = value
                break

        if not isinstance(answer_raw, str):
            return None

        normalized = answer_raw.strip().upper()
        return normalized if normalized in VALID_LETTERS else None

    def build_prompt(self) -> str:
        letters = ["A", "B", "C", "D", "E"]
        choices = "\n".join(f"{letters[i]}. {answer}" for i, answer in enumerate(self.answers))

        return (
            "Przemyśl pytanie krok po kroku, a następnie wybierz poprawną odpowiedź spośród możliwych.\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{self.question}</PYTANIE>\n"
            f"<ODPOWIEDZI>{choices}</ODPOWIEDZI>\n"
        )
