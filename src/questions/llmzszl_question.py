from __future__ import annotations

import json
from typing import Any

from .base import Question, VerificationResult


class LlmzszlQuestion(Question):
    def __init__(self, question: str, answers: list[str], correct_answer_index: int) -> None:
        self.question = question
        self.answers = answers
        self.correct_answer_index = correct_answer_index

    def verify_answer(self, llm_answer: str) -> VerificationResult:
        predicted_index = self._extract_answer_index(llm_answer)
        if predicted_index is None:
            return VerificationResult.ERROR

        if predicted_index == self.correct_answer_index:
            return VerificationResult.CORRECT

        return VerificationResult.INCORRECT

    @staticmethod
    def _extract_answer_index(text: str) -> int | None:
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

        if isinstance(answer_raw, int) and 0 <= answer_raw <= 3:
            return answer_raw

        if not isinstance(answer_raw, str):
            return None

        normalized = answer_raw.strip().upper()
        return {"A": 0, "B": 1, "C": 2, "D": 3}.get(normalized)
    

    def build_prompt(self) -> str:
        question = self.question
        answers = self.answers

        letters = ["A", "B", "C", "D"]
        choices = "\n".join(f"{letters[i]}. {answer}" for i, answer in enumerate(answers))

        return (
            "Przemyśl pytanie krok po kroku, a następnie wybierz poprawną odpowiedź spośród 4 możliwych.\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
            f"<ODPOWIEDZI>{choices}</ODPOWIEDZI>\n"
        )
