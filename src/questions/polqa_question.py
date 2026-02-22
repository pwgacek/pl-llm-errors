from __future__ import annotations

import ast

from .base import Question, VerificationResult


class PolqaQuestion(Question):
    """PolQA open-ended question with a context passage and a list of accepted free-text answers."""

    def __init__(self, question: str, context: str, answers: list[str]) -> None:
        self.question = question
        self.context = context
        self.answers = answers

    def verify_answer(self, llm_answer: str) -> VerificationResult:
        normalized = llm_answer.strip().lower()
        if not normalized:
            return VerificationResult.ERROR

        for accepted in self.answers:
            if accepted.strip().lower() in normalized:
                return VerificationResult.CORRECT

        return VerificationResult.INCORRECT

    @staticmethod
    def parse_answers(raw: str) -> list[str]:
        """Parse a Python list serialized as a string, e.g. \"['tak', 'yes']\"."""
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        except (ValueError, SyntaxError):
            pass
        return [raw]
    
    def build_prompt(self) -> str:
        question = self.question
        answers = self.answers

        return (
            "Odpowiedz na pytanie na podstawie załączonego kontekstu.\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"WYRAŻENIE\"}\n"
            f"<KONTEKST>{self.context}</KONTEKST>\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
            
        )