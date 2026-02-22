from __future__ import annotations

from .base import Question, VerificationResult


class CdsQuestion(Question):
    def __init__(self, sentence_a: str, sentence_b: str, entailment_judgment: str) -> None:
        self.sentence_a = sentence_a
        self.sentence_b = sentence_b
        self.entailment_judgment = entailment_judgment.upper()

    def verify_answer(self, llm_answer: str) -> VerificationResult:
        # Expected answers: NEUTRAL, CONTRADICTION, ENTAILMENT
        normalized = llm_answer.strip().upper()
        
        # Try to extract from JSON format first
        import json
        try:
            payload = json.loads(llm_answer)
            answer = payload.get("odpowiedź", "").strip().upper()
            if answer in ["NEUTRAL", "CONTRADICTION", "ENTAILMENT"]:
                normalized = answer
        except json.JSONDecodeError:
            pass

        if normalized == self.entailment_judgment:
            return VerificationResult.CORRECT
        elif normalized in ["NEUTRAL", "CONTRADICTION", "ENTAILMENT"]:
            return VerificationResult.INCORRECT
        else:
            return VerificationResult.ERROR

    def build_prompt(self) -> str:
        return (
            "Przeanalizuj krok po kroku relację między dwoma zdaniami i określ, czy drugie zdanie jest konsekwencją pierwszego (ENTAILMENT), "
            "sprzeciwia się mu (CONTRADICTION), czy też nie ma jasnej relacji (NEUTRAL).\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"RELATION\"} gdzie RELATION to ENTAILMENT, CONTRADICTION lub NEUTRAL.\n\n"
            f"<ZDANIE_A>{self.sentence_a}</ZDANIE_A>\n"
            f"<ZDANIE_B>{self.sentence_b}</ZDANIE_B>\n"
        )