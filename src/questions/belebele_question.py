from __future__ import annotations

from .llmzszl_question import LlmzszlQuestion


class BelebeleQuestion(LlmzszlQuestion):
    def __init__(self, question: str, answers: list[str], correct_answer_index: int, context: str) -> None:
        super().__init__(question, answers, correct_answer_index)
        self.context = context

    def build_prompt(self) -> str:
        question = self.question
        answers = self.answers

        letters = ["A", "B", "C", "D"]
        choices = "\n".join(f"{letters[i]}. {answer}" for i, answer in enumerate(answers))

        return (
            "Przemyśl pytanie krok po kroku korzystając z kontekstu, a następnie wybierz poprawną odpowiedź spośród 4 możliwych.\n"
            "Odpowiedz w formacie: {\"odpowiedź\": \"LITERA\"}\n"
            f"<KONTEKST>{self.context}</KONTEKST>\n"
            f"<PYTANIE>{question}</PYTANIE>\n"
            f"<ODPOWIEDZI>{choices}</ODPOWIEDZI>\n"
            
        )