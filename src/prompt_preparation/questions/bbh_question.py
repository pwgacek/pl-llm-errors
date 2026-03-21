from .base import Question


class BBHQuestion(Question):
    def __init__(self, text: str, options: list[str], answer: str) -> None:
        super().__init__()
        self.text = text
        self.options = options
        self.answer = answer

    def build_prompt(self, error_generator) -> str:
        letters = ["A", "B", "C", "D", "E", "F", "G"]
        options = [error_generator.apply(option) for option in self.options]
        choices = "\n".join(f"{letters[i]}. {option}" for i, option in enumerate(options))
        text = error_generator.apply(self.text)

        return (
            "Wybierz poprawną odpowiedź spośród podanych.\n"
            "Przemyśl swoją odpowiedź krok po kroku.\n"
            "Odpowiedź powinna mieć format: {\"odpowiedź\": \"LITERA\"}\n"
            f"<PYTANIE>{text}</PYTANIE>\n"
            f"<ODPOWIEDZI>\n{choices}\n</ODPOWIEDZI>\n"
            
        )