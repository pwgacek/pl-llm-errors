from .base import ErrorGenerator


class IdentityGenerator(ErrorGenerator):
    """Error generator that does nothing and can carry answer permutation metadata."""

    def __init__(
        self,
        answer_permutations: dict[str, list[int]] | None = None,
        temperature: float | None = None,
    ) -> None:
        self.answer_permutations = answer_permutations or {}
        self.temperature = temperature

    def apply(self, text: str) -> str:
        return text
