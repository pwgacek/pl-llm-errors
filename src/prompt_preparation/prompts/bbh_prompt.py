from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Prompt, render_template


class BBHPrompt(Prompt):
    def __init__(self, text: str, correct_order: list[list[str]]) -> None:
        self.text = text
        self.correct_order = correct_order

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        text = error_generator.apply(self.text)
        return render_template("bbh", {"text": text})