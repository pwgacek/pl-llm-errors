from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Prompt, render_template


class MatematykaRozszerzonaCKEPrompt(Prompt):
    def __init__(self, text: str, key: str, points: int | str) -> None:
        self.text = text
        self.key = key
        self.points = points

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        text = error_generator.apply(self.text)
        return render_template("matematyka_rozszerzona_cke", {"text": text})
