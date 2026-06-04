from __future__ import annotations

from ..errors.base import ErrorGenerator
from .base import Prompt, render_template


class IFEvalPrompt(Prompt):
    def __init__(
        self,
        prompt: str,
        instruction_id_list: list[str],
        kwargs: list[dict[str, object]],
    ) -> None:
        self.prompt = prompt
        self.instruction_id_list = instruction_id_list
        self.kwargs = kwargs

    def build_prompt(self, error_generator: ErrorGenerator) -> str:
        prompt_text = error_generator.apply(self.prompt)
        return render_template("ifeval", {"prompt": prompt_text})
